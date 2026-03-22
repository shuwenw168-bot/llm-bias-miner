"""
Subgroup Discovery for LLM Fairness Auditing
─────────────────────────────────────────────
Identifies demographic subgroups where the LLM's behavior deviates
most from the overall population. Uses beam search to efficiently
explore the space of attribute combinations.

This is the "needle in a haystack" method: even if the model looks
fair on average, certain *intersections* of demographics may reveal
significant bias (e.g., elderly + female + STEM topic).
"""

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class Subgroup:
    """A discovered subgroup with disproportionate LLM behavior."""
    description: dict[str, str]  # e.g., {"gender": "female", "age": "elderly"}
    size: int                    # Number of instances in subgroup
    target_mean: float           # Mean of target metric in subgroup
    population_mean: float       # Mean of target metric in full population
    quality_score: float         # WRAcc or chi-squared score
    effect_size: float           # Cohen's d
    p_value: float               # Statistical significance
    target_std: float = 0.0      # Std of target in subgroup

    @property
    def depth(self) -> int:
        return len(self.description)

    def __str__(self):
        desc = " & ".join(f"{k}={v}" for k, v in self.description.items())
        direction = "↑" if self.target_mean > self.population_mean else "↓"
        return (
            f"[{desc}] (n={self.size})\n"
            f"  mean={self.target_mean:.3f} vs pop={self.population_mean:.3f} {direction} "
            f"| d={self.effect_size:.2f} | p={self.p_value:.4f} | q={self.quality_score:.4f}"
        )


class SubgroupBiasMiner:
    """Discover demographic subgroups where the LLM shows disproportionate bias.

    Algorithm (Beam Search):
    1. Start with all single-attribute selectors: {gender=female}, {race=Black}, ...
    2. Score each by quality measure (WRAcc or chi-squared)
    3. Keep top-k (beam_width) subgroups
    4. Refine by adding one more attribute → {gender=female, age=elderly}
    5. Repeat until max_depth or no improvement
    """

    DEMOGRAPHIC_COLUMNS = [
        "gender", "race_ethnicity", "age_group", "socioeconomic", "religion"
    ]

    def __init__(
        self,
        beam_width: int = 10,
        max_depth: int = 3,
        min_subgroup_size: int = 30,
        quality_measure: str = "wracc",
    ):
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.min_subgroup_size = min_subgroup_size
        self.quality_measure = quality_measure

    def _get_selectors(self, df: pd.DataFrame) -> list[tuple[str, str]]:
        """Extract all possible (column, value) selectors from demographics."""
        selectors = []
        for col in self.DEMOGRAPHIC_COLUMNS:
            if col in df.columns:
                for val in df[col].dropna().unique():
                    selectors.append((col, val))

        # Also include context columns
        for col in ["category", "occupation", "field"]:
            if col in df.columns:
                for val in df[col].dropna().unique():
                    selectors.append((col, val))

        return selectors

    def _select_subgroup(
        self, df: pd.DataFrame, description: dict[str, str]
    ) -> pd.DataFrame:
        """Select rows matching the subgroup description."""
        mask = pd.Series(True, index=df.index)
        for col, val in description.items():
            mask &= df[col] == val
        return df[mask]

    def _compute_quality(
        self,
        subgroup_values: np.ndarray,
        population_values: np.ndarray,
        subgroup_size: int,
        population_size: int,
    ) -> float:
        """Compute quality score for a subgroup."""
        if self.quality_measure == "wracc":
            # Weighted Relative Accuracy
            p = subgroup_size / population_size
            sg_mean = np.mean(subgroup_values)
            pop_mean = np.mean(population_values)
            return p * (sg_mean - pop_mean)

        elif self.quality_measure == "chi2":
            # Chi-squared based quality
            sg_mean = np.mean(subgroup_values)
            pop_mean = np.mean(population_values)
            pop_std = np.std(population_values)
            if pop_std == 0:
                return 0.0
            return subgroup_size * ((sg_mean - pop_mean) / pop_std) ** 2

        else:
            raise ValueError(f"Unknown quality measure: {self.quality_measure}")

    def _compute_effect_size(
        self, subgroup_values: np.ndarray, complement_values: np.ndarray
    ) -> float:
        """Compute Cohen's d effect size."""
        n1, n2 = len(subgroup_values), len(complement_values)
        if n1 < 2 or n2 < 2:
            return 0.0

        mean1, mean2 = np.mean(subgroup_values), np.mean(complement_values)
        std1, std2 = np.std(subgroup_values, ddof=1), np.std(complement_values, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(
            ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
        )

        if pooled_std == 0:
            return 0.0
        return abs(mean1 - mean2) / pooled_std

    def discover(
        self,
        df: pd.DataFrame,
        target: str = "toxicity_score",
        return_all: bool = False,
    ) -> list[Subgroup]:
        """Run beam search subgroup discovery.

        Args:
            df: Enriched DataFrame from ResponseAnalyzer.
            target: Column name of the metric to analyze.
            return_all: If True, return all candidates (not just significant).

        Returns:
            List of Subgroup objects sorted by quality score.
        """
        print(f"Subgroup discovery on '{target}' "
              f"(beam={self.beam_width}, depth={self.max_depth})")

        selectors = self._get_selectors(df)
        population_values = df[target].dropna().values
        population_mean = np.mean(population_values)
        population_size = len(population_values)

        print(f"  Population: n={population_size}, mean={population_mean:.4f}")
        print(f"  Selectors: {len(selectors)} single attributes")

        # Initialize beam with single-attribute subgroups
        beam: list[tuple[dict, float]] = []  # (description, quality)

        for col, val in selectors:
            desc = {col: val}
            sg_df = self._select_subgroup(df, desc)
            if len(sg_df) < self.min_subgroup_size:
                continue

            sg_values = sg_df[target].dropna().values
            quality = self._compute_quality(
                sg_values, population_values, len(sg_values), population_size
            )
            beam.append((desc, abs(quality)))

        # Sort and keep top beam_width
        beam.sort(key=lambda x: x[1], reverse=True)
        beam = beam[: self.beam_width]

        # Iterative refinement
        for depth in range(2, self.max_depth + 1):
            candidates = []
            for desc, _ in beam:
                # Try adding each selector not already in description
                for col, val in selectors:
                    if col in desc:
                        continue
                    new_desc = {**desc, col: val}

                    # Skip if we've already seen this combination
                    desc_key = tuple(sorted(new_desc.items()))
                    if any(
                        tuple(sorted(d.items())) == desc_key
                        for d, _ in candidates + beam
                    ):
                        continue

                    sg_df = self._select_subgroup(df, new_desc)
                    if len(sg_df) < self.min_subgroup_size:
                        continue

                    sg_values = sg_df[target].dropna().values
                    quality = self._compute_quality(
                        sg_values, population_values, len(sg_values), population_size
                    )
                    candidates.append((new_desc, abs(quality)))

            # Merge and keep best
            all_candidates = beam + candidates
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beam = all_candidates[: self.beam_width]

            print(f"  Depth {depth}: {len(candidates)} candidates, "
                  f"best quality={beam[0][1]:.4f}")

        # Convert to Subgroup objects with statistical testing
        results = []
        for desc, quality in beam:
            sg_df = self._select_subgroup(df, desc)
            sg_values = sg_df[target].dropna().values
            complement_values = df[~df.index.isin(sg_df.index)][target].dropna().values

            # Two-sample t-test
            if len(sg_values) >= 2 and len(complement_values) >= 2:
                _, p_value = stats.ttest_ind(sg_values, complement_values)
            else:
                p_value = 1.0

            effect_size = self._compute_effect_size(sg_values, complement_values)

            subgroup = Subgroup(
                description=desc,
                size=len(sg_values),
                target_mean=float(np.mean(sg_values)),
                population_mean=float(population_mean),
                quality_score=quality,
                effect_size=effect_size,
                p_value=p_value,
                target_std=float(np.std(sg_values)),
            )
            results.append(subgroup)

        # Filter significant results
        if not return_all:
            results = [r for r in results if r.p_value < 0.05 and r.effect_size > 0.2]

        results.sort(key=lambda x: x.quality_score, reverse=True)
        print(f"  Found {len(results)} significant subgroups")
        return results

    def results_to_dataframe(self, subgroups: list[Subgroup]) -> pd.DataFrame:
        """Convert subgroup results to DataFrame."""
        rows = []
        for sg in subgroups:
            desc_str = " & ".join(f"{k}={v}" for k, v in sg.description.items())
            rows.append({
                "subgroup": desc_str,
                "depth": sg.depth,
                "size": sg.size,
                "target_mean": sg.target_mean,
                "population_mean": sg.population_mean,
                "delta": sg.target_mean - sg.population_mean,
                "effect_size": sg.effect_size,
                "p_value": sg.p_value,
                "quality_score": sg.quality_score,
            })
        return pd.DataFrame(rows)

    def summarize(self, subgroups: list[Subgroup], target: str = "target") -> str:
        """Generate human-readable summary."""
        if not subgroups:
            return "No significant biased subgroups found."

        lines = [
            f"Found {len(subgroups)} biased subgroups for '{target}'.\n",
            "Top 5 most biased subgroups:\n",
        ]
        for i, sg in enumerate(subgroups[:5], 1):
            lines.append(f"  {i}. {sg}\n")

        return "\n".join(lines)
