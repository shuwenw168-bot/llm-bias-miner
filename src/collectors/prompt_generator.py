"""
Prompt Generator for LLM Bias Auditing
───────────────────────────────────────
Generates systematically varied prompts by filling demographic slots
in templates. Produces a structured DataFrame where each row is one
prompt with its demographic metadata — enabling downstream mining.
"""

import itertools
import random
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml


class PromptGenerator:
    """Generate bias-audit prompts from demographic templates.

    Each prompt is a combination of:
      template × gender × race × age × socioeconomic × (occupation/field)

    This combinatorial approach ensures we can mine for interaction effects
    between demographic dimensions — the core of intersectional bias analysis.
    """

    def __init__(self, templates_path: str = "data/prompts/templates.yaml"):
        with open(templates_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.demographics = self.config["demographic_attributes"]
        self.templates = self.config["templates"]
        self.names = self.config["name_pools"]["neutral"]

    def generate_all(
        self,
        max_per_template: Optional[int] = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate all prompt combinations across demographics and templates.

        Args:
            max_per_template: Cap prompts per template (for faster experiments).
                              None = generate all combinations.
            seed: Random seed for reproducibility.

        Returns:
            DataFrame with columns:
              - prompt_id, template_name, category, prompt_text
              - gender, race_ethnicity, age_group, socioeconomic
              - occupation/field (if applicable)
        """
        random.seed(seed)
        all_rows = []
        prompt_id = 0

        for tpl_name, tpl_config in self.templates.items():
            template_str = tpl_config["template"]
            category = tpl_config["category"]

            # Determine which demographic dimensions this template uses
            used_dims = {}
            for dim_name, dim_values in self.demographics.items():
                if "{" + dim_name + "}" in template_str:
                    used_dims[dim_name] = dim_values

            # Get context-specific values (occupation, field, etc.)
            context_values = {}
            for key in ["occupations", "fields"]:
                if key in tpl_config:
                    context_values[key] = tpl_config[key]

            # Generate all demographic combinations
            dim_names = list(used_dims.keys())
            dim_value_lists = list(used_dims.values())
            demo_combos = list(itertools.product(*dim_value_lists))

            # Add context dimension if present
            context_key = None
            if "occupations" in context_values:
                context_key = "occupation"
                context_list = context_values["occupations"]
            elif "fields" in context_values:
                context_key = "field"
                context_list = context_values["fields"]
            else:
                context_list = [None]

            combos = list(itertools.product(demo_combos, context_list))

            # Optionally cap the number of prompts
            if max_per_template and len(combos) > max_per_template:
                combos = random.sample(combos, max_per_template)

            for demo_combo, context_val in combos:
                # Build the substitution dict
                subs = dict(zip(dim_names, demo_combo))
                subs["name"] = random.choice(self.names)

                if context_key:
                    subs[context_key] = context_val

                # Fill the template
                prompt_text = template_str
                for k, v in subs.items():
                    prompt_text = prompt_text.replace("{" + k + "}", str(v))

                # Clean up whitespace
                prompt_text = " ".join(prompt_text.split())

                # Build row with demographic metadata
                row = {
                    "prompt_id": prompt_id,
                    "template_name": tpl_name,
                    "category": category,
                    "prompt_text": prompt_text,
                }
                # Add each demographic as a separate column for mining
                for dim_name, dim_value in zip(dim_names, demo_combo):
                    row[dim_name] = dim_value

                if context_key:
                    row[context_key] = context_val

                all_rows.append(row)
                prompt_id += 1

        df = pd.DataFrame(all_rows)
        print(f"Generated {len(df)} prompts across {len(self.templates)} templates")
        print(f"Demographic dimensions: {list(self.demographics.keys())}")
        print(f"Template categories: {df['category'].unique().tolist()}")
        return df

    def generate_sample(self, n: int = 200, seed: int = 42) -> pd.DataFrame:
        """Generate a small stratified sample for quick experiments."""
        full_df = self.generate_all(seed=seed)
        if len(full_df) <= n:
            return full_df

        # Stratified sample: equal representation per category
        sample = full_df.groupby("category", group_keys=False).apply(
            lambda x: x.sample(min(len(x), n // full_df["category"].nunique()),
                               random_state=seed)
        )
        return sample.reset_index(drop=True)

    def get_demographic_columns(self) -> list[str]:
        """Return the list of demographic column names."""
        return list(self.demographics.keys())


if __name__ == "__main__":
    gen = PromptGenerator()
    df = gen.generate_all(max_per_template=50)
    print(f"\nSample prompt:\n{df.iloc[0]['prompt_text']}")
    print(f"\nDemographic metadata:\n{df.iloc[0].to_dict()}")
