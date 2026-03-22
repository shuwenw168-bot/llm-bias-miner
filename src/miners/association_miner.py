"""
Association Rule Mining for LLM Bias Discovery
───────────────────────────────────────────────
Adapts the Apriori algorithm to discover frequent co-occurrence
patterns between demographic attributes in prompts and measurable
properties of LLM responses.

Key insight: We treat each (prompt, response) pair as a "transaction"
where demographic attributes and response features are "items."
Mined rules reveal systematic associations like:
  {gender=female, topic=STEM} → {sentiment=condescending}
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


@dataclass
class BiasRule:
    """A discovered bias association rule with metadata."""
    antecedent: frozenset
    consequent: frozenset
    support: float
    confidence: float
    lift: float
    conviction: float
    demographic_attrs: list[str]  # Which demographics are in the antecedent
    response_attrs: list[str]     # Which response features are in the consequent
    num_instances: int            # How many data points match this rule
    p_value: Optional[float] = None

    def __str__(self):
        ant = ", ".join(sorted(self.antecedent))
        con = ", ".join(sorted(self.consequent))
        return (
            f"{{{ant}}} → {{{con}}}\n"
            f"  support={self.support:.3f}, confidence={self.confidence:.3f}, "
            f"lift={self.lift:.2f}, n={self.num_instances}"
        )


class AssociationBiasMiner:
    """Mine association rules that reveal LLM bias patterns.

    The method:
    1. Discretize response features (sentiment, toxicity, regard, etc.)
    2. Encode each row as a "transaction" of demographic + response items
    3. Run Apriori to find frequent itemsets
    4. Extract rules where demographics → response features
    5. Filter for statistically significant bias patterns
    """

    # Columns that represent demographic prompt attributes
    DEMOGRAPHIC_COLUMNS = [
        "gender", "race_ethnicity", "age_group", "socioeconomic", "religion"
    ]

    # Columns that represent response features (discretized)
    RESPONSE_COLUMNS = [
        "sentiment_bin", "toxicity_bin", "regard_bin",
        "length_bin", "is_refusal",
    ]

    def __init__(
        self,
        min_support: float = 0.03,
        min_confidence: float = 0.5,
        min_lift: float = 1.3,
        require_demographic_antecedent: bool = True,
    ):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.require_demographic_antecedent = require_demographic_antecedent

    def _prepare_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert DataFrame rows into one-hot encoded transaction format.

        Each demographic value and response feature becomes an "item."
        Example row: ["gender=female", "race=Black", "sentiment=negative", ...]
        """
        transactions = []

        # Identify which columns are available
        demo_cols = [c for c in self.DEMOGRAPHIC_COLUMNS if c in df.columns]
        resp_cols = [c for c in self.RESPONSE_COLUMNS if c in df.columns]

        # Also include category and occupation/field if present
        context_cols = [c for c in ["category", "occupation", "field"] if c in df.columns]

        all_cols = demo_cols + resp_cols + context_cols

        for _, row in df.iterrows():
            items = []
            for col in all_cols:
                val = row.get(col)
                if pd.notna(val) and val != "":
                    items.append(f"{col}={val}")
            transactions.append(items)

        # One-hot encode using TransactionEncoder
        te = TransactionEncoder()
        te_array = te.fit_transform(transactions)
        return pd.DataFrame(te_array, columns=te.columns_)

    def mine(self, df: pd.DataFrame) -> list[BiasRule]:
        """Run the full association rule mining pipeline.

        Args:
            df: Enriched DataFrame from ResponseAnalyzer.

        Returns:
            List of BiasRule objects sorted by lift (descending).
        """
        print(f"Mining association rules (support≥{self.min_support}, "
              f"confidence≥{self.min_confidence}, lift≥{self.min_lift})")

        # Step 1: Prepare transactions
        one_hot_df = self._prepare_transactions(df)
        print(f"  Transaction matrix: {one_hot_df.shape[0]} rows × {one_hot_df.shape[1]} items")

        # Step 2: Find frequent itemsets
        frequent_itemsets = apriori(
            one_hot_df,
            min_support=self.min_support,
            use_colnames=True,
            max_len=5,
        )
        print(f"  Frequent itemsets found: {len(frequent_itemsets)}")

        if len(frequent_itemsets) == 0:
            print("  No frequent itemsets found. Try lowering min_support.")
            return []

        # Step 3: Generate association rules
        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=self.min_confidence,
        )
        print(f"  Raw rules generated: {len(rules)}")

        # Step 4: Filter and annotate
        bias_rules = self._filter_bias_rules(rules, df)
        print(f"  Bias rules after filtering: {len(bias_rules)}")

        return sorted(bias_rules, key=lambda r: r.lift, reverse=True)

    def _filter_bias_rules(
        self, rules: pd.DataFrame, original_df: pd.DataFrame
    ) -> list[BiasRule]:
        """Filter rules to keep only bias-relevant patterns."""
        demo_prefixes = tuple(f"{c}=" for c in self.DEMOGRAPHIC_COLUMNS)
        resp_prefixes = tuple(f"{c}=" for c in self.RESPONSE_COLUMNS)

        bias_rules = []
        for _, rule in rules.iterrows():
            if rule["lift"] < self.min_lift:
                continue

            ant = rule["antecedents"]
            con = rule["consequents"]

            # Classify items
            ant_demo = [x for x in ant if x.startswith(demo_prefixes)]
            ant_resp = [x for x in ant if x.startswith(resp_prefixes)]
            con_demo = [x for x in con if x.startswith(demo_prefixes)]
            con_resp = [x for x in con if x.startswith(resp_prefixes)]

            # We want rules: demographics → response features
            if self.require_demographic_antecedent:
                if not ant_demo or not con_resp:
                    continue
                # Skip if consequent has demographics (not interesting)
                if con_demo:
                    continue

            bias_rules.append(BiasRule(
                antecedent=ant,
                consequent=con,
                support=rule["support"],
                confidence=rule["confidence"],
                lift=rule["lift"],
                conviction=rule.get("conviction", float("inf")),
                demographic_attrs=ant_demo,
                response_attrs=con_resp + ant_resp,
                num_instances=int(rule["support"] * len(original_df)),
            ))

        return bias_rules

    def get_top_rules(
        self, rules: list[BiasRule], n: int = 20
    ) -> list[BiasRule]:
        """Return the top N rules by lift."""
        return rules[:n]

    def rules_to_dataframe(self, rules: list[BiasRule]) -> pd.DataFrame:
        """Convert rules to a DataFrame for easy inspection."""
        rows = []
        for r in rules:
            rows.append({
                "antecedent": " & ".join(sorted(r.antecedent)),
                "consequent": " & ".join(sorted(r.consequent)),
                "support": r.support,
                "confidence": r.confidence,
                "lift": r.lift,
                "conviction": r.conviction,
                "demographic_attrs": ", ".join(r.demographic_attrs),
                "response_attrs": ", ".join(r.response_attrs),
                "n_instances": r.num_instances,
            })
        return pd.DataFrame(rows)

    def summarize(self, rules: list[BiasRule]) -> str:
        """Generate a human-readable summary of findings."""
        if not rules:
            return "No significant bias association rules found."

        lines = [
            f"Found {len(rules)} bias association rules.",
            f"Top 5 rules by lift:\n",
        ]
        for i, rule in enumerate(rules[:5], 1):
            lines.append(f"  {i}. {rule}\n")

        # Summarize which demographics appear most
        demo_counts = {}
        for r in rules:
            for d in r.demographic_attrs:
                demo_counts[d] = demo_counts.get(d, 0) + 1

        lines.append("Demographic attributes most involved in bias rules:")
        for attr, count in sorted(demo_counts.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"  {attr}: {count} rules")

        return "\n".join(lines)
