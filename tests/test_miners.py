"""
Unit Tests for LLM-Bias-Miner
──────────────────────────────
Tests the core mining methods using small synthetic data.
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.collectors.sample_generator import generate_sample_data
from src.miners.association_miner import AssociationBiasMiner
from src.miners.subgroup_miner import SubgroupBiasMiner
from src.miners.statistical_tester import StatisticalBiasTester
from src.metrics.fairness_metrics import FairnessMetrics


@pytest.fixture
def sample_df():
    """Generate a small sample dataset for testing."""
    return generate_sample_data(n_prompts=200, seed=42)


class TestSampleGenerator:
    def test_generates_correct_size(self):
        df = generate_sample_data(n_prompts=100, seed=42)
        assert len(df) == 100

    def test_has_required_columns(self, sample_df):
        required = [
            "prompt_id", "gender", "race_ethnicity", "age_group",
            "category", "response_text", "sentiment_label",
            "toxicity_score", "regard_positive", "sentiment_bin",
        ]
        for col in required:
            assert col in sample_df.columns, f"Missing column: {col}"

    def test_demographics_present(self, sample_df):
        assert sample_df["gender"].nunique() >= 2
        assert sample_df["race_ethnicity"].nunique() >= 2
        assert sample_df["category"].nunique() >= 2

    def test_metrics_in_range(self, sample_df):
        assert sample_df["toxicity_score"].between(0, 1).all()
        assert sample_df["sentiment_positive"].between(0, 1).all()
        assert sample_df["regard_positive"].between(0, 1).all()

    def test_reproducibility(self):
        df1 = generate_sample_data(100, seed=42)
        df2 = generate_sample_data(100, seed=42)
        pd.testing.assert_frame_equal(df1, df2)


class TestAssociationMiner:
    def test_mines_rules(self, sample_df):
        miner = AssociationBiasMiner(
            min_support=0.02, min_confidence=0.3, min_lift=1.1
        )
        rules = miner.mine(sample_df)
        assert isinstance(rules, list)

    def test_rules_have_correct_structure(self, sample_df):
        miner = AssociationBiasMiner(
            min_support=0.02, min_confidence=0.3, min_lift=1.1
        )
        rules = miner.mine(sample_df)
        if rules:
            r = rules[0]
            assert hasattr(r, "antecedent")
            assert hasattr(r, "consequent")
            assert hasattr(r, "lift")
            assert r.lift >= 1.1

    def test_to_dataframe(self, sample_df):
        miner = AssociationBiasMiner(
            min_support=0.02, min_confidence=0.3, min_lift=1.1
        )
        rules = miner.mine(sample_df)
        df = miner.rules_to_dataframe(rules)
        assert isinstance(df, pd.DataFrame)
        if len(df) > 0:
            assert "lift" in df.columns
            assert "confidence" in df.columns


class TestSubgroupMiner:
    def test_discovers_subgroups(self, sample_df):
        miner = SubgroupBiasMiner(beam_width=5, max_depth=2, min_subgroup_size=10)
        subgroups = miner.discover(sample_df, target="toxicity_score", return_all=True)
        assert isinstance(subgroups, list)

    def test_subgroup_has_correct_fields(self, sample_df):
        miner = SubgroupBiasMiner(beam_width=5, max_depth=2, min_subgroup_size=10)
        subgroups = miner.discover(sample_df, target="toxicity_score", return_all=True)
        if subgroups:
            sg = subgroups[0]
            assert isinstance(sg.description, dict)
            assert sg.size > 0
            assert isinstance(sg.effect_size, float)
            assert isinstance(sg.p_value, float)

    def test_respects_min_size(self, sample_df):
        miner = SubgroupBiasMiner(beam_width=5, max_depth=2, min_subgroup_size=50)
        subgroups = miner.discover(sample_df, target="toxicity_score", return_all=True)
        for sg in subgroups:
            assert sg.size >= 50


class TestStatisticalTester:
    def test_pairwise_tests(self, sample_df):
        tester = StatisticalBiasTester(alpha=0.05, min_group_size=10)
        results = tester.test_pairwise(sample_df)
        assert isinstance(results, list)

    def test_results_have_corrected_pvalues(self, sample_df):
        tester = StatisticalBiasTester(alpha=0.05, min_group_size=10)
        results = tester.test_pairwise(sample_df)
        if results:
            r = results[0]
            assert hasattr(r, "p_value_corrected")
            assert r.p_value_corrected >= r.p_value  # Correction increases p

    def test_omnibus_tests(self, sample_df):
        tester = StatisticalBiasTester(alpha=0.05, min_group_size=10)
        omnibus = tester.test_omnibus(sample_df)
        assert isinstance(omnibus, pd.DataFrame)
        if len(omnibus) > 0:
            assert "h_statistic" in omnibus.columns
            assert "p_value" in omnibus.columns

    def test_bonferroni_correction(self, sample_df):
        tester = StatisticalBiasTester(correction_method="bonferroni", min_group_size=10)
        results = tester.test_pairwise(sample_df)
        # Bonferroni should produce higher corrected p-values
        for r in results:
            assert r.p_value_corrected >= r.p_value


class TestFairnessMetrics:
    def test_demographic_parity(self, sample_df):
        fm = FairnessMetrics()
        dp = fm.demographic_parity(sample_df, "gender", "toxicity_score")
        assert isinstance(dp, pd.DataFrame)
        assert len(dp) > 0

    def test_disparate_impact(self, sample_df):
        fm = FairnessMetrics()
        di = fm.disparate_impact_ratio(sample_df, "gender", "toxicity_score", 0.3)
        assert "ratio" in di
        if di["ratio"] is not None:
            assert 0 <= di["ratio"] <= 1

    def test_summary_table(self, sample_df):
        fm = FairnessMetrics()
        summary = fm.summary_table(sample_df)
        assert isinstance(summary, pd.DataFrame)
        if len(summary) > 0:
            assert "fairness_concern" in summary.columns
