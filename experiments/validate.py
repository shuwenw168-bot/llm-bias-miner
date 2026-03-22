"""
Quick validation script — verifies the project structure
and core logic works. Can run with minimal dependencies.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def check_structure():
    """Verify all expected files exist."""
    root = Path(__file__).parent.parent
    expected = [
        "README.md",
        "LICENSE",
        "requirements.txt",
        "pyproject.toml",
        "config/default_config.yaml",
        "data/prompts/templates.yaml",
        "src/__init__.py",
        "src/collectors/__init__.py",
        "src/collectors/prompt_generator.py",
        "src/collectors/llm_querier.py",
        "src/collectors/sample_generator.py",
        "src/miners/__init__.py",
        "src/miners/association_miner.py",
        "src/miners/subgroup_miner.py",
        "src/miners/anomaly_miner.py",
        "src/miners/statistical_tester.py",
        "src/metrics/__init__.py",
        "src/metrics/response_analyzer.py",
        "src/metrics/fairness_metrics.py",
        "src/visualization/__init__.py",
        "src/visualization/bias_plots.py",
        "src/reporting/__init__.py",
        "src/reporting/model_card.py",
        "experiments/run_full_audit.py",
        "tests/test_miners.py",
    ]

    missing = []
    for f in expected:
        if not (root / f).exists():
            missing.append(f)

    if missing:
        print(f"FAIL: Missing {len(missing)} files:")
        for f in missing:
            print(f"  - {f}")
        return False
    else:
        print(f"OK: All {len(expected)} expected files present.")
        return True


def check_sample_generator():
    """Test sample data generation."""
    from src.collectors.sample_generator import generate_sample_data

    df = generate_sample_data(n_prompts=100, seed=42)

    checks = {
        "row count": len(df) == 100,
        "has prompt_id": "prompt_id" in df.columns,
        "has gender": "gender" in df.columns,
        "has race_ethnicity": "race_ethnicity" in df.columns,
        "has toxicity_score": "toxicity_score" in df.columns,
        "has sentiment_bin": "sentiment_bin" in df.columns,
        "toxicity in [0,1]": df["toxicity_score"].between(0, 1).all(),
        "multiple genders": df["gender"].nunique() >= 2,
        "multiple races": df["race_ethnicity"].nunique() >= 2,
        "has bias signal": (
            df[df["category"] == "criminal_justice"].groupby("race_ethnicity")["toxicity_score"].mean().std() > 0.005
            if len(df[df["category"] == "criminal_justice"]) > 5 else True
        ),
    }

    all_pass = True
    for name, result in checks.items():
        status = "OK" if result else "FAIL"
        print(f"  {status}: {name}")
        if not result:
            all_pass = False

    return all_pass


def check_statistical_tester():
    """Test statistical testing framework."""
    from src.collectors.sample_generator import generate_sample_data
    from src.miners.statistical_tester import StatisticalBiasTester

    df = generate_sample_data(n_prompts=300, seed=42)
    tester = StatisticalBiasTester(alpha=0.05, min_group_size=10)
    results = tester.test_pairwise(df)
    results_df = tester.results_to_dataframe(results)

    checks = {
        "returns results": len(results) > 0,
        "has p_value_corrected": "p_value_corrected" in results_df.columns if len(results_df) > 0 else True,
        "correction increases p": all(r.p_value_corrected >= r.p_value for r in results) if results else True,
        "effect_size non-negative": all(r.effect_size >= 0 for r in results),
    }

    all_pass = True
    for name, result in checks.items():
        status = "OK" if result else "FAIL"
        print(f"  {status}: {name}")
        if not result:
            all_pass = False

    return all_pass


def check_subgroup_miner():
    """Test subgroup discovery."""
    from src.collectors.sample_generator import generate_sample_data
    from src.miners.subgroup_miner import SubgroupBiasMiner

    df = generate_sample_data(n_prompts=300, seed=42)
    miner = SubgroupBiasMiner(beam_width=5, max_depth=2, min_subgroup_size=10)
    subgroups = miner.discover(df, target="toxicity_score", return_all=True)

    checks = {
        "returns subgroups": len(subgroups) > 0,
        "has description": all(isinstance(s.description, dict) for s in subgroups),
        "has effect_size": all(isinstance(s.effect_size, float) for s in subgroups),
        "respects min size": all(s.size >= 10 for s in subgroups),
    }

    all_pass = True
    for name, result in checks.items():
        status = "OK" if result else "FAIL"
        print(f"  {status}: {name}")
        if not result:
            all_pass = False

    return all_pass


def check_fairness_metrics():
    """Test fairness metrics."""
    from src.collectors.sample_generator import generate_sample_data
    from src.metrics.fairness_metrics import FairnessMetrics

    df = generate_sample_data(n_prompts=200, seed=42)
    fm = FairnessMetrics()

    dp = fm.demographic_parity(df, "gender", "toxicity_score")
    summary = fm.summary_table(df)

    checks = {
        "parity returns groups": len(dp) > 0,
        "summary has concern levels": "fairness_concern" in summary.columns if len(summary) > 0 else True,
    }

    all_pass = True
    for name, result in checks.items():
        status = "OK" if result else "FAIL"
        print(f"  {status}: {name}")
        if not result:
            all_pass = False

    return all_pass


if __name__ == "__main__":
    print("=" * 50)
    print("LLM-Bias-Miner Validation")
    print("=" * 50)

    results = {}

    print("\n[1] Project structure:")
    results["structure"] = check_structure()

    print("\n[2] Sample data generator:")
    results["sample_gen"] = check_sample_generator()

    print("\n[3] Statistical tester:")
    results["stat_tester"] = check_statistical_tester()

    print("\n[4] Subgroup miner:")
    results["subgroup"] = check_subgroup_miner()

    print("\n[5] Fairness metrics:")
    results["fairness"] = check_fairness_metrics()

    print("\n" + "=" * 50)
    passed = sum(results.values())
    total = len(results)
    print(f"Result: {passed}/{total} checks passed")

    if passed == total:
        print("All validations PASSED ✓")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"Failed: {', '.join(failed)}")

    sys.exit(0 if passed == total else 1)
