"""
LLM-Bias-Miner: Full Audit Pipeline
────────────────────────────────────
Runs the complete bias audit: prompt generation → LLM querying →
response analysis → four mining methods → visualization → report.

Usage:
    # With sample data (no GPU/API needed):
    python experiments/run_full_audit.py --use-sample-data

    # With a Hugging Face model:
    python experiments/run_full_audit.py --model "meta-llama/Llama-3.1-8B-Instruct"

    # With OpenAI API:
    python experiments/run_full_audit.py --model "openai:gpt-4o-mini"
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.collectors.prompt_generator import PromptGenerator
from src.collectors.sample_generator import generate_sample_data
from src.metrics.fairness_metrics import FairnessMetrics
from src.miners.association_miner import AssociationBiasMiner
from src.miners.subgroup_miner import SubgroupBiasMiner
from src.miners.statistical_tester import StatisticalBiasTester
from src.visualization.bias_plots import BiasPlotter
from src.reporting.model_card import BiasReportGenerator

# Lazy imports — these require torch/transformers, only loaded when needed:
# from src.collectors.llm_querier import LLMQuerier
# from src.metrics.response_analyzer import ResponseAnalyzer
# from src.miners.anomaly_miner import AnomalyBiasMiner

def _torch_available() -> bool:
    try:
        import torch
        return True
    except ImportError:
        return False


def load_config(config_path: str = "config/default_config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_audit(
    model_name: str = "sample",
    config_path: str = "config/default_config.yaml",
    use_sample_data: bool = False,
    n_prompts: int = 500,
    skip_embeddings: bool = False,
    data_path: str = None,
):
    """Run the complete bias audit pipeline.

    Args:
        model_name: HF model ID, "openai:model-name", or "sample".
        config_path: Path to config YAML.
        use_sample_data: If True, use synthetic data (no GPU/API needed).
        n_prompts: Number of prompts to generate.
        skip_embeddings: Skip embedding computation (faster, no anomaly detection).
        data_path: Path to pre-collected response data (skips collection step).
    """
    start_time = time.time()
    config = load_config(config_path)

    print("=" * 60)
    print("  LLM-Bias-Miner: Systematic Bias Audit")
    print("=" * 60)

    # ── Step 1: Get Response Data ──
    print("\n[1/6] Preparing response data...")

    if data_path:
        # Load pre-collected data
        print(f"  Loading from {data_path}")
        df = pd.read_json(data_path, orient="records")
        model_name = df.attrs.get("model_name", model_name)

    elif use_sample_data:
        # Generate synthetic data with embedded biases
        print("  Generating synthetic data with embedded bias patterns...")
        df = generate_sample_data(n_prompts=n_prompts)
        model_name = "synthetic-sample-v1"

    else:
        # Generate prompts and collect real responses
        print(f"  Model: {model_name}")
        generator = PromptGenerator(config["prompts"]["templates_path"])
        prompts_df = generator.generate_sample(n=n_prompts)

        from src.collectors.llm_querier import LLMQuerier
        querier = LLMQuerier(
            model_name=model_name,
            max_new_tokens=config["model"]["max_new_tokens"],
            temperature=config["model"]["temperature"],
            num_repeats=config["model"].get("num_repeats", 1),
        )
        df = querier.collect(
            prompts_df,
            save_path="results/raw_responses.json",
        )

    print(f"  Responses: {len(df)} rows")

    # ── Step 2: Analyze Responses ──
    print("\n[2/6] Analyzing responses with HF pipelines...")

    # Check if data already has analysis columns (sample data does)
    needs_analysis = "sentiment_label" not in df.columns

    if needs_analysis:
        from src.metrics.response_analyzer import ResponseAnalyzer
        analyzer = ResponseAnalyzer(
            sentiment_model=config["analysis"]["sentiment_model"],
            toxicity_model=config["analysis"]["toxicity_model"],
            regard_model=config["analysis"]["regard_model"],
            embedding_model=config["analysis"]["embedding_model"],
            batch_size=config["analysis"]["batch_size"],
        )
        df = analyzer.analyze(df, compute_embeddings=not skip_embeddings)
    else:
        print("  Data already has analysis columns, skipping HF pipeline.")
        # Generate synthetic embeddings for sample data (only if torch is available for anomaly detection)
        if not skip_embeddings and _torch_available() and "embeddings" not in (df.attrs if hasattr(df, 'attrs') else {}):
            print("  Generating synthetic embeddings for anomaly detection...")
            rng = np.random.RandomState(42)
            n = len(df)
            dim = 384
            embeddings = rng.normal(0, 1, (n, dim)).astype(np.float32)

            # Shift embeddings for biased groups to make them detectable
            for i, row in df.iterrows():
                if row.get("race_ethnicity") == "a Black person" and row.get("category") == "criminal_justice":
                    embeddings[i] += rng.normal(0.5, 0.2, dim)
                if row.get("gender") == "a woman" and row.get("category") == "employment":
                    embeddings[i] += rng.normal(0.3, 0.2, dim)

            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            df.attrs["embeddings"] = embeddings

    # Save enriched data
    results_dir = Path(config["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 3: Run Mining Methods ──
    print("\n[3/6] Running bias mining methods...")

    # 3a. Association Rule Mining
    print("\n--- Association Rule Mining ---")
    assoc_cfg = config["association_mining"]
    assoc_miner = AssociationBiasMiner(
        min_support=assoc_cfg["min_support"],
        min_confidence=assoc_cfg["min_confidence"],
        min_lift=assoc_cfg["min_lift"],
        require_demographic_antecedent=assoc_cfg["require_demographic_antecedent"],
    )
    assoc_rules = assoc_miner.mine(df)
    assoc_rules_df = assoc_miner.rules_to_dataframe(assoc_rules)
    assoc_summary = assoc_miner.summarize(assoc_rules)
    print(assoc_summary)

    # 3b. Subgroup Discovery
    print("\n--- Subgroup Discovery ---")
    sg_cfg = config["subgroup_discovery"]
    sg_miner = SubgroupBiasMiner(
        beam_width=sg_cfg["beam_width"],
        max_depth=sg_cfg["max_depth"],
        min_subgroup_size=sg_cfg["min_subgroup_size"],
        quality_measure=sg_cfg["quality_measure"],
    )

    all_subgroups = []
    for target in ["toxicity_score", "sentiment_negative", "regard_negative"]:
        if target in df.columns:
            subgroups = sg_miner.discover(df, target=target)
            all_subgroups.extend(subgroups)

    subgroups_df = sg_miner.results_to_dataframe(all_subgroups)
    sg_summary = sg_miner.summarize(all_subgroups, "multiple metrics")
    print(sg_summary)

    # 3c. Anomaly Detection
    print("\n--- Anomaly Detection ---")
    anomaly_summary = ""
    anomaly_groups = []
    if not skip_embeddings and df.attrs.get("embeddings") is not None:
        try:
            from src.miners.anomaly_miner import AnomalyBiasMiner
        except ImportError:
            print("  Skipped (torch not installed). Install with: pip install torch")
            skip_embeddings = True

    if not skip_embeddings and df.attrs.get("embeddings") is not None:
        anom_cfg = config["anomaly_detection"]
        anom_miner = AnomalyBiasMiner(
            encoder_dims=anom_cfg["encoder_dims"],
            decoder_dims=anom_cfg["decoder_dims"],
            epochs=anom_cfg["epochs"],
            learning_rate=anom_cfg["learning_rate"],
            batch_size=anom_cfg["batch_size"],
            anomaly_percentile=anom_cfg["anomaly_percentile"],
        )
        anomaly_groups = anom_miner.detect(df, df.attrs["embeddings"])
        anomaly_summary = anom_miner.summarize(anomaly_groups)
        print(anomaly_summary)

        # Add anomaly scores to df for visualization
        if "anomaly_score" not in df.columns:
            scores = anom_miner._compute_anomaly_scores(anom_miner.model, df.attrs["embeddings"])
            df["anomaly_score"] = scores
    else:
        print("  Skipped (no embeddings). Use --no-skip-embeddings to enable.")
        anomaly_summary = "Anomaly detection was skipped (no embeddings computed)."

    # 3d. Statistical Testing
    print("\n--- Statistical Testing ---")
    stat_cfg = config["statistical_testing"]
    stat_tester = StatisticalBiasTester(
        alpha=stat_cfg["alpha"],
        correction_method=stat_cfg["correction_method"],
        min_group_size=stat_cfg["min_group_size"],
        effect_size_threshold=stat_cfg["effect_size_threshold"],
    )
    stat_results = stat_tester.test_pairwise(df)
    stat_results_df = stat_tester.results_to_dataframe(stat_results)
    omnibus_df = stat_tester.test_omnibus(df)
    stat_summary = stat_tester.summarize(stat_results)
    print(stat_summary)

    # ── Step 4: Compute Fairness Metrics ──
    print("\n[4/6] Computing fairness metrics...")
    fairness = FairnessMetrics()
    fairness_summary = fairness.summary_table(df)
    print(fairness_summary.to_string(index=False) if len(fairness_summary) > 0 else "  No fairness metrics computed.")

    # ── Step 5: Generate Visualizations ──
    print("\n[5/6] Generating visualizations...")
    plotter = BiasPlotter(
        output_dir=config["output"]["figures_dir"],
        dpi=config["output"]["figure_dpi"],
        fmt=config["output"]["figure_format"],
    )
    plotter.generate_all(
        df=df,
        rules_df=assoc_rules_df,
        subgroups_df=subgroups_df,
        stats_df=stat_results_df,
        fairness_df=fairness_summary,
    )

    # ── Step 6: Generate Report ──
    print("\n[6/6] Generating bias audit report...")
    reporter = BiasReportGenerator(output_dir=config["output"]["results_dir"])
    config["demographics_used"] = {
        "gender": df["gender"].unique().tolist() if "gender" in df.columns else [],
        "race_ethnicity": df["race_ethnicity"].unique().tolist() if "race_ethnicity" in df.columns else [],
        "age_group": df["age_group"].unique().tolist() if "age_group" in df.columns else [],
    }
    report = reporter.generate(
        df=df,
        config=config,
        assoc_summary=assoc_summary,
        subgroup_summary=sg_summary,
        anomaly_summary=anomaly_summary,
        stats_summary=stat_summary,
        rules_df=assoc_rules_df,
        subgroups_df=subgroups_df,
        stats_df=stat_results_df,
        fairness_df=fairness_summary,
        model_name=model_name,
    )

    # ── Save all results ──
    assoc_rules_df.to_csv(results_dir / "association_rules.csv", index=False)
    subgroups_df.to_csv(results_dir / "subgroup_findings.csv", index=False)
    stat_results_df.to_csv(results_dir / "statistical_tests.csv", index=False)
    omnibus_df.to_csv(results_dir / "omnibus_tests.csv", index=False)
    fairness_summary.to_csv(results_dir / "fairness_summary.csv", index=False)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  Audit complete in {elapsed:.1f}s")
    print(f"  Results: {results_dir}/")
    print(f"  Report:  {results_dir}/bias_report.md")
    print(f"  Figures: {config['output']['figures_dir']}/")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="LLM-Bias-Miner: Systematic Bias Audit")
    parser.add_argument("--model", type=str, default="sample",
                        help="HF model ID, 'openai:model-name', or 'sample'")
    parser.add_argument("--config", type=str, default="config/default_config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--use-sample-data", action="store_true",
                        help="Use synthetic sample data (no GPU/API needed)")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to pre-collected response data JSON")
    parser.add_argument("--n-prompts", type=int, default=500,
                        help="Number of prompts to generate")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip embedding computation (faster, no anomaly detection)")

    args = parser.parse_args()

    # Default to sample data if no model specified
    use_sample = args.use_sample_data or args.model == "sample"

    run_audit(
        model_name=args.model,
        config_path=args.config,
        use_sample_data=use_sample,
        n_prompts=args.n_prompts,
        skip_embeddings=args.skip_embeddings,
        data_path=args.data,
    )


if __name__ == "__main__":
    main()
