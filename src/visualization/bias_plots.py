"""
Bias Visualization — Publication-Ready Figures
───────────────────────────────────────────────
Generates figures for bias audit reports and papers.
All figures follow FAccT / NeurIPS style guidelines.
"""

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns


# ── Style Configuration ──
PALETTE = {
    "positive": "#2ecc71",
    "negative": "#e74c3c",
    "neutral": "#95a5a6",
    "primary": "#2c3e50",
    "accent": "#3498db",
    "warning": "#f39c12",
}

def setup_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
    })
    sns.set_palette("Set2")

setup_style()


class BiasPlotter:
    """Generate publication-ready bias visualization figures."""

    def __init__(
        self,
        output_dir: str = "results/figures",
        dpi: int = 300,
        fmt: str = "png",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.fmt = fmt

    def _save(self, fig: plt.Figure, name: str):
        path = self.output_dir / f"{name}.{self.fmt}"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved: {path}")

    def plot_demographic_heatmap(
        self,
        df: pd.DataFrame,
        metric: str = "toxicity_score",
        row_attr: str = "gender",
        col_attr: str = "race_ethnicity",
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Heatmap of a metric across two demographic dimensions.

        This is often the most revealing visualization: it shows
        intersectional bias patterns at a glance.
        """
        pivot = df.pivot_table(
            values=metric, index=row_attr, columns=col_attr, aggfunc="mean"
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn_r",
            center=df[metric].mean(),
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title(title or f"Mean {metric} by {row_attr} × {col_attr}")
        ax.set_ylabel(row_attr.replace("_", " ").title())
        ax.set_xlabel(col_attr.replace("_", " ").title())
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        self._save(fig, f"heatmap_{metric}_{row_attr}_{col_attr}")
        return fig

    def plot_metric_distribution(
        self,
        df: pd.DataFrame,
        metric: str = "toxicity_score",
        group_by: str = "gender",
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Violin + strip plot showing distribution of a metric per group."""
        fig, ax = plt.subplots(figsize=(10, 5))

        # Clean labels
        order = sorted(df[group_by].dropna().unique())

        sns.violinplot(
            data=df, x=group_by, y=metric, order=order,
            inner=None, alpha=0.3, ax=ax,
        )
        sns.stripplot(
            data=df, x=group_by, y=metric, order=order,
            size=2, alpha=0.4, jitter=True, ax=ax,
        )

        # Add mean markers
        means = df.groupby(group_by)[metric].mean()
        for i, group in enumerate(order):
            if group in means.index:
                ax.plot(i, means[group], "D", color="red", markersize=8, zorder=5)

        # Population mean line
        pop_mean = df[metric].mean()
        ax.axhline(pop_mean, color="red", linestyle="--", alpha=0.5, label=f"Population mean: {pop_mean:.3f}")

        ax.set_title(title or f"{metric} Distribution by {group_by}")
        ax.set_xlabel(group_by.replace("_", " ").title())
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.legend()
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        self._save(fig, f"distribution_{metric}_{group_by}")
        return fig

    def plot_association_rules(
        self,
        rules_df: pd.DataFrame,
        top_n: int = 15,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Horizontal bar chart of top association rules by lift."""
        if len(rules_df) == 0:
            return None

        top = rules_df.head(top_n).copy()
        top["label"] = top["antecedent"] + " → " + top["consequent"]
        top = top.sort_values("lift")

        fig, ax = plt.subplots(figsize=(12, max(4, len(top) * 0.45)))

        colors = [PALETTE["negative"] if l > 2 else PALETTE["warning"] if l > 1.5
                  else PALETTE["accent"] for l in top["lift"]]

        bars = ax.barh(top["label"], top["lift"], color=colors, edgecolor="white")

        # Add confidence labels
        for bar, conf in zip(bars, top["confidence"]):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                    f"conf={conf:.2f}", va="center", fontsize=9)

        ax.axvline(1.0, color="gray", linestyle="--", alpha=0.5, label="Lift = 1 (no association)")
        ax.set_xlabel("Lift")
        ax.set_title(title or "Top Bias Association Rules")
        ax.legend()

        self._save(fig, "association_rules_top")
        return fig

    def plot_subgroup_results(
        self,
        subgroups_df: pd.DataFrame,
        top_n: int = 10,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Dot plot of subgroup discovery results."""
        if len(subgroups_df) == 0:
            return None

        top = subgroups_df.head(top_n).copy()
        top = top.sort_values("effect_size")

        fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.5)))

        colors = ["#e74c3c" if p < 0.01 else "#f39c12" if p < 0.05
                  else "#95a5a6" for p in top["p_value"]]

        ax.scatter(top["effect_size"], range(len(top)), c=colors, s=top["size"] * 2,
                   alpha=0.7, edgecolors="black", linewidth=0.5, zorder=5)

        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["subgroup"])
        ax.set_xlabel("Effect Size (Cohen's d)")
        ax.set_title(title or "Biased Subgroups (size ∝ group count)")
        ax.axvline(0.3, color="orange", linestyle="--", alpha=0.5, label="d=0.3 (small-medium)")
        ax.axvline(0.8, color="red", linestyle="--", alpha=0.5, label="d=0.8 (large)")
        ax.legend()

        self._save(fig, "subgroup_discovery")
        return fig

    def plot_anomaly_scores(
        self,
        df: pd.DataFrame,
        group_by: str = "gender",
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Box plot of anomaly scores by demographic group."""
        if "anomaly_score" not in df.columns:
            return None

        fig, ax = plt.subplots(figsize=(10, 5))

        order = sorted(df[group_by].dropna().unique())
        sns.boxplot(data=df, x=group_by, y="anomaly_score", order=order, ax=ax)

        # Threshold line
        threshold = df["anomaly_score"].quantile(0.95)
        ax.axhline(threshold, color="red", linestyle="--", alpha=0.5,
                   label=f"95th percentile: {threshold:.4f}")

        ax.set_title(title or f"Anomaly Scores by {group_by}")
        ax.set_xlabel(group_by.replace("_", " ").title())
        ax.set_ylabel("Reconstruction Error")
        ax.legend()
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        self._save(fig, f"anomaly_scores_{group_by}")
        return fig

    def plot_fairness_summary(
        self,
        summary_df: pd.DataFrame,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Summary dashboard of fairness metrics."""
        if len(summary_df) == 0:
            return None

        fig, ax = plt.subplots(figsize=(10, max(4, len(summary_df) * 0.4)))

        # Color by concern level
        color_map = {"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#2ecc71"}
        colors = [color_map.get(c, "#95a5a6") for c in summary_df["fairness_concern"]]

        labels = summary_df["demographic"] + " | " + summary_df["metric"]
        labels = labels.values[::-1]
        gaps = summary_df["max_gap"].values[::-1]
        colors = colors[::-1]

        ax.barh(labels, gaps, color=colors, edgecolor="white")

        ax.set_xlabel("Maximum Fairness Gap")
        ax.set_title(title or "Fairness Gap Summary Across All Dimensions")

        # Legend
        patches = [mpatches.Patch(color=c, label=l) for l, c in color_map.items()]
        ax.legend(handles=patches, title="Concern Level")

        self._save(fig, "fairness_summary")
        return fig

    def plot_statistical_significance(
        self,
        results_df: pd.DataFrame,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Volcano plot: effect size vs -log10(p-value)."""
        if len(results_df) == 0:
            return None

        fig, ax = plt.subplots(figsize=(10, 7))

        results_df = results_df.copy()
        results_df["neg_log_p"] = -np.log10(results_df["p_value_corrected"].clip(lower=1e-50))

        colors = ["#e74c3c" if sig else "#95a5a6"
                  for sig in results_df["is_significant"]]

        ax.scatter(
            results_df["effect_size"],
            results_df["neg_log_p"],
            c=colors, alpha=0.6, edgecolors="white", s=40,
        )

        # Threshold lines
        ax.axhline(-np.log10(0.05), color="gray", linestyle="--", alpha=0.4, label="p=0.05")
        ax.axvline(0.3, color="orange", linestyle="--", alpha=0.4, label="d=0.3")

        # Label significant points
        sig_df = results_df[results_df["is_significant"]].nlargest(8, "neg_log_p")
        for _, row in sig_df.iterrows():
            label = f"{row['attribute']}:{row['metric']}"
            ax.annotate(label, (row["effect_size"], row["neg_log_p"]),
                       fontsize=7, alpha=0.8,
                       xytext=(5, 5), textcoords="offset points")

        ax.set_xlabel("Effect Size (|Cohen's d|)")
        ax.set_ylabel("-log₁₀(adjusted p-value)")
        ax.set_title(title or "Bias Significance: Effect Size vs Statistical Significance")
        ax.legend()

        self._save(fig, "volcano_plot")
        return fig

    def generate_all(
        self,
        df: pd.DataFrame,
        rules_df: Optional[pd.DataFrame] = None,
        subgroups_df: Optional[pd.DataFrame] = None,
        stats_df: Optional[pd.DataFrame] = None,
        fairness_df: Optional[pd.DataFrame] = None,
    ):
        """Generate all available visualizations."""
        print("\nGenerating visualizations...")

        demo_cols = [c for c in ["gender", "race_ethnicity", "age_group"]
                     if c in df.columns]

        # 1. Heatmaps
        for metric in ["toxicity_score", "sentiment_positive", "regard_negative"]:
            if metric in df.columns and len(demo_cols) >= 2:
                self.plot_demographic_heatmap(df, metric, demo_cols[0], demo_cols[1])

        # 2. Distributions
        for metric in ["toxicity_score", "sentiment_positive"]:
            if metric in df.columns:
                for attr in demo_cols:
                    self.plot_metric_distribution(df, metric, attr)

        # 3. Association rules
        if rules_df is not None and len(rules_df) > 0:
            self.plot_association_rules(rules_df)

        # 4. Subgroup discovery
        if subgroups_df is not None and len(subgroups_df) > 0:
            self.plot_subgroup_results(subgroups_df)

        # 5. Anomaly scores
        if "anomaly_score" in df.columns:
            for attr in demo_cols:
                self.plot_anomaly_scores(df, attr)

        # 6. Fairness summary
        if fairness_df is not None and len(fairness_df) > 0:
            self.plot_fairness_summary(fairness_df)

        # 7. Statistical volcano plot
        if stats_df is not None and len(stats_df) > 0:
            self.plot_statistical_significance(stats_df)

        print(f"All figures saved to {self.output_dir}/")
