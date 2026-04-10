import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common import (
    attach_embedding_features,
    build_base_context,
    split_xy,
    stage,
)
from models.lr import train_lr_with_sweep


EMBEDDING_MODELS = {
    "e5_small": "intfloat/multilingual-e5-small",
    "e5_base": "intfloat/multilingual-e5-base",
    "e5_large": "intfloat/multilingual-e5-large",
    "e5_large_instruct": "intfloat/multilingual-e5-large-instruct",
}
DEFAULT_BOOTSTRAP_SAMPLES = 1000
DEFAULT_RANDOM_SEED = 42
PLOTS_DIR = "figs"
CACHE_DIR = "experiments/cache"
FEATURE_SET_CONFIGS = [
    {
        "feature_set": "metadata_only",
        "display_name": "Metadata-only (T+C+M+F)",
        "requires_embedding": False,
    },
    {
        "feature_set": "embedding_only",
        "display_name": "Embedding-only (E)",
        "requires_embedding": True,
    },
    {
        "feature_set": "full_model",
        "display_name": "Full model (E+T+C+M+F)",
        "requires_embedding": True,
    },
]


def _safe_roc_auc(y_true, y_score):
    unique_values = np.unique(y_true)
    if unique_values.size < 2:
        return None
    return roc_auc_score(y_true, y_score)


def make_bootstrap_indices(n, n_bootstrap, seed):
    """Generate a fixed set of bootstrap indices for fair comparison across models."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, size=n) for _ in range(n_bootstrap)]


def bootstrap_auc_ci(
    y_true,
    y_score,
    n_bootstrap=DEFAULT_BOOTSTRAP_SAMPLES,
    ci=0.95,
    random_seed=DEFAULT_RANDOM_SEED,
    return_samples=False,
):
    rng = np.random.default_rng(random_seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    sample_size = len(y_true)
    auc_samples = []

    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, sample_size, size=sample_size)
        auc_value = _safe_roc_auc(y_true[sample_idx], y_score[sample_idx])
        if auc_value is not None:
            auc_samples.append(auc_value)

    auc_samples = np.asarray(auc_samples, dtype=np.float64)
    alpha = 1.0 - ci
    ci_low, ci_high = np.quantile(auc_samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    
    out = {
        "mean": float(np.mean(auc_samples)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_valid": int(len(auc_samples)),
    }
    if return_samples:
        out["samples"] = auc_samples
    return out


def bootstrap_auc_diff_ci(
    y_true,
    score_a,
    score_b,
    n_bootstrap=DEFAULT_BOOTSTRAP_SAMPLES,
    ci=0.95,
    random_seed=DEFAULT_RANDOM_SEED,
):
    rng = np.random.default_rng(random_seed)
    y_true = np.asarray(y_true)
    score_a = np.asarray(score_a)
    score_b = np.asarray(score_b)
    sample_size = len(y_true)
    diff_samples = []

    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, sample_size, size=sample_size)
        auc_a = _safe_roc_auc(y_true[sample_idx], score_a[sample_idx])
        auc_b = _safe_roc_auc(y_true[sample_idx], score_b[sample_idx])
        if auc_a is not None and auc_b is not None:
            diff_samples.append(auc_a - auc_b)

    alpha = 1.0 - ci
    ci_low, ci_high = np.quantile(diff_samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "mean_diff": float(np.mean(diff_samples)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_valid": len(diff_samples),
    }


def build_feature_matrix(context, feature_set):
    if feature_set == "metadata_only":
        return np.concatenate(
            [
                context["X_topiccat"],
                context["channel_feature_matrix"],
                context["X_topic_monthly"],
                context["X_framing"],
            ],
            axis=1,
        )
    if feature_set == "embedding_only":
        return context["X_embed"]
    if feature_set == "full_model":
        return context["X_all"]

    raise ValueError(f"Unknown feature_set: {feature_set}")


def save_benchmark_results(results_df, predictions_by_key, y_test, cache_dir=CACHE_DIR):
    """Save benchmark results to cache for faster reloading."""
    os.makedirs(cache_dir, exist_ok=True)
    
    # Save results DataFrame as CSV
    csv_path = os.path.join(cache_dir, "results_df.csv")
    results_df.to_csv(csv_path, index=False)
    
    # Save predictions and y_test as pickle
    predictions_path = os.path.join(cache_dir, "predictions.pkl")
    with open(predictions_path, "wb") as f:
        pickle.dump({"predictions_by_key": predictions_by_key, "y_test": y_test}, f)
    
    print(f"\n💾 Results saved to cache: {cache_dir}")
    return csv_path, predictions_path


def load_benchmark_results(cache_dir=CACHE_DIR):
    """Load cached benchmark results if available."""
    csv_path = os.path.join(cache_dir, "results_df.csv")
    predictions_path = os.path.join(cache_dir, "predictions.pkl")
    
    if not os.path.exists(csv_path) or not os.path.exists(predictions_path):
        return None, None, None
    
    # Load results DataFrame
    results_df = pd.read_csv(csv_path)
    
    # Load predictions and y_test
    with open(predictions_path, "rb") as f:
        data = pickle.load(f)
        predictions_by_key = data["predictions_by_key"]
        y_test = data["y_test"]
    
    print(f"\n✅ Loaded cached results from: {cache_dir}")
    return results_df, predictions_by_key, y_test


def build_feature_matrix(context, feature_set):
    if feature_set == "metadata_only":
        return np.concatenate(
            [
                context["X_topiccat"],
                context["channel_feature_matrix"],
                context["X_topic_monthly"],
                context["X_framing"],
            ],
            axis=1,
        )
    if feature_set == "embedding_only":
        return context["X_embed"]
    if feature_set == "full_model":
        return context["X_all"]

    raise ValueError(f"Unknown feature_set: {feature_set}")


def save_scaling_plots(results_df, output_dir=PLOTS_DIR):
    os.makedirs(output_dir, exist_ok=True)

    dim_plot_path = os.path.join(output_dir, "embedding_dim_vs_test_auc.png")
    time_plot_path = os.path.join(output_dir, "embedding_time_vs_test_auc.png")
    plot_df = results_df[
        results_df["feature_set"].isin(["embedding_only", "full_model"])
    ].copy()

    color_map = {
        "embedding_only": "#1f77b4",
        "full_model": "#d62728",
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for feature_set, subset in plot_df.groupby("feature_set"):
        ax.scatter(
            subset["embedding_dim"],
            subset["auc_test"],
            s=70,
            color=color_map[feature_set],
            label=feature_set,
        )
    for row in plot_df.itertuples():
        ax.annotate(
            f"{row.model}:{row.feature_set}",
            (row.embedding_dim, row.auc_test),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Test AUC")
    ax.set_title("Scaling: Embedding Dimension vs Test AUC")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(dim_plot_path, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for feature_set, subset in plot_df.groupby("feature_set"):
        ax.scatter(
            subset["embed_time_sec"],
            subset["auc_test"],
            s=70,
            color=color_map[feature_set],
            label=feature_set,
        )
    for row in plot_df.itertuples():
        ax.annotate(
            f"{row.model}:{row.feature_set}",
            (row.embed_time_sec, row.auc_test),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    ax.set_xlabel("Embedding Time (sec)")
    ax.set_ylabel("Test AUC")
    ax.set_title("Scaling: Embedding Time vs Test AUC")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(time_plot_path, dpi=200)
    plt.close(fig)

    return {
        "dim_plot_path": dim_plot_path,
        "time_plot_path": time_plot_path,
    }


def save_model_performance_ranking(results_df, output_dir=PLOTS_DIR):
    """Create a ranking plot showing model performance with error bars."""
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "model_performance_ranking.png")

    # Exclude metadata_only
    df = results_df[results_df["feature_set"] != "metadata_only"].copy()

    # Sort by test AUC
    df = df.sort_values("auc_test")

    x = np.arange(len(df))
    y = df["auc_test"].to_numpy()

    yerr = np.vstack([
        df["auc_test"] - df["auc_test_ci_low"],
        df["auc_test_ci_high"] - df["auc_test"]
    ])

    labels = [row.model for row in df.itertuples()]

    # Color mapping
    color_map = {
        "embedding_only": "#1f77b4",
        "full_model": "#d62728",
    }

    colors = [color_map[row.feature_set] for row in df.itertuples()]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Error bars
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="none",
        ecolor="gray",
        elinewidth=2,
        capsize=4,
        zorder=1
    )

    # Points
    ax.scatter(
        x,
        y,
        c=colors,
        s=100,
        zorder=2
    )

    # X-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")

    ax.set_ylabel("Test AUC")
    ax.set_title("Embedding Model Performance")

    ax.grid(axis="y", alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Embedding only',
               markerfacecolor="#1f77b4", markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Full model',
               markerfacecolor="#d62728", markersize=10),
    ]

    ax.legend(handles=legend_elements)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    return plot_path


def save_roc_with_confidence_bands(results_df, predictions_by_key, y_test, output_dir=PLOTS_DIR):
    """Plot ROC curves with bootstrap confidence bands for best models."""
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "roc_curves_with_ci.png")

    best_df = select_best_rows_by_feature_set(results_df)
    color_map = {
        "metadata_only": "#6c757d",
        "embedding_only": "#1f77b4",
        "full_model": "#d62728",
    }
    display_names = {
        "metadata_only": "Metadata",
        "embedding_only": "Embedding",
        "full_model": "Full",
    }
    
    y_test = np.asarray(y_test)
    n = len(y_test)
    
    # Generate bootstrap indices
    boot_idx_list = make_bootstrap_indices(n, DEFAULT_BOOTSTRAP_SAMPLES, DEFAULT_RANDOM_SEED)
    
    # Common FPR for interpolation
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(7, 6))
    
    for row in best_df.itertuples():
        scores = np.asarray(predictions_by_key[(row.model, row.feature_set)])
        color = color_map.get(row.feature_set, "#1f77b4")
        
        # Bootstrap ROC curves
        tprs = []
        for idx in boot_idx_list:
            y_boot = y_test[idx]
            scores_boot = scores[idx]
            
            if len(np.unique(y_boot)) < 2:
                continue
                
            fpr, tpr, _ = roc_curve(y_boot, scores_boot)
            # Interpolate to common FPR
            tpr_interp = np.interp(mean_fpr, fpr, tpr)
            tpr_interp[0] = 0.0
            tprs.append(tpr_interp)
        
        if not tprs:
            continue
            
        tprs = np.array(tprs)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        
        # Confidence bands
        tpr_lower = np.quantile(tprs, 0.025, axis=0)
        tpr_upper = np.quantile(tprs, 0.975, axis=0)
        
        # Plot mean ROC
        label = f"{display_names.get(row.feature_set, row.feature_set)} ({row.auc_test:.3f})"
        ax.plot(mean_fpr, mean_tpr, linewidth=2, color=color, label=label)
        
        # Plot confidence band
        ax.fill_between(
            mean_fpr, 
            tpr_lower, 
            tpr_upper, 
            color=color, 
            alpha=0.15,
            linewidth=0
        )

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, alpha=0.5, label="Random")
    
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves with 95% Confidence Bands")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    
    return plot_path


def compute_bootstrap_rank_stability(results_df, predictions_by_key, y_test):
    """Compute bootstrap ranking probability for each model configuration."""
    df = results_df[results_df["feature_set"] != "metadata_only"].copy()
    
    y_test = np.asarray(y_test)
    n = len(y_test)
    
    # Generate shared bootstrap indices
    boot_idx_list = make_bootstrap_indices(n, DEFAULT_BOOTSTRAP_SAMPLES, DEFAULT_RANDOM_SEED)
    
    def auc_on_idx(idx, scores):
        return _safe_roc_auc(y_test[idx], scores[idx])
    
    # Get all model configurations
    configs = [(row.model, row.feature_set) for row in df.itertuples()]
    config_labels = [f"{row.model} ({row.feature_set})" for row in df.itertuples()]
    
    # Count rankings across bootstrap samples
    rank_counts = {i: {rank: 0 for rank in range(1, len(configs) + 1)} for i in range(len(configs))}
    
    for idx in boot_idx_list:
        # Compute AUC for each config on this bootstrap sample
        aucs = []
        for config in configs:
            if config in predictions_by_key:
                scores = np.asarray(predictions_by_key[config])
                auc = auc_on_idx(idx, scores)
                aucs.append(auc if auc is not None else 0.0)
            else:
                aucs.append(0.0)
        
        # Rank them (higher AUC = better rank)
        ranks = np.argsort(np.argsort(aucs))[::-1] + 1  # 1 = best
        
        for i, rank in enumerate(ranks):
            rank_counts[i][rank] += 1
    
    # Compute probabilities
    results = []
    for i, label in enumerate(config_labels):
        prob_best = rank_counts[i][1] / DEFAULT_BOOTSTRAP_SAMPLES
        prob_top3 = sum(rank_counts[i][r] for r in [1, 2, 3]) / DEFAULT_BOOTSTRAP_SAMPLES
        mean_rank = sum(r * count for r, count in rank_counts[i].items()) / DEFAULT_BOOTSTRAP_SAMPLES
        
        results.append({
            "model_config": label,
            "prob_best": prob_best,
            "prob_top3": prob_top3,
            "mean_rank": mean_rank,
        })
    
    rank_df = pd.DataFrame(results).sort_values("prob_best", ascending=False)
    
    print("\n" + "=" * 80)
    print("BOOTSTRAP RANK STABILITY ANALYSIS")
    print("=" * 80)
    print(rank_df.to_string(index=False))
    
    return rank_df


def save_auc_difference_boxplot(results_df, predictions_by_key, y_test, output_dir=PLOTS_DIR):
    """Create a boxplot showing ΔAUC (full - embedding) distribution for each model."""
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "auc_difference_boxplot.png")

    # Exclude metadata_only
    df = results_df[results_df["feature_set"] != "metadata_only"].copy()
    
    model_order = ["e5_small", "e5_base", "e5_large", "e5_large_instruct"]
    label_map = {
        "e5_small": "small",
        "e5_base": "base",
        "e5_large": "large",
        "e5_large_instruct": "instruct",
    }
    
    y_test = np.asarray(y_test)
    n = len(y_test)

    # Generate shared bootstrap indices
    boot_idx_list = make_bootstrap_indices(n, DEFAULT_BOOTSTRAP_SAMPLES, DEFAULT_RANDOM_SEED)

    def auc_on_idx(idx, scores):
        return _safe_roc_auc(y_test[idx], scores[idx])

    # Compute ΔAUC for each model
    delta_data = []
    delta_labels = []
    delta_point_estimates = []
    significance_markers = []
    
    for model in model_order:
        # Get embedding and full scores
        emb_key = (model, "embedding_only")
        full_key = (model, "full_model")
        
        if emb_key not in predictions_by_key or full_key not in predictions_by_key:
            continue
        
        emb_scores = np.asarray(predictions_by_key[emb_key])
        full_scores = np.asarray(predictions_by_key[full_key])
        
        # Compute bootstrap ΔAUC samples
        delta_samples = []
        for idx in boot_idx_list:
            auc_full = auc_on_idx(idx, full_scores)
            auc_emb = auc_on_idx(idx, emb_scores)
            if auc_full is not None and auc_emb is not None:
                delta_samples.append(auc_full - auc_emb)
        
        delta_samples = np.asarray(delta_samples)
        delta_data.append(delta_samples)
        delta_labels.append(label_map[model])
        
        # Point estimate
        emb_auc = df[(df["model"] == model) & (df["feature_set"] == "embedding_only")]["auc_test"].iloc[0]
        full_auc = df[(df["model"] == model) & (df["feature_set"] == "full_model")]["auc_test"].iloc[0]
        delta_point = full_auc - emb_auc
        delta_point_estimates.append(delta_point)
        
        # Statistical significance test
        ci_low, ci_high = np.quantile(delta_samples, [0.025, 0.975])
        is_significant = (ci_low > 0)  # 95% CI does not include 0
        significance_markers.append(is_significant)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Create boxplot
    positions = np.arange(len(delta_labels))
    bp = ax.boxplot(
        delta_data,
        positions=positions,
        patch_artist=True,
        showfliers=False,
        widths=0.6,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(color="gray", linewidth=1.2),
        capprops=dict(color="gray", linewidth=1.2),
        boxprops=dict(linewidth=1.2),
    )

    # Color boxes based on significance
    for patch, is_sig in zip(bp["boxes"], significance_markers):
        color = "#2ecc71" if is_sig else "#95a5a6"  # green if significant, gray otherwise
        patch.set_facecolor(color)
        patch.set_alpha(0.3)
        patch.set_edgecolor(color)

    # Overlay point estimates
    ax.scatter(
        positions, 
        delta_point_estimates, 
        c=["#27ae60" if sig else "#7f8c8d" for sig in significance_markers],
        s=80, 
        zorder=3, 
        edgecolors='black', 
        linewidths=0.8,
        marker='D'
    )

    # Add significance markers
    for i, (pos, is_sig) in enumerate(zip(positions, significance_markers)):
        if is_sig:
            y_max = max([d.max() for d in delta_data])
            ax.text(pos, y_max * 1.05, '*', ha='center', va='bottom', fontsize=16, fontweight='bold')

    # Add horizontal line at 0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Labels and formatting
    ax.set_xticks(positions)
    ax.set_xticklabels(delta_labels)
    ax.set_ylabel("ΔAUC (Full - Embedding)")
    ax.set_title("AUC Improvement from Adding Metadata Features")
    ax.grid(axis="y", alpha=0.25)

    # Add text annotation
    ax.text(
        0.02, 0.98, 
        "* = statistically significant (95% CI > 0)",
        transform=ax.transAxes,
        va='top',
        fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    )

    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    
    # Print significance results
    print("\n" + "=" * 80)
    print("ΔAUC STATISTICAL SIGNIFICANCE TEST")
    print("=" * 80)
    for model, label, delta_pt, samples, is_sig in zip(
        model_order[:len(delta_data)], delta_labels, delta_point_estimates, delta_data, significance_markers
    ):
        ci_low, ci_high = np.quantile(samples, [0.025, 0.975])
        sig_marker = "✓ SIGNIFICANT" if is_sig else "✗ not significant"
        print(f"{label:10s}: ΔAUC={delta_pt:+.4f}, 95% CI=[{ci_low:+.4f}, {ci_high:+.4f}] {sig_marker}")
    
    return plot_path


def save_auc_bootstrap_boxplot(results_df, predictions_by_key, y_test, output_dir=PLOTS_DIR):
    """Create a grouped boxplot showing bootstrap AUC distribution for each model."""
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "auc_bootstrap_boxplot.png")

    # Exclude metadata_only
    df = results_df[results_df["feature_set"] != "metadata_only"].copy()
    
    # Organize data by model, then by feature_set
    model_order = ["e5_small", "e5_base", "e5_large", "e5_large_instruct"]
    label_map = {
        "e5_small": "small",
        "e5_base": "base",
        "e5_large": "large",
        "e5_large_instruct": "instruct",
    }
    
    y_test = np.asarray(y_test)
    n = len(y_test)

    # Generate shared bootstrap indices for fair comparison
    boot_idx_list = make_bootstrap_indices(n, DEFAULT_BOOTSTRAP_SAMPLES, DEFAULT_RANDOM_SEED)

    def auc_on_idx(idx, scores):
        return _safe_roc_auc(y_test[idx], scores[idx])

    # Color mapping
    color_map = {"embedding_only": "#1f77b4", "full_model": "#d62728"}
    
    # Organize data into groups: [model][feature_set] -> samples
    grouped_data = []
    grouped_colors = []
    grouped_positions = []
    grouped_point_estimates = []
    
    offset = 0.2  # Offset between embedding and full within each group
    group_spacing = 1.0  # Space between model groups
    
    for i, model in enumerate(model_order):
        center_pos = i * group_spacing
        
        for feature_set in ["embedding_only", "full_model"]:
            # Find matching row
            mask = (df["model"] == model) & (df["feature_set"] == feature_set)
            if not mask.any():
                continue
                
            row = df[mask].iloc[0]
            key = (model, feature_set)
            
            # Compute bootstrap samples
            scores = np.asarray(predictions_by_key[key])
            samples = []
            for idx in boot_idx_list:
                v = auc_on_idx(idx, scores)
                if v is not None:
                    samples.append(v)
            
            grouped_data.append(np.asarray(samples))
            grouped_colors.append(color_map[feature_set])
            
            # Position: left for embedding, right for full
            pos = center_pos - offset if feature_set == "embedding_only" else center_pos + offset
            grouped_positions.append(pos)
            grouped_point_estimates.append(row["auc_test"])

    fig, ax = plt.subplots(figsize=(8, 5))

    # Create grouped boxplot
    bp = ax.boxplot(
        grouped_data,
        positions=grouped_positions,
        patch_artist=True,
        showfliers=False,
        widths=0.2,
        medianprops=dict(color="black", linewidth=1.3),
        whiskerprops=dict(color="gray", linewidth=1.2),
        capprops=dict(color="gray", linewidth=1.2),
        boxprops=dict(linewidth=1.2),
    )

    # Color the boxes
    for patch, c in zip(bp["boxes"], grouped_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.25)
        patch.set_edgecolor(c)

    # Overlay point estimates
    ax.scatter(
        grouped_positions, 
        grouped_point_estimates, 
        c=grouped_colors, 
        s=55, 
        zorder=3, 
        edgecolors='black', 
        linewidths=0.5
    )

    # Set x-axis labels (model names at center of each group)
    group_centers = [i * group_spacing for i in range(len(model_order))]
    ax.set_xticks(group_centers)
    ax.set_xticklabels([label_map[m] for m in model_order])
    
    ax.set_ylabel("Bootstrap Test AUC")
    ax.set_title("Bootstrap AUC Distribution (shared resamples)")
    ax.grid(axis="y", alpha=0.25)
    ax.set_xlim(-0.5, (len(model_order) - 1) * group_spacing + 0.5)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0],[0], marker="s", color="w", label="Embedding only",
               markerfacecolor="#1f77b4", alpha=0.25, markersize=12, markeredgecolor="#1f77b4"),
        Line2D([0],[0], marker="s", color="w", label="Full model",
               markerfacecolor="#d62728", alpha=0.25, markersize=12, markeredgecolor="#d62728"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def save_forest_plot(results_df, output_dir=PLOTS_DIR):
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "model_auc_forest.png")

    df = results_df.sort_values("auc_test").copy()
    labels = [f"{row.model} ({row.feature_set})" for row in df.itertuples()]
    positions = np.arange(len(labels))
    left_err = df["auc_test"] - df["auc_test_ci_low"]
    right_err = df["auc_test_ci_high"] - df["auc_test"]

    color_map = {
        "metadata_only": "#6c757d",
        "embedding_only": "#1f77b4",
        "full_model": "#d62728",
    }
    colors = [color_map.get(feature_set, "#1f77b4") for feature_set in df["feature_set"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        df["auc_test"],
        positions,
        xerr=[left_err, right_err],
        fmt="none",
        ecolor="gray",
        elinewidth=2,
        capsize=3,
        alpha=0.8,
    )
    ax.scatter(df["auc_test"], positions, c=colors, s=80, zorder=3)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Test AUC (95% CI)")
    ax.set_title("Forest Plot: Model Performance With Confidence Intervals")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def select_best_rows_by_feature_set(results_df):
    best_rows = []
    for feature_set in ["metadata_only", "embedding_only", "full_model"]:
        subset = results_df[results_df["feature_set"] == feature_set]
        if subset.empty:
            continue
        best_rows.append(
            subset.sort_values(["auc_val", "auc_test"], ascending=False).iloc[0]
        )
    return pd.DataFrame(best_rows)


def save_improvement_plot(results_df, output_dir=PLOTS_DIR):
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "model_improvement_path.png")

    best_df = select_best_rows_by_feature_set(results_df)
    stage_order = ["metadata_only", "embedding_only", "full_model"]
    display_names = {
        "metadata_only": "Metadata",
        "embedding_only": "Embedding",
        "full_model": "Full",
    }
    best_df["stage_rank"] = best_df["feature_set"].map({name: idx for idx, name in enumerate(stage_order)})
    best_df = best_df.sort_values("stage_rank")

    x = np.arange(len(best_df))
    y = best_df["auc_test"].to_numpy()
    yerr = np.vstack(
        [
            best_df["auc_test"] - best_df["auc_test_ci_low"],
            best_df["auc_test_ci_high"] - best_df["auc_test"],
        ]
    )
    labels = [display_names.get(feature_set, feature_set) for feature_set in best_df["feature_set"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, color="#2a9d8f", linewidth=2, alpha=0.8)
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o",
        color="#2a9d8f",
        ecolor="#264653",
        elinewidth=2,
        capsize=4,
        markersize=8,
        zorder=3,
    )
    for row, x_pos, y_pos in zip(best_df.itertuples(), x, y):
        ax.annotate(
            row.model,
            (x_pos, y_pos),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test AUC")
    ax.set_title("Model Improvement Path: Metadata -> Embedding -> Full")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def save_roc_curves_plot(results_df, predictions_by_key, y_test, output_dir=PLOTS_DIR):
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "feature_set_roc_curves.png")

    best_df = select_best_rows_by_feature_set(results_df)
    color_map = {
        "metadata_only": "#6c757d",
        "embedding_only": "#1f77b4",
        "full_model": "#d62728",
    }
    display_names = {
        "metadata_only": "Metadata",
        "embedding_only": "Embedding",
        "full_model": "Full",
    }

    fig, ax = plt.subplots(figsize=(7, 6))
    for row in best_df.itertuples():
        scores = predictions_by_key[(row.model, row.feature_set)]
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc_label = f"{display_names.get(row.feature_set, row.feature_set)} ({row.auc_test:.3f})"
        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            color=color_map.get(row.feature_set, "#1f77b4"),
            label=auc_label,
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves: Metadata vs Embedding vs Full")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def summarize_model_diff_ci(results_df, predictions_by_key, y_test):
    reference_row = results_df.sort_values(["auc_val", "auc_test"], ascending=False).iloc[0]
    reference_model = reference_row["model"]
    reference_feature_set = reference_row["feature_set"]
    reference_key = (reference_model, reference_feature_set)
    reference_scores = predictions_by_key[reference_key]

    rows = []
    for row in results_df.itertuples():
        compare_key = (row.model, row.feature_set)
        if compare_key == reference_key:
            continue

        diff_ci = bootstrap_auc_diff_ci(
            y_test,
            reference_scores,
            predictions_by_key[compare_key],
        )
        rows.append(
            {
                "reference_model": reference_model,
                "reference_feature_set": reference_feature_set,
                "compare_model": row.model,
                "compare_feature_set": row.feature_set,
                "auc_diff_mean": diff_ci["mean_diff"],
                "auc_diff_ci_low": diff_ci["ci_low"],
                "auc_diff_ci_high": diff_ci["ci_high"],
                "bootstrap_valid": diff_ci["n_valid"],
            }
        )

    return pd.DataFrame(rows)


def evaluate_embedding_model(
    base_context,
    model_label,
    model_name,
    feature_set,
    prefix_mode="none",
):
    stage(f"Embedding Benchmark: {model_label} | {feature_set}")
    start_time = time.time()

    if feature_set == "metadata_only":
        context = dict(base_context)
        context["embedding_model"] = None
        context["embedding_time_sec"] = 0.0
        x_matrix = build_feature_matrix(context, feature_set)
        embedding_dim = 0
    else:
        context = attach_embedding_features(
            base_context,
            embedding_model=model_name,
            prefix_mode=prefix_mode,
        )
        x_matrix = build_feature_matrix(context, feature_set)
        embedding_dim = context["X_embed"].shape[1]

    split_data = split_xy(
        x_matrix,
        context["y"],
        context["train_idx"],
        context["val_idx"],
        context["test_idx"],
    )

    lr_model, best_c = train_lr_with_sweep(
        split_data["X_train"],
        split_data["y_train"],
        split_data["X_val"],
        split_data["y_val"],
    )

    auc_train = roc_auc_score(
        split_data["y_train"],
        lr_model.predict_proba(split_data["X_train"])[:, 1],
    )
    auc_val = roc_auc_score(
        split_data["y_val"],
        lr_model.predict_proba(split_data["X_val"])[:, 1],
    )
    auc_test = roc_auc_score(
        split_data["y_test"],
        lr_model.predict_proba(split_data["X_test"])[:, 1],
    )
    test_scores = lr_model.predict_proba(split_data["X_test"])[:, 1]
    test_auc_ci = bootstrap_auc_ci(
        split_data["y_test"],
        test_scores,
    )

    elapsed = time.time() - start_time

    return {
        "model": model_label,
        "model_name": model_name,
        "feature_set": feature_set,
        "embedding_dim": embedding_dim,
        "total_dim": x_matrix.shape[1],
        "best_C": best_c,
        "auc_train": auc_train,
        "auc_val": auc_val,
        "auc_test": auc_test,
        "auc_test_ci_low": test_auc_ci["ci_low"],
        "auc_test_ci_high": test_auc_ci["ci_high"],
        "auc_test_bootstrap_mean": test_auc_ci["mean"],
        "bootstrap_valid": test_auc_ci["n_valid"],
        "overfit_gap": auc_train - auc_test,
        "embed_time_sec": context["embedding_time_sec"],
        "elapsed_sec": elapsed,
        "test_scores": test_scores,
        "y_test": split_data["y_test"],
    }


def run_embedding_benchmark(models=None, prefix_mode="none", use_cache=True, force_recompute=False):
    """
    Run embedding benchmark experiment.
    
    Args:
        models: Dictionary of model labels to model names. If None, uses EMBEDDING_MODELS.
        prefix_mode: Prefix mode for embeddings ("none", "query", "passage").
        use_cache: Whether to use cached results if available.
        force_recompute: Force recomputation even if cache exists.
    
    Returns:
        tuple: (results_df, diff_df)
    """
    if models is None:
        models = EMBEDDING_MODELS

    # Try to load cached results first
    if use_cache and not force_recompute:
        results_df, predictions_by_key, y_test = load_benchmark_results()
        if results_df is not None:
            print("🚀 Using cached results. Run with --force to recompute.")
            # Skip computation, go directly to analysis and plotting
            diff_df = summarize_model_diff_ci(results_df, predictions_by_key, y_test)
            rank_df = compute_bootstrap_rank_stability(results_df, predictions_by_key, y_test)
            plot_paths = save_scaling_plots(results_df)
            performance_path = save_model_performance_ranking(results_df)
            forest_path = save_forest_plot(results_df)
            improvement_path = save_improvement_plot(results_df)
            roc_path = save_roc_curves_plot(results_df, predictions_by_key, y_test)
            roc_ci_path = save_roc_with_confidence_bands(results_df, predictions_by_key, y_test)
            boxplot_path = save_auc_bootstrap_boxplot(results_df, predictions_by_key, y_test)
            delta_path = save_auc_difference_boxplot(results_df, predictions_by_key, y_test)
            
            # Print summary results
            print_benchmark_summary(results_df, diff_df, plot_paths, performance_path, 
                                   forest_path, improvement_path, roc_path, roc_ci_path, 
                                   boxplot_path, delta_path)
            
            return results_df, diff_df

    print("🔄 Computing embeddings and running benchmark...")
    base_context = build_base_context()
    rows = []
    predictions_by_key = {}
    y_test = None

    for feature_config in FEATURE_SET_CONFIGS:
        feature_set = feature_config["feature_set"]

        if not feature_config["requires_embedding"]:
            result = evaluate_embedding_model(
                base_context,
                model_label="metadata_baseline",
                model_name="metadata_only",
                feature_set=feature_set,
                prefix_mode=prefix_mode,
            )
            predictions_by_key[(result["model"], result["feature_set"])] = result["test_scores"]
            if y_test is None:
                y_test = result["y_test"]
            rows.append(result)
            print(
                f"{feature_config['display_name']}: dim={result['total_dim']}, "
                f"VAL AUC={result['auc_val']:.4f}, TEST AUC={result['auc_test']:.4f}, "
                f"95% CI=[{result['auc_test_ci_low']:.4f}, {result['auc_test_ci_high']:.4f}]"
            )
            continue

        for model_label, model_name in models.items():
            result = evaluate_embedding_model(
                base_context,
                model_label=model_label,
                model_name=model_name,
                feature_set=feature_set,
                prefix_mode=prefix_mode,
            )
            predictions_by_key[(result["model"], result["feature_set"])] = result["test_scores"]
            if y_test is None:
                y_test = result["y_test"]

            rows.append(result)
            print(
                f"{model_label} [{feature_set}]: dim={result['total_dim']}, "
                f"VAL AUC={result['auc_val']:.4f}, TEST AUC={result['auc_test']:.4f}, "
                f"95% CI=[{result['auc_test_ci_low']:.4f}, {result['auc_test_ci_high']:.4f}]"
            )

    raw_results_df = pd.DataFrame(rows)
    results_df = raw_results_df.drop(columns=["test_scores", "y_test"]).sort_values(
        ["feature_set", "embedding_dim", "auc_val"],
        ascending=[True, True, False],
    )
    
    # Save results to cache if caching is enabled
    if use_cache:
        save_benchmark_results(results_df, predictions_by_key, y_test)
    
    diff_df = summarize_model_diff_ci(results_df, predictions_by_key, y_test)
    rank_df = compute_bootstrap_rank_stability(results_df, predictions_by_key, y_test)
    plot_paths = save_scaling_plots(results_df)
    performance_path = save_model_performance_ranking(results_df)
    forest_path = save_forest_plot(results_df)
    improvement_path = save_improvement_plot(results_df)
    roc_path = save_roc_curves_plot(results_df, predictions_by_key, y_test)
    roc_ci_path = save_roc_with_confidence_bands(results_df, predictions_by_key, y_test)
    boxplot_path = save_auc_bootstrap_boxplot(results_df, predictions_by_key, y_test)
    delta_path = save_auc_difference_boxplot(results_df, predictions_by_key, y_test)
    
    # Print summary results
    print_benchmark_summary(results_df, diff_df, plot_paths, performance_path, 
                           forest_path, improvement_path, roc_path, roc_ci_path, 
                           boxplot_path, delta_path)
    
    return results_df, diff_df


def print_benchmark_summary(results_df, diff_df, plot_paths, performance_path, 
                            forest_path, improvement_path, roc_path, roc_ci_path, 
                            boxplot_path, delta_path):
    """Print comprehensive benchmark summary."""
    print("\n" + "=" * 80)
    print("EMBEDDING MODEL BENCHMARK")
    print("=" * 80)
    print(
        results_df[
            [
                "feature_set",
                "model",
                "embedding_dim",
                "total_dim",
                "best_C",
                "auc_train",
                "auc_val",
                "auc_test",
                "auc_test_ci_low",
                "auc_test_ci_high",
                "overfit_gap",
                "embed_time_sec",
                "elapsed_sec",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("DIMENSION IMPACT (grouped by feature_set, embedding_dim)")
    print("=" * 80)
    dim_summary = (
        results_df.groupby(["feature_set", "embedding_dim"])[["auc_val", "auc_test"]]
        .agg(["mean", "max", "count"])
        .round(4)
    )
    print(dim_summary.to_string())

    scaling_df = results_df[results_df["feature_set"].isin(["embedding_only", "full_model"])]
    for feature_set, subset in scaling_df.groupby("feature_set"):
        dim_auc_corr = subset["embedding_dim"].corr(subset["auc_test"])
        if pd.notna(dim_auc_corr):
            print(f"\n[{feature_set}] Correlation between embedding_dim and TEST AUC: {dim_auc_corr:.4f}")

        time_auc_corr = subset["embed_time_sec"].corr(subset["auc_test"])
        if pd.notna(time_auc_corr):
            print(f"[{feature_set}] Correlation between embed_time and TEST AUC: {time_auc_corr:.4f}")

    if not diff_df.empty:
        print("\n" + "=" * 80)
        print("BOOTSTRAP CI FOR MODEL AUC DIFFERENCE")
        print("=" * 80)
        print(diff_df.to_string(index=False))

    best_row = results_df.sort_values("auc_val", ascending=False).iloc[0]
    print(
        f"\nBest by VAL AUC: {best_row['model']} [{best_row['feature_set']}] "
        f"(dim={int(best_row['total_dim'])}, "
        f"VAL={best_row['auc_val']:.4f}, TEST={best_row['auc_test']:.4f})"
    )
    
    print("\n" + "=" * 80)
    print("SAVED PLOTS")
    print("=" * 80)
    print(f"Scaling plots: {plot_paths['dim_plot_path']}, {plot_paths['time_plot_path']}")
    print(f"Performance ranking: {performance_path}")
    print(f"Forest plot: {forest_path}")
    print(f"Improvement plot: {improvement_path}")
    print(f"ROC curves: {roc_path}")
    print(f"ROC with CI bands: {roc_ci_path}")
    print(f"Bootstrap boxplot: {boxplot_path}")
    print(f"ΔAUC difference plot: {delta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run embedding benchmark experiment with caching support."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation even if cached results exist.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching (recompute and don't save).",
    )
    parser.add_argument(
        "--prefix-mode",
        type=str,
        default="none",
        choices=["none", "query", "passage"],
        help="Prefix mode for embeddings.",
    )
    
    args = parser.parse_args()
    
    use_cache = not args.no_cache
    force_recompute = args.force
    
    run_embedding_benchmark(
        prefix_mode=args.prefix_mode, 
        use_cache=use_cache, 
        force_recompute=force_recompute
    )
