import argparse
import shutil
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DEFAULT_CACHE_DIR = Path("experiments/cache")
DEFAULT_OUTPUT_DIR = Path("figs/paper")
PRIMARY_BAR_COLOR = "#2F4858"
SECONDARY_BAR_COLOR = "#567C8D"
EDGE_COLOR = "#243447"
GRID_COLOR = "#D8DEE9"
TEXT_COLOR = "#1F2933"
XAI_FIG_WIDTH = 9.6
XAI_FIG_HEIGHT = 6.8

CHANNEL_DISPLAY_NAMES = {
    "東森新聞 CH51": "EBC News",
    "三立LIVE新聞": "SET News",
    "台視新聞 TTV NEWS": "TTV News",
    "TVBS NEWS": "TVBS News",
    "中天新聞": "Cti News",
}


sns.set_theme(style="whitegrid", context="talk")


def _load_csv(path):
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path)


def _display_channel_name(name):
    return CHANNEL_DISPLAY_NAMES.get(str(name), str(name))


def _ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save(fig, output_path):
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _copy_if_exists(src_path, dst_path):
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    if not src_path.exists():
        return None
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)
    return str(dst_path)


def _style_axes(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, fontsize=16, pad=14, color=TEXT_COLOR, fontweight="semibold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12, color=TEXT_COLOR)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, color=TEXT_COLOR)
    ax.tick_params(axis="both", labelsize=11, colors=TEXT_COLOR)
    ax.grid(alpha=0.55, axis="y", color=GRID_COLOR, linewidth=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9AA5B1")
    ax.spines["bottom"].set_color("#9AA5B1")
    ax.set_facecolor("#FAFBFC")


def _dynamic_barh_figure(n_rows, base_width=8.6, min_height=4.8, max_height=12.5):
    height = min(max(min_height, 0.4 * max(n_rows, 4) + 1.2), max_height)
    fig, ax = plt.subplots(figsize=(base_width, height))
    fig.patch.set_facecolor("white")
    return fig, ax


def _fixed_xai_figure():
    fig, ax = plt.subplots(figsize=(XAI_FIG_WIDTH, XAI_FIG_HEIGHT))
    fig.patch.set_facecolor("white")
    return fig, ax


def plot_main_benchmark(cache_dir, output_dir):
    summary_df = _load_csv(Path(cache_dir) / "train_models_multiseed_summary.csv")
    if summary_df is None or summary_df.empty:
        return None

    df = summary_df[summary_df["model"] != "Dummy"].copy()
    df = df.sort_values("auc_test_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    y = np.arange(len(df))
    colors = ["#9db4c0"] * len(df)
    colors[-1] = "#d97757"

    ax.barh(
        y,
        df["auc_test_mean"],
        xerr=df["auc_test_std"].fillna(0.0),
        color=colors,
        edgecolor="#334155",
        capsize=4,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(df["model"])
    ax.set_xlim(max(0.72, df["auc_test_mean"].min() - 0.015), min(0.85, df["auc_test_mean"].max() + 0.015))
    _style_axes(ax, "Main Benchmark Across Models", xlabel="Test AUC (mean +/- std)", ylabel="Model")

    return _save(fig, Path(output_dir) / "paper_main_benchmark.png")


def plot_label_sensitivity(cache_dir, output_dir):
    summary_df = _load_csv(Path(cache_dir) / "label_sensitivity" / "label_sensitivity_summary.csv")
    if summary_df is None or summary_df.empty:
        return None

    df = summary_df[summary_df["model"] != "Dummy"].copy()
    top_models = (
        df.groupby("model", as_index=False)["auc_test_mean"]
        .mean()
        .sort_values("auc_test_mean", ascending=False)["model"]
        .head(4)
        .tolist()
    )
    df = df[df["model"].isin(top_models)].copy()

    palette = ["#d97757", "#4c78a8", "#54a24b", "#b279a2"]
    color_map = {model: palette[i % len(palette)] for i, model in enumerate(top_models)}

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for model in top_models:
        subset = df[df["model"] == model].sort_values("label_percentile")
        ax.plot(
            subset["label_percentile"],
            subset["auc_test_mean"],
            marker="o",
            linewidth=2,
            color=color_map[model],
            label=model,
        )

    ax.set_xticks(sorted(df["label_percentile"].unique()))
    ax.set_ylim(df["auc_test_mean"].min() - 0.02, df["auc_test_mean"].max() + 0.02)
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.22)
    _style_axes(ax, "Label Sensitivity", xlabel="Positive Label Percentile", ylabel="Mean Test AUC")

    return _save(fig, Path(output_dir) / "paper_label_sensitivity.png")


def plot_loco_heatmap(cache_dir, output_dir):
    summary_df = _load_csv(Path(cache_dir) / "leave_one_channel_out" / "leave_one_channel_out_summary.csv")
    if summary_df is None or summary_df.empty:
        return None

    df = summary_df[summary_df["model"] != "Dummy"].copy()
    pivot_df = df.pivot_table(
        index="holdout_channel",
        columns="model",
        values="auc_test_mean",
        aggfunc="mean",
    )
    model_order = pivot_df.mean(axis=0).sort_values(ascending=False).index.tolist()
    pivot_df = pivot_df[model_order]
    pivot_df = pivot_df.rename(index=_display_channel_name)

    values = pivot_df.to_numpy()
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    im = ax.imshow(values, aspect="auto", cmap="YlGnBu", vmin=np.nanmin(values), vmax=np.nanmax(values))

    ax.set_xticks(np.arange(len(pivot_df.columns)))
    ax.set_xticklabels(pivot_df.columns, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(pivot_df.index)))
    ax.set_yticklabels(pivot_df.index)
    _style_axes(ax, "Leave-One-Channel-Out Generalization", xlabel="Model", ylabel="Holdout Channel")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=8, color="#0f172a")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Mean Test AUC")
    return _save(fig, Path(output_dir) / "paper_loco_heatmap.png")


def plot_channel_feature_study(cache_dir, output_dir):
    summary_df = _load_csv(Path(cache_dir) / "channel_feature_study" / "channel_feature_study_summary.csv")
    if summary_df is None or summary_df.empty:
        summary_df = _load_csv(Path(cache_dir) / "channel_prior_study" / "channel_prior_study_summary.csv")
    if summary_df is None or summary_df.empty:
        return None

    preferred_order = [
        "context_only",
        "metadata_only",
        "channel_only",
        "semantic_only",
        "framing_only",
        "semantic_plus_framing",
        "semantic_plus_channel",
        "full_model",
    ]
    df = summary_df.copy()
    df["order"] = df["feature_set"].apply(lambda x: preferred_order.index(x) if x in preferred_order else len(preferred_order))
    df = df.sort_values(["order", "auc_test_mean"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(df))
    colors = ["#9db4c0"] * len(df)
    if "full_model" in df["feature_set"].values:
        colors[df.index[df["feature_set"] == "full_model"][0]] = "#d97757"

    ax.bar(
        x,
        df["auc_test_mean"],
        yerr=df["auc_test_std"].fillna(0.0),
        color=colors,
        edgecolor="#334155",
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["feature_set"], rotation=25, ha="right")
    ax.set_ylim(df["auc_test_mean"].min() - 0.03, df["auc_test_mean"].max() + 0.03)
    _style_axes(ax, "Channel Feature Ablation", ylabel="Mean Test AUC")

    return _save(fig, Path(output_dir) / "paper_channel_feature_study.png")


def plot_ablation_summary(cache_dir, output_dir):
    ablation_df = _load_csv(Path(cache_dir) / "ablation" / "ablation_results.csv")
    if ablation_df is None or ablation_df.empty:
        return None

    df = ablation_df.copy()
    full_test = float(df.loc[df["exp"] == "FULL_all", "auc_test"].iloc[0]) if "FULL_all" in set(df["exp"]) else df["auc_test"].max()
    df["delta_vs_full_test"] = df["auc_test"] - full_test
    top_df = df.sort_values("auc_test", ascending=False).head(18).sort_values("auc_test", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.2), sharey=False)

    colors = ["#d97757" if exp == "FULL_all" else "#8fb8c9" for exp in top_df["exp"]]
    axes[0].barh(top_df["exp"], top_df["auc_test"], color=colors, edgecolor="#334155")
    _style_axes(axes[0], "Top Ablation Configurations", xlabel="Test AUC", ylabel="Experiment")
    axes[0].grid(alpha=0.22, axis="x")

    delta_df = df.sort_values("delta_vs_full_test", ascending=True).tail(18)
    delta_colors = ["#d97757" if val >= 0 else "#9ca3af" for val in delta_df["delta_vs_full_test"]]
    axes[1].barh(delta_df["exp"], delta_df["delta_vs_full_test"], color=delta_colors, edgecolor="#334155")
    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1)
    _style_axes(axes[1], "Delta vs Full Model", xlabel="Test AUC Delta", ylabel="Experiment")
    axes[1].grid(alpha=0.22, axis="x")

    return _save(fig, Path(output_dir) / "paper_ablation_summary.png")


def plot_embedding_summary(cache_dir, output_dir):
    results_df = _load_csv(Path(cache_dir) / "results_df.csv")
    if results_df is None or results_df.empty:
        return None

    df = results_df.copy()
    feature_order = ["metadata_only", "embedding_only", "full_model"]
    best_df = (
        df.sort_values("auc_test", ascending=False)
        .groupby("feature_set", as_index=False)
        .first()
    )
    best_df["order"] = best_df["feature_set"].apply(lambda x: feature_order.index(x) if x in feature_order else len(feature_order))
    best_df = best_df.sort_values("order")

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(best_df))
    ax.bar(
        x,
        best_df["auc_test"],
        color=["#7aa6c2", "#90be6d", "#d97757"],
        edgecolor="#334155",
    )
    ax.errorbar(
        x,
        best_df["auc_test_bootstrap_mean"],
        yerr=[
            best_df["auc_test_bootstrap_mean"] - best_df["auc_test_ci_low"],
            best_df["auc_test_ci_high"] - best_df["auc_test_bootstrap_mean"],
        ],
        fmt="none",
        ecolor="#334155",
        capsize=4,
    )
    labels = [
        f"{row.feature_set}\n{row.model}"
        for row in best_df.itertuples()
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(best_df["auc_test_ci_low"].min() - 0.03, best_df["auc_test_ci_high"].max() + 0.03)
    _style_axes(ax, "Best Representation Per Feature Family", ylabel="Test AUC")

    return _save(fig, Path(output_dir) / "paper_embedding_summary.png")


def plot_shap_summary(output_dir):
    shap_csv = _load_csv(Path("figs") / "shap_group_summary.csv")
    if shap_csv is None or shap_csv.empty:
        return None

    df = shap_csv.head(20).copy()
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    fig, ax = _dynamic_barh_figure(len(df), base_width=8.8, min_height=5.6)
    ax.barh(df["group_name"], df["mean_abs_shap"], color=PRIMARY_BAR_COLOR, edgecolor=EDGE_COLOR)
    _style_axes(ax, "Top Feature Groups by SHAP", xlabel="Mean |SHAP| Per Dimension", ylabel="Feature Group")
    ax.grid(alpha=0.22, axis="x")
    ax.invert_yaxis()

    return _save(fig, Path(output_dir) / "paper_shap_groups.png")


def collect_xgb_explainability_figures(output_dir):
    output_dir = Path(output_dir)
    figure_paths = {
        "xgb_shap_groups": _copy_if_exists(Path("figs") / "shap_family_importance_mean.png", output_dir / "paper_xgb_shap_groups.png"),
        "xgb_shap_theory": _copy_if_exists(Path("figs") / "shap_theory_importance_total.png", output_dir / "paper_xgb_shap_theory.png"),
        "xgb_shap_feature_importance": _copy_if_exists(Path("figs") / "shap_metadata_importance.png", output_dir / "paper_xgb_shap_feature_importance.png"),
        "xgb_shap_beeswarm": _copy_if_exists(Path("figs") / "shap_official_beeswarm.png", output_dir / "paper_xgb_shap_beeswarm.png"),
        "xgb_shap_bar": _copy_if_exists(Path("figs") / "shap_official_bar.png", output_dir / "paper_xgb_shap_bar.png"),
    }
    return figure_paths


def plot_lr_xai_summary(cache_dir, output_dir):
    summary_df = _load_csv(Path(cache_dir) / "xai" / "lr_family_permutation_summary.csv")
    if summary_df is None or summary_df.empty:
        return None

    df = summary_df.head(12).copy()
    df = df.sort_values("auc_drop_mean", ascending=False).reset_index(drop=True)
    fig, ax = _fixed_xai_figure()
    labels = [fill(str(label).replace("_", " "), width=20) for label in df["group_name"]]
    values = df["auc_drop_mean"].to_numpy()
    colors = [PRIMARY_BAR_COLOR] * len(df)
    if len(colors) > 0:
        colors[0] = SECONDARY_BAR_COLOR
    plot_df = df.copy()
    plot_df["_label_wrapped"] = labels
    order = plot_df["_label_wrapped"].tolist()
    sns.barplot(
        data=plot_df,
        x="auc_drop_mean",
        y="_label_wrapped",
        order=order,
        orient="h",
        palette=colors,
        edgecolor=EDGE_COLOR,
        linewidth=1.1,
        ax=ax,
    )
    _style_axes(ax, "LR Family-Level Importance", xlabel="Mean Test AUC Drop", ylabel="Feature Family")
    ax.grid(alpha=0.55, axis="x", color=GRID_COLOR, linewidth=0.9)
    x_max = float(np.nanmax(values)) if len(values) else 0.0
    ax.set_xlim(0, x_max * 1.12 if x_max > 0 else 1.0)
    for idx, value in enumerate(values):
        ax.text(value + x_max * 0.015, idx, f"{value:.3f}", va="center", ha="left", fontsize=10, color=TEXT_COLOR)
    return _save(fig, Path(output_dir) / "paper_lr_xai_summary.png")


def plot_lr_feature_permutation(cache_dir, output_dir):
    summary_df = _load_csv(Path(cache_dir) / "xai" / "lr_feature_permutation_summary.csv")
    if summary_df is None or summary_df.empty:
        return None

    df = summary_df.head(18).copy()
    df = df.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    fig, ax = _fixed_xai_figure()
    labels = [fill(str(label).replace("_", " "), width=24) for label in df["feature_name"]]
    values = df["importance_mean"].to_numpy()
    plot_df = df.copy()
    plot_df["_label_wrapped"] = labels
    order = plot_df["_label_wrapped"].tolist()
    sns.barplot(
        data=plot_df,
        x="importance_mean",
        y="_label_wrapped",
        order=order,
        orient="h",
        color=SECONDARY_BAR_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=1.1,
        ax=ax,
    )
    _style_axes(ax, "LR Top Feature Permutation Importance", xlabel="Mean Test AUC Drop", ylabel="Feature")
    ax.grid(alpha=0.55, axis="x", color=GRID_COLOR, linewidth=0.9)
    x_max = float(np.nanmax(values)) if len(values) else 0.0
    ax.set_xlim(0, x_max * 1.12 if x_max > 0 else 1.0)
    for idx, value in enumerate(values):
        ax.text(value + x_max * 0.015, idx, f"{value:.3f}", va="center", ha="left", fontsize=9.5, color=TEXT_COLOR)
    return _save(fig, Path(output_dir) / "paper_lr_feature_permutation.png")


def plot_lr_per_channel_heatmap(cache_dir, output_dir):
    per_channel_df = _load_csv(Path(cache_dir) / "xai" / "lr_per_channel_family_importance.csv")
    if per_channel_df is None or per_channel_df.empty:
        return None

    family_order = ["embedding", "framing", "topic", "topic_monthly", "channel_context", "channel_metadata"]
    pivot_df = per_channel_df.pivot_table(
        index="channel",
        columns="group_name",
        values="auc_drop_mean",
        aggfunc="mean",
    )
    ordered_cols = [col for col in family_order if col in pivot_df.columns] + [col for col in pivot_df.columns if col not in family_order]
    pivot_df = pivot_df[ordered_cols]
    pivot_df = pivot_df.rename(index=_display_channel_name)

    fig, ax = _fixed_xai_figure()
    fig.patch.set_facecolor("white")
    heatmap_df = pivot_df.copy()
    heatmap_df.columns = [label.replace("_", " ") for label in heatmap_df.columns]
    cmap = sns.light_palette(PRIMARY_BAR_COLOR, as_cmap=True)
    sns.heatmap(
        heatmap_df,
        cmap=cmap,
        annot=True,
        fmt=".3f",
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Mean Test AUC Drop", "shrink": 0.9},
        ax=ax,
    )
    _style_axes(ax, "LR Family Importance Across Channels", xlabel="Feature Family", ylabel="Channel")
    ax.grid(False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    return _save(fig, Path(output_dir) / "paper_lr_per_channel_heatmap.png")


def _plot_stability_heatmap(summary_df, output_path, title):
    if summary_df is None or summary_df.empty:
        return None

    labels = sorted(set(summary_df["left"]).union(set(summary_df["right"])))
    corr_matrix = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)
    jaccard_matrix = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)

    for label in labels:
        corr_matrix.loc[label, label] = 1.0
        jaccard_matrix.loc[label, label] = 1.0

    for row in summary_df.itertuples():
        corr_matrix.loc[row.left, row.right] = row.spearman_rank_corr
        corr_matrix.loc[row.right, row.left] = row.spearman_rank_corr
        jaccard_matrix.loc[row.left, row.right] = row.topk_jaccard
        jaccard_matrix.loc[row.right, row.left] = row.topk_jaccard

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    fig.patch.set_facecolor("white")

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Spearman rank correlation", "shrink": 0.85},
        ax=axes[0],
    )
    _style_axes(axes[0], f"{title}: Rank Correlation", xlabel="", ylabel="")
    axes[0].grid(False)

    sns.heatmap(
        jaccard_matrix,
        annot=True,
        fmt=".3f",
        cmap="YlOrBr",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Top-k Jaccard", "shrink": 0.85},
        ax=axes[1],
    )
    _style_axes(axes[1], f"{title}: Top-k Overlap", xlabel="", ylabel="")
    axes[1].grid(False)

    for ax in axes:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    return _save(fig, output_path)


def plot_lr_seed_stability(cache_dir, output_dir):
    summary_df = _load_csv(Path(cache_dir) / "xai" / "lr_seed_family_stability.csv")
    if summary_df is None or summary_df.empty:
        return None
    return _plot_stability_heatmap(
        summary_df,
        Path(output_dir) / "paper_lr_seed_stability.png",
        "LR Family Stability Across Random Seeds",
    )


def plot_lr_percentile_stability(cache_dir, output_dir):
    summary_df = _load_csv(Path(cache_dir) / "xai" / "lr_percentile_family_stability.csv")
    if summary_df is None or summary_df.empty:
        return None
    return _plot_stability_heatmap(
        summary_df,
        Path(output_dir) / "paper_lr_percentile_stability.png",
        "LR Family Stability Across Label Thresholds",
    )


def collect_cross_model_validation_figures(cache_dir, output_dir):
    output_dir = Path(output_dir)
    consistency_dir = Path(cache_dir) / "explainability_consistency"
    figure_paths = {
        "cross_model_family_alignment": _copy_if_exists(
            consistency_dir / "family_alignment.png",
            output_dir / "paper_cross_model_family_alignment.png",
        ),
        "cross_model_family_rank_alignment": _copy_if_exists(
            consistency_dir / "family_rank_alignment.png",
            output_dir / "paper_cross_model_family_rank_alignment.png",
        ),
        "cross_model_theory_alignment": _copy_if_exists(
            consistency_dir / "theory_alignment.png",
            output_dir / "paper_cross_model_theory_alignment.png",
        ),
        "cross_model_feature_alignment": _copy_if_exists(
            consistency_dir / "feature_alignment_top12.png",
            output_dir / "paper_cross_model_feature_alignment.png",
        ),
        "cross_model_feature_rank_alignment": _copy_if_exists(
            consistency_dir / "feature_rank_alignment.png",
            output_dir / "paper_cross_model_feature_rank_alignment.png",
        ),
    }
    return figure_paths


def build_figure_manifest(output_dir, figure_paths):
    manifest_rows = [
        {"figure_id": figure_id, "path": path}
        for figure_id, path in figure_paths.items()
        if path is not None
    ]
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = Path(output_dir) / "paper_figures_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    return str(manifest_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--only-xai",
        action="store_true",
        help="Only generate feature-importance and XGBoost explainability figures.",
    )
    args = parser.parse_args()

    output_dir = _ensure_dir(args.output_dir)
    cache_dir = Path(args.cache_dir)

    if args.only_xai:
        figure_paths = {
            "lr_xai_summary": plot_lr_xai_summary(cache_dir, output_dir),
            "lr_feature_permutation": plot_lr_feature_permutation(cache_dir, output_dir),
            "lr_per_channel_heatmap": plot_lr_per_channel_heatmap(cache_dir, output_dir),
            "lr_seed_stability": plot_lr_seed_stability(cache_dir, output_dir),
            "lr_percentile_stability": plot_lr_percentile_stability(cache_dir, output_dir),
            "shap_groups": plot_shap_summary(output_dir),
        }
        figure_paths.update(collect_xgb_explainability_figures(output_dir))
        figure_paths.update(collect_cross_model_validation_figures(cache_dir, output_dir))
    else:
        figure_paths = {
            "main_benchmark": plot_main_benchmark(cache_dir, output_dir),
            "label_sensitivity": plot_label_sensitivity(cache_dir, output_dir),
            "loco_heatmap": plot_loco_heatmap(cache_dir, output_dir),
            "channel_feature_study": plot_channel_feature_study(cache_dir, output_dir),
            "ablation_summary": plot_ablation_summary(cache_dir, output_dir),
            "embedding_summary": plot_embedding_summary(cache_dir, output_dir),
            "shap_groups": plot_shap_summary(output_dir),
            "lr_xai_summary": plot_lr_xai_summary(cache_dir, output_dir),
            "lr_feature_permutation": plot_lr_feature_permutation(cache_dir, output_dir),
            "lr_per_channel_heatmap": plot_lr_per_channel_heatmap(cache_dir, output_dir),
            "lr_seed_stability": plot_lr_seed_stability(cache_dir, output_dir),
            "lr_percentile_stability": plot_lr_percentile_stability(cache_dir, output_dir),
        }
        figure_paths.update(collect_xgb_explainability_figures(output_dir))
        figure_paths.update(collect_cross_model_validation_figures(cache_dir, output_dir))
    manifest_path = build_figure_manifest(output_dir, figure_paths)

    print("Saved paper figures:")
    for figure_id, path in figure_paths.items():
        if path is not None:
            print(f"  {figure_id}: {path}")
        else:
            print(f"  {figure_id}: skipped (missing input files)")
    print("Saved figure manifest:", manifest_path)


if __name__ == "__main__":
    main()
