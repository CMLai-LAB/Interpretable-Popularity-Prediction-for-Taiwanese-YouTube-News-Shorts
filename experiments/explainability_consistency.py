import argparse
import sys
from pathlib import Path
from textwrap import fill

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_OUTPUT_DIR = "experiments/cache/explainability_consistency"
PRIMARY_BAR_COLOR = "#2F4858"
SECONDARY_BAR_COLOR = "#567C8D"
TERTIARY_BAR_COLOR = "#BC4749"
EDGE_COLOR = "#243447"
GRID_COLOR = "#D8DEE9"
TEXT_COLOR = "#1F2933"
FIG_WIDTH = 10.5
FIG_HEIGHT = 7.2


def _get_plotting_modules():
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk")
    return plt


def _normalize_importance(series):
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    total = float(values.sum())
    if total <= 0.0:
        return pd.Series(np.zeros(len(values), dtype=np.float64), index=series.index)
    return values / total


def _safe_spearman(left, right):
    aligned = pd.concat([left, right], axis=1, join="inner").dropna()
    if aligned.empty:
        return np.nan
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman"))


def _topk_jaccard(left_rank, right_rank, k):
    left_top = set(left_rank.nsmallest(k).index)
    right_top = set(right_rank.nsmallest(k).index)
    union = left_top | right_top
    if not union:
        return 1.0
    return float(len(left_top & right_top) / len(union))


def _save_rank_plot(df, item_col, output_path, title):
    if df.empty:
        return None

    plt = _get_plotting_modules()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = df.copy()
    plot_df = plot_df.sort_values(
        ["rank_gap_abs", "combined_rank"],
        ascending=[False, True],
    ).head(15)
    plot_df = plot_df.sort_values("combined_rank", ascending=False)
    plot_df["label"] = [fill(str(value).replace("_", " "), width=24) for value in plot_df[item_col]]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.hlines(
        y=plot_df["label"],
        xmin=plot_df["rank_shap"],
        xmax=plot_df["rank_lr"],
        color=GRID_COLOR,
        linewidth=2.2,
        zorder=1,
    )
    ax.scatter(
        plot_df["rank_shap"],
        plot_df["label"],
        s=90,
        color=PRIMARY_BAR_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=1.0,
        label="XGB SHAP rank",
        zorder=2,
    )
    ax.scatter(
        plot_df["rank_lr"],
        plot_df["label"],
        s=90,
        color=TERTIARY_BAR_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=1.0,
        label="LR permutation rank",
        zorder=3,
    )
    ax.set_title(title, fontsize=16, pad=14, color=TEXT_COLOR, fontweight="semibold")
    ax.set_xlabel("Rank (1 = most important)", fontsize=12, color=TEXT_COLOR)
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=11, colors=TEXT_COLOR)
    ax.tick_params(axis="y", labelsize=11, colors=TEXT_COLOR)
    ax.grid(alpha=0.55, axis="x", color=GRID_COLOR, linewidth=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9AA5B1")
    ax.spines["bottom"].set_color("#9AA5B1")
    ax.set_facecolor("#FAFBFC")
    fig.patch.set_facecolor("white")
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _save_alignment_plot(df, item_col, shap_col, lr_col, output_path, title):
    if df.empty:
        return None

    plt = _get_plotting_modules()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = df.sort_values("combined_rank", ascending=True).head(12).copy()
    plot_df["label"] = [fill(str(value).replace("_", " "), width=22) for value in plot_df[item_col]]
    plot_df = plot_df.iloc[::-1]
    positions = np.arange(len(plot_df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.barh(
        positions - width / 2,
        plot_df[shap_col],
        height=width,
        color=PRIMARY_BAR_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=1.0,
        label="XGB SHAP share",
    )
    ax.barh(
        positions + width / 2,
        plot_df[lr_col],
        height=width,
        color=SECONDARY_BAR_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=1.0,
        label="LR permutation share",
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(plot_df["label"])
    ax.set_title(title, fontsize=16, pad=14, color=TEXT_COLOR, fontweight="semibold")
    ax.set_xlabel("Normalized importance share", fontsize=12, color=TEXT_COLOR)
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=11, colors=TEXT_COLOR)
    ax.tick_params(axis="y", labelsize=11, colors=TEXT_COLOR)
    ax.grid(alpha=0.55, axis="x", color=GRID_COLOR, linewidth=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9AA5B1")
    ax.spines["bottom"].set_color("#9AA5B1")
    ax.set_facecolor("#FAFBFC")
    fig.patch.set_facecolor("white")
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _build_comparison(
    shap_df,
    lr_df,
    *,
    item_col,
    shap_score_col,
    lr_score_col,
    shap_norm_col,
    lr_norm_col,
):
    merged = pd.merge(
        shap_df[[item_col, shap_score_col]].copy(),
        lr_df[[item_col, lr_score_col]].copy(),
        on=item_col,
        how="inner",
    )
    if merged.empty:
        return merged, pd.DataFrame()

    merged[shap_norm_col] = _normalize_importance(merged[shap_score_col])
    merged[lr_norm_col] = _normalize_importance(merged[lr_score_col])
    merged["rank_shap"] = merged[shap_score_col].rank(method="average", ascending=False)
    merged["rank_lr"] = merged[lr_score_col].rank(method="average", ascending=False)
    merged["combined_rank"] = (merged["rank_shap"] + merged["rank_lr"]) / 2.0
    merged["rank_gap"] = merged["rank_shap"] - merged["rank_lr"]
    merged["rank_gap_abs"] = merged["rank_gap"].abs()
    merged["share_gap"] = merged[shap_norm_col] - merged[lr_norm_col]
    merged["share_gap_abs"] = merged["share_gap"].abs()
    merged = merged.sort_values(
        ["combined_rank", "rank_gap_abs", item_col],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    shap_rank = merged.set_index(item_col)["rank_shap"]
    lr_rank = merged.set_index(item_col)["rank_lr"]
    summary_df = pd.DataFrame(
        [
            {
                "n_overlap": int(len(merged)),
                "spearman_rank_corr": _safe_spearman(shap_rank, lr_rank),
                "top5_jaccard": _topk_jaccard(shap_rank, lr_rank, k=5),
                "top10_jaccard": _topk_jaccard(shap_rank, lr_rank, k=10),
                "mean_abs_rank_gap": float(merged["rank_gap_abs"].mean()),
                "mean_abs_share_gap": float(merged["share_gap_abs"].mean()),
            }
        ]
    )
    return merged, summary_df


def run_explainability_consistency(
    context=None,
    shap_output_dir="figs",
    xai_output_dir="experiments/cache/xai",
    output_dir=DEFAULT_OUTPUT_DIR,
    force_rerun=False,
):
    from experiments.common import stage
    from experiments.shap_analysis import run_shap_analysis
    from experiments.xai_analysis import run_xai_analysis

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stage("Explainability Consistency: Collecting Upstream Outputs")
    if force_rerun:
        shap_outputs = run_shap_analysis(context=context)
        xai_outputs = run_xai_analysis(context=context, output_dir=xai_output_dir)
    else:
        shap_outputs = run_shap_analysis(context=context) if not (Path(shap_output_dir) / "shap_family_summary.csv").exists() else None
        xai_outputs = run_xai_analysis(context=context, output_dir=xai_output_dir) if not (Path(xai_output_dir) / "lr_family_permutation_summary.csv").exists() else None

    shap_family_path = Path(shap_output_dir) / "shap_family_summary.csv"
    shap_theory_path = Path(shap_output_dir) / "shap_theory_summary.csv"
    shap_feature_path = Path(shap_output_dir) / "shap_feature_summary.csv"
    lr_family_path = Path(xai_output_dir) / "lr_family_permutation_summary.csv"
    lr_theory_path = Path(xai_output_dir) / "lr_theory_permutation_summary.csv"
    lr_feature_path = Path(xai_output_dir) / "lr_feature_permutation_summary.csv"

    stage("Explainability Consistency: Loading Comparison Tables")
    shap_family_df = pd.read_csv(shap_family_path)
    shap_theory_df = pd.read_csv(shap_theory_path)
    shap_feature_df = pd.read_csv(shap_feature_path)
    lr_family_df = pd.read_csv(lr_family_path)
    lr_theory_df = pd.read_csv(lr_theory_path)
    lr_feature_df = pd.read_csv(lr_feature_path)

    stage("Explainability Consistency: Comparing XGB SHAP vs LR Permutation")
    family_compare_df, family_summary_df = _build_comparison(
        shap_family_df,
        lr_family_df.rename(columns={"group_name": "family"}),
        item_col="family",
        shap_score_col="total_abs_shap",
        lr_score_col="auc_drop_mean",
        shap_norm_col="shap_share",
        lr_norm_col="lr_share",
    )
    theory_compare_df, theory_summary_df = _build_comparison(
        shap_theory_df.rename(columns={"theory_group": "theory_family"}),
        lr_theory_df.rename(columns={"theory_family": "theory_family"}),
        item_col="theory_family",
        shap_score_col="total_abs_shap",
        lr_score_col="auc_drop_mean",
        shap_norm_col="shap_share",
        lr_norm_col="lr_share",
    )
    feature_compare_df, feature_summary_df = _build_comparison(
        shap_feature_df,
        lr_feature_df,
        item_col="feature_name",
        shap_score_col="mean_abs_shap",
        lr_score_col="importance_mean",
        shap_norm_col="shap_share",
        lr_norm_col="lr_share",
    )

    family_compare_path = output_path / "family_alignment.csv"
    family_summary_path = output_path / "family_alignment_summary.csv"
    theory_compare_path = output_path / "theory_alignment.csv"
    theory_summary_path = output_path / "theory_alignment_summary.csv"
    feature_compare_path = output_path / "feature_alignment.csv"
    feature_summary_path = output_path / "feature_alignment_summary.csv"

    family_compare_df.to_csv(family_compare_path, index=False)
    family_summary_df.to_csv(family_summary_path, index=False)
    theory_compare_df.to_csv(theory_compare_path, index=False)
    theory_summary_df.to_csv(theory_summary_path, index=False)
    feature_compare_df.to_csv(feature_compare_path, index=False)
    feature_summary_df.to_csv(feature_summary_path, index=False)

    family_plot_path = _save_alignment_plot(
        family_compare_df,
        "family",
        "shap_share",
        "lr_share",
        output_path / "family_alignment.png",
        "XGB SHAP vs LR Permutation by Feature Family",
    )
    family_rank_plot_path = _save_rank_plot(
        family_compare_df,
        "family",
        output_path / "family_rank_alignment.png",
        "Family Rank Agreement",
    )
    theory_plot_path = _save_alignment_plot(
        theory_compare_df,
        "theory_family",
        "shap_share",
        "lr_share",
        output_path / "theory_alignment.png",
        "XGB SHAP vs LR Permutation by Theory Group",
    )
    feature_plot_path = _save_alignment_plot(
        feature_compare_df,
        "feature_name",
        "shap_share",
        "lr_share",
        output_path / "feature_alignment_top12.png",
        "Top Shared Non-Embedding Features",
    )
    feature_rank_plot_path = _save_rank_plot(
        feature_compare_df,
        "feature_name",
        output_path / "feature_rank_alignment.png",
        "Feature Rank Agreement",
    )

    print("Saved explainability consistency outputs:")
    for path in [
        family_compare_path,
        family_summary_path,
        theory_compare_path,
        theory_summary_path,
        feature_compare_path,
        feature_summary_path,
    ]:
        print(f"  {path}")
    for plot_path in [
        family_plot_path,
        family_rank_plot_path,
        theory_plot_path,
        feature_plot_path,
        feature_rank_plot_path,
    ]:
        if plot_path:
            print(f"  {plot_path}")

    if not family_summary_df.empty:
        print("\nFamily-level agreement:")
        print(family_summary_df.to_string(index=False))
    if not theory_summary_df.empty:
        print("\nTheory-level agreement:")
        print(theory_summary_df.to_string(index=False))
    if not feature_summary_df.empty:
        print("\nFeature-level agreement:")
        print(feature_summary_df.to_string(index=False))

    return {
        "family_compare_df": family_compare_df,
        "family_summary_df": family_summary_df,
        "theory_compare_df": theory_compare_df,
        "theory_summary_df": theory_summary_df,
        "feature_compare_df": feature_compare_df,
        "feature_summary_df": feature_summary_df,
        "paths": {
            "family_compare": str(family_compare_path),
            "family_summary": str(family_summary_path),
            "theory_compare": str(theory_compare_path),
            "theory_summary": str(theory_summary_path),
            "feature_compare": str(feature_compare_path),
            "feature_summary": str(feature_summary_path),
            "family_plot": family_plot_path,
            "family_rank_plot": family_rank_plot_path,
            "theory_plot": theory_plot_path,
            "feature_plot": feature_plot_path,
            "feature_rank_plot": feature_rank_plot_path,
        },
        "upstream_outputs": {
            "shap": shap_outputs,
            "xai": xai_outputs,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shap-output-dir", type=str, default="figs")
    parser.add_argument("--xai-output-dir", type=str, default="experiments/cache/xai")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_explainability_consistency(
        shap_output_dir=args.shap_output_dir,
        xai_output_dir=args.xai_output_dir,
        output_dir=args.output_dir,
        force_rerun=args.force_rerun,
    )
