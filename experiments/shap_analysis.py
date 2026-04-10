import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common import (
    CHANNEL_FEATURE_NAMES,
    NUMERIC_FEATURE_NAMES,
    THEORY_GROUP_LABELS,
    TOPIC_MONTHLY_FEATURE_NAMES,
    build_experiment_context,
    split_xy,
    stage,
)
from models.xgb import train_xgb

try:
    import shap
except ImportError:
    shap = None


OUTPUT_DIR = "figs"
SUMMARY_CSV = "shap_group_summary.csv"
PLOT_FILE = "shap_group_importance.png"
MEAN_PLOT_FILE = "shap_group_importance_mean.png"
TOTAL_PLOT_FILE = "shap_group_importance_total.png"
FAMILY_SUMMARY_CSV = "shap_family_summary.csv"
THEORY_SUMMARY_CSV = "shap_theory_summary.csv"
FAMILY_PLOT_FILE = "shap_family_importance_mean.png"
THEORY_PLOT_FILE = "shap_theory_importance_total.png"
FEATURE_SUMMARY_CSV = "shap_feature_summary.csv"
FEATURE_PLOT_FILE = "shap_metadata_importance.png"
SHAP_BEESWARM_FILE = "shap_metadata_beeswarm.png"
SHAP_OFFICIAL_BEESWARM_FILE = "shap_official_beeswarm.png"
SHAP_OFFICIAL_BAR_FILE = "shap_official_bar.png"
SHAP_SCATTER_PREFIX = "shap_scatter_"


def compute_xgb_shap_values(model, x_matrix):
    booster = model.get_booster()
    dmatrix = xgb.DMatrix(x_matrix)
    contribs = booster.predict(dmatrix, pred_contribs=True)

    # The last column is the bias term.
    return contribs[:, :-1], contribs[:, -1]


def compute_tree_shap_values(model, x_matrix):
    if shap is not None:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_matrix)
        return np.asarray(shap_values), "shap"

    shap_values, _ = compute_xgb_shap_values(model, x_matrix)
    return shap_values, "xgboost_pred_contribs"


def _feature_family(feature_name):
    if feature_name.startswith("embed_"):
        return "embedding"
    if feature_name in TOPIC_MONTHLY_FEATURE_NAMES:
        return "topic_monthly"
    if feature_name.startswith("topic_"):
        return "topic"
    if feature_name in {"channel_size_log", "channel_recent_performance"}:
        return "channel_context"
    if feature_name in CHANNEL_FEATURE_NAMES:
        return "channel_metadata"
    if feature_name in NUMERIC_FEATURE_NAMES:
        return "framing"
    return "visual_object"


def _theory_family(family_name):
    if family_name in {"embedding", "topic"}:
        return THEORY_GROUP_LABELS["SEM"]
    if family_name in {"framing", "topic_monthly"}:
        return THEORY_GROUP_LABELS["FRM"]
    if family_name in {"channel_context", "channel_metadata"}:
        return THEORY_GROUP_LABELS["CTX"]
    return "auxiliary_visual"


def summarize_group_shap(shap_values, feature_groups):
    rows = []

    for group in feature_groups:
        group_values = shap_values[:, group["indices"]]
        total_abs_shap = float(np.abs(group_values).sum(axis=1).mean())
        mean_abs_shap = float(np.abs(group_values).mean())
        mean_shap = float(group_values.mean())

        rows.append(
            {
                "group_name": group["group_name"],
                "n_dims": len(group["indices"]),
                "total_abs_shap": total_abs_shap,
                "mean_abs_shap": mean_abs_shap,
                "mean_shap": mean_shap,
            }
        )

    summary_df = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
    summary_df["rank"] = np.arange(1, len(summary_df) + 1)
    return summary_df[
        [
            "rank",
            "group_name",
            "n_dims",
            "mean_abs_shap",
            "total_abs_shap",
            "mean_shap",
        ]
    ]


def summarize_family_shap(shap_values, feature_names):
    family_to_indices = {}
    for idx, feature_name in enumerate(feature_names):
        family = _feature_family(feature_name)
        if family == "visual_object":
            continue
        family_to_indices.setdefault(family, []).append(idx)

    ordered_families = [
        "embedding",
        "topic",
        "framing",
        "topic_monthly",
        "channel_context",
        "channel_metadata",
    ]
    rows = []
    for family in ordered_families:
        indices = family_to_indices.get(family)
        if not indices:
            continue
        family_values = shap_values[:, indices]
        rows.append(
            {
                "family": family,
                "n_dims": len(indices),
                "mean_abs_shap": float(np.abs(family_values).mean()),
                "total_abs_shap": float(np.abs(family_values).sum(axis=1).mean()),
                "mean_shap": float(family_values.mean()),
            }
        )

    summary_df = pd.DataFrame(rows).sort_values("total_abs_shap", ascending=False).reset_index(drop=True)
    summary_df["rank"] = np.arange(1, len(summary_df) + 1)
    return summary_df[["rank", "family", "n_dims", "mean_abs_shap", "total_abs_shap", "mean_shap"]]


def summarize_theory_shap(family_summary_df):
    theory_df = (
        family_summary_df.assign(theory_group=family_summary_df["family"].map(_theory_family))
        .groupby("theory_group", as_index=False)
        .agg(
            n_dims=("n_dims", "sum"),
            mean_abs_shap=("mean_abs_shap", "mean"),
            total_abs_shap=("total_abs_shap", "sum"),
            mean_shap=("mean_shap", "mean"),
        )
        .sort_values("total_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    theory_df["rank"] = np.arange(1, len(theory_df) + 1)
    return theory_df[["rank", "theory_group", "n_dims", "mean_abs_shap", "total_abs_shap", "mean_shap"]]


def summarize_feature_shap(shap_values, feature_names, exclude_prefixes=("embed_",)):
    rows = []

    for feature_idx, feature_name in enumerate(feature_names):
        if feature_name.startswith(exclude_prefixes):
            continue

        feature_values = shap_values[:, feature_idx]
        rows.append(
            {
                "feature_name": feature_name,
                "mean_abs_shap": float(np.abs(feature_values).mean()),
                "mean_shap": float(feature_values.mean()),
            }
        )

    summary_df = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
    summary_df["rank"] = np.arange(1, len(summary_df) + 1)
    return summary_df[["rank", "feature_name", "mean_abs_shap", "mean_shap"]]


def save_group_shap_plot(summary_df, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, PLOT_FILE)
    mean_plot_path = os.path.join(output_dir, MEAN_PLOT_FILE)
    total_plot_path = os.path.join(output_dir, TOTAL_PLOT_FILE)

    top_df = summary_df.head(20).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    axes[0].barh(top_df["group_name"], top_df["mean_abs_shap"], color="#2a6f97")
    axes[0].set_xlabel("Mean |SHAP| Per Feature Dimension")
    axes[0].set_ylabel("Feature Group")
    axes[0].set_title("Grouped SHAP (Mean Per Dimension)")
    axes[0].grid(axis="x", alpha=0.3)

    axes[1].barh(top_df["group_name"], top_df["total_abs_shap"], color="#d17a22")
    axes[1].set_xlabel("Total |SHAP|")
    axes[1].set_title("Grouped SHAP (Total Contribution)")
    axes[1].grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top_df["group_name"], top_df["mean_abs_shap"], color="#2a6f97")
    ax.set_xlabel("Mean |SHAP| Per Feature Dimension")
    ax.set_ylabel("Feature Group")
    ax.set_title("Grouped SHAP (Mean Per Dimension)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(mean_plot_path, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top_df["group_name"], top_df["total_abs_shap"], color="#d17a22")
    ax.set_xlabel("Total |SHAP|")
    ax.set_ylabel("Feature Group")
    ax.set_title("Grouped SHAP (Total Contribution)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(total_plot_path, dpi=200)
    plt.close(fig)

    return {
        "dual_plot_path": plot_path,
        "mean_plot_path": mean_plot_path,
        "total_plot_path": total_plot_path,
    }


def save_family_shap_plot(summary_df, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, FAMILY_PLOT_FILE)

    plot_df = summary_df.sort_values("total_abs_shap", ascending=False).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.barh(plot_df["family"], plot_df["total_abs_shap"], color="#2a6f97")
    ax.set_xlabel("Total |SHAP|")
    ax.set_ylabel("Feature Family")
    ax.set_title("Family-Level SHAP (Total Contribution)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def save_theory_shap_plot(summary_df, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, THEORY_PLOT_FILE)

    plot_df = summary_df.sort_values("total_abs_shap", ascending=False).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.barh(plot_df["theory_group"], plot_df["total_abs_shap"], color="#d17a22")
    ax.set_xlabel("Total |SHAP|")
    ax.set_ylabel("Theory Group")
    ax.set_title("Theory-Level SHAP (Total Contribution)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def save_summary_csv(summary_df, file_name, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, file_name)
    summary_df.to_csv(csv_path, index=False)
    return csv_path


def save_feature_shap_plot(summary_df, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, FEATURE_PLOT_FILE)

    top_df = summary_df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top_df["feature_name"], top_df["mean_abs_shap"], color="#bc4749")
    ax.set_xlabel("Mean |SHAP|")
    ax.set_ylabel("Non-Embedding Feature")
    ax.set_title("Metadata / Topic SHAP Importance")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    return plot_path


def save_shap_beeswarm_plot(shap_values, x_test, feature_names, output_dir=OUTPUT_DIR):
    if shap is None:
        return None

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, SHAP_BEESWARM_FILE)

    non_embedding_indices = [
        idx for idx, feature_name in enumerate(feature_names)
        if not feature_name.startswith("embed_")
    ]
    if not non_embedding_indices:
        return None

    shap_subset = shap_values[:, non_embedding_indices]
    x_subset = x_test[:, non_embedding_indices]
    feature_subset = [feature_names[idx] for idx in non_embedding_indices]

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_subset,
        x_subset,
        feature_names=feature_subset,
        show=False,
        max_display=15,
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    return plot_path


def _build_non_embedding_explanation(shap_values, x_test, feature_names):
    non_embedding_indices = [
        idx for idx, feature_name in enumerate(feature_names)
        if not feature_name.startswith("embed_")
    ]
    if not non_embedding_indices:
        return None

    shap_subset = shap_values[:, non_embedding_indices]
    x_subset = x_test[:, non_embedding_indices]
    feature_subset = [feature_names[idx] for idx in non_embedding_indices]

    return shap.Explanation(
        values=shap_subset,
        data=x_subset,
        feature_names=feature_subset,
    )


def _disable_grid_for_current_figure():
    fig = plt.gcf()
    for ax in fig.axes:
        ax.grid(False)


def save_official_shap_plots(
    shap_values,
    x_test,
    feature_names,
    feature_summary_df,
    output_dir=OUTPUT_DIR,
    max_display=15,
    top_scatter_features=3,
):
    if shap is None:
        return {}

    os.makedirs(output_dir, exist_ok=True)
    explanation = _build_non_embedding_explanation(shap_values, x_test, feature_names)
    if explanation is None:
        return {}

    output_paths = {}

    beeswarm_path = os.path.join(output_dir, SHAP_OFFICIAL_BEESWARM_FILE)
    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    _disable_grid_for_current_figure()
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=200, bbox_inches="tight")
    plt.close()
    output_paths["official_beeswarm_path"] = beeswarm_path

    bar_path = os.path.join(output_dir, SHAP_OFFICIAL_BAR_FILE)
    plt.figure(figsize=(10, 7))
    shap.plots.bar(explanation, max_display=max_display, show=False)
    _disable_grid_for_current_figure()
    plt.tight_layout()
    plt.savefig(bar_path, dpi=200, bbox_inches="tight")
    plt.close()
    output_paths["official_bar_path"] = bar_path

    scatter_paths = []
    for feature_name in feature_summary_df.head(top_scatter_features)["feature_name"]:
        if feature_name not in explanation.feature_names:
            continue
        scatter_path = os.path.join(
            output_dir,
            f"{SHAP_SCATTER_PREFIX}{feature_name.replace('/', '_').replace(' ', '_')}.png",
        )
        plt.figure(figsize=(8, 6))
        shap.plots.scatter(explanation[:, feature_name], show=False)
        plt.tight_layout()
        plt.savefig(scatter_path, dpi=200, bbox_inches="tight")
        plt.close()
        scatter_paths.append(scatter_path)

    output_paths["scatter_paths"] = scatter_paths
    return output_paths


def run_shap_analysis(context=None):
    if context is None:
        context = build_experiment_context()

    split_data = split_xy(
        context["X_all"],
        context["y"],
        context["train_idx"],
        context["val_idx"],
        context["test_idx"],
    )

    y_train = split_data["y_train"]
    pos_ratio = y_train.mean()
    scale_pos_weight = (1 - pos_ratio) / pos_ratio

    stage("Training XGBoost For SHAP")
    xgb_model = train_xgb(
        split_data["X_train"],
        y_train,
        split_data["X_val"],
        split_data["y_val"],
        scale_pos_weight,
    )

    stage("Computing Grouped SHAP On Test Set")
    shap_values, shap_backend = compute_tree_shap_values(
        xgb_model,
        split_data["X_test"],
    )
    summary_df = summarize_group_shap(shap_values, context["feature_groups"])
    family_summary_df = summarize_family_shap(shap_values, context["feature_names"])
    theory_summary_df = summarize_theory_shap(family_summary_df)
    feature_summary_df = summarize_feature_shap(shap_values, context["feature_names"])

    print("\n" + "=" * 80)
    print("GROUPED SHAP SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("FAMILY-LEVEL SHAP SUMMARY")
    print("=" * 80)
    print(family_summary_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("THEORY-LEVEL SHAP SUMMARY")
    print("=" * 80)
    print(theory_summary_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("NON-EMBEDDING FEATURE SHAP SUMMARY")
    print("=" * 80)
    print(feature_summary_df.head(20).to_string(index=False))

    csv_path = save_summary_csv(summary_df, SUMMARY_CSV)
    family_csv_path = save_summary_csv(family_summary_df, FAMILY_SUMMARY_CSV)
    theory_csv_path = save_summary_csv(theory_summary_df, THEORY_SUMMARY_CSV)
    feature_csv_path = save_summary_csv(feature_summary_df, FEATURE_SUMMARY_CSV)
    group_plot_paths = save_group_shap_plot(summary_df)
    family_plot_path = save_family_shap_plot(family_summary_df)
    theory_plot_path = save_theory_shap_plot(theory_summary_df)
    feature_plot_path = save_feature_shap_plot(feature_summary_df)
    beeswarm_plot_path = save_shap_beeswarm_plot(
        shap_values,
        split_data["X_test"],
        context["feature_names"],
    )
    official_plot_paths = save_official_shap_plots(
        shap_values,
        split_data["X_test"],
        context["feature_names"],
        feature_summary_df,
    )
    print(f"\nSHAP backend: {shap_backend}")
    print(f"\nSaved SHAP summary CSV: {csv_path}")
    print(f"Saved family SHAP summary CSV: {family_csv_path}")
    print(f"Saved theory SHAP summary CSV: {theory_csv_path}")
    print(f"Saved feature SHAP summary CSV: {feature_csv_path}")
    print(f"Saved SHAP dual plot: {group_plot_paths['dual_plot_path']}")
    print(f"Saved SHAP mean plot: {group_plot_paths['mean_plot_path']}")
    print(f"Saved SHAP total plot: {group_plot_paths['total_plot_path']}")
    print(f"Saved family SHAP plot: {family_plot_path}")
    print(f"Saved theory SHAP plot: {theory_plot_path}")
    print(f"Saved feature SHAP plot: {feature_plot_path}")
    if beeswarm_plot_path is not None:
        print(f"Saved SHAP beeswarm plot: {beeswarm_plot_path}")
    else:
        print("SHAP beeswarm plot skipped because `shap` is not installed.")
    if official_plot_paths.get("official_beeswarm_path"):
        print(f"Saved official SHAP beeswarm plot: {official_plot_paths['official_beeswarm_path']}")
        print(f"Saved official SHAP bar plot: {official_plot_paths['official_bar_path']}")
        for scatter_path in official_plot_paths.get("scatter_paths", []):
            print(f"Saved SHAP scatter plot: {scatter_path}")

    return {
        "model": xgb_model,
        "summary_df": summary_df,
        "family_summary_df": family_summary_df,
        "theory_summary_df": theory_summary_df,
        "feature_summary_df": feature_summary_df,
        "shap_values": shap_values,
        "shap_backend": shap_backend,
        "csv_path": csv_path,
        "family_csv_path": family_csv_path,
        "theory_csv_path": theory_csv_path,
        "feature_csv_path": feature_csv_path,
        "plot_path": group_plot_paths["dual_plot_path"],
        "mean_plot_path": group_plot_paths["mean_plot_path"],
        "total_plot_path": group_plot_paths["total_plot_path"],
        "family_plot_path": family_plot_path,
        "theory_plot_path": theory_plot_path,
        "feature_plot_path": feature_plot_path,
        "beeswarm_plot_path": beeswarm_plot_path,
        "official_beeswarm_path": official_plot_paths.get("official_beeswarm_path"),
        "official_bar_path": official_plot_paths.get("official_bar_path"),
        "scatter_paths": official_plot_paths.get("scatter_paths", []),
    }


if __name__ == "__main__":
    run_shap_analysis()
