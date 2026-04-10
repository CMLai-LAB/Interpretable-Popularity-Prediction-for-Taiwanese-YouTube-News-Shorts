import argparse
import sys
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common import (
    CHANNEL_FEATURE_NAMES,
    NUMERIC_FEATURE_NAMES,
    TOPIC_MONTHLY_FEATURE_NAMES,
    THEORY_GROUP_LABELS,
    apply_label_configuration,
    attach_embedding_features,
    build_base_context,
    build_experiment_context,
    split_xy,
    stage,
)
from models.lr import train_lr_with_sweep


DEFAULT_OUTPUT_DIR = "experiments/cache/xai"
DEFAULT_FAMILY_PLOT_TOP_N = 12
DEFAULT_FEATURE_PLOT_TOP_N = 25
DEFAULT_FEATURE_PERM_PLOT_TOP_N = 20
XAI_RAW_FIG_WIDTH = 10.2
XAI_RAW_FIG_HEIGHT = 7.2
PRIMARY_BAR_COLOR = "#2F4858"
SECONDARY_BAR_COLOR = "#567C8D"
EDGE_COLOR = "#243447"
GRID_COLOR = "#D8DEE9"
TEXT_COLOR = "#1F2933"


sns.set_theme(style="whitegrid", context="talk")


def _safe_auc(y_true, scores):
    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        return None
    return float(roc_auc_score(y_true, scores))


def _predict_positive_proba(model, x):
    proba = model.predict_proba(x)
    if proba.ndim != 2:
        raise ValueError("predict_proba must return a 2D array")
    if proba.shape[1] == 1:
        return np.zeros(len(x), dtype=np.float64)
    return proba[:, 1]


def _train_lr_context(context, seed):
    split_data = split_xy(
        context["X_all"],
        context["y"],
        context["train_idx"],
        context["val_idx"],
        context["test_idx"],
    )
    model, best_c = train_lr_with_sweep(
        split_data["X_train"],
        split_data["y_train"],
        split_data["X_val"],
        split_data["y_val"],
        random_state=seed,
    )
    proba_test = _predict_positive_proba(model, split_data["X_test"])
    return {
        "model": model,
        "best_c": float(best_c),
        "split_data": split_data,
        "proba_test": proba_test,
        "auc_test": _safe_auc(split_data["y_test"], proba_test),
    }


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


def _build_family_groups(context):
    family_to_indices = {}
    for idx, feature_name in enumerate(context["feature_names"]):
        family = _feature_family(feature_name)
        family_to_indices.setdefault(family, []).append(idx)

    ordered_families = [
        "embedding",
        "topic",
        "channel_context",
        "channel_metadata",
        "topic_monthly",
        "framing",
        "visual_object",
    ]
    groups = []
    for family in ordered_families:
        indices = family_to_indices.get(family)
        if not indices:
            continue
        groups.append({"group_name": family, "indices": indices})
    return groups


def _summarize_feature_coefficients(coef_vector, feature_names, exclude_prefixes=("embed_",)):
    rows = []
    for feature_name, coef in zip(feature_names, coef_vector):
        if feature_name.startswith(exclude_prefixes):
            continue
        rows.append(
            {
                "feature_name": feature_name,
                "family": _feature_family(feature_name),
                "coef": float(coef),
                "abs_coef": float(abs(coef)),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_coef", ascending=False).reset_index(drop=True)


def _summarize_family_coefficients(coef_vector, feature_names):
    rows = []
    family_to_values = {}
    for feature_name, coef in zip(feature_names, coef_vector):
        family = _feature_family(feature_name)
        family_to_values.setdefault(family, []).append(float(coef))

    for family, values in family_to_values.items():
        arr = np.asarray(values, dtype=np.float64)
        rows.append(
            {
                "family": family,
                "n_dims": int(arr.size),
                "mean_abs_coef": float(np.abs(arr).mean()),
                "total_abs_coef": float(np.abs(arr).sum()),
                "mean_coef": float(arr.mean()),
            }
        )

    return pd.DataFrame(rows).sort_values("mean_abs_coef", ascending=False).reset_index(drop=True)


def _summarize_theory_coefficients(family_summary_df):
    grouped = (
        family_summary_df.assign(theory_family=family_summary_df["family"].map(_theory_family))
        .groupby("theory_family", as_index=False)
        .agg(
            n_dims=("n_dims", "sum"),
            mean_abs_coef=("mean_abs_coef", "mean"),
            total_abs_coef=("total_abs_coef", "sum"),
            mean_coef=("mean_coef", "mean"),
        )
        .sort_values("total_abs_coef", ascending=False)
        .reset_index(drop=True)
    )
    return grouped


def _summarize_permutation_importance(model, x_test, y_test, feature_names, seed, exclude_prefixes=("embed_",)):
    perm = permutation_importance(
        model,
        x_test,
        y_test,
        scoring="roc_auc",
        n_repeats=10,
        random_state=seed,
        n_jobs=1,
    )
    rows = []
    for idx, feature_name in enumerate(feature_names):
        if feature_name.startswith(exclude_prefixes):
            continue
        rows.append(
            {
                "feature_name": feature_name,
                "family": _feature_family(feature_name),
                "importance_mean": float(perm.importances_mean[idx]),
                "importance_std": float(perm.importances_std[idx]),
            }
        )
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def _group_permutation_importance(model, x_test, y_test, groups, seed, n_repeats=20):
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    baseline_scores = _predict_positive_proba(model, x_test)
    baseline_auc = _safe_auc(y_test, baseline_scores)
    if baseline_auc is None:
        return pd.DataFrame(
            [
                {
                    "group_name": group["group_name"],
                    "n_dims": len(group["indices"]),
                    "auc_drop_mean": np.nan,
                    "auc_drop_std": np.nan,
                    "baseline_auc": np.nan,
                }
                for group in groups
            ]
        )

    rng = np.random.default_rng(seed)
    rows = []
    for group in groups:
        auc_drops = []
        for _ in range(n_repeats):
            x_perm = x_test.copy()
            perm_idx = rng.permutation(x_test.shape[0])
            x_perm[:, group["indices"]] = x_perm[perm_idx][:, group["indices"]]
            auc_perm = _safe_auc(y_test, _predict_positive_proba(model, x_perm))
            if auc_perm is not None:
                auc_drops.append(baseline_auc - auc_perm)

        rows.append(
            {
                "group_name": group["group_name"],
                "n_dims": len(group["indices"]),
                "auc_drop_mean": float(np.mean(auc_drops)) if auc_drops else np.nan,
                "auc_drop_std": float(np.std(auc_drops)) if auc_drops else np.nan,
                "baseline_auc": baseline_auc,
            }
        )

    return pd.DataFrame(rows).sort_values("auc_drop_mean", ascending=False).reset_index(drop=True)


def _summarize_theory_permutation_importance(family_perm_df):
    grouped = (
        family_perm_df.assign(theory_family=family_perm_df["group_name"].map(_theory_family))
        .groupby("theory_family", as_index=False)
        .agg(
            n_dims=("n_dims", "sum"),
            auc_drop_mean=("auc_drop_mean", "sum"),
            auc_drop_std=("auc_drop_std", "mean"),
            baseline_auc=("baseline_auc", "mean"),
        )
        .sort_values("auc_drop_mean", ascending=False)
        .reset_index(drop=True)
    )
    return grouped


def _rank_series(summary_df, item_col, score_col):
    ranked = summary_df[[item_col, score_col]].copy()
    ranked["rank"] = ranked[score_col].rank(method="average", ascending=False)
    return ranked.set_index(item_col)["rank"]


def _topk_overlap(rank_a, rank_b, k=10):
    top_a = set(rank_a.nsmallest(k).index)
    top_b = set(rank_b.nsmallest(k).index)
    if not top_a and not top_b:
        return 1.0
    union = top_a | top_b
    if not union:
        return 1.0
    return float(len(top_a & top_b) / len(union))


def _pairwise_stability(rank_series_by_name, top_k=10):
    names = list(rank_series_by_name)
    rows = []
    for i, left_name in enumerate(names):
        for right_name in names[i + 1:]:
            left_rank = rank_series_by_name[left_name]
            right_rank = rank_series_by_name[right_name]
            aligned = pd.concat([left_rank, right_rank], axis=1, join="inner").dropna()
            if aligned.empty:
                spearman = np.nan
            else:
                spearman = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman"))
            rows.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "spearman_rank_corr": spearman,
                    "topk_jaccard": _topk_overlap(left_rank, right_rank, k=top_k),
                }
            )
    return pd.DataFrame(rows)


def _top_local_contributions(x_row, coef_vector, feature_names, top_n=5):
    contribs = x_row * coef_vector
    rows = []
    for feature_name, contribution in zip(feature_names, contribs):
        if feature_name.startswith("embed_"):
            continue
        rows.append((feature_name, float(contribution)))
    rows.sort(key=lambda item: abs(item[1]), reverse=True)
    return rows[:top_n]


def _build_error_cases(records, context, split_data, proba_test, coef_vector, top_n=10):
    test_records = [records[idx] for idx in context["test_idx"]]
    y_test = split_data["y_test"]
    pred_label = (proba_test >= 0.5).astype(int)
    rows = []

    for local_idx, record in enumerate(test_records):
        true_label = int(y_test[local_idx])
        pred = int(pred_label[local_idx])
        if true_label == 0 and pred == 1:
            error_type = "false_positive"
            confidence = float(proba_test[local_idx])
        elif true_label == 1 and pred == 0:
            error_type = "false_negative"
            confidence = float(1.0 - proba_test[local_idx])
        else:
            error_type = "correct"
            confidence = float(max(proba_test[local_idx], 1.0 - proba_test[local_idx]))

        top_contribs = _top_local_contributions(
            split_data["X_test"][local_idx],
            coef_vector,
            context["feature_names"],
            top_n=5,
        )
        rows.append(
            {
                "video_id": record["video_id"],
                "channel": record["channel"],
                "date": record["date"].isoformat(),
                "title_text": record["title_text"],
                "comment_rate": float(record["comment_rate"]),
                "y_true": true_label,
                "y_pred": pred,
                "proba_hot": float(proba_test[local_idx]),
                "error_type": error_type,
                "confidence": confidence,
                "top_contributors": "; ".join(f"{name}:{value:.4f}" for name, value in top_contribs),
            }
        )

    errors_df = pd.DataFrame(rows)
    ranked_errors = (
        errors_df[errors_df["error_type"] != "correct"]
        .sort_values(["error_type", "confidence"], ascending=[True, False])
        .groupby("error_type", as_index=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    summary_df = (
        errors_df.groupby(["channel", "error_type"], as_index=False)
        .size()
        .pivot(index="channel", columns="error_type", values="size")
        .fillna(0)
        .reset_index()
    )
    return ranked_errors, summary_df


def _save_bar_plot(df, label_col, value_col, output_path, title, xlabel):
    if df.empty:
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = df.sort_values(value_col, ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(XAI_RAW_FIG_WIDTH, XAI_RAW_FIG_HEIGHT))
    labels = [fill(str(label).replace("_", " "), width=26) for label in plot_df[label_col]]
    values = plot_df[value_col].to_numpy()
    colors = [PRIMARY_BAR_COLOR] * len(plot_df)
    if len(colors) > 0:
        colors[0] = SECONDARY_BAR_COLOR

    plot_source = plot_df.copy()
    plot_source["_label_wrapped"] = labels
    order = plot_source["_label_wrapped"].tolist()
    sns.barplot(
        data=plot_source,
        x=value_col,
        y="_label_wrapped",
        order=order,
        orient="h",
        palette=colors,
        edgecolor=EDGE_COLOR,
        linewidth=1.1,
        ax=ax,
    )
    ax.set_title(title, fontsize=16, pad=14, color=TEXT_COLOR, fontweight="semibold")
    ax.set_xlabel(xlabel, fontsize=12, color=TEXT_COLOR)
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

    x_max = float(np.nanmax(values)) if len(values) else 0.0
    ax.set_xlim(0, x_max * 1.12 if x_max > 0 else 1.0)
    for idx, value in enumerate(values):
        ax.text(
            value + x_max * 0.015,
            idx,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=10,
            color=TEXT_COLOR,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def run_xai_analysis(
    context=None,
    seeds=(42, 52, 62, 72, 82),
    percentiles=(60, 70, 80, 90),
    primary_seed=42,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if context is None:
        context = build_experiment_context()

    family_groups = _build_family_groups(context)
    coefficient_rows = []
    family_coef_rows = []
    family_perm_rows = []
    feature_perm_rows = []
    family_ranks_by_seed = {}
    feature_ranks_by_seed = {}
    primary_run = None

    for seed in seeds:
        stage(f"LR XAI Analysis (seed={seed})")
        run_output = _train_lr_context(context, seed=seed)
        model = run_output["model"]
        coef_vector = model.coef_.ravel()

        feature_coef_df = _summarize_feature_coefficients(coef_vector, context["feature_names"])
        feature_coef_df.insert(0, "seed", seed)
        coefficient_rows.append(feature_coef_df)

        family_coef_df = _summarize_family_coefficients(coef_vector, context["feature_names"])
        family_coef_df.insert(0, "seed", seed)
        family_coef_rows.append(family_coef_df)

        family_perm_df = _group_permutation_importance(
            model,
            run_output["split_data"]["X_test"],
            run_output["split_data"]["y_test"],
            family_groups,
            seed=seed,
        )
        family_perm_df.insert(0, "seed", seed)
        family_perm_rows.append(family_perm_df)

        feature_perm_df = _summarize_permutation_importance(
            model,
            run_output["split_data"]["X_test"],
            run_output["split_data"]["y_test"],
            context["feature_names"],
            seed=seed,
        )
        feature_perm_df.insert(0, "seed", seed)
        feature_perm_rows.append(feature_perm_df)

        family_ranks_by_seed[f"seed_{seed}"] = _rank_series(
            family_perm_df,
            "group_name",
            "auc_drop_mean",
        )
        feature_ranks_by_seed[f"seed_{seed}"] = _rank_series(
            feature_coef_df,
            "feature_name",
            "abs_coef",
        )

        if seed == primary_seed:
            primary_run = run_output

    all_feature_coef_df = pd.concat(coefficient_rows, ignore_index=True)
    all_family_coef_df = pd.concat(family_coef_rows, ignore_index=True)
    all_family_perm_df = pd.concat(family_perm_rows, ignore_index=True)
    all_feature_perm_df = pd.concat(feature_perm_rows, ignore_index=True)

    feature_coef_summary_df = (
        all_feature_coef_df.groupby(["feature_name", "family"], as_index=False)
        .agg(
            mean_coef=("coef", "mean"),
            coef_std=("coef", "std"),
            mean_abs_coef=("abs_coef", "mean"),
            abs_coef_std=("abs_coef", "std"),
        )
        .sort_values("mean_abs_coef", ascending=False)
        .reset_index(drop=True)
    )
    family_coef_summary_df = (
        all_family_coef_df.groupby(["family", "n_dims"], as_index=False)
        .agg(
            mean_coef=("mean_coef", "mean"),
            mean_abs_coef=("mean_abs_coef", "mean"),
            total_abs_coef=("total_abs_coef", "mean"),
        )
        .sort_values("mean_abs_coef", ascending=False)
        .reset_index(drop=True)
    )
    theory_coef_summary_df = _summarize_theory_coefficients(family_coef_summary_df)
    family_perm_summary_df = (
        all_family_perm_df.groupby(["group_name", "n_dims"], as_index=False)
        .agg(
            auc_drop_mean=("auc_drop_mean", "mean"),
            auc_drop_std=("auc_drop_mean", "std"),
            baseline_auc=("baseline_auc", "mean"),
        )
        .sort_values("auc_drop_mean", ascending=False)
        .reset_index(drop=True)
    )
    theory_perm_summary_df = _summarize_theory_permutation_importance(family_perm_summary_df)
    feature_perm_summary_df = (
        all_feature_perm_df.groupby(["feature_name", "family"], as_index=False)
        .agg(
            importance_mean=("importance_mean", "mean"),
            importance_std=("importance_mean", "std"),
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    seed_family_stability_df = _pairwise_stability(family_ranks_by_seed, top_k=5)
    seed_feature_stability_df = _pairwise_stability(feature_ranks_by_seed, top_k=10)

    stage("LR XAI Analysis: Label Stability")
    shared_base_context = build_base_context(label_percentile=percentiles[0], verbose=False)
    shared_base_context = attach_embedding_features(
        shared_base_context,
        embedding_model=context["embedding_model"],
        x_embed=context["X_embed"],
        embedder=context.get("embedder"),
        embedding_time_sec=context.get("embedding_time_sec", 0.0),
        verbose=False,
    )

    percentile_family_rank = {}
    percentile_feature_rank = {}
    percentile_rows = []
    percentile_family_rows = []

    for percentile in percentiles:
        percentile_context = apply_label_configuration(
            shared_base_context,
            label_percentile=percentile,
            label_train_idx=shared_base_context["train_idx"],
            verbose=False,
        )
        run_output = _train_lr_context(percentile_context, seed=primary_seed)
        coef_vector = run_output["model"].coef_.ravel()
        feature_coef_df = _summarize_feature_coefficients(
            coef_vector,
            percentile_context["feature_names"],
        )
        feature_coef_df.insert(0, "label_percentile", percentile)
        percentile_rows.append(feature_coef_df)

        family_perm_df = _group_permutation_importance(
            run_output["model"],
            run_output["split_data"]["X_test"],
            run_output["split_data"]["y_test"],
            _build_family_groups(percentile_context),
            seed=primary_seed,
        )
        family_perm_df.insert(0, "label_percentile", percentile)
        percentile_family_rows.append(family_perm_df)

        percentile_feature_rank[f"p{percentile}"] = _rank_series(
            feature_coef_df,
            "feature_name",
            "abs_coef",
        )
        percentile_family_rank[f"p{percentile}"] = _rank_series(
            family_perm_df,
            "group_name",
            "auc_drop_mean",
        )

    percentile_feature_df = pd.concat(percentile_rows, ignore_index=True)
    percentile_family_df = pd.concat(percentile_family_rows, ignore_index=True)
    percentile_feature_stability_df = _pairwise_stability(percentile_feature_rank, top_k=10)
    percentile_family_stability_df = _pairwise_stability(percentile_family_rank, top_k=5)

    if primary_run is None:
        primary_run = _train_lr_context(context, seed=primary_seed)

    stage("LR XAI Analysis: Per-channel Importance")
    test_channels = np.asarray(context["channels_array"])[context["test_idx"]]
    per_channel_rows = []
    for channel in sorted(np.unique(test_channels)):
        channel_mask = test_channels == channel
        channel_x = primary_run["split_data"]["X_test"][channel_mask]
        channel_y = primary_run["split_data"]["y_test"][channel_mask]
        perm_df = _group_permutation_importance(
            primary_run["model"],
            channel_x,
            channel_y,
            family_groups,
            seed=primary_seed,
            n_repeats=10,
        )
        perm_df.insert(0, "channel", channel)
        perm_df.insert(1, "sample_size", int(channel_mask.sum()))
        perm_df.insert(2, "channel_auc", _safe_auc(channel_y, _predict_positive_proba(primary_run["model"], channel_x)))
        per_channel_rows.append(perm_df)
    per_channel_family_df = pd.concat(per_channel_rows, ignore_index=True)

    stage("LR XAI Analysis: Error Cases")
    error_cases_df, error_summary_df = _build_error_cases(
        context["records"],
        context,
        primary_run["split_data"],
        primary_run["proba_test"],
        primary_run["model"].coef_.ravel(),
    )

    coefficient_path = output_path / "lr_feature_coefficients_raw.csv"
    coefficient_summary_path = output_path / "lr_feature_coefficients_summary.csv"
    family_coef_path = output_path / "lr_family_coefficients_raw.csv"
    family_coef_summary_path = output_path / "lr_family_coefficients_summary.csv"
    theory_coef_summary_path = output_path / "lr_theory_coefficients_summary.csv"
    family_perm_path = output_path / "lr_family_permutation_raw.csv"
    family_perm_summary_path = output_path / "lr_family_permutation_summary.csv"
    theory_perm_summary_path = output_path / "lr_theory_permutation_summary.csv"
    feature_perm_path = output_path / "lr_feature_permutation_raw.csv"
    feature_perm_summary_path = output_path / "lr_feature_permutation_summary.csv"
    seed_family_stability_path = output_path / "lr_seed_family_stability.csv"
    seed_feature_stability_path = output_path / "lr_seed_feature_stability.csv"
    percentile_feature_path = output_path / "lr_percentile_feature_coefficients.csv"
    percentile_family_path = output_path / "lr_percentile_family_permutation.csv"
    percentile_feature_stability_path = output_path / "lr_percentile_feature_stability.csv"
    percentile_family_stability_path = output_path / "lr_percentile_family_stability.csv"
    per_channel_family_path = output_path / "lr_per_channel_family_importance.csv"
    error_cases_path = output_path / "lr_error_cases.csv"
    error_summary_path = output_path / "lr_error_summary_by_channel.csv"

    all_feature_coef_df.to_csv(coefficient_path, index=False)
    feature_coef_summary_df.to_csv(coefficient_summary_path, index=False)
    all_family_coef_df.to_csv(family_coef_path, index=False)
    family_coef_summary_df.to_csv(family_coef_summary_path, index=False)
    theory_coef_summary_df.to_csv(theory_coef_summary_path, index=False)
    all_family_perm_df.to_csv(family_perm_path, index=False)
    family_perm_summary_df.to_csv(family_perm_summary_path, index=False)
    theory_perm_summary_df.to_csv(theory_perm_summary_path, index=False)
    all_feature_perm_df.to_csv(feature_perm_path, index=False)
    feature_perm_summary_df.to_csv(feature_perm_summary_path, index=False)
    seed_family_stability_df.to_csv(seed_family_stability_path, index=False)
    seed_feature_stability_df.to_csv(seed_feature_stability_path, index=False)
    percentile_feature_df.to_csv(percentile_feature_path, index=False)
    percentile_family_df.to_csv(percentile_family_path, index=False)
    percentile_feature_stability_df.to_csv(percentile_feature_stability_path, index=False)
    percentile_family_stability_df.to_csv(percentile_family_stability_path, index=False)
    per_channel_family_df.to_csv(per_channel_family_path, index=False)
    error_cases_df.to_csv(error_cases_path, index=False)
    error_summary_df.to_csv(error_summary_path, index=False)

    family_plot_path = _save_bar_plot(
        family_perm_summary_df.head(DEFAULT_FAMILY_PLOT_TOP_N),
        "group_name",
        "auc_drop_mean",
        output_path / "lr_family_permutation_summary.png",
        "LR Family-Level Permutation Importance",
        "Mean Test AUC Drop",
    )
    feature_plot_path = _save_bar_plot(
        feature_coef_summary_df.head(DEFAULT_FEATURE_PLOT_TOP_N),
        "feature_name",
        "mean_abs_coef",
        output_path / "lr_feature_coefficients_summary.png",
        "LR Non-Embedding Coefficient Magnitude",
        "Mean |Coefficient|",
    )
    feature_perm_plot_path = _save_bar_plot(
        feature_perm_summary_df.head(DEFAULT_FEATURE_PERM_PLOT_TOP_N),
        "feature_name",
        "importance_mean",
        output_path / "lr_feature_permutation_summary.png",
        "LR Top Feature Permutation Importance",
        "Mean Test AUC Drop",
    )

    print("Saved LR XAI outputs:")
    for path in [
        coefficient_summary_path,
        family_coef_summary_path,
        theory_coef_summary_path,
        family_perm_summary_path,
        theory_perm_summary_path,
        feature_perm_summary_path,
        seed_family_stability_path,
        seed_feature_stability_path,
        percentile_feature_stability_path,
        percentile_family_stability_path,
        per_channel_family_path,
        error_cases_path,
        error_summary_path,
    ]:
        print(f"  {path}")
    if family_plot_path:
        print(f"  {family_plot_path}")
    if feature_plot_path:
        print(f"  {feature_plot_path}")
    if feature_perm_plot_path:
        print(f"  {feature_perm_plot_path}")

    return {
        "feature_coef_summary_df": feature_coef_summary_df,
        "family_coef_summary_df": family_coef_summary_df,
        "theory_coef_summary_df": theory_coef_summary_df,
        "family_perm_summary_df": family_perm_summary_df,
        "theory_perm_summary_df": theory_perm_summary_df,
        "feature_perm_summary_df": feature_perm_summary_df,
        "seed_family_stability_df": seed_family_stability_df,
        "seed_feature_stability_df": seed_feature_stability_df,
        "percentile_feature_stability_df": percentile_feature_stability_df,
        "percentile_family_stability_df": percentile_family_stability_df,
        "per_channel_family_df": per_channel_family_df,
        "error_cases_df": error_cases_df,
        "error_summary_df": error_summary_df,
        "paths": {
            "feature_coef_summary": str(coefficient_summary_path),
            "family_coef_summary": str(family_coef_summary_path),
            "theory_coef_summary": str(theory_coef_summary_path),
            "family_perm_summary": str(family_perm_summary_path),
            "theory_perm_summary": str(theory_perm_summary_path),
            "feature_perm_summary": str(feature_perm_summary_path),
            "seed_family_stability": str(seed_family_stability_path),
            "seed_feature_stability": str(seed_feature_stability_path),
            "percentile_feature_stability": str(percentile_feature_stability_path),
            "percentile_family_stability": str(percentile_family_stability_path),
            "per_channel_family": str(per_channel_family_path),
            "error_cases": str(error_cases_path),
            "error_summary": str(error_summary_path),
            "family_plot": family_plot_path,
            "feature_plot": feature_plot_path,
            "feature_perm_plot": feature_perm_plot_path,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--primary-seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62, 72, 82])
    parser.add_argument("--percentiles", nargs="+", type=int, default=[60, 70, 80, 90])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_xai_analysis(
        seeds=tuple(args.seeds),
        percentiles=tuple(args.percentiles),
        primary_seed=args.primary_seed,
        output_dir=args.output_dir,
    )
