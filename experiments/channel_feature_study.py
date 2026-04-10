import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common import (
    DEFAULT_EMBEDDING_MODEL,
    build_experiment_context,
    build_feature_blocks,
    concatenate_feature_blocks,
    split_xy,
    stage,
)
from models.lr import train_lr_with_sweep


DEFAULT_FEATURE_SETS = [
    ("context_only", ["C_CTX"]),
    ("metadata_only", ["C_META"]),
    ("channel_only", ["CTX"]),
    ("semantic_only", ["SEM"]),
    ("framing_only", ["FRM"]),
    ("semantic_plus_framing", ["SEM", "FRM"]),
    ("semantic_plus_channel", ["SEM", "CTX"]),
    ("full_model", ["SEM", "FRM", "CTX"]),
]


def _safe_roc_auc(y_true, y_score):
    if np.unique(y_true).size < 2:
        return None
    return roc_auc_score(y_true, y_score)


def bootstrap_auc_diff_ci(y_true, score_a, score_b, n_bootstrap=1000, seed=42, ci=0.95):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    score_a = np.asarray(score_a)
    score_b = np.asarray(score_b)
    n = len(y_true)
    diffs = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        auc_a = _safe_roc_auc(y_true[idx], score_a[idx])
        auc_b = _safe_roc_auc(y_true[idx], score_b[idx])
        if auc_a is not None and auc_b is not None:
            diffs.append(auc_a - auc_b)

    diffs = np.asarray(diffs, dtype=np.float64)
    alpha = 1.0 - ci
    ci_low, ci_high = np.quantile(diffs, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "mean_diff": float(np.mean(diffs)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_valid": int(len(diffs)),
        "significant": bool(ci_low > 0 or ci_high < 0),
    }


def _collect_metrics(y_train, y_val, y_test, proba_train, proba_val, proba_test):
    return {
        "auc_train": roc_auc_score(y_train, proba_train),
        "auc_val": roc_auc_score(y_val, proba_val),
        "auc_test": roc_auc_score(y_test, proba_test),
        "ap_test": average_precision_score(y_test, proba_test),
        "brier_train": brier_score_loss(y_train, proba_train),
        "brier_val": brier_score_loss(y_val, proba_val),
        "brier_test": brier_score_loss(y_test, proba_test),
        "overfit_gap": roc_auc_score(y_train, proba_train) - roc_auc_score(y_test, proba_test),
    }


def run_channel_feature_study(
    context=None,
    seeds=(42, 52, 62, 72, 82),
    feature_sets=DEFAULT_FEATURE_SETS,
    output_dir="experiments/cache/channel_feature_study",
):
    if context is None:
        context = build_experiment_context(embedding_model=DEFAULT_EMBEDDING_MODEL)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    blocks = build_feature_blocks(context)
    split_data = split_xy(
        context["X_all"],
        context["y"],
        context["train_idx"],
        context["val_idx"],
        context["test_idx"],
    )

    rows = []
    predictions = {}

    for seed in seeds:
        stage(f"Channel Feature Study (seed={seed})")
        for feature_set_name, keys in feature_sets:
            x_all = concatenate_feature_blocks(blocks, keys)
            x_train = x_all[context["train_idx"]]
            x_val = x_all[context["val_idx"]]
            x_test = x_all[context["test_idx"]]

            lr_model, best_c = train_lr_with_sweep(
                x_train,
                split_data["y_train"],
                x_val,
                split_data["y_val"],
                random_state=seed,
            )

            proba_train = lr_model.predict_proba(x_train)[:, 1]
            proba_val = lr_model.predict_proba(x_val)[:, 1]
            proba_test = lr_model.predict_proba(x_test)[:, 1]
            predictions.setdefault(feature_set_name, []).append(proba_test)

            row = {
                "seed": seed,
                "feature_set": feature_set_name,
                "keys": "+".join(keys),
                "dim": int(x_all.shape[1]),
                "best_C": float(best_c),
            }
            row.update(
                _collect_metrics(
                    split_data["y_train"],
                    split_data["y_val"],
                    split_data["y_test"],
                    proba_train,
                    proba_val,
                    proba_test,
                )
            )
            rows.append(row)

    raw_df = pd.DataFrame(rows)
    summary_df = (
        raw_df.groupby(["feature_set", "keys", "dim"], as_index=False)
        .agg(
            auc_test_mean=("auc_test", "mean"),
            auc_test_std=("auc_test", "std"),
            auc_val_mean=("auc_val", "mean"),
            ap_test_mean=("ap_test", "mean"),
            brier_test_mean=("brier_test", "mean"),
            overfit_gap_mean=("overfit_gap", "mean"),
        )
        .sort_values("auc_test_mean", ascending=False)
        .reset_index(drop=True)
    )

    best_feature_set = summary_df.iloc[0]["feature_set"]
    y_test = split_data["y_test"]
    ensemble_scores = {
        feature_set: np.vstack(score_list).mean(axis=0)
        for feature_set, score_list in predictions.items()
    }

    significance_rows = []
    best_scores = ensemble_scores[best_feature_set]
    for feature_set, scores in ensemble_scores.items():
        if feature_set == best_feature_set:
            continue
        diff = bootstrap_auc_diff_ci(y_test, best_scores, scores, seed=42)
        significance_rows.append(
            {
                "best_feature_set": best_feature_set,
                "compare_to": feature_set,
                "mean_auc_diff": diff["mean_diff"],
                "ci_low": diff["ci_low"],
                "ci_high": diff["ci_high"],
                "significant": diff["significant"],
                "n_valid": diff["n_valid"],
            }
        )
    significance_df = pd.DataFrame(significance_rows)
    if not significance_df.empty:
        significance_df = significance_df.sort_values("mean_auc_diff", ascending=False)

    raw_path = output_path / "channel_feature_study_raw.csv"
    summary_path = output_path / "channel_feature_study_summary.csv"
    significance_path = output_path / "channel_feature_study_significance.csv"

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    significance_df.to_csv(significance_path, index=False)

    print("Saved channel feature raw metrics:", raw_path)
    print("Saved channel feature summary:", summary_path)
    print("Saved channel feature significance:", significance_path)

    return {
        "raw_df": raw_df,
        "summary_df": summary_df,
        "significance_df": significance_df,
        "best_feature_set": best_feature_set,
        "paths": {
            "raw": str(raw_path),
            "summary": str(summary_path),
            "significance": str(significance_path),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 52, 62, 72, 82],
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/cache/channel_feature_study",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_channel_feature_study(
        seeds=tuple(args.seeds),
        output_dir=args.output_dir,
    )
