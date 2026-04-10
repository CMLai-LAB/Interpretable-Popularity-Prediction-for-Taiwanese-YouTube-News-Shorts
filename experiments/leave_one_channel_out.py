import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common import (
    DEFAULT_EMBEDDING_MODEL,
    attach_embedding_features,
    build_base_context,
    stage,
)
from experiments.train_models import run_multi_seed_benchmark


def _slugify_channel_name(name):
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_") or "channel"


def _build_holdout_indices(records, holdout_channel):
    fit_train_idx = []
    label_train_idx = []
    val_idx = []
    test_idx = []

    for idx, record in enumerate(records):
        date_value = record["date"]
        is_holdout = record["channel"] == holdout_channel

        if date_value.year == 2025 and date_value.month <= 10:
            # Use all historical data to define channel-specific label thresholds.
            label_train_idx.append(idx)
            if is_holdout:
                continue
            fit_train_idx.append(idx)
        elif date_value.year == 2025 and date_value.month >= 11 and is_holdout:
            val_idx.append(idx)
        elif date_value.year == 2026 and is_holdout:
            test_idx.append(idx)

    return (
        np.asarray(fit_train_idx),
        np.asarray(label_train_idx),
        np.asarray(val_idx),
        np.asarray(test_idx),
    )


def run_leave_one_channel_out(
    seeds=(42, 52, 62, 72, 82),
    embedding_model=DEFAULT_EMBEDDING_MODEL,
    prefix_mode="none",
    label_percentile=80,
    output_dir="experiments/cache/leave_one_channel_out",
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stage("Preparing Shared Embeddings For Leave-One-Channel-Out")
    shared_context = build_base_context(label_percentile=label_percentile)
    shared_context = attach_embedding_features(
        shared_context,
        embedding_model=embedding_model,
        prefix_mode=prefix_mode,
    )

    records = shared_context["records"]
    channels = sorted(np.unique([record["channel"] for record in records]))

    summary_rows = []
    raw_rows = []
    significance_rows = []
    fold_rows = []

    for holdout_channel in channels:
        fit_train_idx, label_train_idx, val_idx, test_idx = _build_holdout_indices(
            records,
            holdout_channel,
        )

        if len(fit_train_idx) == 0 or len(label_train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
            print(f"Skipping {holdout_channel}: insufficient samples for LOCO split.")
            continue

        stage(f"Running Leave-One-Channel-Out: holdout={holdout_channel}")
        holdout_context = build_base_context(
            records=records,
            train_idx=fit_train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            label_percentile=label_percentile,
            label_train_idx=label_train_idx,
            verbose=False,
        )
        holdout_context = attach_embedding_features(
            holdout_context,
            embedding_model=embedding_model,
            prefix_mode=prefix_mode,
            x_embed=shared_context["X_embed"],
            embedder=shared_context["embedder"],
            embedding_time_sec=shared_context["embedding_time_sec"],
            verbose=False,
        )

        run_output = run_multi_seed_benchmark(
            context=holdout_context,
            seeds=seeds,
            output_dir=output_path / _slugify_channel_name(holdout_channel),
        )

        test_y = holdout_context["y"][test_idx]
        val_y = holdout_context["y"][val_idx]
        fold_rows.append(
            {
                "holdout_channel": holdout_channel,
                "train_size": len(fit_train_idx),
                "label_train_size": len(label_train_idx),
                "val_size": len(val_idx),
                "test_size": len(test_idx),
                "val_positive_rate": float(val_y.mean()),
                "test_positive_rate": float(test_y.mean()),
            }
        )

        summary_df = run_output["summary_df"].copy()
        summary_df.insert(0, "holdout_channel", holdout_channel)
        summary_rows.append(summary_df)

        raw_df = run_output["raw_df"].copy()
        raw_df.insert(0, "holdout_channel", holdout_channel)
        raw_rows.append(raw_df)

        significance_df = run_output["significance_df"].copy()
        if not significance_df.empty:
            significance_df.insert(0, "holdout_channel", holdout_channel)
            significance_rows.append(significance_df)

    all_summary_df = pd.concat(summary_rows, ignore_index=True)
    all_raw_df = pd.concat(raw_rows, ignore_index=True)
    all_significance_df = (
        pd.concat(significance_rows, ignore_index=True)
        if significance_rows else pd.DataFrame()
    )
    fold_df = pd.DataFrame(fold_rows)

    aggregate_df = (
        all_summary_df.groupby("model", as_index=False)
        .agg(
            auc_test_macro_mean=("auc_test_mean", "mean"),
            auc_test_macro_std=("auc_test_mean", "std"),
            auc_val_macro_mean=("auc_val_mean", "mean"),
            ap_test_macro_mean=("ap_test_mean", "mean"),
            brier_test_macro_mean=("brier_test_mean", "mean"),
            ndcg10_macro_mean=("ndcg10_mean", "mean"),
            ndcg20_macro_mean=("ndcg20_mean", "mean"),
            spearman_macro_mean=("spearman_mean", "mean"),
            kendall_tau_macro_mean=("kendall_tau_mean", "mean"),
            train_time_sec_macro_mean=("train_time_sec_mean", "mean"),
            infer_test_time_sec_macro_mean=("infer_test_time_sec_mean", "mean"),
            test_latency_ms_per_sample_macro_mean=("test_latency_ms_per_sample_mean", "mean"),
            model_artifact_size_bytes_macro_mean=("model_artifact_size_bytes_mean", "mean"),
            model_param_count_macro_mean=("model_param_count_mean", "mean"),
        )
        .sort_values("auc_test_macro_mean", ascending=False)
        .reset_index(drop=True)
    )

    summary_path = output_path / "leave_one_channel_out_summary.csv"
    raw_path = output_path / "leave_one_channel_out_raw.csv"
    significance_path = output_path / "leave_one_channel_out_significance.csv"
    aggregate_path = output_path / "leave_one_channel_out_aggregate.csv"
    folds_path = output_path / "leave_one_channel_out_folds.csv"

    all_summary_df.to_csv(summary_path, index=False)
    all_raw_df.to_csv(raw_path, index=False)
    all_significance_df.to_csv(significance_path, index=False)
    aggregate_df.to_csv(aggregate_path, index=False)
    fold_df.to_csv(folds_path, index=False)

    print("Saved LOCO summary:", summary_path)
    print("Saved LOCO raw metrics:", raw_path)
    print("Saved LOCO significance:", significance_path)
    print("Saved LOCO aggregate summary:", aggregate_path)
    print("Saved LOCO fold metadata:", folds_path)

    return {
        "summary_df": all_summary_df,
        "raw_df": all_raw_df,
        "significance_df": all_significance_df,
        "aggregate_df": aggregate_df,
        "fold_df": fold_df,
        "paths": {
            "summary": str(summary_path),
            "raw": str(raw_path),
            "significance": str(significance_path),
            "aggregate": str(aggregate_path),
            "folds": str(folds_path),
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
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
    )
    parser.add_argument(
        "--prefix-mode",
        type=str,
        default="none",
        choices=["none", "e5"],
    )
    parser.add_argument(
        "--label-percentile",
        type=int,
        default=80,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/cache/leave_one_channel_out",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_leave_one_channel_out(
        seeds=tuple(args.seeds),
        embedding_model=args.embedding_model,
        prefix_mode=args.prefix_mode,
        label_percentile=args.label_percentile,
        output_dir=args.output_dir,
    )
