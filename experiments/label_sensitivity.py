import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common import (
    DEFAULT_EMBEDDING_MODEL,
    apply_label_configuration,
    attach_embedding_features,
    build_base_context,
    stage,
)
from experiments.train_models import run_multi_seed_benchmark


def run_label_sensitivity(
    percentiles=(60, 70, 80, 90),
    seeds=(42, 52, 62, 72, 82),
    embedding_model=DEFAULT_EMBEDDING_MODEL,
    prefix_mode="none",
    output_dir="experiments/cache/label_sensitivity",
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stage("Preparing Shared Context For Label Sensitivity")
    shared_context = build_base_context(label_percentile=percentiles[0])
    shared_context = attach_embedding_features(
        shared_context,
        embedding_model=embedding_model,
        prefix_mode=prefix_mode,
    )

    summary_rows = []
    significance_rows = []
    raw_rows = []

    for percentile in percentiles:
        stage(f"Running Label Sensitivity @ percentile={percentile}")
        percentile_context = apply_label_configuration(
            shared_context,
            label_percentile=percentile,
            label_train_idx=shared_context["train_idx"],
            verbose=False,
        )
        run_output = run_multi_seed_benchmark(
            context=percentile_context,
            seeds=seeds,
            output_dir=output_path / f"p{percentile}",
        )

        summary_df = run_output["summary_df"].copy()
        summary_df.insert(0, "label_percentile", percentile)
        summary_rows.append(summary_df)

        significance_df = run_output["significance_df"].copy()
        if not significance_df.empty:
            significance_df.insert(0, "label_percentile", percentile)
            significance_rows.append(significance_df)

        raw_df = run_output["raw_df"].copy()
        raw_df.insert(0, "label_percentile", percentile)
        raw_rows.append(raw_df)

    all_summary_df = pd.concat(summary_rows, ignore_index=True)
    all_raw_df = pd.concat(raw_rows, ignore_index=True)
    all_significance_df = (
        pd.concat(significance_rows, ignore_index=True)
        if significance_rows else pd.DataFrame()
    )

    summary_path = output_path / "label_sensitivity_summary.csv"
    raw_path = output_path / "label_sensitivity_raw.csv"
    significance_path = output_path / "label_sensitivity_significance.csv"

    all_summary_df.to_csv(summary_path, index=False)
    all_raw_df.to_csv(raw_path, index=False)
    all_significance_df.to_csv(significance_path, index=False)

    print("Saved label sensitivity summary:", summary_path)
    print("Saved label sensitivity raw metrics:", raw_path)
    print("Saved label sensitivity significance:", significance_path)

    return {
        "summary_df": all_summary_df,
        "raw_df": all_raw_df,
        "significance_df": all_significance_df,
        "paths": {
            "summary": str(summary_path),
            "raw": str(raw_path),
            "significance": str(significance_path),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--percentiles",
        nargs="+",
        type=int,
        default=[60, 70, 80, 90],
    )
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
        "--output-dir",
        type=str,
        default="experiments/cache/label_sensitivity",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_label_sensitivity(
        percentiles=tuple(args.percentiles),
        seeds=tuple(args.seeds),
        embedding_model=args.embedding_model,
        prefix_mode=args.prefix_mode,
        output_dir=args.output_dir,
    )
