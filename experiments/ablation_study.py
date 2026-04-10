import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ablation import run_ablation
from experiments.common import (
    build_experiment_context,
    build_feature_blocks,
    stage,
)
from models.lr import run_lr_ablation


def run_ablation_study(context=None, output_dir="experiments/cache/ablation"):
    if context is None:
        context = build_experiment_context()

    stage("Running Ablation Study on Feature Combinations")
    blocks = build_feature_blocks(context)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\nFeature Block Dimensions:")
    for name, feat in blocks.items():
        print(f"  {name}: {feat.shape[1]} dims")

    ablation_df = run_ablation(
        blocks,
        context["y"],
        context["train_idx"],
        context["val_idx"],
        context["test_idx"],
        run_lr_ablation,
    )

    full_val = float(
        ablation_df.loc[ablation_df["exp"] == "FULL_all", "auc_val"].values[0]
    )
    full_test = float(
        ablation_df.loc[ablation_df["exp"] == "FULL_all", "auc_test"].values[0]
    )
    ablation_df["delta_val_vs_full"] = ablation_df["auc_val"] - full_val
    ablation_df["delta_test_vs_full"] = ablation_df["auc_test"] - full_test

    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS (sorted by VAL AUC)")
    print("=" * 80)

    ablation_sorted = ablation_df.sort_values("auc_val", ascending=False)
    print(
        ablation_sorted[
            [
                "exp",
                "keys",
                "dim",
                "best_C",
                "auc_train",
                "auc_val",
                "auc_test",
                "gap_train_test",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("DELTA vs FULL MODEL (positive = better than FULL)")
    print("=" * 80)

    delta_sorted = ablation_df.sort_values("delta_val_vs_full", ascending=False)
    print(
        delta_sorted[
            [
                "exp",
                "delta_val_vs_full",
                "delta_test_vs_full",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    top3_val = ablation_sorted.head(3)["exp"].tolist()
    print("\nTop 3 by VAL AUC:")
    for rank, exp_name in enumerate(top3_val, start=1):
        row = ablation_df[ablation_df["exp"] == exp_name].iloc[0]
        print(f"  {rank}. {exp_name}: {row['auc_val']:.4f} (TEST: {row['auc_test']:.4f})")

    full_dim = int(ablation_df.loc[ablation_df["exp"] == "FULL_all", "dim"].values[0])
    threshold = full_val - 0.005
    good_minimal = ablation_df[
        (ablation_df["auc_val"] >= threshold) & (ablation_df["dim"] < full_dim)
    ].sort_values("dim")

    if len(good_minimal) > 0:
        print(f"\nMinimal feature sets (VAL AUC >= {threshold:.4f}):")
        for _, row in good_minimal.iterrows():
            print(f"  - {row['exp']}: {row['dim']} dims, VAL AUC = {row['auc_val']:.4f}")
    else:
        print(f"\nNo simpler model achieves VAL AUC >= {threshold:.4f}")

    print("\n" + "=" * 80)
    ablation_path = output_path / "ablation_results.csv"
    ablation_rounded_path = output_path / "ablation_results_rounded_3dp.csv"
    key_findings = pd.DataFrame(
        [
            {
                "best_experiment": ablation_sorted.iloc[0]["exp"],
                "best_val_auc": ablation_sorted.iloc[0]["auc_val"],
                "best_test_auc": ablation_sorted.iloc[0]["auc_test"],
                "full_model_val_auc": full_val,
                "full_model_test_auc": full_test,
            }
        ]
    )
    findings_path = output_path / "ablation_key_findings.csv"
    findings_rounded_path = output_path / "ablation_key_findings_rounded_3dp.csv"

    ablation_df.to_csv(ablation_path, index=False)
    key_findings.to_csv(findings_path, index=False)
    ablation_df.round(3).to_csv(ablation_rounded_path, index=False)
    key_findings.round(3).to_csv(findings_rounded_path, index=False)

    print("Saved ablation results:", ablation_path)
    print("Saved rounded ablation results:", ablation_rounded_path)
    print("Saved ablation key findings:", findings_path)
    print("Saved rounded ablation key findings:", findings_rounded_path)
    return ablation_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-feature-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="experiments/cache/ablation")
    args = parser.parse_args()

    context = build_experiment_context(object_feature_path=args.object_feature_path)
    run_ablation_study(context=context, output_dir=args.output_dir)
