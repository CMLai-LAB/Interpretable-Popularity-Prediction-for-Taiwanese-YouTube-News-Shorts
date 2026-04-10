import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ablation_study import run_ablation_study
from experiments.channel_feature_study import run_channel_feature_study
from experiments.common import build_experiment_context, stage
from experiments.embedding_benchmark import run_embedding_benchmark
from experiments.label_sensitivity import run_label_sensitivity
from experiments.leave_one_channel_out import run_leave_one_channel_out
from experiments.shap_analysis import run_shap_analysis
from experiments.train_models import run_multi_seed_benchmark
from experiments.xai_analysis import run_xai_analysis


def run_paper_suite(
    run_main=True,
    run_channel_features=True,
    run_ablation=True,
    run_label=True,
    run_loco=True,
    run_shap=True,
    run_xai=True,
    run_embedding=False,
):
    stage("Building Shared Context For Paper Suite")
    context = build_experiment_context()
    outputs = {}

    if run_main:
        stage("Paper Suite: Main Multi-seed Benchmark")
        outputs["main"] = run_multi_seed_benchmark(context=context)

    if run_channel_features:
        stage("Paper Suite: Channel Feature Study")
        outputs["channel_features"] = run_channel_feature_study(context=context)

    if run_ablation:
        stage("Paper Suite: Feature Ablation")
        outputs["ablation"] = run_ablation_study(context=context)

    if run_label:
        stage("Paper Suite: Label Sensitivity")
        outputs["label_sensitivity"] = run_label_sensitivity()

    if run_loco:
        stage("Paper Suite: Leave-One-Channel-Out")
        outputs["leave_one_channel_out"] = run_leave_one_channel_out()

    if run_shap:
        stage("Paper Suite: SHAP Analysis")
        outputs["shap"] = run_shap_analysis(context=context)

    if run_xai:
        stage("Paper Suite: LR XAI Analysis")
        outputs["xai"] = run_xai_analysis(context=context)

    if run_embedding:
        stage("Paper Suite: Embedding Benchmark")
        outputs["embedding"] = run_embedding_benchmark()

    return outputs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-main", action="store_true")
    parser.add_argument("--skip-channel-features", action="store_true")
    parser.add_argument("--skip-prior", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-label", action="store_true")
    parser.add_argument("--skip-loco", action="store_true")
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--skip-xai", action="store_true")
    parser.add_argument("--run-embedding", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_paper_suite(
        run_main=not args.skip_main,
        run_channel_features=not (args.skip_channel_features or args.skip_prior),
        run_ablation=not args.skip_ablation,
        run_label=not args.skip_label,
        run_loco=not args.skip_loco,
        run_shap=not args.skip_shap,
        run_xai=not args.skip_xai,
        run_embedding=args.run_embedding,
    )
