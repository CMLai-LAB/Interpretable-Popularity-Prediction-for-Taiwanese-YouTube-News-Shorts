from experiments.channel_feature_study import (
    DEFAULT_FEATURE_SETS,
    parse_args,
    run_channel_feature_study,
)


def run_channel_prior_study(
    context=None,
    seeds=(42, 52, 62, 72, 82),
    feature_sets=DEFAULT_FEATURE_SETS,
    output_dir="experiments/cache/channel_prior_study",
):
    return run_channel_feature_study(
        context=context,
        seeds=seeds,
        feature_sets=feature_sets,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    args = parse_args()
    run_channel_prior_study(
        seeds=tuple(args.seeds),
        output_dir=args.output_dir,
    )
