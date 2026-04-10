# Paper Results Package

This folder contains the curated experiment outputs that directly support the manuscript in [paper](/home/yeley/youtube_shorts/paper).

## Structure

- `tables/`: summary CSV files used to report the main quantitative findings
- `figures/`: paper-ready figures and figure manifest files

## Included result tables

- `train_models_multiseed_summary.csv`: main benchmark across LR, XGBoost, MLP, and Random Forest
- `train_models_significance.csv`: significance comparisons for the main benchmark
- `ablation_results.csv`: feature ablation results
- `channel_feature_study_summary.csv`: channel-feature study summary
- `label_sensitivity_summary.csv`: robustness across percentile thresholds
- `leave_one_channel_out_summary.csv`: leave-one-channel-out generalization summary
- `embedding_scale_results.csv`: embedding-scale robustness results across multilingual-E5 variants
- `lr_family_permutation_summary.csv`: Logistic Regression family-level permutation importance
- `lr_feature_permutation_summary.csv`: Logistic Regression feature-level permutation importance
- `lr_per_channel_family_importance.csv`: per-channel family-level importance
- `lr_percentile_family_stability.csv`: percentile-based stability of family-level importance
- `lr_theory_permutation_summary.csv`: theory-level grouped permutation importance
- `theory_alignment_summary.csv`: theory-level cross-model alignment summary
- `family_alignment_summary.csv`: family-level cross-model alignment summary
- `feature_alignment_summary.csv`: feature-level cross-model alignment summary

## Notes

- Large caches, embeddings, model artifacts, and diagnostic intermediate files remain under ignored directories and are not part of this package.
- The source files for these outputs remain in `experiments/cache/` and `figs/paper/`.
