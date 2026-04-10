import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

RAW_PATH = "experiments/cache/train_models_multiseed_raw.csv"
SUMMARY_PATH = "experiments/cache/train_models_multiseed_summary.csv"
OUT_DIR = Path("figs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

name_map = {
    "LR": "Logistic Regression",
    "RF": "Random Forest",
    "XGB": "XGBoost",
    "MLP": "MLP",
}
order = ["Logistic Regression", "Random Forest", "XGBoost", "MLP"]

sns.set_theme(style="whitegrid", font_scale=1.1)

# -----------------------------
# 1. raw.csv -> boxplot
# -----------------------------
raw_df = pd.read_csv(RAW_PATH)
raw_df = raw_df[raw_df["model"].isin(["LR", "RF", "XGB", "MLP"])].copy()
raw_df["model_pretty"] = raw_df["model"].map(name_map)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.boxplot(
    data=raw_df,
    x="model_pretty",
    y="auc_test",
    order=order,
    showfliers=False,
    ax=axes[0],
)
axes[0].set_title("Test ROC-AUC Across Seeds")
axes[0].set_xlabel("")
axes[0].set_ylabel("ROC-AUC")
axes[0].tick_params(axis="x", rotation=20)

sns.boxplot(
    data=raw_df,
    x="model_pretty",
    y="ap_test",
    order=order,
    showfliers=False,
    ax=axes[1],
)
axes[1].set_title("Test Average Precision Across Seeds")
axes[1].set_xlabel("")
axes[1].set_ylabel("Average Precision")
axes[1].tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig(OUT_DIR / "model_boxplot_from_raw.png", dpi=300, bbox_inches="tight")
plt.close()

# -----------------------------
# 2. summary.csv -> mean ± std
# -----------------------------
summary_df = pd.read_csv(SUMMARY_PATH)
summary_df = summary_df[summary_df["model"].isin(["LR", "RF", "XGB", "MLP"])].copy()
summary_df["model_pretty"] = summary_df["model"].map(name_map)
summary_df = summary_df.set_index("model_pretty").loc[order].reset_index()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(
    summary_df["model_pretty"],
    summary_df["auc_test_mean"],
    yerr=summary_df["auc_test_std"],
    capsize=5,
)
axes[0].set_title("Test ROC-AUC (Mean ± Std)")
axes[0].set_xlabel("")
axes[0].set_ylabel("ROC-AUC")
axes[0].tick_params(axis="x", rotation=20)

axes[1].bar(
    summary_df["model_pretty"],
    summary_df["ap_test_mean"],
    yerr=summary_df["ap_test_std"],
    capsize=5,
)
axes[1].set_title("Test Average Precision (Mean ± Std)")
axes[1].set_xlabel("")
axes[1].set_ylabel("Average Precision")
axes[1].tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig(OUT_DIR / "model_bar_from_summary.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved:")
print(OUT_DIR / "model_boxplot_from_raw.png")
print(OUT_DIR / "model_bar_from_summary.png")
