import argparse
import io
import time
import random
import sys
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibration import evaluate_calibration
from classification import evaluate_auc
from experiments.common import build_experiment_context, split_xy, stage
# from models.dummy import train_dummy
# from models.lgbm import train_lgbm_or_fallback
from models.lr import train_lr_with_sweep
from models.mlp import train_mlp
from models.rf import train_rf
from models.xgb import train_xgb
from ranking import evaluate_ranking
from summary import evaluate_model


def _timed_call(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def _predict_positive_proba(model, x):
    proba = model.predict_proba(x)
    if proba.ndim != 2:
        raise ValueError("predict_proba must return a 2D array")
    if proba.shape[1] == 1:
        return np.zeros(len(x), dtype=np.float64)
    return proba[:, 1]


def _timed_predict_positive_proba(model, x):
    start = time.perf_counter()
    proba = _predict_positive_proba(model, x)
    elapsed = time.perf_counter() - start
    return proba, elapsed


def _timed_mlp_predict(model, x, device):
    start = time.perf_counter()
    with torch.no_grad():
        x_tensor = torch.tensor(x).float().to(device)
        proba = torch.sigmoid(model(x_tensor)).cpu().numpy().flatten()
    elapsed = time.perf_counter() - start
    return proba, elapsed


def _estimate_model_artifact_size_bytes(model):
    try:
        payload = pickle.dumps(model)
        return len(payload)
    except Exception:
        return None


def _estimate_model_param_count(model):
    if hasattr(model, "parameters"):
        try:
            return int(sum(p.numel() for p in model.parameters()))
        except Exception:
            return None
    if hasattr(model, "coef_") and hasattr(model, "intercept_"):
        return int(np.size(model.coef_) + np.size(model.intercept_))
    if hasattr(model, "estimators_"):
        try:
            return int(sum(getattr(est, "tree_", None).node_count for est in model.estimators_.ravel()))
        except Exception:
            return None
    return None


def _set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collect_metrics(y_train, y_val, y_test, proba_train, proba_val, proba_test):
    auc_metrics = evaluate_auc(y_train, y_val, y_test, proba_train, proba_val, proba_test)
    cal_metrics = evaluate_calibration(y_train, y_val, y_test, proba_train, proba_val, proba_test)
    ranking_metrics = evaluate_ranking(y_test, proba_test)
    return {**auc_metrics, **cal_metrics, **ranking_metrics}


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


def run_train_models(context=None, seed=42, verbose=True):
    if context is None:
        context = build_experiment_context()

    _set_global_seed(seed)

    split_data = split_xy(
        context["X_all"],
        context["y"],
        context["train_idx"],
        context["val_idx"],
        context["test_idx"],
    )

    x_train = split_data["X_train"]
    x_val = split_data["X_val"]
    x_test = split_data["X_test"]
    y_train = split_data["y_train"]
    y_val = split_data["y_val"]
    y_test = split_data["y_test"]

    pos_ratio = y_train.mean()
    scale_pos_weight = (1 - pos_ratio) / max(pos_ratio, 1e-8)

    if verbose:
        stage(f"Training Models (seed={seed})")

    xgb_model, xgb_train_time_sec = _timed_call(
        train_xgb,
        x_train,
        y_train,
        x_val,
        y_val,
        scale_pos_weight,
        random_state=seed,
    )
    xgb_proba_train, xgb_infer_train_time_sec = _timed_predict_positive_proba(xgb_model, x_train)
    xgb_proba_val, xgb_infer_val_time_sec = _timed_predict_positive_proba(xgb_model, x_val)
    xgb_proba_test, xgb_infer_test_time_sec = _timed_predict_positive_proba(xgb_model, x_test)

    # Dummy baseline and HistGBDT(fallback) are currently disabled because the
    # paper's main benchmark focuses on the four primary models: LR, RF, XGB, and MLP.

    rf_model, rf_train_time_sec = _timed_call(train_rf, x_train, y_train, x_val, y_val, random_state=seed)
    rf_proba_train, rf_infer_train_time_sec = _timed_predict_positive_proba(rf_model, x_train)
    rf_proba_val, rf_infer_val_time_sec = _timed_predict_positive_proba(rf_model, x_val)
    rf_proba_test, rf_infer_test_time_sec = _timed_predict_positive_proba(rf_model, x_test)

    (lr_model, best_c), lr_train_time_sec = _timed_call(
        train_lr_with_sweep,
        x_train,
        y_train,
        x_val,
        y_val,
        random_state=seed,
    )
    lr_proba_train, lr_infer_train_time_sec = _timed_predict_positive_proba(lr_model, x_train)
    lr_proba_val, lr_infer_val_time_sec = _timed_predict_positive_proba(lr_model, x_val)
    lr_proba_test, lr_infer_test_time_sec = _timed_predict_positive_proba(lr_model, x_test)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mlp_model, mlp_train_time_sec = _timed_call(
        train_mlp,
        x_train,
        y_train,
        x_val,
        y_val,
        seed=seed,
    )
    proba_mlp_train, mlp_infer_train_time_sec = _timed_mlp_predict(mlp_model, x_train, device)
    proba_mlp_val, mlp_infer_val_time_sec = _timed_mlp_predict(mlp_model, x_val, device)
    proba_mlp_test, mlp_infer_test_time_sec = _timed_mlp_predict(mlp_model, x_test, device)

    scores_by_model = {
        "RF": {"train": rf_proba_train, "val": rf_proba_val, "test": rf_proba_test},
        "XGB": {"train": xgb_proba_train, "val": xgb_proba_val, "test": xgb_proba_test},
        "LR": {"train": lr_proba_train, "val": lr_proba_val, "test": lr_proba_test},
        "MLP": {"train": proba_mlp_train, "val": proba_mlp_val, "test": proba_mlp_test},
    }

    efficiency_by_model = {
        "RF": {
            "train_time_sec": rf_train_time_sec,
            "infer_train_time_sec": rf_infer_train_time_sec,
            "infer_val_time_sec": rf_infer_val_time_sec,
            "infer_test_time_sec": rf_infer_test_time_sec,
            "test_latency_ms_per_sample": 1000.0 * rf_infer_test_time_sec / max(len(x_test), 1),
            "model_artifact_size_bytes": _estimate_model_artifact_size_bytes(rf_model),
            "model_param_count": _estimate_model_param_count(rf_model),
        },
        "XGB": {
            "train_time_sec": xgb_train_time_sec,
            "infer_train_time_sec": xgb_infer_train_time_sec,
            "infer_val_time_sec": xgb_infer_val_time_sec,
            "infer_test_time_sec": xgb_infer_test_time_sec,
            "test_latency_ms_per_sample": 1000.0 * xgb_infer_test_time_sec / max(len(x_test), 1),
            "model_artifact_size_bytes": _estimate_model_artifact_size_bytes(xgb_model),
            "model_param_count": _estimate_model_param_count(xgb_model),
        },
        "LR": {
            "train_time_sec": lr_train_time_sec,
            "infer_train_time_sec": lr_infer_train_time_sec,
            "infer_val_time_sec": lr_infer_val_time_sec,
            "infer_test_time_sec": lr_infer_test_time_sec,
            "test_latency_ms_per_sample": 1000.0 * lr_infer_test_time_sec / max(len(x_test), 1),
            "model_artifact_size_bytes": _estimate_model_artifact_size_bytes(lr_model),
            "model_param_count": _estimate_model_param_count(lr_model),
        },
        "MLP": {
            "train_time_sec": mlp_train_time_sec,
            "infer_train_time_sec": mlp_infer_train_time_sec,
            "infer_val_time_sec": mlp_infer_val_time_sec,
            "infer_test_time_sec": mlp_infer_test_time_sec,
            "test_latency_ms_per_sample": 1000.0 * mlp_infer_test_time_sec / max(len(x_test), 1),
            "model_artifact_size_bytes": _estimate_model_artifact_size_bytes(mlp_model),
            "model_param_count": _estimate_model_param_count(mlp_model),
        },
    }

    metrics_by_model = {}
    for model_name, score_dict in scores_by_model.items():
        metrics_by_model[model_name] = _collect_metrics(
            y_train,
            y_val,
            y_test,
            score_dict["train"],
            score_dict["val"],
            score_dict["test"],
        )
        metrics_by_model[model_name].update(efficiency_by_model[model_name])

    if verbose:
        for model_name in ["XGB", "RF", "LR", "MLP"]:
            print(f"{model_name} TEST AUC:", metrics_by_model[model_name]["auc_test"])
        print("Best LR C:", best_c)
        print("Best XGB config:", getattr(xgb_model, "best_config_", None))
        print("Best RF config:", getattr(rf_model, "best_config_", None))

        stage("Model Performance Summary")
        evaluate_model("XGB", y_train, y_val, y_test, xgb_proba_train, xgb_proba_val, xgb_proba_test)
        evaluate_model("Random Forest", y_train, y_val, y_test, rf_proba_train, rf_proba_val, rf_proba_test)
        evaluate_model("Logistic Regression", y_train, y_val, y_test, lr_proba_train, lr_proba_val, lr_proba_test)
        evaluate_model("MLP", y_train, y_val, y_test, proba_mlp_train, proba_mlp_val, proba_mlp_test)

        print("\n=== Ranking Metrics ===")
        for model_name, scores in scores_by_model.items():
            print(model_name, evaluate_ranking(y_test, scores["test"]))

        print("\n=== Efficiency Metrics ===")
        for model_name in ["XGB", "RF", "LR", "MLP"]:
            metrics = metrics_by_model[model_name]
            print(
                model_name,
                {
                    "train_time_sec": round(metrics["train_time_sec"], 4),
                    "infer_test_time_sec": round(metrics["infer_test_time_sec"], 6),
                    "test_latency_ms_per_sample": round(metrics["test_latency_ms_per_sample"], 6),
                    "model_artifact_size_bytes": metrics["model_artifact_size_bytes"],
                    "model_param_count": metrics["model_param_count"],
                },
            )

    return {
        "context": context,
        "seed": seed,
        "split_data": split_data,
        "models": {
            "rf": rf_model,
            "xgb": xgb_model,
            "lr": lr_model,
            "mlp": mlp_model,
        },
        "scores": {
            "rf_train": rf_proba_train,
            "rf_val": rf_proba_val,
            "rf_test": rf_proba_test,
            "xgb_train": xgb_proba_train,
            "xgb_val": xgb_proba_val,
            "xgb_test": xgb_proba_test,
            "lr_train": lr_proba_train,
            "lr_val": lr_proba_val,
            "lr_test": lr_proba_test,
            "mlp_train": proba_mlp_train,
            "mlp_val": proba_mlp_val,
            "mlp_test": proba_mlp_test,
        },
        "scores_by_model": scores_by_model,
        "metrics_by_model": metrics_by_model,
        "best_c": best_c,
    }


def run_multi_seed_benchmark(
    context=None,
    seeds=(42, 52, 62, 72, 82),
    output_dir="experiments/cache",
    n_bootstrap=1000,
):
    if context is None:
        context = build_experiment_context()

    rows = []
    run_outputs = []

    for seed in seeds:
        out = run_train_models(context=context, seed=seed, verbose=False)
        run_outputs.append(out)
        for model_name, metrics in out["metrics_by_model"].items():
            row = {"seed": seed, "model": model_name}
            row.update(metrics)
            rows.append(row)

    raw_df = pd.DataFrame(rows)

    std_df = (
        raw_df.groupby("model", as_index=False)
        .agg(
            auc_test_std=("auc_test", "std"),
            auc_val_std=("auc_val", "std"),
            ap_test_std=("ap_test", "std"),
            brier_test_std=("brier_test", "std"),
            p10_std=("p10", "std"),
            r10_std=("r10", "std"),
            p20_std=("p20", "std"),
            r20_std=("r20", "std"),
            ndcg10_std=("ndcg10", "std"),
            ndcg20_std=("ndcg20", "std"),
            hit10_std=("hit10", "std"),
            hit20_std=("hit20", "std"),
            spearman_std=("spearman", "std"),
            kendall_tau_std=("kendall_tau", "std"),
            train_time_sec_std=("train_time_sec", "std"),
            infer_test_time_sec_std=("infer_test_time_sec", "std"),
            test_latency_ms_per_sample_std=("test_latency_ms_per_sample", "std"),
            model_artifact_size_bytes_std=("model_artifact_size_bytes", "std"),
            model_param_count_std=("model_param_count", "std"),
        )
    )
    raw_df = raw_df.merge(std_df, on="model", how="left")

    agg_df = (
        raw_df.groupby("model", as_index=False)
        .agg(
            auc_test_mean=("auc_test", "mean"),
            auc_test_std=("auc_test_std", "mean"),
            auc_val_mean=("auc_val", "mean"),
            auc_val_std=("auc_val_std", "mean"),
            ap_test_mean=("ap_test", "mean"),
            ap_test_std=("ap_test_std", "mean"),
            brier_test_mean=("brier_test", "mean"),
            brier_test_std=("brier_test_std", "mean"),
            p10_mean=("p10", "mean"),
            p10_std=("p10_std", "mean"),
            r10_mean=("r10", "mean"),
            r10_std=("r10_std", "mean"),
            p20_mean=("p20", "mean"),
            p20_std=("p20_std", "mean"),
            r20_mean=("r20", "mean"),
            r20_std=("r20_std", "mean"),
            ndcg10_mean=("ndcg10", "mean"),
            ndcg10_std=("ndcg10_std", "mean"),
            ndcg20_mean=("ndcg20", "mean"),
            ndcg20_std=("ndcg20_std", "mean"),
            hit10_mean=("hit10", "mean"),
            hit10_std=("hit10_std", "mean"),
            hit20_mean=("hit20", "mean"),
            hit20_std=("hit20_std", "mean"),
            spearman_mean=("spearman", "mean"),
            spearman_std=("spearman_std", "mean"),
            kendall_tau_mean=("kendall_tau", "mean"),
            kendall_tau_std=("kendall_tau_std", "mean"),
            train_time_sec_mean=("train_time_sec", "mean"),
            train_time_sec_std=("train_time_sec_std", "mean"),
            infer_test_time_sec_mean=("infer_test_time_sec", "mean"),
            infer_test_time_sec_std=("infer_test_time_sec_std", "mean"),
            test_latency_ms_per_sample_mean=("test_latency_ms_per_sample", "mean"),
            test_latency_ms_per_sample_std=("test_latency_ms_per_sample_std", "mean"),
            model_artifact_size_bytes_mean=("model_artifact_size_bytes", "mean"),
            model_artifact_size_bytes_std=("model_artifact_size_bytes_std", "mean"),
            model_param_count_mean=("model_param_count", "mean"),
            model_param_count_std=("model_param_count_std", "mean"),
        )
        .sort_values("auc_test_mean", ascending=False)
        .reset_index(drop=True)
    )

    best_model = agg_df.iloc[0]["model"]
    y_test = run_outputs[0]["split_data"]["y_test"]

    ensemble_scores = {}
    for model_name in run_outputs[0]["scores_by_model"].keys():
        stacked = np.vstack([out["scores_by_model"][model_name]["test"] for out in run_outputs])
        ensemble_scores[model_name] = stacked.mean(axis=0)

    significance_rows = []
    best_scores = ensemble_scores[best_model]
    for model_name, model_scores in ensemble_scores.items():
        if model_name == best_model:
            continue
        diff = bootstrap_auc_diff_ci(y_test, best_scores, model_scores, n_bootstrap=n_bootstrap, seed=42)
        significance_rows.append(
            {
                "best_model": best_model,
                "compare_to": model_name,
                "mean_auc_diff": diff["mean_diff"],
                "ci_low": diff["ci_low"],
                "ci_high": diff["ci_high"],
                "significant": diff["significant"],
                "n_valid": diff["n_valid"],
            }
        )

    significance_df = pd.DataFrame(significance_rows).sort_values("mean_auc_diff", ascending=False)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    raw_path = output_path / "train_models_multiseed_raw.csv"
    agg_path = output_path / "train_models_multiseed_summary.csv"
    sig_path = output_path / "train_models_significance.csv"
    raw_rounded_path = output_path / "train_models_multiseed_raw_rounded_3dp.csv"
    agg_rounded_path = output_path / "train_models_multiseed_summary_rounded_3dp.csv"
    sig_rounded_path = output_path / "train_models_significance_rounded_3dp.csv"
    ensemble_path = output_path / "train_models_ensemble_scores.pkl"

    raw_df.to_csv(raw_path, index=False)
    agg_df.to_csv(agg_path, index=False)
    significance_df.to_csv(sig_path, index=False)

    raw_df.round(3).to_csv(raw_rounded_path, index=False)
    agg_df.round(3).to_csv(agg_rounded_path, index=False)
    significance_df.round(3).to_csv(sig_rounded_path, index=False)

    with open(ensemble_path, "wb") as f:
        pickle.dump({"y_test": y_test, "ensemble_scores": ensemble_scores}, f)

    print("Saved raw multi-seed metrics:", raw_path)
    print("Saved aggregated summary:", agg_path)
    print("Saved significance table:", sig_path)
    print("Saved rounded raw multi-seed metrics:", raw_rounded_path)
    print("Saved rounded aggregated summary:", agg_rounded_path)
    print("Saved rounded significance table:", sig_rounded_path)
    print("Saved ensemble scores:", ensemble_path)
    print("Best model by mean test AUC:", best_model)

    return {
        "raw_df": raw_df,
        "summary_df": agg_df,
        "significance_df": significance_df,
        "best_model": best_model,
        "paths": {
            "raw": str(raw_path),
            "summary": str(agg_path),
            "significance": str(sig_path),
            "raw_rounded_3dp": str(raw_rounded_path),
            "summary_rounded_3dp": str(agg_rounded_path),
            "significance_rounded_3dp": str(sig_rounded_path),
            "ensemble_scores": str(ensemble_path),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-feature-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi-seed", action="store_true")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 52, 62, 72, 82],
    )
    parser.add_argument("--output-dir", type=str, default="experiments/cache")
    args = parser.parse_args()

    context = build_experiment_context(object_feature_path=args.object_feature_path)
    if args.multi_seed:
        run_multi_seed_benchmark(
            context=context,
            seeds=tuple(args.seeds),
            output_dir=args.output_dir,
        )
    else:
        run_train_models(context=context, seed=args.seed)
