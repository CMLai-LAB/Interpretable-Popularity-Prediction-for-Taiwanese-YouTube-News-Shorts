try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


def train_lgbm_or_fallback(
    X_train, y_train, X_val, y_val, scale_pos_weight, random_state=42
):
    if lgb is not None:
        candidate_configs = [
            {
                "n_estimators": 1400,
                "max_depth": 4,
                "learning_rate": 0.03,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "reg_lambda": 2.0,
                "reg_alpha": 1.0,
                "min_child_samples": 30,
            },
            {
                "n_estimators": 1800,
                "max_depth": 6,
                "learning_rate": 0.02,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "reg_lambda": 2.0,
                "reg_alpha": 0.5,
                "min_child_samples": 50,
            },
        ]

        best_model = None
        best_config = None
        best_auc = -1.0
        for cfg in candidate_configs:
            model = lgb.LGBMClassifier(
                n_estimators=cfg["n_estimators"],
                max_depth=cfg["max_depth"],
                learning_rate=cfg["learning_rate"],
                subsample=cfg["subsample"],
                colsample_bytree=cfg["colsample_bytree"],
                reg_lambda=cfg["reg_lambda"],
                reg_alpha=cfg["reg_alpha"],
                min_child_samples=cfg["min_child_samples"],
                scale_pos_weight=scale_pos_weight,
                random_state=random_state,
                n_jobs=-1,
            )
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            )
            auc_val = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
            if auc_val > best_auc:
                best_auc = auc_val
                best_model = model
                best_config = cfg

        best_model.best_config_ = best_config
        return best_model, "LightGBM"

    candidate_configs = [
        {"learning_rate": 0.03, "max_iter": 500, "max_depth": 4, "min_samples_leaf": 30},
        {"learning_rate": 0.02, "max_iter": 700, "max_depth": 6, "min_samples_leaf": 40},
        {"learning_rate": 0.05, "max_iter": 400, "max_depth": 3, "min_samples_leaf": 20},
    ]
    sample_weight = (y_train * scale_pos_weight) + (1 - y_train)

    best_model = None
    best_config = None
    best_auc = -1.0
    for cfg in candidate_configs:
        model = HistGradientBoostingClassifier(
            learning_rate=cfg["learning_rate"],
            max_iter=cfg["max_iter"],
            max_depth=cfg["max_depth"],
            min_samples_leaf=cfg["min_samples_leaf"],
            random_state=random_state,
        )
        model.fit(X_train, y_train, sample_weight=sample_weight)
        auc_val = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        if auc_val > best_auc:
            best_auc = auc_val
            best_model = model
            best_config = cfg

    best_model.best_config_ = best_config
    return best_model, "HistGBDT(fallback)"
