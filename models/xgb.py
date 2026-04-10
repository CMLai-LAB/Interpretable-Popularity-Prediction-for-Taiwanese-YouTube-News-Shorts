import xgboost as xgb
from sklearn.metrics import roc_auc_score


def train_xgb(X_train, y_train, X_val, y_val, scale_pos_weight, random_state=42):
    candidate_configs = [
        {
            "n_estimators": 1400,
            "max_depth": 3,
            "min_child_weight": 10,
            "learning_rate": 0.03,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_lambda": 2.0,
            "reg_alpha": 1.0,
            "gamma": 1.0,
        },
        {
            "n_estimators": 1600,
            "max_depth": 4,
            "min_child_weight": 20,
            "learning_rate": 0.02,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_lambda": 2.0,
            "reg_alpha": 1.0,
            "gamma": 1.0,
        },
        {
            "n_estimators": 1000,
            "max_depth": 3,
            "min_child_weight": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "reg_alpha": 0.5,
            "gamma": 0.0,
        },
    ]

    best_model = None
    best_config = None
    best_auc = -1.0

    for cfg in candidate_configs:
        model = xgb.XGBClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_child_weight=cfg["min_child_weight"],
            learning_rate=cfg["learning_rate"],
            subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample_bytree"],
            reg_lambda=cfg["reg_lambda"],
            reg_alpha=cfg["reg_alpha"],
            gamma=cfg["gamma"],
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="auc",
            early_stopping_rounds=50,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        auc_val = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        if auc_val > best_auc:
            best_auc = auc_val
            best_model = model
            best_config = cfg

    if best_model is not None:
        best_model.best_config_ = best_config
    return best_model
