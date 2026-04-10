from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def train_rf(X_train, y_train, X_val=None, y_val=None, random_state=42):
    candidate_configs = [
        {"n_estimators": 500, "max_depth": 12, "min_samples_leaf": 8, "max_features": "sqrt"},
        {"n_estimators": 500, "max_depth": 16, "min_samples_leaf": 4, "max_features": 0.5},
        {"n_estimators": 800, "max_depth": None, "min_samples_leaf": 2, "max_features": "sqrt"},
    ]

    if X_val is None or y_val is None:
        cfg = candidate_configs[0]
        model = RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_leaf=cfg["min_samples_leaf"],
            max_features=cfg["max_features"],
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        model.best_config_ = cfg
        return model

    best_model = None
    best_config = None
    best_auc = -1.0
    for cfg in candidate_configs:
        model = RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_leaf=cfg["min_samples_leaf"],
            max_features=cfg["max_features"],
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        auc_val = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        if auc_val > best_auc:
            best_auc = auc_val
            best_model = model
            best_config = cfg

    best_model.best_config_ = best_config
    return best_model
