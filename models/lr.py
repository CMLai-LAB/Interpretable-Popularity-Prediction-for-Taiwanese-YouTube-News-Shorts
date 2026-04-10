from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np


def train_lr_with_sweep(X_train, y_train, X_val, y_val, random_state=42):

    best_auc = -1
    best_model = None
    best_C = None

    # Finer log-scale search around regularization strength.
    for C in np.logspace(-3, 2, 13):
        lr = LogisticRegression(
            C=float(C),
            max_iter=4000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=random_state
        )
        lr.fit(X_train, y_train)

        auc_val = roc_auc_score(y_val, lr.predict_proba(X_val)[:,1])

        if auc_val > best_auc:
            best_auc = auc_val
            best_model = lr
            best_C = C

    return best_model, best_C

def run_lr_ablation(Xtr, ytr, Xv, yv, Xt, yt, C_list=(0.01,0.1,0.5,1,5,10), random_state=42):
    """Train LR with C sweep on val set"""
    best = {"val_auc": -1, "C": None, "model": None}
    for Cval in C_list:
        lr_model = LogisticRegression(
            C=Cval,
            max_iter=4000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=random_state
        )
        lr_model.fit(Xtr, ytr)
        pv = lr_model.predict_proba(Xv)[:,1]
        val_auc = roc_auc_score(yv, pv)
        if val_auc > best["val_auc"]:
            best.update(val_auc=val_auc, C=Cval, model=lr_model)

    # Get predictions on all sets with best model
    lr_best = best["model"]
    ptr = lr_best.predict_proba(Xtr)[:,1]
    pv  = lr_best.predict_proba(Xv)[:,1]
    pt  = lr_best.predict_proba(Xt)[:,1]
    
    return {
        "best_C": best["C"],
        "auc_train": roc_auc_score(ytr, ptr),
        "auc_val":   roc_auc_score(yv,  pv),
        "auc_test":  roc_auc_score(yt,  pt),
    }
