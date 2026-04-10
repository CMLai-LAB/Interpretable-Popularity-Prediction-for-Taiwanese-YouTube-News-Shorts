from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate_auc(y_train, y_val, y_test, proba_train, proba_val, proba_test):

    return {
        "auc_train": roc_auc_score(y_train, proba_train),
        "auc_val":   roc_auc_score(y_val, proba_val),
        "auc_test":  roc_auc_score(y_test, proba_test),
        "overfit_gap": roc_auc_score(y_train, proba_train) - roc_auc_score(y_test, proba_test),
        "ap_test": average_precision_score(y_test, proba_test),
    }