from sklearn.metrics import brier_score_loss

def evaluate_calibration(y_train, y_val, y_test, proba_train, proba_val, proba_test):

    return {
        "brier_train": brier_score_loss(y_train, proba_train),
        "brier_val":   brier_score_loss(y_val, proba_val),
        "brier_test":  brier_score_loss(y_test, proba_test),
    }