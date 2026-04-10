from classification import evaluate_auc
from calibration import evaluate_calibration

def evaluate_model(name, y_train, y_val, y_test, proba_train, proba_val, proba_test):

    auc_metrics = evaluate_auc(
        y_train, y_val, y_test,
        proba_train, proba_val, proba_test
    )

    cal_metrics = evaluate_calibration(
        y_train, y_val, y_test,
        proba_train, proba_val, proba_test
    )

    print(f"\n=== {name} Performance ===")

    print(f"AUC TRAIN: {auc_metrics['auc_train']:.4f}")
    print(f"AUC VAL  : {auc_metrics['auc_val']:.4f}")
    print(f"AUC TEST : {auc_metrics['auc_test']:.4f}")
    print(f"Overfit gap: {auc_metrics['overfit_gap']:.4f}")

    print(f"Brier TRAIN: {cal_metrics['brier_train']:.4f}")
    print(f"Brier VAL  : {cal_metrics['brier_val']:.4f}")
    print(f"Brier TEST : {cal_metrics['brier_test']:.4f}")

    print(f"AP TEST: {auc_metrics['ap_test']:.4f}")

    return {**auc_metrics, **cal_metrics}