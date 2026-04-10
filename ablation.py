import numpy as np
import pandas as pd


def run_ablation(blocks, y, train_idx, val_idx, test_idx, run_lr_ablation):
    experiments = [
        # Theory-aligned comparisons
        ("SEM_only", ["SEM"]),
        ("FRM_only", ["FRM"]),
        ("CTX_only", ["CTX"]),
        ("SEM+FRM", ["SEM", "FRM"]),
        ("SEM+CTX", ["SEM", "CTX"]),
        ("FRM+CTX", ["FRM", "CTX"]),
        ("SEM+FRM+CTX", ["SEM", "FRM", "CTX"]),

        # Channel diagnostics
        ("CTX_context_only", ["C_CTX"]),
        ("CTX_metadata_only", ["C_META"]),
        ("CTX_full_channel", ["C"]),

        # Finer-grained diagnostics
        ("SEM_embedding_only", ["E"]),
        ("SEM_topic_only", ["T"]),
        ("FRM_month_only", ["M"]),
        ("FRM_surface_only", ["F"]),
        ("SEM_embedding+framing_surface", ["E", "F"]),
        ("FULL_no_channel", ["SEM", "FRM"]),
        ("FULL_no_context", ["SEM", "FRM", "C_META"]),
        ("FULL_no_metadata", ["SEM", "FRM", "C_CTX"]),
        ("FULL_all", ["SEM", "FRM", "CTX"]),
    ]
    if "V_OBJ" in blocks:
        experiments.extend(
            [
                ("OBJ_only", ["V_OBJ"]),
                ("SEM+OBJ", ["SEM", "V_OBJ"]),
                ("FULL_with_obj", ["SEM", "FRM", "CTX", "V_OBJ"]),
            ]
        )

    rows = []

    for name, keys in experiments:
        X_concat = np.concatenate([blocks[k] for k in keys], axis=1).astype(np.float32)

        Xtr = X_concat[train_idx]
        Xv  = X_concat[val_idx]
        Xt  = X_concat[test_idx]

        ytr = y[train_idx]
        yv  = y[val_idx]
        yt  = y[test_idx]

        out = run_lr_ablation(Xtr, ytr, Xv, yv, Xt, yt)

        out["exp"] = name
        out["keys"] = "+".join(keys)
        out["dim"] = X_concat.shape[1]
        out["gap_train_test"] = out["auc_train"] - out["auc_test"]
        rows.append(out)

        print(f"Completed {name} with AUC TEST: {out['auc_test']:.4f}")

    return pd.DataFrame(rows)
