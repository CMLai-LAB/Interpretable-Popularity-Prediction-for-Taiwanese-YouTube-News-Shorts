import copy
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, y_train, X_val=None, y_val=None,
              epochs=100, batch_size=256, lr=1e-3, patience=10, seed=42):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = MLP(X_train.shape[1]).to(device)

    Xtr = torch.tensor(X_train).float()
    ytr = torch.tensor(y_train).float().unsqueeze(1)

    dataset = torch.utils.data.TensorDataset(Xtr, ytr)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    pos_ratio = float(y_train.mean())
    pos_weight = (1.0 - pos_ratio) / max(pos_ratio, 1e-8)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    best_val_auc = -1
    patience_counter = 0
    best_state_dict = None

    for epoch in range(epochs):

        # ===== Training =====
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        # ===== Validation =====
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                Xv = torch.tensor(X_val).float().to(device)
                val_logits = model(Xv)
                val_probs = torch.sigmoid(val_logits).cpu().numpy().flatten()
                val_auc = roc_auc_score(y_val, val_probs)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state_dict = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

    # restore best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model
