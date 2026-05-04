import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    CKPT_PATH,
    DATA_PATH,
    EPOCHS,
    HIDDEN_DIM,
    INPUT_DIM,
    LR,
    LSTM_HIDDEN,
    PRED_HORIZON,
    TCN_LAYERS,
    WINDOW_SIZE,
)
from dataset import build_datasets
from model import ResourcePredictor
from utils import save_checkpoint


def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            total += criterion(model(x), y).item()
    return total / max(len(loader), 1)


def train():
    train_set, val_set, test_set, mean, std, columns, _ = build_datasets(DATA_PATH)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResourcePredictor(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        tcn_layers=TCN_LAYERS,
        lstm_hidden=LSTM_HIDDEN,
        pred_horizon=PRED_HORIZON,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val_loss = float("inf")

    config = {
        "input_dim": INPUT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "tcn_layers": TCN_LAYERS,
        "lstm_hidden": LSTM_HIDDEN,
        "pred_horizon": PRED_HORIZON,
        "window_size": WINDOW_SIZE,
    }

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{EPOCHS}, train={train_loss:.6f}, val={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(CKPT_PATH, model, mean, std, columns, config)
            print(f"Saved best model to {CKPT_PATH}")

    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"best_val={best_val_loss:.6f}, test={test_loss:.6f}")


if __name__ == "__main__":
    train()
