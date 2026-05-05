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
    MODEL_MODE,
    PRED_HORIZON,
    TCN_LAYERS,
    WINDOW_SIZE,
)
from dataset import build_datasets
from model import ResourcePredictor
from utils import load_checkpoint, save_checkpoint


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.sqrt(torch.mean((pred - target) ** 2) + self.eps)


def evaluate(model, loader, criterion, device):
    # 验证/测试阶段，只计算 loss，不更新参数
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            total += criterion(model(x), y).item()
    return total / max(len(loader), 1)


def train():
    # 读取数据，并构造训练/验证/测试数据集
    train_set, val_set, test_set, mean, std, columns, _ = build_datasets(DATA_PATH)

    # DataLoader 负责按 batch 取数据
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # 选择 GPU 或 CPU，并创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = ResourcePredictor(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        tcn_layers=TCN_LAYERS,
        lstm_hidden=LSTM_HIDDEN,
        pred_horizon=PRED_HORIZON,
    ).to(device)

    # RMSELoss 衡量预测值和真实值的差距
    criterion = RMSELoss()
    # AdamW 根据 loss 更新模型参数，并用 weight_decay 做轻微正则化
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        min_lr=1e-5,
    )
    best_val_loss = float("inf")
    bad_epochs = 0
    patience = 30

    # 保存模型时一起保存结构参数，推理时要用
    config = {
        "input_dim": INPUT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "tcn_layers": TCN_LAYERS,
        "lstm_hidden": LSTM_HIDDEN,
        "model_mode": MODEL_MODE,
        "pred_horizon": PRED_HORIZON,
        "window_size": WINDOW_SIZE,
    }

    for epoch in range(EPOCHS):
        # 训练阶段：前向计算、反向传播、更新参数
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            # x: 历史窗口 [batch, T, 3]
            # y: 真实未来 [batch, H, 3]
            x = x.to(device)
            y = y.to(device)

            pred = model(x)              # 模型预测未来 H 步
            loss = criterion(pred, y)    # 计算预测和真实值的误差

            optimizer.zero_grad()        # 清空上一轮残留的梯度
            loss.backward()              # 根据 loss 计算每个参数该怎么改
            optimizer.step()             # AdamW 按梯度更新模型参数

            train_loss += loss.item()    # 累加当前 batch 的 loss 数值

        # 每轮训练后，在验证集上检查效果
        train_loss /= max(len(train_loader), 1)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{EPOCHS}, train_rmse={train_loss:.6f}, val_rmse={val_loss:.6f}")

        # 只保存验证集效果最好的模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            bad_epochs = 0
            save_checkpoint(CKPT_PATH, model, mean, std, columns, config)
            print(f"Saved best model to {CKPT_PATH}")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # 训练结束后，加载验证集效果最好的模型，再用测试集做最终评估
    ckpt = load_checkpoint(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"best_val_rmse={best_val_loss:.6f}, best_test_rmse={test_loss:.6f}")


if __name__ == "__main__":
    train()
