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

    # MSELoss 衡量预测值和真实值的差距
    criterion = nn.MSELoss()
    # Adam 根据 loss 更新模型参数
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val_loss = float("inf")

    # 保存模型时一起保存结构参数，推理时要用
    config = {
        "input_dim": INPUT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "tcn_layers": TCN_LAYERS,
        "lstm_hidden": LSTM_HIDDEN,
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
            optimizer.step()             # Adam 按梯度更新模型参数

            train_loss += loss.item()    # 累加当前 batch 的 loss 数值

        # 每轮训练后，在验证集上检查效果
        train_loss /= max(len(train_loader), 1)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{EPOCHS}, train_loss={train_loss:.6f}, val={val_loss:.6f}")

        # 只保存验证集效果最好的模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(CKPT_PATH, model, mean, std, columns, config)
            print(f"Saved best model to {CKPT_PATH}")

    # 训练结束后，用测试集做一次最终评估
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"best_val={best_val_loss:.6f}, test={test_loss:.6f}")


if __name__ == "__main__":
    train()
