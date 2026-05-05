from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent

# 每个时间步的资源维度：带宽、CPU、内存
INPUT_DIM = 3

# ChannelMix 和 TCN 使用的特征维度
HIDDEN_DIM = 64

# TCN 时间卷积块数量
TCN_LAYERS = 3

# LSTM 隐藏状态维度
LSTM_HIDDEN = 64

# 模型预测相对最后一个历史点的增量，再加回基准值
MODEL_MODE = "last_value_residual"

# 输入历史窗口长度 T
WINDOW_SIZE = 60

# 预测未来 H 个时间步
PRED_HORIZON = 10

# 每次训练送入模型的样本数
BATCH_SIZE = 64

# 训练轮数
EPOCHS = 500

# Adam 优化器学习率
# Adam 优化器就是训练神经网络时用来更新模型参数的方法
LR = 1e-3

# 按时间顺序切分训练集比例
TRAIN_RATIO = 0.70

# 按时间顺序切分验证集比例，剩余部分为测试集
VAL_RATIO = 0.15

# CSV 中参与预测的三列资源指标
RESOURCE_COLUMNS = ("Bandwidth", "CPU", "MEM")

# 默认数据文件路径
DATA_PATH = ROOT_DIR / "data" / "resource.csv"

# 训练后保存的最佳模型路径
CKPT_PATH = ROOT_DIR / "checkpoints" / "best_model.pt"
