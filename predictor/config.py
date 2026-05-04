from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent

INPUT_DIM = 3
HIDDEN_DIM = 64
TCN_LAYERS = 3
LSTM_HIDDEN = 64

WINDOW_SIZE = 60
PRED_HORIZON = 10

BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

RESOURCE_COLUMNS = ("Bandwidth", "CPU", "MEM")
DATA_PATH = ROOT_DIR / "data" / "resource.csv"
CKPT_PATH = ROOT_DIR / "checkpoints" / "best_model.pt"
