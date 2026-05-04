import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(self, in_dim=3, hidden=64, num_layers=1, H=10, out_dim=3):
        super().__init__()
        self.H = H
        self.out_dim = out_dim
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, H * out_dim),
        )

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h_last = h[-1]
        y = self.head(h_last)
        return y.view(-1, self.H, self.out_dim)
