import torch
import torch.nn as nn


class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = out[:, :, :x.size(-1)]
        return self.relu(out + x)


class ResourcePredictor(nn.Module):
    def __init__(
        self,
        input_dim=3,
        hidden_dim=64,
        tcn_layers=3,
        lstm_hidden=64,
        pred_horizon=10,
    ):
        super().__init__()
        self.pred_horizon = pred_horizon

        self.channel_mix = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        self.tcn = nn.Sequential(
            *[
                TCNBlock(
                    channels=hidden_dim,
                    kernel_size=3,
                    dilation=2 ** i,
                )
                for i in range(tcn_layers)
            ]
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            batch_first=True,
        )

        self.head_b = nn.Linear(lstm_hidden, pred_horizon)
        self.head_c = nn.Linear(lstm_hidden, pred_horizon)
        self.head_m = nn.Linear(lstm_hidden, pred_horizon)

    def forward(self, x):
        z = self.channel_mix(x)
        z = z.transpose(1, 2)
        z = self.tcn(z)
        z = z.transpose(1, 2)

        _, (h_n, _) = self.lstm(z)
        s = h_n[-1]

        pred_b = self.head_b(s)
        pred_c = self.head_c(s)
        pred_m = self.head_m(s)
        return torch.stack([pred_b, pred_c, pred_m], dim=-1)
