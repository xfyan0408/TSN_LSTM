from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1, bias: bool = True):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.0,
        norm: str = "layer",
        activation: str = "relu",
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError("activation must be 'relu' or 'gelu'")

        self.norm1 = nn.LayerNorm(out_ch) if norm == "layer" else None
        self.norm2 = nn.LayerNorm(out_ch) if norm == "layer" else None
        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None

    def _apply_ln(self, y: torch.Tensor, ln: Optional[nn.LayerNorm]) -> torch.Tensor:
        if ln is None:
            return y
        y = y.transpose(1, 2)
        y = ln(y)
        y = y.transpose(1, 2)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self._apply_ln(y, self.norm1)
        y = self.act(y)
        y = self.dropout(y)

        y = self.conv2(y)
        y = self._apply_ln(y, self.norm2)
        y = self.act(y)
        y = self.dropout(y)

        res = x if self.downsample is None else self.downsample(x)
        return y + res


class TCNBackbone(nn.Module):
    def __init__(
        self,
        channels: List[int],
        kernel_size: int = 3,
        dilations: Optional[List[int]] = None,
        dropout: float = 0.0,
        norm: str = "layer",
        activation: str = "relu",
    ):
        super().__init__()
        n_blocks = len(channels) - 1
        if dilations is None:
            dilations = [2 ** i for i in range(n_blocks)]
        if len(dilations) != n_blocks:
            raise ValueError("dilations length must match number of TCN blocks")

        blocks = []
        for i in range(n_blocks):
            blocks.append(
                TemporalBlock(
                    in_ch=channels[i],
                    out_ch=channels[i + 1],
                    kernel_size=kernel_size,
                    dilation=dilations[i],
                    dropout=dropout,
                    norm=norm,
                    activation=activation,
                )
            )
        self.net = nn.Sequential(*blocks)
        self.dilations = dilations
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def receptive_field(self) -> int:
        k = self.kernel_size
        return 1 + sum(2 * (k - 1) * d for d in self.dilations)


class TCNLSTMForecaster(nn.Module):
    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 3,
        horizon: int = 10,
        d_model: int = 32,
        embed_activation: str = "relu",
        tcn_layers: int = 5,
        tcn_kernel_size: int = 3,
        tcn_dilations: Optional[List[int]] = None,
        tcn_dropout: float = 0.1,
        tcn_norm: str = "layer",
        tcn_activation: str = "relu",
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        lstm_dropout: float = 0.0,
        mlp_hidden: int = 64,
        separate_heads: bool = False,
        residual_to_last: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.horizon = horizon
        self.residual_to_last = residual_to_last
        self.separate_heads = separate_heads

        self.embed = nn.Linear(in_dim, d_model)
        if embed_activation == "relu":
            self.embed_act = nn.ReLU()
        elif embed_activation == "gelu":
            self.embed_act = nn.GELU()
        elif embed_activation == "none":
            self.embed_act = nn.Identity()
        else:
            raise ValueError("embed_activation must be relu/gelu/none")

        channels = [d_model] + [d_model] * tcn_layers
        self.tcn = TCNBackbone(
            channels=channels,
            kernel_size=tcn_kernel_size,
            dilations=tcn_dilations,
            dropout=tcn_dropout,
            norm=tcn_norm,
            activation=tcn_activation,
        )

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )

        def make_mlp(out_size: int):
            return nn.Sequential(
                nn.Linear(lstm_hidden, mlp_hidden),
                nn.ReLU(),
                nn.Linear(mlp_hidden, out_size),
            )

        if separate_heads:
            self.head_b = make_mlp(horizon)
            self.head_c = make_mlp(horizon)
            self.head_m = make_mlp(horizon)
        else:
            self.head = make_mlp(horizon * out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, d = x.shape
        if d != self.in_dim:
            raise ValueError(f"expected input dim {self.in_dim}, got {d}")

        z = self.embed_act(self.embed(x))
        z = z.transpose(1, 2)
        z = self.tcn(z)
        z = z.transpose(1, 2)

        _, (h, _) = self.lstm(z)
        h_last = h[-1]

        if self.separate_heads:
            yb = self.head_b(h_last).unsqueeze(-1)
            yc = self.head_c(h_last).unsqueeze(-1)
            ym = self.head_m(h_last).unsqueeze(-1)
            y = torch.cat([yb, yc, ym], dim=-1)
        else:
            b = x.size(0)
            y = self.head(h_last).view(b, self.horizon, self.out_dim)

        if self.residual_to_last:
            base = x[:, -1:, :].repeat(1, self.horizon, 1)
            y = base + y

        return y
