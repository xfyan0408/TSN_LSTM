import torch
import torch.nn as nn


class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        # 卷积后输出会变短,需要补空位,让卷积后长度不丢
        self.padding = (kernel_size - 1) * dilation
        # 多通道卷积
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
        )
        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 带膨胀的一维卷积[标准多通道卷积]
        out = self.conv(x)
        # 裁剪到原时间长度
        out = out[:, :, :x.size(-1)]
        # 残差连接,再经过ReLU
        return self.relu(out + x)

# lstm + rcn 的模型
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
        """
        ChannelMix
        3 个输入维度混合成 hidden_dim 维特征:
        输入 x
        ↓
        Linear：线性变换，把 3 维变成 hidden_dim 维
        ↓
        ReLU：非线性激活，把负数变成 0
        ↓
        输出 z 
        """
        self.channel_mix = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        self.tcn = nn.Sequential(
            *[ # * 列表展开
                TCNBlock(
                    channels=hidden_dim,
                    kernel_size=3,
                    dilation=2 ** i, # 幂运算
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
        # x 是归一化后的历史窗口，base 也在归一化空间里。
        base = x[:, -1:, :].repeat(1, self.pred_horizon, 1)

        z = self.channel_mix(x)
        z = z.transpose(1, 2) # 交换 z 的第 1 维和第 2 维。
        z = self.tcn(z)
        z = z.transpose(1, 2) # 交换 z 的第 1 维和第 2 维。

        """
        output = 每个时间步的输出
        h_n    = 最后时刻的隐藏状态:LSTM 在当前时刻对外输出的状态表示。
        c_n    = 最后时刻的记忆状态:LSTM 内部保存的长期记忆。
        """ 
        _, (h_n, _) = self.lstm(z)
        s = h_n[-1]

        delta_b = self.head_b(s)
        delta_c = self.head_c(s)
        delta_m = self.head_m(s)
        delta = torch.stack([delta_b, delta_c, delta_m], dim=-1)
        return base + delta
