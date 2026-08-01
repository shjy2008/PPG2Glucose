"""Ali et al. CBHFF model (JBHI 2025): multi-view attention + cascaded BiLSTM fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super().__init__()
        self.gate_c = nn.Sequential()
        self.gate_c.add_module('flatten', Flatten())
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range(len(gate_channels) - 2):
            self.gate_c.add_module(
                'gate_c_fc_%d' % i, nn.Linear(gate_channels[i], gate_channels[i + 1])
            )
            self.gate_c.add_module(
                'gate_c_bn_%d' % (i + 1), nn.BatchNorm1d(gate_channels[i + 1])
            )
            self.gate_c.add_module('gate_c_relu_%d' % (i + 1), nn.ReLU())
        self.gate_c.add_module(
            'gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1])
        )

    def forward(self, in_tensor):
        avg_pool = F.avg_pool1d(
            in_tensor, in_tensor.size(2), stride=in_tensor.size(2)
        ).squeeze(-1)
        return self.gate_c(avg_pool).unsqueeze(2)


class SpatialGate(nn.Module):
    def __init__(
        self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4
    ):
        super().__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module(
            'gate_s_conv_reduce0',
            nn.Conv1d(gate_channel, gate_channel // reduction_ratio, kernel_size=1),
        )
        self.gate_s.add_module(
            'gate_s_bn_reduce0', nn.BatchNorm1d(gate_channel // reduction_ratio)
        )
        self.gate_s.add_module('gate_s_relu_reduce0', nn.ReLU())
        for i in range(dilation_conv_num):
            self.gate_s.add_module(
                'gate_s_conv_di_%d' % i,
                nn.Conv1d(
                    gate_channel // reduction_ratio,
                    gate_channel // reduction_ratio,
                    kernel_size=3,
                    padding=dilation_val,
                    dilation=dilation_val,
                ),
            )
            self.gate_s.add_module(
                'gate_s_bn_di_%d' % i, nn.BatchNorm1d(gate_channel // reduction_ratio)
            )
            self.gate_s.add_module('gate_s_relu_di_%d' % i, nn.ReLU())
        self.gate_s.add_module(
            'gate_s_conv_final',
            nn.Conv1d(gate_channel // reduction_ratio, 1, kernel_size=1),
        )

    def forward(self, in_tensor):
        return self.gate_s(in_tensor).squeeze(1).unsqueeze(1)


class Multiview_Attention(nn.Module):
    def __init__(self, gate_channel):
        super().__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)

    def forward(self, in_tensor):
        attention_map = torch.sigmoid(
            self.channel_att(in_tensor) + self.spatial_att(in_tensor)
        )
        return in_tensor + in_tensor * attention_map


class BloodGlucosePredictor(nn.Module):
    def __init__(self, n_demographics: int = 0):
        super().__init__()
        self.n_demographics = n_demographics
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.mva1 = Multiview_Attention(16)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.mva2 = Multiview_Attention(32)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2)
        self.dropout2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.mva3 = Multiview_Attention(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(2)
        self.dropout3 = nn.Dropout(0.25)

        self.conv4 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.mva4 = Multiview_Attention(128)
        self.bn4 = nn.BatchNorm1d(128)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        self.lstm_a = nn.LSTM(128, 4, batch_first=True, bidirectional=True)
        self.lstm_b = nn.LSTM(128, 2, batch_first=True, bidirectional=True)
        self.lstm_c = nn.LSTM(128, 1, batch_first=True, bidirectional=True)

        common_size = 128
        self.transform_a = nn.Linear(8, common_size)
        self.transform_b = nn.Linear(4, common_size)
        self.transform_c = nn.Linear(2, common_size)

        self.fc1 = nn.Linear(common_size * 6 + n_demographics, 256)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)

    def forward(self, x, demographics=None):
        x = self.dropout1(self.pool1(F.relu(self.bn1(self.mva1(self.conv1(x))))))
        x = self.dropout2(self.pool2(F.relu(self.bn2(self.mva2(self.conv2(x))))))
        x = self.dropout3(self.pool3(F.relu(self.bn3(self.mva3(self.conv3(x))))))
        x = F.relu(self.bn4(self.mva4(self.conv4(x))))
        x = self.global_max_pool(x).squeeze(-1)

        x_a = self.transform_a(self.lstm_a(x.unsqueeze(1))[0].squeeze(1))
        x_b = self.transform_b(self.lstm_b(x.unsqueeze(1))[0].squeeze(1))
        x_c = self.transform_c(self.lstm_c(x.unsqueeze(1))[0].squeeze(1))

        block1 = torch.cat((x_a, x_b, x_c), dim=-1)
        block2 = torch.cat((x_a * x_b, x_a * x_c, x_b * x_c), dim=-1)
        hof = torch.cat((block1, block2), dim=-1)
        if self.n_demographics:
            hof = torch.cat((hof, demographics), dim=-1)

        # Original release used dropout3 here (0.25), not dropout4 (0.5).
        x = self.dropout3(F.relu(self.fc1(hof)))
        return self.fc5(self.fc4(self.fc3(self.fc2(x))))
