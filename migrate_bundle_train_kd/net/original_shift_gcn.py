import torch
import torch.nn as nn
import numpy as np
import math

# ---------- helpers ----------
def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

class tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Shift_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(Shift_tcn, self).__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Shift_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(Shift_gcn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_point = A.shape[1]

        self.linear = nn.Linear(in_channels, out_channels)
        if self.linear.bias is not None:
            self.linear.bias.data.fill_(0)

        self.Feature_Mask = nn.Parameter(torch.ones(1, self.num_point, in_channels))
        nn.init.constant_(self.Feature_Mask, 0)

        self.bn = nn.BatchNorm1d(self.num_point * out_channels)
        self.relu = nn.ReLU(inplace=True)

        index_array = np.empty(self.num_point * in_channels).astype(np.int64)
        for i in range(self.num_point):
            for j in range(in_channels):
                index_array[i * in_channels + j] = (
                    (i * in_channels + j + j * in_channels) % (in_channels * self.num_point)
                )
        self.shift_in = nn.Parameter(torch.from_numpy(index_array), requires_grad=False)

        index_array = np.empty(self.num_point * out_channels).astype(np.int64)
        for i in range(self.num_point):
            for j in range(out_channels):
                index_array[i * out_channels + j] = (
                    (i * out_channels + j - j * out_channels) % (out_channels * self.num_point)
                )
        self.shift_out = nn.Parameter(torch.from_numpy(index_array), requires_grad=False)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

    def forward(self, x0):
        n, c, t, v = x0.size()
        x = x0.permute(0, 2, 3, 1).contiguous()        # (n,t,v,c)
        x = x.view(n * t, v * c)
        x = torch.index_select(x, 1, self.shift_in)
        x = x.view(n * t, v, c)
        x = x * (torch.tanh(self.Feature_Mask) + 1)
        x = self.linear(x)                             # (n*t, v, out_c)
        x = x.view(n * t, -1)
        x = torch.index_select(x, 1, self.shift_out)
        x = self.bn(x)
        x = x.view(n, t, v, self.out_channels).permute(0, 3, 1, 2).contiguous()
        x = x + self.down(x0)
        x = self.relu(x)
        return x

class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = Shift_gcn(in_channels, out_channels, A)
        self.tcn1 = Shift_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)

# ---------- main backbone ----------
class OriginalShiftGCN(nn.Module):
    """
    兼容两种输出：
      - return logits, feature_vector
      - 若 return_map=True：return logits, feature_map(N,C,T',V',M), feature_vector
    """
    def __init__(self, num_class=60, num_point=25, num_person=2,
                 graph=None, graph_args=dict(), in_channels=3,
                 return_map: bool = False):
        super(OriginalShiftGCN, self).__init__()
        if graph is None:
            raise ValueError("graph must be provided")
        Graph = import_class(graph)
        self.graph = Graph(**graph_args)

        self.num_point = self.graph.num_node
        A = self.graph.A
        self.return_map_default = bool(return_map)

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * self.num_point)

        self.l1  = TCN_GCN_unit(in_channels, 64,  A, residual=False)
        self.l2  = TCN_GCN_unit(64,  64,  A)
        self.l3  = TCN_GCN_unit(64,  64,  A)
        self.l4  = TCN_GCN_unit(64,  64,  A)
        self.l5  = TCN_GCN_unit(64,  128, A, stride=2)   # T downsample
        self.l6  = TCN_GCN_unit(128, 128, A)
        self.l7  = TCN_GCN_unit(128, 128, A)
        self.l8  = TCN_GCN_unit(128, 256, A, stride=2)   # T downsample
        self.l9  = TCN_GCN_unit(256, 256, A)
        self.l10 = TCN_GCN_unit(256, 256, A)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x, return_map: bool | None = None):
        """
        x: (N,C,T,V,M)
        """
        if return_map is None:
            return_map = self.return_map_default

        N, C, T, V, M = x.size()

        # (N,C,T,V,M) -> BN over (N, M*V*C, T)
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        # backbone
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)   # T'
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)   # T''
        x = self.l9(x)
        x = self.l10(x)

        C_new = x.size(1)
        T_new = x.size(2)
        V_new = x.size(3)

        # vector head: GAP over (T',V) and over persons M
        feature_vector = x.view(N, M, C_new, -1).mean(3).mean(1)  # (N, C_new)
        logits = self.fc(feature_vector)

        if return_map:
            # (N*M,C,T',V) -> (N, M, C, T', V) -> (N,C,T',V,M)
            feature_map = x.view(N, M, C_new, T_new, V_new).permute(0, 2, 3, 4, 1).contiguous()
            return logits, feature_map, feature_vector
        else:
            return logits, feature_vector
