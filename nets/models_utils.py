from collections import OrderedDict

import torch.nn as nn
import torch


# 这是标准的CBA模块
def conv2d(filter_in, filter_out, kernel_size, groups=1, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    # OrderedDict有序字典，OrderedDict(key1=value1,key2=value2,.....),也可以按下面的方式传入传入一个包含元组的列表
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, groups=groups,
                           bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.ReLU6(inplace=True)),
    ]))


# 这里包含了dw卷积和pw卷积,因此是一个完整的深度可分离卷积
def conv_dw(filter_in, filter_out, stride=1):
    return nn.Sequential(
        nn.Conv2d(filter_in, filter_in, 3, stride, 1, groups=filter_in, bias=False),
        nn.BatchNorm2d(filter_in),
        nn.ReLU6(inplace=True),

        nn.Conv2d(filter_in, filter_out, 1, 1, 0, bias=False),
        nn.BatchNorm2d(filter_out),
        nn.ReLU6(inplace=True),
    )


# 三次卷积块，原先中间的卷积是3 * 3的卷积，这里换成了深度可分离卷积
def make_three_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv_dw(filters_list[0], filters_list[1]),  # 深度可分离卷积
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m


# 五次卷积块，3 * 3卷积换成了深度可分离卷积
def make_five_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv_dw(filters_list[0], filters_list[1]),
        conv2d(filters_list[1], filters_list[0], 1),
        conv_dw(filters_list[0], filters_list[1]),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m


# ---------------------------------------------------#
#   SPP结构，利用不同大小的池化核进行池化
#   池化后堆叠
# ---------------------------------------------------#
class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size // 2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)  # 经过多级池化后，将这些经过不同窗口大小池化后的特征进行堆叠。

        return features


# ---------------------------------------------------#
#   卷积 + 上采样
# ---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x, ):
        x = self.upsample(x)
        return x


# ---------------------------------------------------#
#   最后获得yolov4的输出
# ---------------------------------------------------#
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        conv_dw(in_filters, filters_list[0]),

        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m


# ---------------------以下是yolox的module-----------------------
# silu激活函数
class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


# h-swish激活函数
class HS(nn.Module):

    def __init__(self):
        super(HS, self).__init__()

    def forward(self, inputs):
        clip = torch.clamp(inputs + 3, 0, 6) / 6
        return inputs * clip


# 根据name返回指定的激活函数
def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = SiLU()
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == "hs":
        module = HS()
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


# 基本的卷积模块:CBS模块， fuseforward表示不使用BN层
class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        # return self.act(self.bn(self.conv(x)))
        return self.act(x)
    def fuseforward(self, x):
        return self.act(self.conv(x))


# DWConv：包含一次dw卷积和pw卷积，也就是一次完整的深度可分离卷积
class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels, act=act, )
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class SPPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)  # 4 * 512 = 2024
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)  # 多了一个卷积核大小为1卷积
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


# --------------------------------------------------#
#   残差结构的构建，小的残差结构
# --------------------------------------------------#
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, depthwise=False, act="silu", ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # 这里expansion为1所以隐藏channel等于out_channel
        Conv = DWConv if depthwise else BaseConv
        # --------------------------------------------------#
        #   利用1x1卷积进行通道数的缩减。缩减率一般是50%
        # --------------------------------------------------#
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        # --------------------------------------------------#
        #   利用3x3卷积进行通道数的拓张。并且完成特征提取
        # --------------------------------------------------#
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=False, act="silu", ):
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # 隐藏层为输出channel的一半
        # --------------------------------------------------#
        #   主干部分的初次卷积
        # --------------------------------------------------#
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        # --------------------------------------------------#
        #   大的残差边部分的初次卷积
        # --------------------------------------------------#
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        # -----------------------------------------------#
        #   对堆叠的结果进行卷积的处理
        # -----------------------------------------------#
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)

        # --------------------------------------------------#
        #   根据循环的次数构建上述Bottleneck残差结构
        # --------------------------------------------------#
        module_list = [Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act) for _ in
                       range(n)]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        # -------------------------------#
        #   x_1是主干部分
        # -------------------------------#
        x_1 = self.conv1(x)
        # -------------------------------#
        #   x_2是大的残差边部分
        # -------------------------------#
        x_2 = self.conv2(x)

        # -----------------------------------------------#
        #   主干部分利用残差结构堆叠继续进行特征提取
        # -----------------------------------------------#
        x_1 = self.m(x_1)
        # -----------------------------------------------#
        #   主干部分和大的残差边部分进行堆叠
        # -----------------------------------------------#
        x = torch.cat((x_1, x_2), dim=1)
        # -----------------------------------------------#
        #   对堆叠的结果进行卷积的处理
        # -----------------------------------------------#
        return self.conv3(x)
