import torch
from torch import nn

"""
@description：the networks in this file are used to 消融实验
"""


class conv2d(nn.Module):
    """
    卷积模块：conv2d = Conv2d + BatchNorm2d + ReLU
    """

    def __init__(self, input_channels, output_channels, kernel_size, pad, stride, groups=1):
        super(conv2d, self).__init__()
        self.conv_2d = nn.Conv2d(in_channels=input_channels, out_channels=output_channels,
                                 kernel_size=kernel_size, padding=pad, stride=stride, groups=groups)
        self.bn = nn.BatchNorm2d(output_channels)
        self.active = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.conv_2d(x)
        out = self.bn(out)
        out = self.active(out)
        return out


class TA(nn.Module):
    """
    通道注意力模块
    """

    def __init__(self, in_channel, groups, ks, ratio=9):
        super(TA, self).__init__()
        self.c_weight = nn.Conv2d(in_channels=in_channel, out_channels=in_channel,
                                  kernel_size=ks, padding=0, stride=(1, 1),
                                  bias=False, groups=in_channel)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_channel, (in_channel // ratio) * 3, (1, 1), bias=False, groups=groups),
            nn.ReLU(),
            nn.Conv2d((in_channel // ratio) * 3, in_channel, (1, 1), bias=False, groups=groups)
        )
        self.active = nn.Sigmoid()

    def forward(self, x):
        weight = self.c_weight(x)
        weight = self.shared_MLP(weight)
        weight = self.active(weight)
        out = weight * x
        return out


class TA2(nn.Module):
    """
    通道注意力模块，使用了两个卷积核去生成权重
    """

    def __init__(self, in_channel, groups, ks, ratio=9):
        super(TA2, self).__init__()
        self.c_weight1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel,
                                   kernel_size=ks, padding=0, stride=(1, 1),
                                   bias=False, groups=in_channel)
        self.c_weight2 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel,
                                   kernel_size=ks, padding=0, stride=(1, 1),
                                   bias=False, groups=in_channel)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_channel, (in_channel // ratio) * 3, (1, 1), bias=False, groups=groups),
            nn.ReLU(),
            nn.Conv2d((in_channel // ratio) * 3, in_channel, (1, 1), bias=False, groups=groups)
        )
        self.active = nn.Sigmoid()

    def forward(self, x):
        weight1 = self.c_weight1(x)
        weight1 = self.shared_MLP(weight1)
        weight2 = self.c_weight2(x)
        weight2 = self.shared_MLP(weight2)
        weight = self.active(weight1 + weight2)
        out = weight * x
        return out


class TA3(nn.Module):
    """
    通道注意力模块，使用了三个卷积核去生成权重
    """

    def __init__(self, in_channel, groups, ks, ratio=9):
        super(TA3, self).__init__()
        self.c_weight1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel,
                                   kernel_size=ks, padding=0, stride=(1, 1),
                                   bias=False, groups=in_channel)
        self.c_weight2 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel,
                                   kernel_size=ks, padding=0, stride=(1, 1),
                                   bias=False, groups=in_channel)
        self.c_weight3 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel,
                                   kernel_size=ks, padding=0, stride=(1, 1),
                                   bias=False, groups=in_channel)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_channel, (in_channel // ratio) * 3, (1, 1), bias=False, groups=groups),
            nn.ReLU(),
            nn.Conv2d((in_channel // ratio) * 3, in_channel, (1, 1), bias=False, groups=groups)
        )
        self.active = nn.Sigmoid()

    def forward(self, x):
        weight1 = self.c_weight1(x)
        weight1 = self.shared_MLP(weight1)
        weight2 = self.c_weight2(x)
        weight2 = self.shared_MLP(weight2)
        weight3 = self.c_weight2(x)
        weight3 = self.shared_MLP(weight3)
        weight = self.active(weight1 + weight2 + weight3)
        out = weight * x
        return out


class TA_without_shared_mlp(nn.Module):
    """
    通道注意力模块，去掉了shared mlp
    """

    def __init__(self, in_channel, groups, ks, ratio=9):
        super(TA_without_shared_mlp, self).__init__()
        self.c_weight = nn.Conv2d(in_channels=in_channel, out_channels=in_channel,
                                  kernel_size=ks, padding=0, stride=(1, 1),
                                  bias=False, groups=in_channel)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_channel, (in_channel // ratio) * 3, (1, 1), bias=False, groups=groups),
            nn.ReLU(),
            nn.Conv2d((in_channel // ratio) * 3, in_channel, (1, 1), bias=False, groups=groups)
        )
        self.active = nn.Sigmoid()

    def forward(self, x):
        weight = self.c_weight(x)
        # weight = self.shared_MLP(weight)
        weight = self.active(weight)
        out = weight * x
        return out


class SA(nn.Module):
    """
    空间注意力模块
    """

    def __init__(self, in_channels, groups, ks):
        super(SA, self).__init__()
        self.s_weight = nn.Conv2d(in_channels=in_channels, out_channels=groups,
                                  kernel_size=ks, padding=0, stride=(1, 1),
                                  bias=False, groups=groups)
        self.group = groups
        self.active = nn.Sigmoid()

    def forward(self, x):
        if self.group == 3:
            x_list = x.chunk(3, dim=1)
            weight = self.active(self.s_weight(x))
            weight_list = weight.chunk(3, dim=1)
            x_list[0].data *= weight_list[0]
            x_list[1].data *= weight_list[1]
            x_list[2].data *= weight_list[2]
            x_list = torch.cat(x_list, dim=1)
            out = x_list
            return out
        else:
            weight = self.s_weight(x)
            weight = self.active(weight)
            out = weight * x
            return out


class SA2(nn.Module):
    """
    空间注意力模块，使用两个卷积模块
    """

    def __init__(self, in_channels, groups, ks):
        super(SA2, self).__init__()
        self.s_weight1 = nn.Conv2d(in_channels=in_channels, out_channels=groups,
                                   kernel_size=ks, padding=0, stride=(1, 1),
                                   bias=False, groups=groups)
        self.s_weight2 = nn.Conv2d(in_channels=in_channels, out_channels=groups,
                                   kernel_size=ks, padding=0, stride=(1, 1),
                                   bias=False, groups=groups)
        self.group = groups
        self.active = nn.Sigmoid()

    def forward(self, x):
        if self.group == 3:
            x_list = x.chunk(3, dim=1)
            weight1 = self.s_weight1(x)
            weight2 = self.s_weight2(x)
            weight = self.active(weight1 + weight2)
            weight_list = weight.chunk(3, dim=1)
            x_list[0].data *= weight_list[0]
            x_list[1].data *= weight_list[1]
            x_list[2].data *= weight_list[2]
            x_list = torch.cat(x_list, dim=1)
            out = x_list
            return out
        else:
            weight1 = self.s_weight1(x)
            weight2 = self.s_weight2(x)
            weight = self.active(weight1 + weight2)
            out = weight * x
            return out


class SA3(nn.Module):
    """
    空间注意力模块
    """

    def __init__(self, in_channels, groups, ks):
        super(SA3, self).__init__()
        self.s_weight1 = nn.Conv2d(in_channels=in_channels, out_channels=groups,
                                   kernel_size=ks, padding=0, stride=(1, 1),
                                   bias=False, groups=groups)
        self.s_weight2 = nn.Conv2d(in_channels=in_channels, out_channels=groups,
                                   kernel_size=ks, padding=0, stride=(1, 1),
                                   bias=False, groups=groups)
        self.s_weight3 = nn.Conv2d(in_channels=in_channels, out_channels=groups,
                                   kernel_size=ks, padding=0, stride=(1, 1),
                                   bias=False, groups=groups)
        self.group = groups
        self.active = nn.Sigmoid()

    def forward(self, x):
        if self.group == 3:
            x_list = x.chunk(3, dim=1)
            weight1 = self.s_weight1(x)
            weight2 = self.s_weight2(x)
            weight3 = self.s_weight3(x)
            weight = self.active(weight1 + weight2 + weight3)
            weight_list = weight.chunk(3, dim=1)
            x_list[0].data *= weight_list[0]
            x_list[1].data *= weight_list[1]
            x_list[2].data *= weight_list[2]
            x_list = torch.cat(x_list, dim=1)
            out = x_list
            return out
        else:
            weight1 = self.s_weight1(x)
            weight2 = self.s_weight2(x)
            weight = self.active(weight1 + weight2)
            out = weight * x
            return out


class RAM(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, feature_size,
                 groups, use_1x1conv=False, strides=(1, 1), use_TA=True, use_SA=True):
        super().__init__()
        self.conv1 = conv2d(input_channels, output_channels, groups=groups,
                            kernel_size=kernel_size, pad=1, stride=strides)
        self.conv2 = conv2d(output_channels, output_channels, groups=groups,
                            kernel_size=kernel_size, pad=1, stride=(1, 1))
        self.TA = None
        self.SA = None
        self.conv3 = None
        if use_TA:
            self.TA = TA(output_channels, groups, ks=feature_size)
        if use_SA:
            self.SA = SA(output_channels, groups, ks=(1, 1))
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, output_channels, groups=groups,
                                   kernel_size=(1, 1), stride=strides)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.TA:
            out = self.TA(out)
        if self.SA:
            out = self.SA(out)
        if self.conv3:
            x = self.conv3(x)
        out1 = out + x
        return out1


class RAM2(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, feature_size,
                 groups, use_1x1conv=False, strides=(1, 1)):
        super().__init__()
        self.conv1 = conv2d(input_channels, output_channels, groups=groups,
                            kernel_size=kernel_size, pad=1, stride=strides)
        self.conv2 = conv2d(output_channels, output_channels, groups=groups,
                            kernel_size=kernel_size, pad=1, stride=(1, 1))
        self.TA = TA2(output_channels, groups, ks=feature_size)
        self.SA = SA2(output_channels, groups, ks=(1, 1))

        if use_1x1conv:
            self.conv3 = conv2d(input_channels, output_channels, groups=groups,
                                kernel_size=(1, 1), pad=0, stride=strides)
        else:
            self.conv3 = None

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.TA(out)
        out = self.SA(out)
        if self.conv3:
            x = self.conv3(x)
        out += x
        return out


class RAM3(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, feature_size,
                 groups, use_1x1conv=False, strides=(1, 1)):
        super().__init__()
        self.conv1 = conv2d(input_channels, output_channels, groups=groups,
                            kernel_size=kernel_size, pad=1, stride=strides)
        self.conv2 = conv2d(output_channels, output_channels, groups=groups,
                            kernel_size=kernel_size, pad=1, stride=(1, 1))
        self.TA = TA3(output_channels, groups, ks=feature_size)
        self.SA = SA3(output_channels, groups, ks=(1, 1))

        if use_1x1conv:
            self.conv3 = conv2d(input_channels, output_channels, groups=groups,
                                kernel_size=(1, 1), pad=0, stride=strides)
        else:
            self.conv3 = None

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.TA(out)
        out = self.SA(out)
        if self.conv3:
            x = self.conv3(x)
        out += x
        return out


class RAM_without_shared_mlp(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, feature_size,
                 groups, use_1x1conv=False, strides=(1, 1), use_TA=True, use_SA=True):
        super().__init__()
        self.conv1 = conv2d(input_channels, output_channels, groups=groups,
                            kernel_size=kernel_size, pad=1, stride=strides)
        self.conv2 = conv2d(output_channels, output_channels, groups=groups,
                            kernel_size=kernel_size, pad=1, stride=(1, 1))
        self.TA = None
        self.SA = None
        self.conv3 = None
        if use_TA:
            self.TA = TA_without_shared_mlp(output_channels, groups, ks=feature_size)
        if use_SA:
            self.SA = SA(output_channels, groups, ks=(1, 1))
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, output_channels, groups=groups,
                                   kernel_size=(1, 1), stride=strides)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.TA:
            out = self.TA(out)
        if self.SA:
            out = self.SA(out)
        if self.conv3:
            x = self.conv3(x)
        out1 = out + x
        return out1


class Attention_module(nn.Module):
    def __init__(self, input_channels, output_channels, feature_size, strides=(1, 1),
                 groups=3, use_TA=True, use_SA=True):
        super(Attention_module, self).__init__()
        self.DRAM = RAM(input_channels=input_channels, output_channels=output_channels, kernel_size=(3, 3),
                        strides=(2, 2), feature_size=feature_size, groups=groups, use_1x1conv=True,
                        use_TA=use_TA, use_SA=use_SA)
        self.RAM1 = RAM(input_channels=output_channels, output_channels=output_channels, kernel_size=(3, 3),
                        feature_size=feature_size, groups=groups, use_TA=use_TA, use_SA=use_SA)
        self.RAM2 = RAM(input_channels=output_channels, output_channels=output_channels, kernel_size=(3, 3),
                        feature_size=feature_size, groups=1, use_TA=use_TA, use_SA=use_SA)

    def forward(self, x):
        out = self.DRAM(x)
        out = self.RAM1(out)
        out = self.RAM2(out)
        return out


class Attention_module2(nn.Module):
    def __init__(self, input_channels, output_channels, feature_size, groups=3):
        super(Attention_module2, self).__init__()
        self.DRAM = RAM2(input_channels=input_channels, output_channels=output_channels, kernel_size=(3, 3),
                         strides=(2, 2), feature_size=feature_size, groups=groups, use_1x1conv=True)
        self.RAM1 = RAM2(input_channels=output_channels, output_channels=output_channels, kernel_size=(3, 3),
                         feature_size=feature_size, groups=groups)
        self.RAM2 = RAM2(input_channels=output_channels, output_channels=output_channels, kernel_size=(3, 3),
                         feature_size=feature_size, groups=1)

    def forward(self, x):
        out = self.DRAM(x)
        out = self.RAM1(out)
        out = self.RAM2(out)
        return out


class Attention_module3(nn.Module):
    def __init__(self, input_channels, output_channels, feature_size, groups=3):
        super(Attention_module3, self).__init__()
        self.DRAM = RAM3(input_channels=input_channels, output_channels=output_channels, kernel_size=(3, 3),
                         strides=(2, 2), feature_size=feature_size, groups=groups, use_1x1conv=True)
        self.RAM1 = RAM3(input_channels=output_channels, output_channels=output_channels, kernel_size=(3, 3),
                         feature_size=feature_size, groups=groups)
        self.RAM2 = RAM3(input_channels=output_channels, output_channels=output_channels, kernel_size=(3, 3),
                         feature_size=feature_size, groups=1)

    def forward(self, x):
        out = self.DRAM(x)
        out = self.RAM1(out)
        out = self.RAM2(out)
        return out


class Attention_module_without_shared_mlp(nn.Module):
    def __init__(self, input_channels, output_channels, feature_size, groups=3, use_TA=True, use_SA=True):
        super(Attention_module_without_shared_mlp, self).__init__()
        self.DRAM = RAM_without_shared_mlp(input_channels=input_channels, output_channels=output_channels,
                                           kernel_size=(3, 3), strides=(2, 2), feature_size=feature_size,
                                           groups=groups, use_1x1conv=True, use_TA=use_TA, use_SA=use_SA)
        self.RAM1 = RAM_without_shared_mlp(input_channels=output_channels, output_channels=output_channels,
                                           kernel_size=(3, 3), feature_size=feature_size, groups=groups,
                                           use_TA=use_TA, use_SA=use_SA)
        self.RAM2 = RAM_without_shared_mlp(input_channels=output_channels, output_channels=output_channels,
                                           kernel_size=(3, 3), feature_size=feature_size, groups=1,
                                           use_TA=use_TA, use_SA=use_SA)

    def forward(self, x):
        out = self.DRAM(x)
        out = self.RAM1(out)
        out = self.RAM2(out)
        return out


def RAM_Net():
    b1 = conv2d(input_channels=3, output_channels=33, kernel_size=(3, 3), pad=1, groups=3, stride=(2, 2))  # out-->32*32
    b2 = nn.Sequential(Attention_module(input_channels=33, output_channels=66, feature_size=(16, 16)),  # out-->16*16
                       Attention_module(input_channels=66, output_channels=132, feature_size=(8, 8)),  # out-->8*8
                       Attention_module(input_channels=132, output_channels=264, feature_size=(4, 4)),  # out-->4*4
                       Attention_module(input_channels=264, output_channels=528, feature_size=(2, 2)))  # out-->2*2
    new_net = nn.Sequential(b1, b2, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(528, 2))
    return new_net


def RAM_Net_Without_TA():
    b1 = conv2d(input_channels=3, output_channels=33, kernel_size=(3, 3), pad=1, groups=3, stride=(2, 2))  # out-->32*32
    b2 = nn.Sequential(Attention_module(input_channels=33, output_channels=66, feature_size=(16, 16),
                                        use_TA=False),  # out-->16*16
                       Attention_module(input_channels=66, output_channels=132, feature_size=(8, 8),
                                        use_TA=False),  # out-->8*8
                       Attention_module(input_channels=132, output_channels=264, feature_size=(4, 4),
                                        use_TA=False),  # out-->4*4
                       Attention_module(input_channels=264, output_channels=528, feature_size=(2, 2),
                                        use_TA=False))  # out-->2*2
    new_net = nn.Sequential(b1, b2, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(528, 2))
    return new_net


def RAM_Net_Without_SA():
    b1 = conv2d(input_channels=3, output_channels=33, kernel_size=(3, 3), pad=1, groups=3, stride=(2, 2))  # out-->32*32
    b2 = nn.Sequential(Attention_module(input_channels=33, output_channels=66, feature_size=(16, 16),
                                        use_SA=False),  # out-->16*16
                       Attention_module(input_channels=66, output_channels=132, feature_size=(8, 8),
                                        use_SA=False),  # out-->8*8
                       Attention_module(input_channels=132, output_channels=264, feature_size=(4, 4),
                                        use_SA=False),  # out-->4*4
                       Attention_module(input_channels=264, output_channels=528, feature_size=(2, 2),
                                        use_SA=False))  # out-->2*2
    new_net = nn.Sequential(b1, b2, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(528, 2))
    return new_net


def RAM_Net_Without_SA_TA():
    b1 = conv2d(input_channels=3, output_channels=33, kernel_size=(3, 3), pad=1, groups=3, stride=(2, 2))  # out-->32*32
    b2 = nn.Sequential(Attention_module(input_channels=33, output_channels=66, feature_size=(16, 16),
                                        use_TA=False, use_SA=False),  # out-->16*16
                       Attention_module(input_channels=66, output_channels=132, feature_size=(8, 8),
                                        use_TA=False, use_SA=False),  # out-->8*8
                       Attention_module(input_channels=132, output_channels=264, feature_size=(4, 4),
                                        use_TA=False, use_SA=False),  # out-->4*4
                       Attention_module(input_channels=264, output_channels=528, feature_size=(2, 2),
                                        use_TA=False, use_SA=False))  # out-->2*2
    new_net = nn.Sequential(b1, b2, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(528, 2))
    return new_net


def RAM_Net_2kernels():
    b1 = conv2d(input_channels=3, output_channels=33, kernel_size=(3, 3), pad=1, groups=3, stride=(2, 2))  # out-->32*32
    b2 = nn.Sequential(Attention_module2(input_channels=33, output_channels=66, feature_size=(16, 16)),  # out-->16*16
                       Attention_module2(input_channels=66, output_channels=132, feature_size=(8, 8)),  # out-->8*8
                       Attention_module2(input_channels=132, output_channels=264, feature_size=(4, 4)),  # out-->4*4
                       Attention_module2(input_channels=264, output_channels=528, feature_size=(2, 2)))  # out-->2*2
    new_net = nn.Sequential(b1, b2, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(528, 2))
    return new_net


def RAM_Net_3kernels():
    b1 = conv2d(input_channels=3, output_channels=33, kernel_size=(3, 3), pad=1, groups=3, stride=(2, 2))  # out-->32*32
    b2 = nn.Sequential(Attention_module3(input_channels=33, output_channels=66, feature_size=(16, 16)),  # out-->16*16
                       Attention_module3(input_channels=66, output_channels=132, feature_size=(8, 8)),  # out-->8*8
                       Attention_module3(input_channels=132, output_channels=264, feature_size=(4, 4)),  # out-->4*4
                       Attention_module3(input_channels=264, output_channels=528, feature_size=(2, 2)))  # out-->2*2
    new_net = nn.Sequential(b1, b2, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(528, 2))
    return new_net


def RAM_Net_without_shared_mlp():
    b1 = conv2d(input_channels=3, output_channels=33, kernel_size=(3, 3), pad=1, groups=3, stride=(2, 2))  # out-->32*32
    b2 = nn.Sequential(Attention_module_without_shared_mlp(input_channels=33, output_channels=66,
                                                           feature_size=(16, 16)),  # out-->16*16
                       Attention_module_without_shared_mlp(input_channels=66, output_channels=132,
                                                           feature_size=(8, 8)),  # out-->8*8
                       Attention_module_without_shared_mlp(input_channels=132, output_channels=264,
                                                           feature_size=(4, 4)),  # out-->4*4
                       Attention_module_without_shared_mlp(input_channels=264, output_channels=528,
                                                           feature_size=(2, 2)))  # out-->2*2
    new_net = nn.Sequential(b1, b2, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(528, 2))
    return new_net
