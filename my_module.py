import torch
from torch import nn
from torch.nn import functional as F


class Self_ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=9):
        super(Self_ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, (channel // ratio)*3, 1, bias=False, groups=3),
            nn.ReLU(),
            nn.Conv2d((channel // ratio)*3, channel, 1, bias=False, groups=3)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class Self_SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(Self_SpatialAttentionModule, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7,
                                stride=1, padding=3)
        self.conv2d_1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7,
                                stride=1, padding=3)
        self.conv2d_2 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7,
                                stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_list = x.chunk(3, dim=1)
        avgout_0 = torch.mean(x_list[0].data, dim=1, keepdim=True)
        maxout_0, _ = torch.max(x_list[0].data, dim=1, keepdim=True)
        avgout_1 = torch.mean(x_list[1].data, dim=1, keepdim=True)
        maxout_1, _ = torch.max(x_list[1].data, dim=1, keepdim=True)
        avgout_2 = torch.mean(x_list[2].data, dim=1, keepdim=True)
        maxout_2, _ = torch.max(x_list[2].data, dim=1, keepdim=True)
        out_0 = torch.cat([avgout_0, maxout_0], dim=1)
        out_0 = self.sigmoid(self.conv2d_0(out_0))
        x_list[0].data *= out_0
        out_1 = torch.cat([avgout_1, maxout_1], dim=1)
        out_1 = self.sigmoid(self.conv2d_1(out_1))
        x_list[1].data *= out_1
        out_2 = torch.cat([avgout_2, maxout_2], dim=1)
        out_2 = self.sigmoid(self.conv2d_2(out_2))
        x_list[2].data *= out_2

        x_list = torch.cat(x_list, dim=1)
        return x_list


class Self_CBAM(nn.Module):
    def __init__(self, channel):
        super(Self_CBAM, self).__init__()
        self.channel_attention = Self_ChannelAttentionModule(channel)
        self.spatial_attention = Self_SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out)
        return out


class Cross_ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=9):
        super(Cross_ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, (channel // ratio)*3, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d((channel // ratio)*3, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class Cross_SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(Cross_SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1,
                                kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out1 = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out1))
        return out


class Cross_CBAM(nn.Module):
    def __init__(self, channel):
        super(Cross_CBAM, self).__init__()
        self.channel_attention = Cross_ChannelAttentionModule(channel)
        self.spatial_attention = Cross_SpatialAttentionModule()

    def forward(self, x):
        out1 = self.channel_attention(x) * x
        out2 = self.spatial_attention(x) * out1
        return out2


class Self_CBAMResidual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, groups=3,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, groups=3,
                               kernel_size=3, padding=1)
        self.cbamself = Self_CBAM(num_channels)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, groups=3,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        Y = self.cbamself(Y)
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


class Cross_CBAMResidual(nn.Module):
    def __init__(self, input_channels, num_channels, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        self.cbam = Cross_CBAM(num_channels)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        Y = self.cbam(Y)
        Y += X
        return F.relu(Y)
