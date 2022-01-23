import torch
from torch import nn
from torch.nn import functional as F


class Self_ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=9):
        super(Self_ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, (channel // ratio) * 3, 1, bias=False, groups=3),
            nn.ReLU(),
            nn.Conv2d((channel // ratio) * 3, channel, 1, bias=False, groups=3)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class Self_SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(Self_SpatialAttentionModule, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(3, 9),
                                  stride=(1, 1), padding=(1, 4))
        self.conv2d_1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(3, 9),
                                  stride=(1, 1), padding=(1, 4))
        self.conv2d_2 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(3, 9),
                                  stride=(1, 1), padding=(1, 4))
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


class Self_SpatialAttentionModule1(nn.Module):
    def __init__(self, in_channels, kernel_size, padding):
        super(Self_SpatialAttentionModule1, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=kernel_size,
                                  groups=3, stride=(1, 1), padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x *= self.conv2d_0(x)
        return x


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
            nn.Conv2d(channel, (channel // ratio) * 3, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d((channel // ratio) * 3, channel, 1, bias=False)
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
                                kernel_size=(3, 9), stride=(1, 1), padding=(1, 4))
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


class Self_ObjectDetect_CBAMResidual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False,
                 strides=(1, 1), padding=(1, 4), kernel_size=(3, 9)):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, groups=3,
                               kernel_size=kernel_size, padding=padding, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, groups=3,
                               kernel_size=kernel_size, padding=padding)
        self.cbamself = Self_CBAM(num_channels)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, groups=3,
                                   kernel_size=kernel_size, stride=strides, padding=padding)
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


class Cross_ObjectDetect_CBAMResidual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 padding=(1, 4), kernel_size=(3, 9)):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=kernel_size, padding=padding)
        self.cbam = Cross_CBAM(num_channels)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        Y = self.cbam(Y)
        Y += X
        return F.relu(Y)


class Out_Layer(nn.Module):
    """
    对输出进行处理
    """
    def __init__(self, anchors, stride_x, stride_y):
        super(Out_Layer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.stride_x = stride_x  # layer stride 特征图上一步对应原图上的步距 [32, 16, 8]
        self.stride_y = stride_y
        self.na = len(anchors)  # number of anchors (3)
        self.no = 5  # number of outputs (5: x, y, w, h, obj,)
        self.nx, self.ny, self.ng = 0, 0, (0, 0)  # initialize number of x, y gridpoints
        # 将anchors大小缩放到grid尺度
        self.anchors[:, 0] /= self.stride_x
        self.anchors[:, 1] /= self.stride_y
        # batch_size, na, grid_h, grid_w, wh,
        # 值为1的维度对应的值不是固定值，后续操作可根据broadcast广播机制自动扩充
        self.anchor_wh = self.anchors.view(1, self.na, 1, 1, 2)
        self.grid = None

    def create_grids(self, ng=(16, 8), device="cpu"):
        """
        更新grids信息并生成新的grids参数
        :param ng: 特征图大小
        :param device:
        :return:
        """
        self.nx, self.ny = ng
        self.ng = torch.tensor(ng, dtype=torch.float)

        # build xy offsets 构建每个cell处的anchor的xy偏移量(在feature map上的)
        if not self.training:  # 训练模式不需要回归到最终预测boxes
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device),
                                     torch.arange(self.nx, device=device)])
            # batch_size, na, grid_h, grid_w, wh
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

    def forward(self, p):

        bs, _, ny, nx = p.shape  # batch_size, predict_param(255), grid(13), grid(13)
        if (self.nx, self.ny) != (nx, ny) or self.grid is None:  # fix no grid bug
            self.create_grids((nx, ny), p.device)

        # view: (batch_size, 255, 13, 13) -> (batch_size, 3, 85, 13, 13)
        # permute: (batch_size, 3, 85, 13, 13) -> (batch_size, 3, 13, 13, 85)
        # [bs, anchor, grid, grid, xywh + obj + classes]
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p
        else:  # inference，如果是测试的话
            # [bs, anchor, grid, grid, xywh + obj + classes]
            io = p.clone()  # inference output
            self.anchor_wh = self.anchor_wh.to(p.device)
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy 计算在feature map上的xy坐标
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method 计算在feature map上的wh
            io[..., [0, 2]] *= self.stride_x  # 换算映射回原图尺度
            io[..., [1, 3]] *= self.stride_y  # 换算映射回原图尺度
            torch.sigmoid_(io[..., 4:])
            return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 128, 15]
