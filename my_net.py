import my_module
import my_module as mm
from torch import nn


# 定义attention block
def attention_block(input_channels, num_channels, num_self_cbam_residuals,
                    kernel_size, padding, first_block=False):
    blk = []
    for i in range(num_self_cbam_residuals):
        if i == 0 and not first_block:
            blk.append(mm.Self_CBAMResidual(input_channels, num_channels, kernel_size, padding, 3,
                                            use_1x1conv=True, strides=2))
        else:
            blk.append(mm.Self_CBAMResidual(num_channels, num_channels, kernel_size, padding, 3))
    blk.append(mm.Cross_CBAMResidual(num_channels, num_channels, kernel_size, padding))
    return blk


# 定义attention block
def attention_block_no_group(input_channels, num_channels, num_self_cbam_residuals,
                    kernel_size, padding, first_block=False):
    blk = []
    for i in range(num_self_cbam_residuals):
        if i == 0 and not first_block:
            blk.append(mm.Cross_CBAMResidual_for_no_group(input_channels, num_channels, kernel_size, padding, 3,
                                             use_1x1conv=True, strides=2))
        else:
            blk.append(mm.Cross_CBAMResidual_for_no_group(num_channels, num_channels, kernel_size, padding, 3))
    blk.append(mm.Cross_CBAMResidual(num_channels, num_channels, kernel_size, padding))
    return blk


def new_cbam_net(kernel_size, padding):
    b1 = nn.Sequential(nn.Conv2d(3, 63, kernel_size=(3, 3), padding=1, groups=3, stride=(2, 2)),
                       nn.BatchNorm2d(63), nn.ReLU())  # out-->32*32
    b2 = nn.Sequential(*attention_block(63, 63, 2, kernel_size, padding))  # out-->16*16
    b3 = nn.Sequential(*attention_block(63, 126, 2, kernel_size, padding))  # out-->8*8
    b4 = nn.Sequential(*attention_block(126, 252, 2, kernel_size, padding))  # out-->4*4
    b5 = nn.Sequential(*attention_block(252, 504, 2, kernel_size, padding))  # out-->2*2
    new_net = nn.Sequential(b1, b2, b3, b4, b5,
                            nn.AdaptiveAvgPool2d((1, 1)),
                            nn.Flatten(), nn.Linear(504, 2))
    return new_net


def RAM_Net_P(kernel_size, padding):
    b1 = nn.Sequential(nn.Conv2d(3, 33, kernel_size=(3, 3), padding=1, groups=3, stride=(2, 2)),
                       nn.BatchNorm2d(33), nn.ReLU())  # out-->32*32
    b2 = nn.Sequential(*attention_block(33, 66, 2, kernel_size, padding))  # out-->16*16
    b3 = nn.Sequential(*attention_block(66, 132, 2, kernel_size, padding))  # out-->8*8
    b4 = nn.Sequential(*attention_block(132, 264, 2, kernel_size, padding))  # out-->4*4
    b5 = nn.Sequential(*attention_block(264, 528, 2, kernel_size, padding))  # out-->2*2
    new_net = nn.Sequential(b1, b2, b3, b4, b5,
                            nn.AdaptiveAvgPool2d((1, 1)),
                            nn.Flatten(), nn.Linear(528, 2))
    return new_net


def new_cbam_net_s1(kernel_size, padding):
    """
    比原始的网络少一个大层
    """
    b1 = nn.Sequential(nn.Conv2d(3, 63, kernel_size=3, padding=1, groups=3, stride=2),
                       nn.BatchNorm2d(63), nn.ReLU())  # 32
    b2 = nn.Sequential(*attention_block(63, 126, 2, kernel_size, padding))  # 16
    b3 = nn.Sequential(*attention_block(126, 252, 2, kernel_size, padding))  # 8
    b4 = nn.Sequential(*attention_block(252, 504, 2, kernel_size, padding))  # 4
    new_net = nn.Sequential(b1, b2, b3, b4,
                            nn.AdaptiveAvgPool2d((1, 1)),
                            nn.Flatten(), nn.Linear(504, 2))
    return new_net


def new_cbam_net_ss2(kernel_size=3, padding=1):
    """
    比原始的网络少两个大层
    """
    b1 = nn.Sequential(nn.Conv2d(3, 63, kernel_size=3, padding=1, groups=3, stride=2),
                       nn.BatchNorm2d(63), nn.ReLU())  # 32
    b2 = nn.Sequential(*attention_block(63, 126, 2, kernel_size, padding))  # 16
    b3 = nn.Sequential(*attention_block(126, 252, 2, kernel_size, padding))  # 8
    new_net = nn.Sequential(b1, b2, b3,
                            nn.AdaptiveAvgPool2d((1, 1)),
                            nn.Flatten(), nn.Linear(252, 2))
    return new_net


def new_cbam_net_ss2_no_group(kernel_size=3, padding=1):
    """
    比原始的网络少两个大层
    """
    b1 = nn.Sequential(nn.Conv2d(3, 63, kernel_size=3, padding=1, groups=3, stride=2),
                       nn.BatchNorm2d(63), nn.ReLU())  # 32
    b2 = nn.Sequential(*attention_block_no_group(63, 126, 2, kernel_size, padding))  # 16
    b3 = nn.Sequential(*attention_block_no_group(126, 252, 2, kernel_size, padding))  # 8
    new_net = nn.Sequential(b1, b2, b3,
                            nn.AdaptiveAvgPool2d((1, 1)),
                            nn.Flatten(), nn.Linear(252, 2))
    return new_net


def new_cbam_net_s2(kernel_size=3, padding=1):
    """
    每一个大的层中比原始的网络少一个小的层即RAM
    """
    b1 = nn.Sequential(nn.Conv2d(3, 63, kernel_size=3, padding=1, groups=3, stride=2),
                       nn.BatchNorm2d(63), nn.ReLU())  # 32
    b2 = nn.Sequential(*attention_block(63, 63, 1, kernel_size, padding, first_block=True))  # 32
    b3 = nn.Sequential(*attention_block(63, 126, 1, kernel_size, padding))  # 16
    b4 = nn.Sequential(*attention_block(126, 252, 1, kernel_size, padding))  # 8
    b5 = nn.Sequential(*attention_block(252, 504, 1, kernel_size, padding))  # 4
    new_net = nn.Sequential(b1, b2, b3, b4, b5,
                            nn.AdaptiveAvgPool2d((1, 1)),
                            nn.Flatten(), nn.Linear(504, 2))
    return new_net


def train_siamese_net(kernel_size, padding):
    b1 = nn.Sequential(nn.Conv2d(3, 63, kernel_size=3, padding=1, groups=3, stride=2),
                       nn.BatchNorm2d(63), nn.ReLU())
    b2 = nn.Sequential(*attention_block(63, 63, 2, kernel_size, padding, first_block=True))
    b3 = nn.Sequential(*attention_block(63, 126, 2, kernel_size, padding))
    b4 = nn.Sequential(*attention_block(126, 252, 2, kernel_size, padding))
    b5 = nn.Sequential(*attention_block(252, 504, 2, kernel_size, padding))
    net = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten())
    return net


#  定义object detect attention block
def object_detect_attention_block(input_channels, num_channels, num_self_cbam_residuals,
                                  stride, kernel_size, padding, groups):
    blk = []
    for i in range(num_self_cbam_residuals):
        if i == 0:
            blk.append(mm.Self_ObjectDetect_CBAMResidual(input_channels, num_channels, kernel_size=kernel_size,
                                                         use_1x1conv=True, strides=stride, padding=padding, groups=groups))
        else:
            blk.append(mm.Self_ObjectDetect_CBAMResidual(num_channels, num_channels, kernel_size=kernel_size, padding=padding, groups=groups))
    blk.append(mm.Cross_ObjectDetect_CBAMResidual(num_channels, num_channels, kernel_size=kernel_size, padding=padding))
    return blk


def object_detect_new_cbam_net(anchors, kernel_size, padding, groups):
    b1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=(3, 3), padding=(1, 1), groups=groups, stride=(1, 1)),
                       nn.BatchNorm2d(32), nn.ReLU())  # out-->64*512
    b2 = nn.Sequential(*object_detect_attention_block(32, 64, 2, stride=(2, 2), kernel_size=kernel_size, padding=padding, groups=groups))  # out-->32*256
    b3 = nn.Sequential(*object_detect_attention_block(64, 128, 2, stride=(1, 2), kernel_size=kernel_size, padding=padding, groups=groups))  # out-->32*128
    b4 = nn.Sequential(*object_detect_attention_block(128, 256, 2, stride=(2, 2), kernel_size=kernel_size, padding=padding, groups=groups))  # out-->16*64
    b5 = nn.Sequential(*object_detect_attention_block(256, 128, 2, stride=(1, 2), kernel_size=kernel_size, padding=padding, groups=groups))  # out-->16*32
    b6 = nn.Sequential(*object_detect_attention_block(128, 64, 2, stride=(2, 2), kernel_size=kernel_size, padding=padding, groups=groups))  # out-->8*16
    b7 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=(1, 1)), nn.ReLU())  # out-->32*8*16
    b8 = nn.Sequential(nn.Conv2d(32, 5, kernel_size=(1, 1)), nn.ReLU())  # out-->5*8*16
    b9 = my_module.Out_Layer(anchors=anchors, stride_x=32, stride_y=8)
    new_net = nn.Sequential(b1, b2, b3, b4, b5, b6, b7, b8, b9)
    return new_net
