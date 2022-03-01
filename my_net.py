import my_module
import my_module as mm
from torch import nn


# 定义attention block
def attention_block(input_channels, num_channels, num_self_cbam_residuals,
                    kernel_size, padding, first_block=False):
    blk = []
    for i in range(num_self_cbam_residuals):
        if i == 0 and not first_block:
            blk.append(mm.Self_CBAMResidual(input_channels, num_channels, kernel_size, padding,
                                            use_1x1conv=True, strides=2))
        else:
            blk.append(mm.Self_CBAMResidual(num_channels, num_channels, kernel_size, padding))
    blk.append(mm.Cross_CBAMResidual(num_channels, num_channels, kernel_size, padding))
    return blk


def new_cbam_net(kernel_size, padding):
    b1 = nn.Sequential(nn.Conv2d(3, 63, kernel_size=3, padding=1, groups=3, stride=2),
                       nn.BatchNorm2d(63), nn.ReLU())
    b2 = nn.Sequential(*attention_block(63, 63, 2, kernel_size, padding, first_block=True))
    b3 = nn.Sequential(*attention_block(63, 126, 2, kernel_size, padding))
    b4 = nn.Sequential(*attention_block(126, 252, 2, kernel_size, padding))
    b5 = nn.Sequential(*attention_block(252, 504, 2, kernel_size, padding))
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
                                  stride, kernel_size):
    blk = []
    for i in range(num_self_cbam_residuals):
        if i == 0:
            blk.append(mm.Self_ObjectDetect_CBAMResidual(input_channels, num_channels, kernel_size=kernel_size,
                                                         use_1x1conv=True, strides=stride))
        else:
            blk.append(mm.Self_ObjectDetect_CBAMResidual(num_channels, num_channels, kernel_size=kernel_size))
    blk.append(mm.Cross_ObjectDetect_CBAMResidual(num_channels, num_channels))
    return blk


def object_detect_new_cbam_net(anchors, kernel_size):
    b1 = nn.Sequential(nn.Conv2d(3, 63, kernel_size=(3, 9), padding=(1, 4), groups=3, stride=(1, 1)),
                       nn.BatchNorm2d(63), nn.ReLU())  # out-->63*512
    b2 = nn.Sequential(*object_detect_attention_block(63, 63, 2, stride=(2, 2), kernel_size=kernel_size))  # out-->32*256
    b3 = nn.Sequential(*object_detect_attention_block(63, 126, 2, stride=(1, 2), kernel_size=kernel_size))  # out-->32*128
    b4 = nn.Sequential(*object_detect_attention_block(126, 126, 2, stride=(2, 2), kernel_size=kernel_size))  # out-->16*64
    b5 = nn.Sequential(*object_detect_attention_block(126, 252, 2, stride=(1, 2), kernel_size=kernel_size))  # out-->16*32
    b6 = nn.Sequential(*object_detect_attention_block(252, 252, 2, stride=(2, 2), kernel_size=kernel_size))  # out-->8*16
    b7 = nn.Sequential(nn.Conv2d(252, 15, kernel_size=(1, 1)), nn.BatchNorm2d(15), nn.ReLU())  # out-->15*8*16
    b8 = my_module.Out_Layer(anchors=anchors, stride_x=32, stride_y=8)
    new_net = nn.Sequential(b1, b2, b3, b4, b5, b6, b7, b8)
    return new_net
