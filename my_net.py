import my_module
import my_module as mm
from torch import nn


# 定义attention block
def attention_block(input_channels, num_channels, num_self_cbam_residuals, first_block=False):
    blk = []
    for i in range(num_self_cbam_residuals):
        if i == 0 and not first_block:
            blk.append(mm.Self_CBAMResidual(input_channels, num_channels,
                                            use_1x1conv=True, strides=2))
        else:
            blk.append(mm.Self_CBAMResidual(num_channels, num_channels))
    blk.append(mm.Cross_CBAMResidual(num_channels, num_channels))
    return blk


def new_cbam_net():
    b1 = nn.Sequential(nn.Conv2d(3, 63, kernel_size=3, padding=1, groups=3, stride=2),
                       nn.BatchNorm2d(63), nn.ReLU())
    b2 = nn.Sequential(*attention_block(63, 63, 2, first_block=True))
    b3 = nn.Sequential(*attention_block(63, 126, 2))
    b4 = nn.Sequential(*attention_block(126, 252, 2))
    b5 = nn.Sequential(*attention_block(252, 504, 2))
    new_net = nn.Sequential(b1, b2, b3, b4, b5,
                            nn.AdaptiveAvgPool2d((1, 1)),
                            nn.Flatten(), nn.Linear(504, 2))
    return new_net


def train_siamese_net():
    b1 = nn.Sequential(nn.Conv2d(3, 63, kernel_size=3, padding=1, groups=3, stride=2),
                       nn.BatchNorm2d(63), nn.ReLU())
    b2 = nn.Sequential(*attention_block(63, 63, 2, first_block=True))
    b3 = nn.Sequential(*attention_block(63, 126, 2))
    b4 = nn.Sequential(*attention_block(126, 252, 2))
    b5 = nn.Sequential(*attention_block(252, 504, 2))
    net = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten())
    return net


#  定义object detect attention block
def object_detect_attention_block(input_channels, num_channels, num_self_cbam_residuals, stride):
    blk = []
    for i in range(num_self_cbam_residuals):
        if i == 0:
            blk.append(mm.Self_ObjectDetect_CBAMResidual(input_channels, num_channels,
                                                         use_1x1conv=True, strides=stride))
        else:
            blk.append(mm.Self_ObjectDetect_CBAMResidual(num_channels, num_channels))
    blk.append(mm.Cross_ObjectDetect_CBAMResidual(num_channels, num_channels))
    return blk


def object_detect_new_cbam_net(anchors):
    b1 = nn.Sequential(nn.Conv2d(3, 63, kernel_size=(3, 9), padding=(1, 4), groups=3, stride=(1, 1)),
                       nn.BatchNorm2d(63), nn.ReLU())  # out-->63*512
    b2 = nn.Sequential(*object_detect_attention_block(63, 63, 2, stride=(2, 2)))  # out-->32*256
    b3 = nn.Sequential(*object_detect_attention_block(63, 126, 2, stride=(1, 2)))  # out-->32*128
    b4 = nn.Sequential(*object_detect_attention_block(126, 126, 2, stride=(2, 2)))  # out-->16*64
    b5 = nn.Sequential(*object_detect_attention_block(126, 252, 2, stride=(1, 2)))  # out-->16*32
    b6 = nn.Sequential(*object_detect_attention_block(252, 252, 2, stride=(2, 2)))  # out-->8*16
    b7 = nn.Sequential(nn.Conv2d(252, 15, kernel_size=(1, 1)), nn.BatchNorm2d(15), nn.ReLU())  # out-->15*8*16
    b8 = my_module.Out_Layer(anchors=anchors, stride_x=32, stride_y=8)
    new_net = nn.Sequential(b1, b2, b3, b4, b5, b6, b7, b8)
    return new_net
