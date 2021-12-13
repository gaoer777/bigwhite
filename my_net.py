import my_module as mm
from torch import nn


# 定义残差层
def attention_block(input_channels, num_channels, num_self_cbam_residuals
                    , first_block=False):
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
    b1 = nn.Sequential(nn.Conv2d(3, 63, kernel_size=7, padding=3, groups=3, stride=2),
                       nn.BatchNorm2d(63), nn.ReLU())
    b2 = nn.Sequential(*attention_block(63, 63, 2, first_block=True))
    b3 = nn.Sequential(*attention_block(63, 126, 2))
    b4 = nn.Sequential(*attention_block(126, 252, 2))
    b5 = nn.Sequential(*attention_block(252, 504, 2))
    new_net = nn.Sequential(b1, b2, b3, b4, b5,
                            nn.AdaptiveAvgPool2d((1, 1)),
                            nn.Flatten(), nn.Linear(504, 2))
    return new_net
