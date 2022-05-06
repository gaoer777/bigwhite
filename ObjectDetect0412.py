import torch
from torch import nn
import my_module
from torch.nn import functional as F


class object_detect_attention_block(nn.Module):
    def __init__(self, input_channels, num_channels, kernel_size
                 , strides, padding, use_1x1conv=True):
        super(object_detect_attention_block, self).__init__()
        self.CrossRes1 = my_module.Cross_ObjectDetect_CBAMResidual(input_channels, num_channels, strides, kernel_size=kernel_size
                                                                   , padding=padding, use_1x1conv=use_1x1conv)
        self.CrossRes2 = my_module.Cross_ObjectDetect_CBAMResidual(num_channels, num_channels, 1, kernel_size=kernel_size
                                                                   , padding=padding)

    def forward(self, x):
        out = self.CrossRes1(x)
        out = self.CrossRes2(out)
        return out


class convSet(nn.Module):
    def __init__(self, in_channels, med_channels):
        super(convSet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, med_channels, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(med_channels, 5, kernel_size=(1, 1))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out


class ODAB(nn.Module):  # object_detect_attention_block
    def __init__(self):
        super(ODAB, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, stride=(2, 2), kernel_size=(3, 3), padding=(1, 1))  # out-->32*256
        self.attention_block1 = object_detect_attention_block(16, 32, strides=(2, 2), kernel_size=(3, 3), padding=1)  # out-->16*128
        self.attention_block2 = object_detect_attention_block(32, 64, strides=(1, 2), kernel_size=(3, 3), padding=1)  # out-->16*64
        self.attention_block3 = object_detect_attention_block(64, 128, strides=(2, 2), kernel_size=(3, 3), padding=1)  # out-->8*32
        self.attention_block4 = object_detect_attention_block(128, 256, strides=(1, 2), kernel_size=(3, 3), padding=1)  # out-->8*16
        self.attention_block5 = object_detect_attention_block(256, 512, strides=(2, 2), kernel_size=(3, 3), padding=1)  # out-->4*8
        self.upv1 = nn.ConvTranspose2d(512, 256, kernel_size=(2, 2), stride=(2, 2))  # out-->8*16
        self.upv2 = nn.ConvTranspose2d(256, 128, kernel_size=(1, 2), stride=(1, 2))  # out-->8*32
        self.upv3 = nn.ConvTranspose2d(256, 64, kernel_size=(2, 2), stride=(2, 2))  # out-->16*64
        self.upv4 = nn.ConvTranspose2d(64, 32, kernel_size=(1, 2), stride=(1, 2))  # out-->16*128
        self.out_conv1 = convSet(512, 32)
        self.out_conv2 = convSet(256, 32)
        self.out_conv3 = convSet(64, 32)
        self.out_layer1 = my_module.Out_Layer(anchors=[[60, 50]], stride_x=64, stride_y=16)
        self.out_layer2 = my_module.Out_Layer(anchors=[[30, 30]], stride_x=16, stride_y=8)
        self.out_layer3 = my_module.Out_Layer(anchors=[[15, 20]], stride_x=4, stride_y=4)

    def forward(self, x):
        cov1 = self.conv1(x)
        attention1 = self.attention_block1(cov1)
        attention2 = self.attention_block2(attention1)
        attention3 = self.attention_block3(attention2)
        attention4 = self.attention_block4(attention3)
        attention5 = self.attention_block5(attention4)
        upconv1 = self.upv1(attention5)
        upconv2 = self.upv2(upconv1)
        upconv3 = self.upv3(torch.cat((attention3, upconv2), dim=1))
        upconv4 = self.upv4(upconv3)
        outconv1 = self.out_conv1(attention5)
        outconv2 = self.out_conv2(torch.cat((attention3, upconv2), dim=1))
        outconv3 = self.out_conv3(torch.cat((attention1, upconv4), dim=1))
        out1 = self.out_layer1(outconv1)
        out2 = self.out_layer2(outconv2)
        out3 = self.out_layer3(outconv3)
        out = [out1, out2, out3]
        return out



