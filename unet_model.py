
import torch
from torch import nn
import numpy as np

class Unet2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        ch = np.array([32, 64, 128, 256, 512])
        self.ch = ch




        self.conv1 = self.double_conv(in_channels, ch[0], 7, 3)
        self.conv2 = self.double_conv(ch[0], ch[1], 3, 1)
        self.conv3 = self.double_conv(ch[1], ch[2], 3, 1)
        self.conv4 = self.double_conv(ch[2], ch[3], 3, 1)
        self.conv5 = self.double_conv(ch[3], ch[4], 3, 1)

        self.upconv4 = self.double_conv(ch[3]*2, ch[3], 3, 1)
        self.upconv3 = self.double_conv(ch[2]*2, ch[2], 3, 1)
        self.upconv2 = self.double_conv(ch[1]*2, ch[1], 3, 1)
        self.upconv1 = self.double_conv(ch[0]*2, ch[0], 3, 1)

        self.last_conv = self.single_conv(ch[0], out_channels, 3, 1)

        self.conv_transp54 =torch.nn.ConvTranspose2d(self.ch[4], self.ch[3], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_transp43 =torch.nn.ConvTranspose2d(self.ch[3], self.ch[2], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_transp32 =torch.nn.ConvTranspose2d(self.ch[2], self.ch[1], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_transp21 =torch.nn.ConvTranspose2d(self.ch[1], self.ch[0], kernel_size=3, stride=2, padding=1, output_padding=1)




    def __call__(self, x):
        # downsampling part
        out1 = self.conv1(x)
        in2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(out1)
        out2 = self.conv2(in2)
        in3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(out2)
        out3 = self.conv3(in3)
        in4 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(out3)
        out4 = self.conv4(in4)
        in5 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(out4)
        out5 = self.conv5(in5)

        up4_in = self.conv_transp54(out5)


        cat4 = torch.cat([up4_in,out4], 1)

        up4_out = self.upconv4(cat4)


        up3_in = self.conv_transp43(up4_out)
        cat3 = torch.cat([up3_in,out3], 1)
        up3_out = self.upconv3(cat3)

        up2_in = self.conv_transp32(up3_out)
        cat2 = torch.cat([up2_in, out2], 1)
        up2_out = self.upconv2(cat2)

        up1_in = self.conv_transp21(up2_out)
        cat1 = torch.cat([up1_in, out1], 1)
        out = self.upconv1(cat1)

        out = self.last_conv(out)

        return out




    def double_conv(self, in_channels, out_channels, kernel_size, padding):

        seq = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),

            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
                                 )

        return seq

    def single_conv(self, in_channels, out_channels, kernel_size, padding):

        seq = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU()
                                 )

        return seq

