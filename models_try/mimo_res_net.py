from models_try.mimo_dip import MIMO_DIP, HandlerCat, handler_feature_size
import torch.nn as nn
import torch
import torch.nn.functional as F
from models.common import *


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=True, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class SCM(nn.Module):
    def __init__(self, input_channel, out_plane, num_res):
        super(SCM, self).__init__()
        self.main = nn.Sequential()
        for i in range(num_res):
            self.main.add(BasicConv(input_channel, out_plane, kernel_size=3, stride=1, relu=True))
            input_channel = out_plane

    def forward(self, x):
        return self.main(x)


class ScaleOutputBlock(nn.Module):
    def __init__(self, in_channel, out_channel, num_res, need_1x1=True):
        super(ScaleOutputBlock, self).__init__()
        self.layers = nn.Sequential()
        for i in range(num_res):
            self.layers.add(conv(in_channel, out_channel, 3, 1, bias=True, pad='zero'))
            self.layers.add(bn(out_channel))
            self.layers.add(act("LeakyReLU"))

            if need_1x1:
                self.layers.add(conv(out_channel, out_channel, 1, bias=True, pad='zero'))
                self.layers.add(bn(out_channel))
                self.layers.add(act("LeakyReLU"))
            in_channel = out_channel

    def forward(self, x):
        return self.layers(x)


class OutBaseConv(nn.Module):
    def __init__(self, in_channel, out_channel, num_res, need_sigmoid=True):
        super(OutBaseConv, self).__init__()
        self.layers = nn.Sequential()
        self.layers.add(conv(in_channel, out_channel, 3))
        if need_sigmoid:
            self.layers.add(nn.Sigmoid())

    def forward(self, x):
        return self.layers(x)


class Self_MSNet(MIMO_DIP):
    def __init__(self, input_channel, out_channel):
        super(Self_MSNet, self).__init__(input_channel, out_channel)
        base_channel = 128
        self.SCM = nn.ModuleList([
            SCM(input_channel, base_channel, 2),
            SCM(input_channel, base_channel, 5),
            SCM(input_channel, base_channel, 8),
            SCM(input_channel, base_channel, 11),
        ])
        self.OutScales = nn.ModuleList([
            ScaleOutputBlock(base_channel, base_channel, 3),
            ScaleOutputBlock(base_channel, base_channel, 2),
            ScaleOutputBlock(base_channel, base_channel, 1),
        ])

    def forward(self, x):
        x_1 = x[3]
        x_2 = x[2]
        x_4 = x[1]
        x_8 = x[0]

        skip_z2 = self.Skip1(x_2)
        z2 = self.SCM[0](x_2)

        z4 = self.SCM[1](x_4)
        skip_z4 = self.Skip1(x_4)

        z8 = self.SCM[2](x_8)
        skip_z8 = self.Skip1(x_8)

        res1 = self.Encoder[0](x_1)
        [res1, z2] = handler_feature_size([res1, z2])
        res1 = self.FAM(res1, z2)

        res2 = self.Encoder[1](res1)
        [res2, z4] = handler_feature_size([res2, z4])
        res2 = self.FAM(res2, z4)

        res3 = self.Encoder[2](res2)
        [res3, z8] = handler_feature_size([res3, z8])
        res3 = self.FAM(res3, z8)

        res4 = self.Encoder[3](res3)

        res5 = self.Encoder[4](res4)

        up4 = self.UpSample(res5)
        skip4 = self.Skip2(res4)
        up4 = HandlerCat([up4, skip4], 1)
        up4 = self.Decoder[0](up4)

        up3 = self.UpSample(up4)
        skip3 = self.Skip2(res3)
        up3 = HandlerCat([up3, skip3])
        up3 = self.Decoder[1](up3)

        up3_o = self.OutScales[0](up3)
        up3_o = HandlerCat([up3_o, skip_z8], 1)
        up3_o = self.Decoder[0](up3_o)

        out8 = self.OutConv(up3_o)
        h8, w8 = x_8.shape[2], x_8.shape[3]
        out_h8, out_w8 = out8.shape[2], out8.shape[3]
        out8 = F.pad(out8, (w8 - out_w8, 0, h8 - out_h8, 0))

        up2 = self.UpSample(up3)
        skip2 = self.Skip2(res2)
        up2 = HandlerCat([up2, skip2])
        up2 = self.Decoder[2](up2)

        up2_o = self.OutScales[0](up2)
        up2_o = HandlerCat([up2_o, skip_z4], 1)
        up2_o = self.Decoder[0](up2_o)

        out4 = self.OutConv(up2_o)
        h4, w4 = x_4.shape[2], x_4.shape[3]
        out_h4, out_w4 = out4.shape[2], out4.shape[3]
        out4 = F.pad(out4, (w4 - out_w4, 0, h4 - out_h4, 0))

        up1 = self.UpSample(up2)
        skip1 = self.Skip2(res1)
        up1 = HandlerCat([ up1, skip1])
        up1 = self.Decoder[3](up1)

        up1_o = self.OutScales[0](up1)
        up1_o = HandlerCat([up1_o, skip_z2], 1)
        up1_o = self.Decoder[0](up1_o)

        out2 = self.OutConv(up1_o)
        h2, w2 = x_2.shape[2], x_2.shape[3]
        out_h2, out_w2 = out2.shape[2], out2.shape[3]
        out2 = F.pad(out2, (w2 - out_w2, 0, h2 - out_h2, 0))

        up0 = self.UpSample(up1)
        skip0 = self.Skip1(x_1)
        up0 = HandlerCat([up0, skip0])
        up0 = self.Decoder[4](up0)
        out1 = self.OutConv(up0)
        h1, w1 = x_1.shape[2], x_1.shape[3]
        out_h1, out_w1 = out1.shape[2], out1.shape[3]
        out1 = F.pad(out1, (w1 - out_w1, 0, h1 - out_h1, 0))

        return [out8, out4, out2, out1]

class Self_MSNet_S(MIMO_DIP): #for ablation study
    def __init__(self, input_channel, out_channel):
        super(Self_MSNet_S, self).__init__(input_channel, out_channel)
        base_channel = 128

    def forward(self, x):
        x_1 = x[0]

        res1 = self.Encoder[0](x_1)
        res1 = self.FAM1(res1) #FAM includes two inputs, and FAM1 includes only one input

        res2 = self.Encoder[1](res1)
        res2 = self.FAM1(res2)

        res3 = self.Encoder[2](res2)
        res3 = self.FAM1(res3)

        res4 = self.Encoder[3](res3)

        res5 = self.Encoder[4](res4)

        up4 = self.UpSample(res5)
        skip4 = self.Skip2(res4)
        up4 = HandlerCat([up4, skip4], 1)
        up4 = self.Decoder[0](up4)

        up3 = self.UpSample(up4)
        skip3 = self.Skip2(res3)
        up3 = HandlerCat([up3, skip3])
        up3 = self.Decoder[1](up3)

        up2 = self.UpSample(up3)
        skip2 = self.Skip2(res2)
        up2 = HandlerCat([up2, skip2])
        up2 = self.Decoder[2](up2)

        up1 = self.UpSample(up2)
        skip1 = self.Skip2(res1)
        up1 = HandlerCat([up1, skip1])
        up1 = self.Decoder[3](up1)

        up0 = self.UpSample(up1)
        skip0 = self.Skip1(x_1)
        up0 = HandlerCat([up0, skip0])
        up0 = self.Decoder[4](up0)
        out1 = self.OutConv(up0)
        h1, w1 = x_1.shape[2], x_1.shape[3]
        out_h1, out_w1 = out1.shape[2], out1.shape[3]
        out1 = F.pad(out1, (w1 - out_w1, 0, h1 - out_h1, 0))

        return out1