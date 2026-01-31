from models.common import *

from models.non_local_dot_product import NONLocalBlock2D
import torch.nn.functional as F


class EBlock(nn.Module):
    def __init__(self, in_channel, out_channel, num_res=2, need_nonlocal=False):
        super(EBlock, self).__init__()
        self.layers = nn.Sequential()

        self.layers.add(conv(in_channel, out_channel, 3, 2, bias=True, pad='zero',
                                 downsample_mode="stride"))
        self.layers.add(bn(out_channel))
        self.layers.add(act("LeakyReLU"))
        for i in range(num_res):
            if need_nonlocal and i == 0:
                self.layers.add(NONLocalBlock2D(in_channels=out_channel))
            self.layers.add(conv(out_channel, out_channel, 3, bias=True, pad='zero'))
            self.layers.add(bn(out_channel))
            self.layers.add(act("LeakyReLU"))

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, in_channel, out_channel, num_res=2, need_1x1=True):
        super(DBlock, self).__init__()
        self.layers = nn.Sequential()
        for i in range(num_res):
            self.layers.add(conv(in_channel, out_channel, 3, 1, bias=True, pad='zero'))
            self.layers.add(bn(out_channel))
            self.layers.add(act("LeakyReLU"))

            if need_1x1 and i == 0:
                self.layers.add(conv(out_channel, out_channel, 1, bias=True, pad='zero'))
                self.layers.add(bn(out_channel))
                self.layers.add(act("LeakyReLU"))
            in_channel = out_channel

    def forward(self, x):
        return self.layers(x)


class Skip(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Skip, self).__init__()
        self.layers = nn.Sequential()
        self.layers.add(conv(in_channel, out_channel, 3, bias=True, pad='zero'))
        self.layers.add(bn(out_channel))
        self.layers.add(act("LeakyReLU"))

    def forward(self, x):
        return self.layers(x)


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


class OutBaseConv(nn.Module):
    def __init__(self, in_channel, out_channel, need_sigmoid=True):
        super(OutBaseConv, self).__init__()
        self.layers = nn.Sequential()
        self.layers.add(conv(in_channel, out_channel, 3))
        if need_sigmoid:
            self.layers.add(nn.Sigmoid())
            # self.layers.add(nn.Tanh()) #if Gaussian distribution is used as input, then Tanh should be used as the activation function, followed by +1 and  /2

    def forward(self, x):
        return self.layers(x)


class SCM(nn.Module):
    def __init__(self, input_channel, out_plane, num_res):
        super(SCM, self).__init__()
        self.main = nn.Sequential()
        input_channel_1 = input_channel
        for i in range(num_res):
            self.main.add(BasicConv(input_channel, out_plane // 2, kernel_size=3, stride=1, relu=True))
            self.main.add(BasicConv(out_plane // 2, out_plane // 2, kernel_size=1, stride=1, relu=True))
            input_channel = out_plane // 2
        self.main.add(BasicConv(input_channel, out_plane - input_channel_1, kernel_size=1, stride=1, relu=True))
        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out

class FAM1(nn.Module):
    def __init__(self, channel):
        super(FAM1, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x):
        out = x + self.merge(x)
        return out


class MIMO_DIP(nn.Module):
    def __init__(self, input_channel, out_channel):
        super(MIMO_DIP, self).__init__()
        base_channel = 128
        skip_channel = 4
        self.Encoder = nn.ModuleList([
            EBlock(input_channel, base_channel),
            EBlock(base_channel, base_channel),
            EBlock(base_channel, base_channel, need_nonlocal=True),
            EBlock(base_channel, base_channel, need_nonlocal=True),
            EBlock(base_channel, base_channel, need_nonlocal=True),
        ])

        self.SCM = nn.ModuleList([
            SCM(input_channel, base_channel, 1),
            SCM(input_channel, base_channel, 2),
            SCM(input_channel, base_channel, 3),
            SCM(input_channel, base_channel, 4),
        ])
        self.FAM = FAM(base_channel)
        self.FAM1 = FAM1(base_channel)
        self.Skip1 = Skip(input_channel, skip_channel)
        self.Skip2 = Skip(base_channel, skip_channel)
        self.UpSample = nn.Upsample(scale_factor=2, mode='nearest')
        self.Decoder = nn.ModuleList([
            DBlock(base_channel + skip_channel, base_channel),
            DBlock(base_channel + skip_channel, base_channel),
            DBlock(base_channel + skip_channel, base_channel),
            DBlock(base_channel + skip_channel, base_channel),
            DBlock(base_channel + skip_channel, base_channel),
        ])
        self.OutConv = OutBaseConv(base_channel, out_channel, need_sigmoid=True)

    def forward(self, x):
        x_1 = x[4]
        x_2 = x[3]
        x_4 = x[2]
        x_8 = x[1]
        x_16 = x[0]

        z2 = self.SCM[0](x_2)
        z4 = self.SCM[1](x_4)
        z8 = self.SCM[2](x_8)
        z16 = self.SCM[3](x_16)

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
        [res4, z16] = handler_feature_size([res4, z16])
        res4 = self.FAM(res4, z16)

        res5 = self.Encoder[4](res4)

        up4 = self.UpSample(res5)
        skip4 = self.Skip2(res4)
        up4 = HandlerCat([up4, skip4], 1)
        up4 = self.Decoder[0](up4)
        out16 = self.OutConv(up4)
        h16, w16 = x_16.shape[2], x_16.shape[3]
        out_h16, out_w16 = out16.shape[2], out16.shape[3]
        out16 = F.pad(out16, (w16 - out_w16, 0, h16 - out_h16, 0))

        up3 = self.UpSample(up4)
        skip3 = self.Skip2(res3)
        up3 = HandlerCat([up3, skip3])
        up3 = self.Decoder[1](up3)
        out8 = self.OutConv(up3)
        h8, w8 = x_8.shape[2], x_8.shape[3]
        out_h8, out_w8 = out8.shape[2], out8.shape[3]
        out8 = F.pad(out8, (w8 - out_w8, 0, h8 - out_h8, 0))

        up2 = self.UpSample(up3)
        skip2 = self.Skip2(res2)
        up2 = HandlerCat([up2, skip2])
        up2 = self.Decoder[2](up2)
        out4 = self.OutConv(up2)
        h4, w4 = x_4.shape[2], x_4.shape[3]
        out_h4, out_w4 = out4.shape[2], out4.shape[3]
        out4 = F.pad(out4, (w4 - out_w4, 0, h4 - out_h4, 0))

        up1 = self.UpSample(up2)
        skip1 = self.Skip2(res1)
        up1 = HandlerCat([up1, skip1])
        up1 = self.Decoder[3](up1)
        out2 = self.OutConv(up1)
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

        return [out16, out8, out4, out2, out1]


def HandlerCat(inputs, dim=1):
    inputs_shapes2 = [x.shape[2] for x in inputs]
    inputs_shapes3 = [x.shape[3] for x in inputs]
    if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(
            np.array(inputs_shapes3) == min(inputs_shapes3)):
        inputs_ = inputs
    else:
        target_shape2 = min(inputs_shapes2)
        target_shape3 = min(inputs_shapes3)

        inputs_ = []
        for inp in inputs:
            diff2 = (inp.size(2) - target_shape2) // 2
            diff3 = (inp.size(3) - target_shape3) // 2
            inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])

    return torch.cat(inputs_, dim=dim)


def handler_feature_size(inputs):
    inputs_shapes2 = [x.shape[2] for x in inputs]
    inputs_shapes3 = [x.shape[3] for x in inputs]
    if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(
            np.array(inputs_shapes3) == min(inputs_shapes3)):
        inputs_ = inputs
    else:
        target_shape2 = min(inputs_shapes2)
        target_shape3 = min(inputs_shapes3)

        inputs_ = []
        for inp in inputs:
            diff2 = (inp.size(2) - target_shape2) // 2
            diff3 = (inp.size(3) - target_shape3) // 2
            inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])
    return inputs_
