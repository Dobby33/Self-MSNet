import torch.nn.functional as F
from .mimo_layes import *
from .mimo_unet import *


class MidConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MidConvBlock, self).__init__()
        layers = [
            BasicConv(in_channel, in_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(in_channel, out_channel, kernel_size=3, relu=True, stride=1),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MIMOUNetSimple1(nn.Module):
    def __init__(self, num_res=8):
        super(MIMOUNetSimple1, self).__init__()

        base_channel = 32

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel * 2, num_res),
            EBlock(base_channel * 4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel * 1),
            AFF(base_channel * 7, base_channel * 2)
        ])
        self.MidConv = nn.ModuleList(
            [
                MidConvBlock(base_channel * 4, base_channel * 2),
                MidConvBlock(base_channel * 2, base_channel * 1)
            ]
        )

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_1 = x[2]
        x_2 = x[1]
        x_4 = x[0]
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)
        outputs = list()

        x_ = self.feat_extract[0](x_1)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = F.interpolate(z, [z2.shape[2], z2.shape[3]])
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = F.interpolate(z, [z4.shape[2], z4.shape[3]])
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)
        mid_z3 = self.MidConv[0](z)
        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        z_ = self.Sigmoid(z_)
        z_ = F.interpolate(z_, [x_4.shape[2], x_4.shape[3]])
        outputs.append(z_)

        mid_z3 = F.interpolate(mid_z3, [res2.shape[2], res2.shape[3]])
        res2 = res2 + mid_z3
        mid_z2 = self.MidConv[1](res2)
        z = F.interpolate(z, [res2.shape[2], res2.shape[3]])
        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        z_ = self.Sigmoid(z_)
        outputs.append(z_)

        mid_z2 = F.interpolate(mid_z2, [res1.shape[2], res1.shape[3]])
        res1 = res1 + mid_z2
        z = F.interpolate(z, [res1.shape[2], res1.shape[3]])
        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        z = self.Sigmoid(z)
        outputs.append(z)

        return outputs


class MIMOUNetSimple2(nn.Module):

    def __init__(self, num_res=8):
        super(MIMOUNetSimple2, self).__init__()

        base_channel = 32

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel * 2, num_res),
            EBlock(base_channel * 4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel * 1),
            AFF(base_channel * 7, base_channel * 2)
        ])

        self.MidConv = nn.ModuleList(
            [
                MidConvBlock(base_channel * 4, base_channel * 2),
                MidConvBlock(base_channel * 2, base_channel * 1)
            ]
        )

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)
        self.FAM3 = FAM(base_channel)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_1 = x[2]
        x_2 = x[1]
        x_4 = x[0]
        outputs = list()

        x_ = self.feat_extract[0](x_1)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.Encoder[2](z)
        mid_z3 = self.MidConv[0](z)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        z_ = self.Sigmoid(z_)
        z_ = F.interpolate(z_, [x_4.shape[2], x_4.shape[3]])
        outputs.append(z_)

        mid_z3 = F.interpolate(mid_z3, [res2.shape[2], res2.shape[3]])
        res2 = res2 + mid_z3
        mid_z2 = self.MidConv[1](res2)
        z = F.interpolate(z, [res2.shape[2], res2.shape[3]])
        z = torch.cat([z, res2], dim=1)

        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        z_ = self.Sigmoid(z_)
        z_ = F.interpolate(z_, [x_2.shape[2], x_2.shape[3]])
        outputs.append(z_)

        res1 = res1 + mid_z2
        z = F.interpolate(z, [res1.shape[2], res1.shape[3]])
        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        z = self.Sigmoid(z)
        outputs.append(z)

        return outputs

class MIMOUNetSimple3(nn.Module):
    def __init__(self, num_res=8):
        super(MIMOUNetSimple3, self).__init__()

        base_channel = 32

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel * 2, num_res),
            EBlock(base_channel * 4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel * 1),
            AFF(base_channel * 7, base_channel * 2)
        ])
        self.MidConv = nn.ModuleList(
            [
                MidConvBlock(base_channel * 4, base_channel * 2),
                MidConvBlock(base_channel * 2, base_channel * 1)
            ]
        )

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_1 = x[2]
        x_2 = x[1]
        x_4 = x[0]
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)
        outputs = list()

        x_ = self.feat_extract[0](x_1)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = F.interpolate(z, [z2.shape[2], z2.shape[3]])
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = F.interpolate(z, [z4.shape[2], z4.shape[3]])
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)
        mid_z3 = self.MidConv[0](z)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        z_ = self.Sigmoid(z_)
        z_ = F.interpolate(z_, [x_4.shape[2], x_4.shape[3]])
        outputs.append(z_)

        mid_z3 = F.interpolate(mid_z3, [res2.shape[2], res2.shape[3]])
        res2 = res2 + mid_z3
        mid_z2 = self.MidConv[1](res2)
        z = F.interpolate(z, [res2.shape[2], res2.shape[3]])
        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        z_ = self.Sigmoid(z_)
        outputs.append(z_)

        mid_z2 = F.interpolate(mid_z2, [res1.shape[2], res1.shape[3]])
        res1 = res1 + mid_z2
        z = F.interpolate(z, [res1.shape[2], res1.shape[3]])
        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        z = self.Sigmoid(z)
        outputs.append(z)

        return outputs


class MIMOUNet_Up_mutiply(nn.Module):
    def __init__(self, num_res=8):
        super(MIMOUNet_Up_mutiply, self).__init__()

        base_channel = 32

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel * 2, num_res),
            EBlock(base_channel * 4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 2, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel * 1),
            AFF(base_channel * 7, base_channel * 2)
        ])
        self.MidConv = nn.ModuleList(
            [
                MidConvBlock(base_channel * 4, base_channel * 2),
                MidConvBlock(base_channel * 2, base_channel * 1)
            ]
        )

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)
        self.FAM3 = FAM(base_channel * 1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_1 = x[2]
        x_2 = x[1]
        x_4 = x[0]
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)
        outputs = list()

        x_ = self.feat_extract[0](x_1)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = F.interpolate(z, [z2.shape[2], z2.shape[3]])
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = F.interpolate(z, [z4.shape[2], z4.shape[3]])
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)
        mid_z3 = self.MidConv[0](z)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        z_ = self.Sigmoid(z_)
        z_ = F.interpolate(z_, [x_4.shape[2], x_4.shape[3]])
        outputs.append(z_)

        mid_z3 = F.interpolate(mid_z3, [res2.shape[2], res2.shape[3]])
        mid_z2 = self.MidConv[1](res2)
        z = F.interpolate(z, [res2.shape[2], res2.shape[3]])
        z = self.FAM2(z, res2)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        z_ = self.Sigmoid(z_)
        outputs.append(z_)

        mid_z2 = F.interpolate(mid_z2, [res1.shape[2], res1.shape[3]])

        z = F.interpolate(z, [res1.shape[2], res1.shape[3]])
        z = self.FAM3(z, res1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        z = self.Sigmoid(z)
        outputs.append(z)

        return outputs