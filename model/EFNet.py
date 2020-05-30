import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model.resnet import ResNet50
from model.resnet import ResNet18


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes)
        )

    def forward(self, x):
        x = self.conv_bn(x)
        return x


class Reduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Reduction, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)


class conv_upsample(nn.Module):
    def __init__(self, channel):
        super(conv_upsample, self).__init__()
        self.conv = BasicConv2d(channel, channel, 1)

    def forward(self, x, target):
        if x.size()[2:] != target.size()[2:]:
            x = self.conv(F.upsample(x, size=target.size()[2:], mode='bilinear', align_corners=True))
        return x


class DenseFusion(nn.Module):
    # Cross Refinement Unit
    def __init__(self, channel):
        super(DenseFusion, self).__init__()
        self.conv1 = conv_upsample(channel)
        self.conv2 = conv_upsample(channel)
        self.conv3 = conv_upsample(channel)
        self.conv4 = conv_upsample(channel)
        self.conv5 = conv_upsample(channel)
        self.conv6 = conv_upsample(channel)
        self.conv7 = conv_upsample(channel)
        self.conv8 = conv_upsample(channel)
        self.conv9 = conv_upsample(channel)
        self.conv10 = conv_upsample(channel)
        self.conv11 = conv_upsample(channel)
        self.conv12 = conv_upsample(channel)

        self.conv_f1 = nn.Sequential(
            BasicConv2d(5 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f2 = nn.Sequential(
            BasicConv2d(4 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f3 = nn.Sequential(
            BasicConv2d(3 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f4 = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )

        self.conv_f5 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f6 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f7 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f8 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )

    def forward(self, x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4):
        x_sf1 = x_s1 + self.conv_f1(torch.cat((x_s1, x_e1,
                                               self.conv1(x_e2, x_s1),
                                               self.conv2(x_e3, x_s1),
                                               self.conv3(x_e4, x_s1)), 1))
        x_sf2 = x_s2 + self.conv_f2(torch.cat((x_s2, x_e2,
                                               self.conv4(x_e3, x_s2),
                                               self.conv5(x_e4, x_s2)), 1))
        x_sf3 = x_s3 + self.conv_f3(torch.cat((x_s3, x_e3,
                                               self.conv6(x_e4, x_s3)), 1))
        x_sf4 = x_s4 + self.conv_f4(torch.cat((x_s4, x_e4), 1))

        x_ef1 = x_e1 + self.conv_f5(x_e1 * x_s1 *
                                    self.conv7(x_s2, x_e1) *
                                    self.conv8(x_s3, x_e1) *
                                    self.conv9(x_s4, x_e1))
        x_ef2 = x_e2 + self.conv_f6(x_e2 * x_s2 *
                                    self.conv10(x_s3, x_e2) *
                                    self.conv11(x_s4, x_e2))
        x_ef3 = x_e3 + self.conv_f7(x_e3 * x_s3 *
                                    self.conv12(x_s4, x_e3))
        x_ef4 = x_e4 + self.conv_f8(x_e4 * x_s4)

        return x_sf1, x_sf2, x_sf3, x_sf4, x_ef1, x_ef2, x_ef3, x_ef4


class ConcatOutput(nn.Module):
    def __init__(self, channel):
        super(ConcatOutput, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_cat1 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.conv_cat2 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.conv_cat3 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.output = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )

    def forward(self, x1, x2, x3, x4):
        x3 = torch.cat((x3, self.conv_upsample1(self.upsample(x4))), 1)
        x3 = self.conv_cat1(x3)

        x2 = torch.cat((x2, self.conv_upsample2(self.upsample(x3))), 1)
        x2 = self.conv_cat2(x2)

        x1 = torch.cat((x1, self.conv_upsample3(self.upsample(x2))), 1)
        x1 = self.conv_cat3(x1)

        x = self.output(x1)
        return x


class SCRN(nn.Module):
    # Stacked Cross Refinement Network
    def __init__(self, channel=32):
        super(SCRN, self).__init__()
        self.resnet = ResNet50()
        self.reduce_s1 = Reduction(256, channel)
        self.reduce_s2 = Reduction(512, channel)
        self.reduce_s3 = Reduction(1024, channel)
        self.reduce_s4 = Reduction(2048, channel)

        self.reduce_e1 = Reduction(256, channel)
        self.reduce_e2 = Reduction(512, channel)
        self.reduce_e3 = Reduction(1024, channel)
        self.reduce_e4 = Reduction(2048, channel)

        self.df1 = DenseFusion(channel)
        self.df2 = DenseFusion(channel)
        self.df3 = DenseFusion(channel)
        self.df4 = DenseFusion(channel)

        self.output_s = ConcatOutput(channel)
        self.output_e = ConcatOutput(channel)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.initialize_weights()

    def forward(self, x):
        size = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        # feature abstraction
        x_s1 = self.reduce_s1(x1)
        x_s2 = self.reduce_s2(x2)
        x_s3 = self.reduce_s3(x3)
        x_s4 = self.reduce_s4(x4)

        x_e1 = self.reduce_e1(x1)
        x_e2 = self.reduce_e2(x2)
        x_e3 = self.reduce_e3(x3)
        x_e4 = self.reduce_e4(x4)

        # four cross refinement units
        x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4 = self.df1(x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4)
        x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4 = self.df2(x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4)
        x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4 = self.df3(x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4)
        x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4 = self.df4(x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4)

        # feature aggregation using u-net
        pred_s = self.output_s(x_s1, x_s2, x_s3, x_s4)
        pred_e = self.output_e(x_e1, x_e2, x_e3, x_e4)

        pred_s = F.upsample(pred_s, size=size, mode='bilinear', align_corners=True)
        pred_e = F.upsample(pred_e, size=size, mode='bilinear', align_corners=True)

        return pred_s, pred_e, x_s1, x_s2, x_s3, x_s4

    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        self.resnet.load_state_dict(res50.state_dict(), False)


class CFC(nn.Module):
    def __init__(self, channel):
        super(CFC, self).__init__()
        self.conv1 = BasicConv2d(channel, channel, 1, padding=0)
        self.conv2 = BasicConv2d(channel, channel, 1, padding=0)
        self.conv3 = BasicConv2d(2 * channel, channel, 1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, s, d):
        s_conv = self.conv1(s)
        s_conv = self.relu(s_conv)
        d_conv = self.conv2(d)
        d_conv = self.relu(d_conv)
        sd1 = torch.cat((s_conv, d_conv), 1)
        sd2 = self.conv3(sd1)
        sd2 = self.relu(sd2)
        sd_last = torch.cat((sd2, sd1), 1)
        return sd_last


class EFNet(nn.Module):
    def __init__(self, channel=32, alpha=0.5):
        super(EFNet, self).__init__()

        self.rgb_extractor = SCRN(channel=channel)
        self.resnet_depth = ResNet18()
        self.alpha = alpha

        self.sd1 = CFC(channel)
        self.sd2 = CFC(channel)
        self.sd3 = CFC(channel)
        self.sd4 = CFC(channel)

        self.output_ds = ConcatOutput(3 * channel)

        self.conv_upsample1 = BasicConv2d(2 * channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(4 * channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(8 * channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(16 * channel, channel, 3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, d):
        size = x.size()[2:]
        pred_s, pred_e, x_s1, x_s2, x_s3, x_s4 = self.rgb_extractor(x)

        pred_mask_sig = torch.sigmoid(pred_s)
        dd = d * self.alpha + (1 - self.alpha) * pred_mask_sig * d

        d_cor = self.resnet_depth.conv1(dd)
        d_cor = self.resnet_depth.bn1(d_cor)
        d_cor = self.resnet_depth.relu(d_cor)
        d_cor = self.resnet_depth.maxpool(d_cor)

        d1 = self.resnet_depth.layer1(d_cor)
        d2 = self.resnet_depth.layer2(d1)
        d3 = self.resnet_depth.layer3(d2)
        d4 = self.resnet_depth.layer4(d3)

        dx_1 = self.sd1(x_s1, self.conv_upsample1(d1))
        dx_2 = self.sd2(x_s2, self.conv_upsample2(d2))
        dx_3 = self.sd3(x_s3, self.conv_upsample3(d3))
        dx_4 = self.sd4(x_s4, self.conv_upsample4(d4))

        pred_ds = self.output_ds(dx_1, dx_2, dx_3, dx_4)
        pred_ds = F.upsample(pred_ds, size=size, mode='bilinear', align_corners=True)

        return pred_s, pred_e, pred_ds
