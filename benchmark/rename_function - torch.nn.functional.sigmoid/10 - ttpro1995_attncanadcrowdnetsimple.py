import torch.nn as nn
import torch
from torchvision import models
import collections
import torch.nn.functional as F
import os
from .deform_conv_v2 import DeformConv2d
from .squeeze_and_excitation import ChannelSpatialSELayer, SpatialSELayer
# from dcn.modules.deform_conv import DeformConvPack, ModulatedDeformConvPack


class AttnCanAdcrowdNetSimpleV1(nn.Module):
    def __init__(self, load_weights=False):
        super(AttnCanAdcrowdNetSimpleV1, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend = make_layers(self.frontend_feat)

        # freeze vgg layer
        for param in self.frontend.parameters():
            param.requires_grad = False

        self.csSE = ChannelSpatialSELayer(num_channels=512, reduction_ratio=1)

        self.concat_filter_layer = nn.Conv2d(1024, 512, kernel_size=3, padding=2, dilation=2)

        # we skip one formation of deformconv
        # self.deform_conv_1_3 = DeformConv2d(512, 256, kernel_size=3, stride=1, padding=1)
        # self.deform_conv_1_5 = DeformConv2d(512, 256, kernel_size=5, stride=1, padding=2)
        # self.deform_conv_1_7 = DeformConv2d(512, 256, kernel_size=7, stride=1, padding=3)
        self.concat_filter_layer_1 = nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2)

        self.deform_conv_2_3 = DeformConv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.deform_conv_2_5 = DeformConv2d(256, 128, kernel_size=5, stride=1, padding=2)
        # self.deform_conv_2_7 = DeformConv2d(256, 128, kernel_size=7, stride=1, padding=3)
        self.concat_filter_layer_2 = nn.Conv2d(128 * 2, 128, kernel_size=3, padding=2, dilation=2)

        self.deform_conv_3_3 = DeformConv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.deform_conv_3_5 = DeformConv2d(128, 64, kernel_size=5, stride=1, padding=2)
        # self.deform_conv_3_7 = DeformConv2d(128, 64, kernel_size=7, stride=1, padding=3)
        self.concat_filter_layer_3 = nn.Conv2d(64 * 2, 64, kernel_size=3, padding=2, dilation=2)

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.conv1_1 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv1_2 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv2_1 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv2_2 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv3_1 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv3_2 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv6_1 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv6_2 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            fsd = collections.OrderedDict()
            for i in range(len(self.frontend.state_dict().items())):
                temp_key = list(self.frontend.state_dict().items())[i][0]
                fsd[temp_key] = list(mod.state_dict().items())[i][1]
            self.frontend.load_state_dict(fsd)

    def forward(self, x):
        fv = self.frontend(x)

        # concurrent spatial and channel squeeze & excitation
        fv = self.csSE(fv)

        # S=1
        ave1 = nn.functional.adaptive_avg_pool2d(fv, (1, 1))
        ave1 = self.conv1_1(ave1)
        s1 = nn.functional.upsample(ave1, size=(fv.shape[2], fv.shape[3]), mode='bilinear')
        c1 = s1 - fv
        w1 = self.conv1_2(c1)
        w1 = nn.functional.sigmoid(w1)
        # S=2
        ave2 = nn.functional.adaptive_avg_pool2d(fv, (2, 2))
        ave2 = self.conv2_1(ave2)
        s2 = nn.functional.upsample(ave2, size=(fv.shape[2], fv.shape[3]), mode='bilinear')
        c2 = s2 - fv
        w2 = self.conv2_2(c2)
        w2 = nn.functional.sigmoid(w2)
        # S=3
        ave3 = nn.functional.adaptive_avg_pool2d(fv, (3, 3))
        ave3 = self.conv3_1(ave3)
        s3 = nn.functional.upsample(ave3, size=(fv.shape[2], fv.shape[3]), mode='bilinear')
        c3 = s3 - fv
        w3 = self.conv3_2(c3)
        w3 = nn.functional.sigmoid(w3)
        # S=6
        ave6 = nn.functional.adaptive_avg_pool2d(fv, (6, 6))
        ave6 = self.conv6_1(ave6)
        s6 = nn.functional.upsample(ave6, size=(fv.shape[2], fv.shape[3]), mode='bilinear')
        c6 = s6 - fv
        w6 = self.conv6_2(c6)
        w6 = nn.functional.sigmoid(w6)

        fi = (w1 * s1 + w2 * s2 + w3 * s3 + w6 * s6) / (w1 + w2 + w3 + w6 + 0.000000000001)
        x = torch.cat((fv, fi), 1)
        x = F.relu(self.concat_filter_layer(x), inplace=True)

        # x3 = self.deform_conv_1_3(x)
        # x5 = self.deform_conv_1_5(x)
        # x7 = self.deform_conv_1_7(x)
        # x = torch.cat((x3, x5, x7), 1)
        # x = torch.cat((x3, x5), 1)
        x = F.relu(self.concat_filter_layer_1(x), inplace=True)

        x3 = self.deform_conv_2_3(x)
        x5 = self.deform_conv_2_5(x)
        # x7 = self.deform_conv_2_7(x)
        # x = torch.cat((x3, x5, x7), 1)
        x = F.relu(torch.cat((x3, x5), 1), inplace=True)
        x = F.relu(self.concat_filter_layer_2(x), inplace=True)

        x3 = self.deform_conv_3_3(x)
        x5 = self.deform_conv_3_5(x)
        # x7 = self.deform_conv_3_7(x)
        # x = torch.cat((x3, x5, x7), 1)
        x = F.relu(torch.cat((x3, x5), 1), inplace=True)
        x = F.relu(self.concat_filter_layer_3(x), inplace=True)

        x = self.output_layer(x)

        # this cause too much dimension mismatch problem
        # so we desampling label instead
        # x = nn.functional.upsample(x, scale_factor=8, mode='bilinear') / 64.0
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class AttnCanAdcrowdNetSimpleV2(nn.Module):
    """
    why fail: we don't have enought GPU mem (only have 11G) for full size shanghaitech A
    """
    def __init__(self, load_weights=False):
        super(AttnCanAdcrowdNetSimpleV2, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend = make_layers(self.frontend_feat)

        # freeze vgg layer
        for param in self.frontend.parameters():
            param.requires_grad = False

        self.sSE = SpatialSELayer(num_channels=512)

        self.concat_filter_layer = nn.Conv2d(1024, 512, kernel_size=3, padding=2, dilation=2)

        # we skip one formation of deformconv
        # self.deform_conv_1_3 = DeformConv2d(512, 256, kernel_size=3, stride=1, padding=1)
        # self.deform_conv_1_5 = DeformConv2d(512, 256, kernel_size=5, stride=1, padding=2)
        # self.deform_conv_1_7 = DeformConv2d(512, 256, kernel_size=7, stride=1, padding=3)
        self.concat_filter_layer_1 = nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2)

        self.deform_conv_2_3 = DeformConv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.deform_conv_2_5 = DeformConv2d(256, 128, kernel_size=5, stride=1, padding=2)
        # self.deform_conv_2_7 = DeformConv2d(256, 128, kernel_size=7, stride=1, padding=3)
        self.concat_filter_layer_2 = nn.Conv2d(128 * 2, 128, kernel_size=3, padding=2, dilation=2)

        self.deform_conv_3_3 = DeformConv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.deform_conv_3_5 = DeformConv2d(128, 64, kernel_size=5, stride=1, padding=2)
        # self.deform_conv_3_7 = DeformConv2d(128, 64, kernel_size=7, stride=1, padding=3)
        self.concat_filter_layer_3 = nn.Conv2d(64 * 2, 64, kernel_size=3, padding=2, dilation=2)

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.conv1_1 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv1_2 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv2_1 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv2_2 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv3_1 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv3_2 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv6_1 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv6_2 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            fsd = collections.OrderedDict()
            for i in range(len(self.frontend.state_dict().items())):
                temp_key = list(self.frontend.state_dict().items())[i][0]
                fsd[temp_key] = list(mod.state_dict().items())[i][1]
            self.frontend.load_state_dict(fsd)

    def forward(self, x):
        fv = self.frontend(x)

        #  spatial squeeze & excitation
        fv = self.sSE(fv)

        # S=1
        ave1 = nn.functional.adaptive_avg_pool2d(fv, (1, 1))
        ave1 = self.conv1_1(ave1)
        s1 = nn.functional.upsample(ave1, size=(fv.shape[2], fv.shape[3]), mode='bilinear')
        c1 = s1 - fv
        w1 = self.conv1_2(c1)
        w1 = nn.functional.sigmoid(w1)
        # S=2
        ave2 = nn.functional.adaptive_avg_pool2d(fv, (2, 2))
        ave2 = self.conv2_1(ave2)
        s2 = nn.functional.upsample(ave2, size=(fv.shape[2], fv.shape[3]), mode='bilinear')
        c2 = s2 - fv
        w2 = self.conv2_2(c2)
        w2 = nn.functional.sigmoid(w2)
        # S=3
        ave3 = nn.functional.adaptive_avg_pool2d(fv, (3, 3))
        ave3 = self.conv3_1(ave3)
        s3 = nn.functional.upsample(ave3, size=(fv.shape[2], fv.shape[3]), mode='bilinear')
        c3 = s3 - fv
        w3 = self.conv3_2(c3)
        w3 = nn.functional.sigmoid(w3)
        # S=6
        ave6 = nn.functional.adaptive_avg_pool2d(fv, (6, 6))
        ave6 = self.conv6_1(ave6)
        s6 = nn.functional.upsample(ave6, size=(fv.shape[2], fv.shape[3]), mode='bilinear')
        c6 = s6 - fv
        w6 = self.conv6_2(c6)
        w6 = nn.functional.sigmoid(w6)

        fi = (w1 * s1 + w2 * s2 + w3 * s3 + w6 * s6) / (w1 + w2 + w3 + w6 + 0.000000000001)
        x = torch.cat((fv, fi), 1)
        x = F.relu(self.concat_filter_layer(x), inplace=True)

        # x3 = self.deform_conv_1_3(x)
        # x5 = self.deform_conv_1_5(x)
        # x7 = self.deform_conv_1_7(x)
        # x = torch.cat((x3, x5, x7), 1)
        # x = torch.cat((x3, x5), 1)
        x = F.relu(self.concat_filter_layer_1(x), inplace=True)

        x3 = self.deform_conv_2_3(x)
        x5 = self.deform_conv_2_5(x)
        # x7 = self.deform_conv_2_7(x)
        # x = torch.cat((x3, x5, x7), 1)
        x = F.relu(torch.cat((x3, x5), 1), inplace=True)
        x = F.relu(self.concat_filter_layer_2(x), inplace=True)

        x3 = self.deform_conv_3_3(x)
        x5 = self.deform_conv_3_5(x)
        # x7 = self.deform_conv_3_7(x)
        # x = torch.cat((x3, x5, x7), 1)
        x = F.relu(torch.cat((x3, x5), 1), inplace=True)
        x = F.relu(self.concat_filter_layer_3(x), inplace=True)

        x = self.output_layer(x)

        # this cause too much dimension mismatch problem
        # so we desampling label instead
        # x = nn.functional.upsample(x, scale_factor=8, mode='bilinear') / 64.0
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class AttnCanAdcrowdNetSimpleV3(nn.Module):
    """
    """
    def __init__(self, load_weights=False):
        super(AttnCanAdcrowdNetSimpleV3, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend = make_layers(self.frontend_feat)

        # freeze vgg layer
        for param in self.frontend.parameters():
            param.requires_grad = False

        self.sSE = SpatialSELayer(num_channels=512)

        self.concat_filter_layer = nn.Conv2d(1024, 512, kernel_size=3, padding=2, dilation=2)

        # we skip one formation of deformconv
        # self.deform_conv_1_3 = DeformConv2d(512, 256, kernel_size=3, stride=1, padding=1)
        # self.deform_conv_1_5 = DeformConv2d(512, 256, kernel_size=5, stride=1, padding=2)
        # self.deform_conv_1_7 = DeformConv2d(512, 256, kernel_size=7, stride=1, padding=3)
        self.concat_filter_layer_1 = nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2)

        self.dilated_conv_2_3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, dilation=2, padding=2)
        self.dilated_conv_2_5 = nn.Conv2d(256, 128, kernel_size=3, stride=1, dilation=4, padding=4)
        # self.deform_conv_2_7 = DeformConv2d(256, 128, kernel_size=7, stride=1, padding=3)
        self.concat_filter_layer_2 = nn.Conv2d(128 * 2, 128, kernel_size=3, padding=2, dilation=2)

        self.deform_conv_3_3 = DeformConv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.deform_conv_3_5 = DeformConv2d(128, 64, kernel_size=5, stride=1, padding=2)
        # self.deform_conv_3_7 = DeformConv2d(128, 64, kernel_size=7, stride=1, padding=3)
        self.concat_filter_layer_3 = nn.Conv2d(64 * 2, 64, kernel_size=3, padding=2, dilation=2)

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.conv1_1 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv1_2 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv2_1 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv2_2 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv3_1 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv3_2 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv6_1 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv6_2 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            fsd = collections.OrderedDict()
            for i in range(len(self.frontend.state_dict().items())):
                temp_key = list(self.frontend.state_dict().items())[i][0]
                fsd[temp_key] = list(mod.state_dict().items())[i][1]
            self.frontend.load_state_dict(fsd)

    def forward(self, x):
        fv = self.frontend(x)

        #  spatial squeeze & excitation
        fv = self.sSE(fv)

        # S=1
        ave1 = nn.functional.adaptive_avg_pool2d(fv, (1, 1))
        ave1 = self.conv1_1(ave1)
        s1 = nn.functional.upsample(ave1, size=(fv.shape[2], fv.shape[3]), mode='bilinear')
        c1 = s1 - fv
        w1 = self.conv1_2(c1)
        w1 = nn.functional.sigmoid(w1)
        # S=2
        ave2 = nn.functional.adaptive_avg_pool2d(fv, (2, 2))
        ave2 = self.conv2_1(ave2)
        s2 = nn.functional.upsample(ave2, size=(fv.shape[2], fv.shape[3]), mode='bilinear')
        c2 = s2 - fv
        w2 = self.conv2_2(c2)
        w2 = nn.functional.sigmoid(w2)
        # S=3
        ave3 = nn.functional.adaptive_avg_pool2d(fv, (3, 3))
        ave3 = self.conv3_1(ave3)
        s3 = nn.functional.upsample(ave3, size=(fv.shape[2], fv.shape[3]), mode='bilinear')
        c3 = s3 - fv
        w3 = self.conv3_2(c3)
        w3 = nn.functional.sigmoid(w3)
        # S=6
        ave6 = nn.functional.adaptive_avg_pool2d(fv, (6, 6))
        ave6 = self.conv6_1(ave6)
        s6 = nn.functional.upsample(ave6, size=(fv.shape[2], fv.shape[3]), mode='bilinear')
        c6 = s6 - fv
        w6 = self.conv6_2(c6)
        w6 = nn.functional.sigmoid(w6)

        fi = (w1 * s1 + w2 * s2 + w3 * s3 + w6 * s6) / (w1 + w2 + w3 + w6 + 0.000000000001)
        x = torch.cat((fv, fi), 1)
        x = F.relu(self.concat_filter_layer(x), inplace=True)

        # x3 = self.deform_conv_1_3(x)
        # x5 = self.deform_conv_1_5(x)
        # x7 = self.deform_conv_1_7(x)
        # x = torch.cat((x3, x5, x7), 1)
        # x = torch.cat((x3, x5), 1)
        x = F.relu(self.concat_filter_layer_1(x), inplace=True)

        x3 = self.dilated_conv_2_3(x)
        x5 = self.dilated_conv_2_5(x)
        # x7 = self.deform_conv_2_7(x)
        # x = torch.cat((x3, x5, x7), 1)
        x = F.relu(torch.cat((x3, x5), 1), inplace=True)
        x = F.relu(self.concat_filter_layer_2(x), inplace=True)

        x3 = self.deform_conv_3_3(x)
        x5 = self.deform_conv_3_5(x)
        # x7 = self.deform_conv_3_7(x)
        # x = torch.cat((x3, x5, x7), 1)
        x = F.relu(torch.cat((x3, x5), 1), inplace=True)
        x = F.relu(self.concat_filter_layer_3(x), inplace=True)

        x = self.output_layer(x)

        # this cause too much dimension mismatch problem
        # so we desampling label instead
        # x = nn.functional.upsample(x, scale_factor=8, mode='bilinear') / 64.0
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class AttnCanAdcrowdNetSimpleV4(nn.Module):
    """
    compare with v3: add 1 layer (1 branch) of deformable cnn before output layer
    """
    def __init__(self, load_weights=False):
        super(AttnCanAdcrowdNetSimpleV4, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend = make_layers(self.frontend_feat)

        # freeze vgg layer
        for param in self.frontend.parameters():
            param.requires_grad = False

        self.sSE = SpatialSELayer(num_channels=512)

        self.concat_filter_layer = nn.Conv2d(1024, 512, kernel_size=3, padding=2, dilation=2)

        # we skip one formation of deformconv
        # self.deform_conv_1_3 = DeformConv2d(512, 256, kernel_size=3, stride=1, padding=1)
        # self.deform_conv_1_5 = DeformConv2d(512, 256, kernel_size=5, stride=1, padding=2)
        # self.deform_conv_1_7 = DeformConv2d(512, 256, kernel_size=7, stride=1, padding=3)
        self.concat_filter_layer_1 = nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2)

        self.dilated_conv_2_3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, dilation=2, padding=2)
        self.dilated_conv_2_5 = nn.Conv2d(256, 128, kernel_size=3, stride=1, dilation=4, padding=4)
        # self.deform_conv_2_7 = DeformConv2d(256, 128, kernel_size=7, stride=1, padding=3)
        self.concat_filter_layer_2 = nn.Conv2d(128 * 2, 128, kernel_size=3, padding=2, dilation=2)

        self.deform_conv_3_3 = DeformConv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.deform_conv_3_5 = DeformConv2d(128, 64, kernel_size=5, stride=1, padding=2)
        # self.deform_conv_3_7 = DeformConv2d(128, 64, kernel_size=7, stride=1, padding=3)
        self.concat_filter_layer_3 = nn.Conv2d(64 * 2, 64, kernel_size=3, padding=2, dilation=2)

        self.deform_conv_4_3 = DeformConv2d(64, 32, kernel_size=3, stride=1, padding=1)

        self.output_layer = nn.Conv2d(32, 1, kernel_size=1)
        self.conv1_1 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv1_2 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv2_1 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv2_2 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv3_1 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv3_2 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv6_1 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv6_2 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            fsd = collections.OrderedDict()
            for i in range(len(self.frontend.state_dict().items())):
                temp_key = list(self.frontend.state_dict().items())[i][0]
                fsd[temp_key] = list(mod.state_dict().items())[i][1]
            self.frontend.load_state_dict(fsd)

    def forward(self, x):
        fv = self.frontend(x)

        #  spatial squeeze & excitation
        fv = self.sSE(fv)

        # S=1
        ave1 = nn.functional.adaptive_avg_pool2d(fv, (1, 1))
        ave1 = self.conv1_1(ave1)
        s1 = nn.functional.upsample(ave1, size=(fv.shape[2], fv.shape[3]), mode='bilinear')
        c1 = s1 - fv
        w1 = self.conv1_2(c1)
        w1 = nn.functional.sigmoid(w1)
        # S=2
        ave2 = nn.functional.adaptive_avg_pool2d(fv, (2, 2))
        ave2 = self.conv2_1(ave2)
        s2 = nn.functional.upsample(ave2, size=(fv.shape[2], fv.shape[3]), mode='bilinear')
        c2 = s2 - fv
        w2 = self.conv2_2(c2)
        w2 = nn.functional.sigmoid(w2)
        # S=3
        ave3 = nn.functional.adaptive_avg_pool2d(fv, (3, 3))
        ave3 = self.conv3_1(ave3)
        s3 = nn.functional.upsample(ave3, size=(fv.shape[2], fv.shape[3]), mode='bilinear')
        c3 = s3 - fv
        w3 = self.conv3_2(c3)
        w3 = nn.functional.sigmoid(w3)
        # S=6
        ave6 = nn.functional.adaptive_avg_pool2d(fv, (6, 6))
        ave6 = self.conv6_1(ave6)
        s6 = nn.functional.upsample(ave6, size=(fv.shape[2], fv.shape[3]), mode='bilinear')
        c6 = s6 - fv
        w6 = self.conv6_2(c6)
        w6 = nn.functional.sigmoid(w6)

        fi = (w1 * s1 + w2 * s2 + w3 * s3 + w6 * s6) / (w1 + w2 + w3 + w6 + 0.000000000001)
        x = torch.cat((fv, fi), 1)
        x = F.relu(self.concat_filter_layer(x), inplace=True)

        # x3 = self.deform_conv_1_3(x)
        # x5 = self.deform_conv_1_5(x)
        # x7 = self.deform_conv_1_7(x)
        # x = torch.cat((x3, x5, x7), 1)
        # x = torch.cat((x3, x5), 1)
        x = F.relu(self.concat_filter_layer_1(x), inplace=True)

        x3 = self.dilated_conv_2_3(x)
        x5 = self.dilated_conv_2_5(x)
        # x7 = self.deform_conv_2_7(x)
        # x = torch.cat((x3, x5, x7), 1)
        x = F.relu(torch.cat((x3, x5), 1), inplace=True)
        x = F.relu(self.concat_filter_layer_2(x), inplace=True)

        x3 = self.deform_conv_3_3(x)
        x5 = self.deform_conv_3_5(x)
        # x7 = self.deform_conv_3_7(x)
        # x = torch.cat((x3, x5, x7), 1)
        x = F.relu(torch.cat((x3, x5), 1), inplace=True)
        x = F.relu(self.concat_filter_layer_3(x), inplace=True)

        x = F.relu(self.deform_conv_4_3(x))
        x = self.output_layer(x)

        # this cause too much dimension mismatch problem
        # so we desampling label instead
        # x = nn.functional.upsample(x, scale_factor=8, mode='bilinear') / 64.0
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class AttnCanAdcrowdNetSimpleV3(nn.Module):
    """
    """
    def __init__(self, load_weights=False):
        super(AttnCanAdcrowdNetSimpleV3, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend = make_layers(self.frontend_feat)

        # freeze vgg layer
        for param in self.frontend.parameters():
            param.requires_grad = False

        self.sSE = SpatialSELayer(num_channels=512)

        self.concat_filter_layer = nn.Conv2d(1024, 512, kernel_size=3, padding=2, dilation=2)

        # we skip one formation of deformconv
        # self.deform_conv_1_3 = DeformConv2d(512, 256, kernel_size=3, stride=1, padding=1)
        # self.deform_conv_1_5 = DeformConv2d(512, 256, kernel_size=5, stride=1, padding=2)
        # self.deform_conv_1_7 = DeformConv2d(512, 256, kernel_size=7, stride=1, padding=3)
        self.concat_filter_layer_1 = nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2)

        self.dilated_conv_2_3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, dilation=2, padding=2)
        self.dilated_conv_2_5 = nn.Conv2d(256, 128, kernel_size=3, stride=1, dilation=4, padding=4)
        # self.deform_conv_2_7 = DeformConv2d(256, 128, kernel_size=7, stride=1, padding=3)
        self.concat_filter_layer_2 = nn.Conv2d(128 * 2, 128, kernel_size=3, padding=2, dilation=2)

        self.deform_conv_3_3 = DeformConv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.deform_conv_3_5 = DeformConv2d(128, 64, kernel_size=5, stride=1, padding=2)
        # self.deform_conv_3_7 = DeformConv2d(128, 64, kernel_size=7, stride=1, padding=3)
        self.concat_filter_layer_3 = nn.Conv2d(64 * 2, 64, kernel_size=3, padding=2, dilation=2)

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.conv1_1 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv1_2 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv2_1 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv2_2 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv3_1 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv3_2 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv6_1 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv6_2 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            fsd = collections.OrderedDict()
            for i in range(len(self.frontend.state_dict().items())):
                temp_key = list(self.frontend.state_dict().items())[i][0]
                fsd[temp_key] = list(mod.state_dict().items())[i][1]
            self.frontend.load_state_dict(fsd)

    def forward(self, x):
        fv = self.frontend(x)

        #  spatial squeeze & excitation
        fv = self.sSE(fv)

        # S=1
        ave1 = nn.functional.adaptive_avg_pool2d(fv, (1, 1))
        ave1 = self.conv1_1(ave1)
        s1 = nn.functional.upsample(ave1, size=(fv.shape[2], fv.shape[3]), mode='bilinear')
        c1 = s1 - fv
        w1 = self.conv1_2(c1)
        w1 = nn.functional.sigmoid(w1)
        # S=2
        ave2 = nn.functional.adaptive_avg_pool2d(fv, (2, 2))
        ave2 = self.conv2_1(ave2)
        s2 = nn.functional.upsample(ave2, size=(fv.shape[2], fv.shape[3]), mode='bilinear')
        c2 = s2 - fv
        w2 = self.conv2_2(c2)
        w2 = nn.functional.sigmoid(w2)
        # S=3
        ave3 = nn.functional.adaptive_avg_pool2d(fv, (3, 3))
        ave3 = self.conv3_1(ave3)
        s3 = nn.functional.upsample(ave3, size=(fv.shape[2], fv.shape[3]), mode='bilinear')
        c3 = s3 - fv
        w3 = self.conv3_2(c3)
        w3 = nn.functional.sigmoid(w3)
        # S=6
        ave6 = nn.functional.adaptive_avg_pool2d(fv, (6, 6))
        ave6 = self.conv6_1(ave6)
        s6 = nn.functional.upsample(ave6, size=(fv.shape[2], fv.shape[3]), mode='bilinear')
        c6 = s6 - fv
        w6 = self.conv6_2(c6)
        w6 = nn.functional.sigmoid(w6)

        fi = (w1 * s1 + w2 * s2 + w3 * s3 + w6 * s6) / (w1 + w2 + w3 + w6 + 0.000000000001)
        x = torch.cat((fv, fi), 1)
        x = F.relu(self.concat_filter_layer(x), inplace=True)

        # x3 = self.deform_conv_1_3(x)
        # x5 = self.deform_conv_1_5(x)
        # x7 = self.deform_conv_1_7(x)
        # x = torch.cat((x3, x5, x7), 1)
        # x = torch.cat((x3, x5), 1)
        x = F.relu(self.concat_filter_layer_1(x), inplace=True)

        x3 = self.dilated_conv_2_3(x)
        x5 = self.dilated_conv_2_5(x)
        # x7 = self.deform_conv_2_7(x)
        # x = torch.cat((x3, x5, x7), 1)
        x = F.relu(torch.cat((x3, x5), 1), inplace=True)
        x = F.relu(self.concat_filter_layer_2(x), inplace=True)

        x3 = self.deform_conv_3_3(x)
        x5 = self.deform_conv_3_5(x)
        # x7 = self.deform_conv_3_7(x)
        # x = torch.cat((x3, x5, x7), 1)
        x = F.relu(torch.cat((x3, x5), 1), inplace=True)
        x = F.relu(self.concat_filter_layer_3(x), inplace=True)

        x = self.output_layer(x)

        # this cause too much dimension mismatch problem
        # so we desampling label instead
        # x = nn.functional.upsample(x, scale_factor=8, mode='bilinear') / 64.0
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class AttnCanAdcrowdNetSimpleV5(nn.Module):
    """
    we change all deformable cnn to dilated cnn
    """
    def __init__(self, load_weights=False):
        super(AttnCanAdcrowdNetSimpleV5, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend = make_layers(self.frontend_feat)

        # freeze vgg layer
        for param in self.frontend.parameters():
            param.requires_grad = False

        self.sSE = SpatialSELayer(num_channels=512)

        self.concat_filter_layer = nn.Conv2d(1024, 512, kernel_size=3, padding=2, dilation=2)

        self.dilated_conv_1_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, dilation=2, padding=2, bias=False)
        self.dilated_conv_1_2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, dilation=4, padding=4, bias=False)
        self.dilated_conv_1_3 = nn.Conv2d(512, 256, kernel_size=3, stride=1, dilation=6, padding=6, bias=False)
        self.concat_filter_layer_1 = nn.Conv2d(256*3, 256, kernel_size=3, padding=2, dilation=2, bias=False)
        self.bn_c_1 = nn.BatchNorm2d(256 * 3)
        self.bn1 = nn.BatchNorm2d(256)

        self.dilated_conv_2_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, dilation=2, padding=2, bias=False)
        self.dilated_conv_2_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, dilation=4, padding=4, bias=False)
        self.dilated_conv_2_3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, dilation=6, padding=6, bias=False)
        self.concat_filter_layer_2 = nn.Conv2d(128 * 3, 128, kernel_size=3, padding=2, dilation=2, bias=False)
        self.bn_c_2 = nn.BatchNorm2d(128 * 3)
        self.bn2 = nn.BatchNorm2d(128)

        self.dilated_conv_3_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, dilation=2, padding=2, bias=False)
        self.dilated_conv_3_2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, dilation=4, padding=4, bias=False)
        self.dilated_conv_3_3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, dilation=6, padding=6, bias=False)
        self.concat_filter_layer_3 = nn.Conv2d(64 * 3, 64, kernel_size=3, padding=2, dilation=2, bias=False)
        self.bn_c_3 = nn.BatchNorm2d(64 * 3)
        self.bn3 = nn.BatchNorm2d(64)

        # we will test with and without this
        # v6 with have this deform
        # self.deform_conv_4_3 = DeformConv2d(64, 32, kernel_size=3, stride=1, padding=1)

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.conv1_1 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv1_2 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv2_1 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv2_2 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv3_1 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv3_2 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv6_1 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv6_2 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            fsd = collections.OrderedDict()
            for i in range(len(self.frontend.state_dict().items())):
                temp_key = list(self.frontend.state_dict().items())[i][0]
                fsd[temp_key] = list(mod.state_dict().items())[i][1]
            self.frontend.load_state_dict(fsd)

    def forward(self, x):
        fv = self.frontend(x)

        #  spatial squeeze & excitation
        fv = self.sSE(fv)

        # S=1
        ave1 = nn.functional.adaptive_avg_pool2d(fv, (1, 1))
        ave1 = self.conv1_1(ave1)
        s1 = nn.functional.upsample(ave1, size=(fv.shape[2], fv.shape[3]), mode='bilinear')
        c1 = s1 - fv
        w1 = self.conv1_2(c1)
        w1 = nn.functional.sigmoid(w1)
        # S=2
        ave2 = nn.functional.adaptive_avg_pool2d(fv, (2, 2))
        ave2 = self.conv2_1(ave2)
        s2 = nn.functional.upsample(ave2, size=(fv.shape[2], fv.shape[3]), mode='bilinear')
        c2 = s2 - fv
        w2 = self.conv2_2(c2)
        w2 = nn.functional.sigmoid(w2)
        # S=3
        ave3 = nn.functional.adaptive_avg_pool2d(fv, (3, 3))
        ave3 = self.conv3_1(ave3)
        s3 = nn.functional.upsample(ave3, size=(fv.shape[2], fv.shape[3]), mode='bilinear')
        c3 = s3 - fv
        w3 = self.conv3_2(c3)
        w3 = nn.functional.sigmoid(w3)
        # S=6
        ave6 = nn.functional.adaptive_avg_pool2d(fv, (6, 6))
        ave6 = self.conv6_1(ave6)
        s6 = nn.functional.upsample(ave6, size=(fv.shape[2], fv.shape[3]), mode='bilinear')
        c6 = s6 - fv
        w6 = self.conv6_2(c6)
        w6 = nn.functional.sigmoid(w6)

        fi = (w1 * s1 + w2 * s2 + w3 * s3 + w6 * s6) / (w1 + w2 + w3 + w6 + 0.000000000001)
        x = torch.cat((fv, fi), 1)
        x = F.relu(self.concat_filter_layer(x), inplace=True)

        # x3 = self.deform_conv_1_3(x)
        # x5 = self.deform_conv_1_5(x)
        # x7 = self.deform_conv_1_7(x)
        # x = torch.cat((x3, x5, x7), 1)
        # x = torch.cat((x3, x5), 1)
        x1 = self.dilated_conv_1_1(x)
        x2 = self.dilated_conv_1_2(x)
        x3 = self.dilated_conv_1_3(x)
        x = F.relu(self.bn_c_1(torch.cat((x1, x2, x3), 1)), inplace=True)
        x = F.relu(self.bn1(self.concat_filter_layer_1(x)), inplace=True)

        x1 = self.dilated_conv_2_1(x)
        x2 = self.dilated_conv_2_2(x)
        x3 = self.dilated_conv_2_3(x)
        x = F.relu(self.bn_c_2(torch.cat((x1, x2, x3), 1)), inplace=True)
        x = F.relu(self.bn2(self.concat_filter_layer_2(x)), inplace=True)

        x1 = self.dilated_conv_3_1(x)
        x2 = self.dilated_conv_3_2(x)
        x3 = self.dilated_conv_3_3(x)
        x = F.relu(self.bn_c_3(torch.cat((x1, x2, x3), 1)), inplace=True)
        x = F.relu(self.bn3(self.concat_filter_layer_3(x)), inplace=True)

        # x = F.relu(self.deform_conv_4_3(x))
        x = self.output_layer(x)

        # this cause too much dimension mismatch problem
        # so we desampling label instead
        # x = nn.functional.upsample(x, scale_factor=8, mode='bilinear') / 64.0
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
