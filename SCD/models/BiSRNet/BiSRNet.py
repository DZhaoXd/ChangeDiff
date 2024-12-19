import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.resnet import _cfg


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SR(nn.Module):
    '''Spatial reasoning module'''

    # codes from DANet 'Dual attention network for scene segmentation'
    def __init__(self, in_dim):
        super(SR, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        ''' inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW) '''
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = x + self.gamma * out

        return out


class CotSR(nn.Module):
    # codes derived from DANet 'Dual attention network for scene segmentation'
    def __init__(self, in_dim):
        super(CotSR, self).__init__()
        self.chanel_in = in_dim

        self.query_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.query_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        ''' inputs :
                x1 : input feature maps( B X C X H X W)
                x2 : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW) '''
        m_batchsize, C, height, width = x1.size()

        q1 = self.query_conv1(x1).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        k1 = self.key_conv1(x1).view(m_batchsize, -1, width * height)
        v1 = self.value_conv1(x1).view(m_batchsize, -1, width * height)

        q2 = self.query_conv2(x2).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        k2 = self.key_conv2(x2).view(m_batchsize, -1, width * height)
        v2 = self.value_conv2(x2).view(m_batchsize, -1, width * height)

        energy1 = torch.bmm(q1, k2)
        attention1 = self.softmax(energy1)
        out1 = torch.bmm(v2, attention1.permute(0, 2, 1))
        out1 = out1.view(m_batchsize, C, height, width)

        energy2 = torch.bmm(q2, k1)
        attention2 = self.softmax(energy2)
        out2 = torch.bmm(v1, attention2.permute(0, 2, 1))
        out2 = out2.view(m_batchsize, C, height, width)

        out1 = x1 + self.gamma1 * out1
        out2 = x2 + self.gamma2 * out2

        return out1, out2


class BiSRNet(nn.Module):
    def __init__(self, bc_token_length=1, sc_token_length=7):
        super(BiSRNet, self).__init__()
        #
        in_channels = [64, 64, 128, 256, 512]
        de_channel_c2 = 64
        de_channel_c5 = 128
        # config_resnet = _cfg(url='', file='/mnt/disk_d/backbone_weights/resnet34d_ra2-f8dcfcaf.pth')
        # self.context_encoder = timm.create_model(
        #     'resnet34d', features_only=True, pretrained=True, pretrained_cfg=config_resnet)
        pre_path = '/data2/yjy/SCD/models/SCanNet/resnet18d_ra2-48a79e06.pth'
        self.context_encoder = timm.create_model('resnet18d', features_only=True, output_stride=8, pretrained=True,
                                                 pretrained_cfg_overlay=dict(file=pre_path))
        self.feature_extraction_c5 = nn.Sequential(
            nn.Conv2d(in_channels[4], de_channel_c5, kernel_size=1, bias=False),
            nn.BatchNorm2d(de_channel_c5),
            nn.ReLU()
        )
        self.SR1 = SR(de_channel_c5)
        self.SR2 = SR(de_channel_c5)
        self.CotSR = CotSR(de_channel_c5)
        self.classifier1 = nn.Conv2d(de_channel_c5, sc_token_length, kernel_size=1)
        self.classifier2 = nn.Conv2d(de_channel_c5, sc_token_length, kernel_size=1)

        self.resCD = self._make_layer(ResBlock, de_channel_c5 * 2, de_channel_c5, 6, stride=1)
        self.classifierCD = nn.Sequential(
            nn.Conv2d(de_channel_c5, de_channel_c5 // 2, kernel_size=1),
            nn.BatchNorm2d(de_channel_c5 // 2),
            nn.ReLU(),
            nn.Conv2d(de_channel_c5 // 2, bc_token_length, kernel_size=1)
        )

        # for param in self.context_encoder.parameters():
        #     param.requires_grad = False

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, t1, t2):
        t1_c1, t1_c2, t1_c3, t1_c4, t1_c5 = self.context_encoder(t1)
        t2_c1, t2_c2, t2_c3, t2_c4, t2_c5 = self.context_encoder(t2)
        #
        t1_c5 = self.feature_extraction_c5(t1_c5)
        t2_c5 = self.feature_extraction_c5(t2_c5)
        t1_c5 = self.SR1(t1_c5)
        t2_c5 = self.SR2(t2_c5)
        #
        td_c5 = self.resCD(torch.cat([t1_c5, t2_c5], dim=1))
        mask_bc = self.classifierCD(td_c5)
        t1_c5, t2_c5 = self.CotSR(t1_c5, t2_c5)
        mask_t1 = self.classifier1(t1_c5)
        mask_t2 = self.classifier2(t2_c5)

        mask_bc = F.interpolate(mask_bc, size=t1.size()[2:], mode='bilinear', align_corners=True)
        mask_t1 = F.interpolate(mask_t1, size=t1.size()[2:], mode='bilinear', align_corners=True)
        mask_t2 = F.interpolate(mask_t2, size=t1.size()[2:], mode='bilinear', align_corners=True)

        return mask_t1, mask_t2, mask_bc


def get_model(bc_token_length=1, sc_token_length=7):
    model = BiSRNet(bc_token_length=bc_token_length,
                    sc_token_length=sc_token_length)

    return model


if __name__ == '__main__':
    t1 = torch.randn(1, 3, 512, 512)
    t2 = torch.randn(1, 3, 512, 512)
    models = get_model()
    mask_t1, mask_t2, mask_bc = models(t1, t2)
    print('1111')
