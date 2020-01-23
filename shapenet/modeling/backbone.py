# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.nn as nn

import torchvision


class ResNetBackbone(nn.Module):
    def __init__(self, net):
        super(ResNetBackbone, self).__init__()
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.stage1 = net.layer1
        self.stage2 = net.layer2
        self.stage3 = net.layer3
        self.stage4 = net.layer4

    def forward(self, imgs):
        feats = self.stem(imgs)
        conv1 = self.stage1(feats)  # 18, 34: 64
        conv2 = self.stage2(conv1)
        conv3 = self.stage3(conv2)
        conv4 = self.stage4(conv3)

        return [conv1, conv2, conv3, conv4]


_FEAT_DIMS = {
    "resnet18": (64, 128, 256, 512),
    "resnet34": (64, 128, 256, 512),
    "resnet50": (256, 512, 1024, 2048),
    "resnet101": (256, 512, 1024, 2048),
    "resnet152": (256, 512, 1024, 2048),
}


def build_backbone(name, pretrained=True):
    resnets = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
    if name in resnets and name in _FEAT_DIMS:
        cnn = getattr(torchvision.models, name)(pretrained=pretrained)
        backbone = ResNetBackbone(cnn)
        feat_dims = _FEAT_DIMS[name]
        return backbone, feat_dims
    else:
        raise ValueError('Unrecognized backbone type "%s"' % name)
