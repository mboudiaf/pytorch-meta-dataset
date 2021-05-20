import torch.nn as nn
from .metamodules import (MetaModule, MetaSequential, MetaConv2d,
                          MetaBatchNorm2d, MetaLinear)
__all__ = ['resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return MetaConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                      padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return MetaConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def BasicBlock(inplanes, planes, stride=1, downsample=None):
    block = MetaSequential(
                    conv3x3(inplanes, planes, stride),
                    MetaBatchNorm2d(planes),
                    nn.ReLU(inplace=True),
                    conv3x3(planes, planes),
                    MetaBatchNorm2d(planes),
                    nn.ReLU(inplace=True),
                            )
    block.expansion = 1
    return block


class Bottleneck(MetaModule):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bottleneck = MetaSequential(
                        conv1x1(inplanes, planes),
                        MetaBatchNorm2d(planes),
                        nn.ReLU(inplace=True),
                        conv3x3(planes, planes, stride),
                        MetaBatchNorm2d(planes),
                        nn.ReLU(inplace=True),
                        conv1x1(planes, planes * self.expansion),
                        MetaBatchNorm2d(planes * self.expansion),
                        nn.ReLU(inplace=True),
                    )

    def forward(self, x, params=None):
        return self.bottleneck(x, params=self.get_subdict(params, 'bottleneck'))


class ResNet(MetaModule):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, expansion=1, **kwargs):
        super(ResNet, self).__init__()
        self.inplanes = 64

        self.features = MetaSequential(
                            MetaConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                            MetaBatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            self._make_layer(block, 64, layers[0], expansion=expansion),
                            self._make_layer(block, 128, layers[1], stride=2, expansion=expansion),
                            self._make_layer(block, 256, layers[2], stride=2, expansion=expansion),
                            self._make_layer(block, 512, layers[3], stride=2, expansion=expansion),
                            nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = MetaLinear(512 * expansion, num_classes)

        for m in self.modules():
            if isinstance(m, MetaConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, MetaBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, expansion=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * expansion:
            downsample = MetaSequential(
                conv1x1(self.inplanes, planes * expansion, stride),
                MetaBatchNorm2d(planes * expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return MetaSequential(*layers)

    def forward(self, inputs, params=None, features=False):
        x = self.features(inputs, params=self.get_subdict(params, 'features'))
        x = x.view((x.size(0), -1))
        if features:
            return x
        logits = self.classifier(x, params=self.get_subdict(params, 'classifier'))
        return logits


def resnet10(**kwargs):
    """Constructs a ResNet-10 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], expansion=1, **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], expansion=1, **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], expansion=1, **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], expansion=4, **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], expansion=4, **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-152 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], expansion=4, **kwargs)
    return model
