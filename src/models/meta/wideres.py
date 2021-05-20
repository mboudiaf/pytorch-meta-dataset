import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from .metamodules import (MetaModule, MetaSequential, MetaConv2d,
                          MetaBatchNorm2d, MetaLinear)


def conv3x3(in_planes, out_planes, stride=1):
    return MetaConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)


def wide_basic(in_planes, planes, dropout_rate, stride=1):
    block = MetaSequential(
                MetaBatchNorm2d(in_planes),
                nn.ReLU(inplace=True),
                MetaConv2d(in_planes, planes, kernel_size=3, padding=1, bias=True),
                nn.Dropout(p=dropout_rate),
                MetaBatchNorm2d(planes),
                nn.ReLU(inplace=True),
                MetaConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
                )
    return block


class Wide_ResNet(MetaModule):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, use_fc=True):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) // 6
        k = widen_factor

        print('| Wide-Resnet %dx%d' % (depth, k))
        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.features = MetaSequential(
                            conv3x3(3, nStages[0]),
                            self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1),
                            self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2),
                            self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2),
                            MetaBatchNorm2d(nStages[3], momentum=0.9),
                            nn.ReLU(inplace=True),
                            nn.AdaptiveAvgPool2d((1, 1)))
        if use_fc:
            self.classifier = MetaLinear(nStages[3], num_classes)
        for m in self.modules():
            if isinstance(m, MetaConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, MetaBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return MetaSequential(*layers)

    def forward(self, inputs, params=None, features=False):
        if params:
            print(params.keys())
        x = self.features(inputs, params=self.get_subdict(params, 'features'))
        x = x.view((x.size(0), -1))
        if features:
            return x
        logits = self.classifier(x, params=self.get_subdict(params, 'classifier'))
        return logits


def wideres2810(num_classes):
    """Constructs a wideres-28-10 model without dropout.
    """
    return Wide_ResNet(28, 10, 0, num_classes)


if __name__ == '__main__':
    net = Wide_ResNet(28, 10, 0.3, 10)
    y = net(Variable(torch.randn(1, 3, 32, 32)))