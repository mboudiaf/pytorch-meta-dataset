import torch.nn as nn
from .metamodules import (MetaModule, MetaSequential, MetaConv2d,
                          MetaBatchNorm2d, MetaLinear)


def conv3x3(in_channels, out_channels, **kwargs):
    return MetaSequential(
        MetaConv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        MetaBatchNorm2d(out_channels, momentum=1., track_running_stats=False),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class conv4(MetaModule):
    def __init__(self, num_classes, in_channels=3, hidden_size=64, **kwargs):
        super(conv4, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        self.features = MetaSequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size)
        )

        self.classifier = MetaLinear(1600, num_classes)

    def forward(self, inputs, params=None, features=False):
        x = self.features(inputs, params=self.get_subdict(params, 'features'))
        x = x.view((x.size(0), -1))
        if features:
            return x
        logits = self.classifier(x, params=self.get_subdict(params, 'classifier'))
        return logits