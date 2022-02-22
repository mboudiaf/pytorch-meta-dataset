from .resnet import *
from .resnet_v2 import *
from .vit import *

from .resnet_v2 import default_cfgs as resnet_v2_config
from .resnet import default_cfgs as resnet_config
from .vit import default_cfgs as vit_config

model_configs = resnet_v2_config
model_configs.update(resnet_config)
model_configs.update(vit_config)