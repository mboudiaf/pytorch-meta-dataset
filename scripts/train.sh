method=$1
arch=$2
data=$3


base_config_path="config/base.yaml"
method_config_path="config/${method}.yaml"

base_sources="['${data}']"
val_sources="['${data}']"
# ===============> Mini <===================

python -m src.train --base_config config/base.yaml \
                    --method_config ${method_config_path} \
                    --opts arch ${arch} \
                           base_sources ${base_sources} \
                           val_sources ${val_sources} \


python -m src.train --base_config config/base.yaml --method_config config/simpleshot.yaml --opts arch resnet18

