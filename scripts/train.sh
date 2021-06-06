method=$1
arch=$2
data=$3


base_config_path="config/base.yaml"
method_config_path="config/${method}.yaml"

base_source=${data}
val_source=${data}
test_source=${data}
# ===============> Mini <===================

python -m src.train --base_config config/base.yaml \
                    --method_config ${method_config_path} \
                    --opts arch ${arch} \
                           base_source ${base_source} \
                           val_source ${val_source} \
                           test_source ${test_source} \
                           debug True
