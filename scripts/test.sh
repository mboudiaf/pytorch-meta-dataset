method=$1
arch=$2
base_source=$3
test_source=$4
visu='True'

if [ -z "${test_source}" ]
    then
        test_source=${base_source}
fi

base_config_path="config/base.yaml"
method_config_path="config/${method}.yaml"

base_source=${base_source}
val_source=${base_source}
test_source=${test_source}


python -m src.eval  --base_config ${base_config_path} \
                    --method_config ${method_config_path} \
                    --opts arch ${arch} \
                         base_source ${base_source} \
                         val_source ${val_source} \
                         test_source ${test_source} \
                         val_episodes 600 \
                         visu ${visu} \
                         extract_batch_size 10 \
                         num_ways 0 \
                         num_support 0 \
                         num_query 0 \
                         val_batch_size 1 # batching is not straightforward when episodes have random formats

