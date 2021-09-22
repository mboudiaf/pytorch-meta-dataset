data_path=$1
all_sources="ilsvrc_2012 aircraft cu_birds dtd fungi mscoco omniglot quickdraw traffic_sign vgg_flower"

for source in ${all_sources}
do
    source_path=${data_path}/${source}
    find ${source_path} -name '*.tfrecords' -type f -exec sh -c 'python3 -m tfrecord.tools.tfrecord2idx $2 ${2%.tfrecords}.index' sh ${source_path} {} \;
done