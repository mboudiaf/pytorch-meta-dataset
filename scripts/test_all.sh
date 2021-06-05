method=$1
arch=$2
all_sources="ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower traffic_sign mscoco "

for source in ${all_sources}
do
    bash scripts/test.sh ${method} ${arch} ilsvrc_2012 ${source}
done
