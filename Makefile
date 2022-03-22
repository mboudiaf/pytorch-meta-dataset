# ------ misc options ---------

exec=PYTHONHASHSEED=0 python
debug=False
override=False
loader=pytorch

# ------ SERVER options --------- (only useful if you want to deploy/import results from remote server)

USER=N/A # set the user of the remote server
SERVER_IP=# set the ip address of the remote server
SERVER_PATH=# set the path where you want all the folders to be dropped

# ------ Simulation hyperparams ------------

n_episodes=600
source=records
mode=test
arch=resnet18
load_from_timm=True
method=finetune
data=variable

method_cfg=config/method/$(method).yaml
data_cfg=config/data/$(data).yaml

base=ilsvrc_2012_v2
val=aircraft
test=aircraft

# ------ Test-time options ------------

visu=False


# ============= Sanity tests =============

# sanity_check:
#        make base_source=mini_imagenet test_source=mini_imagenet data=1_shot test ;\


# ============= Main scripts =============

train:
	$(exec) -m src.train --base_config config/base.yaml \
				--method_config ${method_cfg} \
				--data_config ${data_cfg} \
				--opts \
				base_source $(base) \
				arch $(arch) \
				val_source $(val) \
				debug $(debug) \
				loader_version $(loader) \
				load_from_timm False \

eval:
	$(exec) -m src.eval  --base_config config/base.yaml \
				--method_config ${method_cfg} \
				--data_config ${data_cfg} \
				--opts \
				 base_source ${base} \
				 val_source ${val} \
				 test_source ${test} \
				 val_episodes $(n_episodes) \
				 eval_mode $(mode) \
				 visu ${visu} \
				 val_batch_size 1 \
				 load_from_timm $(load_from_timm) ;\

# ============= Pre-made recipes ===================

benchmark:
	for method in simpleshot bd_cspn tim_gd finetune; do \
		for source in omniglot aircraft cu_birds dtd quickdraw fungi traffic_sign mscoco; do \
		    make method=$${method} test=$${source} eval;\
		done ;\
	done ;\


# ============= Plotting =============

train_plot:
	python -m src.plot --folder checkpoints/base=${base}/val=${val}/arch=${arch}

# ============= Communication with server =============

import_results:
	rsync -avm --include='*' --include='*.csv' --include='*.npy' --include='*.json' \
		    --include='*.pdf' --include='*/' --exclude='*'\
		$(SERVER_IP):${SERVER_PATH}/results ./

import_models:
    rsync -avm --include='*.pth' \
		 --include='*/' \
		 --exclude='*' \
		  $(SERVER_IP):${SERVER_PATH}/checkpoints/ ./checkpoints/


deploy_models:
	rsync -avm --include='*.pth' \
			   --include='*.pth.tar' \
			   --include='*/' \
			   --exclude='*' \
			   ./checkpoints/ $(SERVER_IP):${SERVER_PATH}/checkpoints/


deploy_code:
	rsync -av --exclude plots \
			  --exclude checkpoints/ \
			  --exclude archive/ \
			  --exclude .git \
			  --exclude logs \
			  --exclude results \
			  --exclude *.sublime-project \
			  --exclude *.sublime-workspace \
			  --exclude __pycache__ \
			   ./ $(SERVER_IP):${SERVER_PATH}/

tar_and_deploy_data:
	find ${RECORDS} ! -wholename ${RECORDS} -type d -exec tar -czv -f {}.tar.gz {} \; ;\
	find ${RECORDS} -name '*.tar.gz' -exec rsync -av {} ${SERVER_IP}:${SERVER_PATH}/data/ \; ;\


# ============= Prepare data =============

ilsvrc_2012:
	# Assumes you already have downloaded the ILSVRC2012_img_train.tar file
	mkdir ${DATASRC}/ILSVRC2012_img_train ;\
	tar -xvf /ssd/download/ILSVRC2012_img_train.tar -C ${DATASRC}/ILSVRC2012_img_train ;\
	# Execute the following in the directory ${DATASRC}/ILSVRC2012_img_train
# 	find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
	# Finally, execute what follows
	$(exec) -m src.datasets.conversion.convert_datasets_to_records \
	  --dataset=ilsvrc_2012 \
	  --ilsvrc_2012_data_root=${DATASRC}/ILSVRC2012_img_train \
	  --splits_root=${SPLITS} \
	  --records_root=${RECORDS}

ilsvrc_2012_v2:
	$(exec) -m src.datasets.conversion.convert_datasets_to_records \
	  --dataset=ilsvrc_2012_v2 \
	  --ilsvrc_2012_data_root=${DATASRC}/ILSVRC2012_img_train \
	  --splits_root=${SPLITS} \
	  --records_root=${RECORDS}

omniglot:
	$(exec) -m src.datasets.conversion.convert_datasets_to_records \
	  --dataset=omniglot \
	  --omniglot_data_root=${DATASRC}/omniglot \
	  --splits_root=${SPLITS} \
	  --records_root=${RECORDS}

aircraft:
	$(exec) -m src.datasets.conversion.convert_datasets_to_records \
	  --dataset=aircraft \
	  --aircraft_data_root=${DATASRC}/fgvc-aircraft-2013b \
	  --splits_root=${SPLITS} \
	  --records_root=${RECORDS}

cub:
	$(exec) -m src.datasets.conversion.convert_datasets_to_records \
	  --dataset=cu_birds \
	  --cu_birds_data_root=${DATASRC}/CUB_200_2011 \
	  --splits_root=${SPLITS} \
	  --records_root=${RECORDS}

dtd:
	$(exec) -m src.datasets.conversion.convert_datasets_to_records \
	  --dataset=dtd \
	  --dtd_data_root=${DATASRC}/dtd \
	  --splits_root=${SPLITS} \
	  --records_root=${RECORDS}

quickdraw:
	$(exec) -m src.datasets.conversion.convert_datasets_to_records \
	  --dataset=quickdraw \
	  --quickdraw_data_root=${DATASRC}/quickdraw \
	  --splits_root=${SPLITS} \
	  --records_root=${RECORDS}

fungi:
	$(exec) -m src.datasets.conversion.convert_datasets_to_records \
	  --dataset=fungi \
	  --fungi_data_root=${DATASRC}/fungi \
	  --splits_root=${SPLITS} \
	  --records_root=${RECORDS}

vgg:
	$(exec) -m src.datasets.conversion.convert_datasets_to_records \
	    --dataset=vgg_flower \
	    --vgg_flower_data_root=${DATASRC}/vgg_flower \
	    --splits_root=${SPLITS} \
	    --records_root=${RECORDS}

traffic_sign:
	$(exec) -m src.datasets.conversion.convert_datasets_to_records \
	  --dataset=traffic_sign \
	  --traffic_sign_data_root=${DATASRC}/GTSRB \
	  --splits_root=${SPLITS} \
	  --records_root=${RECORDS}

mscoco:
	cd ${DATASRC}/mscoco/ mkdir train2017
	gsutil -m rsync gs://images.cocodataset.org/train2017 train2017
	gsutil -m cp gs://images.cocodataset.org/annotations/annotations_trainval2017.zip
	unzip annotations_trainval2017.zip
	$(exec) -m src.datasets.conversion.convert_datasets_to_records \
	  --dataset=mscoco \
	  --mscoco_data_root=${DATASRC}/mscoco \
	  --splits_root=${SPLITS} \
	  --records_root=${RECORDS}

mini:
	$(exec) -m src.datasets.conversion.convert_datasets_to_records \
	  --dataset=mini_imagenet \
	  --mini_imagenet_data_root=${DATASRC}/mini_imagenet \
	  --splits_root=${SPLITS} \
	  --records_root=${RECORDS} \
	cp ${RECORDS}/mini_imagenet/dataset_spec.json ${DATASRC}/mini_imagenet ;\


tiered:
	cp ${RECORDS}/tiered_imagenet/dataset_spec.json ${DATASRC}/tiered_imagenet ;\
      $(exec) -m src.datasets.conversion.make_global_labels \
		     --split_root=${DATASRC}/tiered_imagenet/splits/ \
      $(exec) -m src.datasets.conversion.convert_datasets_to_records \
	 --dataset=tiered_imagenet \
	 --tiered_imagenet_data_root=${DATASRC}/tiered_imagenet \
	 --splits_root=${SPLITS} \
	 --records_root=${RECORDS} \


indexes:
	for source in ilsvrc_2012_v2; do \
		source_path=${RECORDS}/$${source} ;\
		find $${source_path} -name '*.tfrecords' -type f -exec sh -c '$(exec)3 -m tfrecord.tools.tfrecord2idx $$2 $${2%.tfrecords}.index' sh $${source_path} {} \; ;\
	done ;\


# ============= Archive results =============


store: # Archive experiments
	python src/utils/list_files.py results/ archive/ tmp.txt
	{ read -r out_files; read -r archive_dir; } < tmp.txt ; \
	for file in $${out_files}; do \
		cp -Rv $${file} $${archive_dir}/ ; \
	done
	rm tmp.txt


restore: # Restore experiments to output/
	python src/utils/list_files.py archive/ results/ tmp.txt ; \
	read -r out_files < tmp.txt ; \
	mkdir -p results/$${folder[1]} ; \
	for file in $${out_files}; do \
		cp -Rv $${file} results/$${folder[1]}/ ; \
	done
	rm tmp.txt
