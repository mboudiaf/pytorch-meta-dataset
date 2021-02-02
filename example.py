import torch
import sacred
from pytorch_meta_dataset.utils import Split
import pytorch_meta_dataset.config as config_lib
import pytorch_meta_dataset.dataset_spec as dataset_spec_lib
from pytorch_meta_dataset.config import dataset_ingredient
from torch.utils.data import DataLoader
import os
import pytorch_meta_dataset.pipeline as pipeline

ex = sacred.Experiment('Model training',
                       ingredients=[dataset_ingredient])


@ex.automain
def main():

    # Define your device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Recovering configurations
    data_config = config_lib.DataConfig()
    episod_config = config_lib.EpisodeDescriptionConfig()

    # Get the data specifications
    datasets = data_config.sources
    if episod_config.num_ways:
        if len(datasets) > 1:
            raise ValueError('For fixed episodes, not tested yet on > 1 dataset')
        use_dag_ontology_list = [False]
    else:
        use_bilevel_ontology_list = [False]*len(datasets)
        # Enable ontology aware sampling for Omniglot and ImageNet.
        if 'omniglot' in datasets:
            use_bilevel_ontology_list[datasets.index('omniglot')] = True
        if 'imagenet' in datasets:
            use_bilevel_ontology_list[datasets.index('imagenet')] = True

        use_bilevel_ontology_list = use_bilevel_ontology_list
        use_dag_ontology_list = [False]*len(datasets)
    episod_config.use_bilevel_ontology_list = use_bilevel_ontology_list
    episod_config.use_dag_ontology_list = use_dag_ontology_list

    all_dataset_specs = []
    for dataset_name in datasets:
        dataset_records_path = os.path.join(data_config.path, dataset_name)
        dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)
        all_dataset_specs.append(dataset_spec)

    # Form an episodic dataset
    split = Split["TRAIN"]
    episodic_dataset = pipeline.make_episode_pipeline(dataset_spec_list=all_dataset_specs,
                                                      split=split,
                                                      data_config=data_config,
                                                      episode_descr_config=episod_config)

    #  If you want to get the total number of classes (i.e from combined datasets)
    num_classes = sum([len(d_spec.get_classes(split=split)) for d_spec in all_dataset_specs])
    print(f"=> There are {num_classes} classes in the combined datasets")

    # Use a standard dataloader
    episodic_loader = DataLoader(dataset=episodic_dataset,
                                 batch_size=1,
                                 num_workers=data_config.num_workers)

    # Training or validation loop
    for i, (support, query, support_labels, query_labels) in enumerate(episodic_loader):
        support, support_labels = support.to(device), support_labels.to(device, non_blocking=True)
        query, query_labels = query.to(device), query_labels.to(device, non_blocking=True)
        # Do some operations
        print("=> Example of episode")
        print("Number of ways: {}   Support size: {}   Query Size: {} \n".format(
                            support_labels.unique().size(0),
                            list(support.size()),
                            list(query.size())))
        break

    # Form a batch dataset
    batch_dataset = pipeline.make_batch_pipeline(dataset_spec_list=all_dataset_specs,
                                                 split=split,
                                                 data_config=data_config)

    # Use a standard dataloader
    batch_loader = DataLoader(dataset=batch_dataset,
                              batch_size=data_config.batch_size,
                              num_workers=data_config.num_workers)
    # Training or validation loop
    for i, (input, target) in enumerate(batch_loader):
        input, target = input.to(device), target.long().to(device, non_blocking=True)
        # Do some operations
        print("=> Example of a batch")
        print(f"Shape of batch: {list(input.size())}")
        break
