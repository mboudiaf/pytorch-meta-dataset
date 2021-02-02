from . import pipeline
from torch.utils.data import DataLoader


def get_episodic_dataloader(split,
                            all_dataset_specs,
                            episod_config,
                            data_config):
    dataset = pipeline.make_episode_pipeline(dataset_spec_list=all_dataset_specs,
                                             split=split,
                                             data_config=data_config,
                                             episode_descr_config=episod_config)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=1,
                             num_workers=data_config.num_workers)
    return data_loader


def get_batch_dataloader(split,
                         all_dataset_specs,
                         data_config):
    dataset = pipeline.make_one_source_batch_pipeline(dataset_spec_list=all_dataset_specs,
                                                      data_config=data_config,
                                                      split=split)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=data_config.batch_size,
                             num_workers=data_config.num_workers)

    return data_loader