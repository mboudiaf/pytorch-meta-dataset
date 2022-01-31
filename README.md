# PyTorch META-DATASET (Few-shot classification benchmark)

This repo contains a PyTorch implementation of [meta-dataset](https://github.com/google-research/meta-dataset) and a unified implementation of some few-shot methods. This repo may be useful to you if you:

- want some pre-trained ImageNet models in PyTorch for META-DATASET;
- want to benchmark your method on META-DATASET (but do not want to mix your PyTorch code with the original TensorFlow implementation);
- are looking for a codebase to visualize few-shot episodes.

**Benefits over original code**:

1. This repo can be properly seeded, allowing to repeat the same random series of episodes if needed;
2. Data shuffling is performed without using a buffer, hence reducing the memory consumption;
3. Better results can be obtained using this repo thanks to an enhanced way of resizing images. More details in the paper.

Note that **this code also includes the original implementation** for comparison (using the PyTorch workaround proposed by the authors). If you wish to use the original implementation, set the option `loader_version: 'tf'` in [`base.yaml`](config/base.yaml#L44) (by default set to `pytorch`).

**Yet to do**:

1. Add more methods
2. Test for the multi-source setting

## Table of contents
* [Table of contents](#table-of-contents)
* [1. Setting up](#1-setting-up)
    * [1.1 Requirements](#11-requirements)
    * [1.2 Data](#12-data)
    * [1.3 Download pre-trained models](#13-download-pre-trained-models)
    * [1.4 Train models from scratch (optional)](#14-train-models-from-scratch-optional)
    * [1.5 Test your models](#15-test-your-models)
* [2. Visualization of results](#2-visualization-of-results)
    * [2.1 Training metrics](#21-training-metrics)
    * [2.2 Inference metrics](#22-inference-metrics)
    * [2.3 Visualization of episodes](#23-visualization-of-episodes)
* [3. Incorporate your own method](#3-incorporate-your-own-method)
* [4. Contributions](#4-contributions)
* [5. Citation](#5-citation)
* [6. Acknowledgements](#6-acknowledgements)


## 1. Setting up

Please carefully follow the instructions below to get started.

### 1.1 Requirements

The present code was developped and tested in Python 3.8. The list of requirements is provided in requirements.txt:
```bash
pip install -r requirements.txt
```

### 1.2 Data

To download the META-DATASET, please follow the details instructions provided at [meta-dataset](https://github.com/google-research/meta-dataset) to obtain the `.tfrecords` converted data. Once done, make sure all converted dataset are in a single folder, and execute the following script to produce index files:

```bash
bash scripts/make_records/make_index_files.sh <path_to_converted_data>
```
This may take a few minutes. Once all this is done, set the `path` variable in [`config/base.yaml`](config/base.yaml#L37) to your data folder.

### 1.3 Download pre-trained models

We provide trained Resnet-18 and WRN-2810 models on the training split of ILSVRC_2012 at [checkpoints](https://drive.google.com/file/d/1Sp7OJEK9-RKnlXjz4DEdM-9BHe1j0rtP/view?usp=sharing). All non-episodic baselines use the same checkpoint, stored in the `standard` folder. The results (averaged over 600 episodes) obtained with the provided Resnet-18 are summarized below:


|  Inductive methods     | Architecture|    ILSVRC  |    Omniglot  |   Aircraft   |    Birds   |   Textures   |  Quick Draw |   Fungi  |  VGG Flower  | Traffic Signs |    MSCOCO   |    Mean    |
|      ---      |      ---             |    ---     |       ---    |     ---      |     ---    |     ---      |    ---      |    ---   |     ---      |      ---      |     ---     |   ---      |
|      Finetune |   Resnet-18          |    59.8    |   **60.5**   |   **63.5**   |  **80.6**  |  **80.9**    |   **61.5**  |    45.2  |   **91.1**   |    **55.1**   |     41.8    |  **64.0**  |
|      ProtoNet |   Resnet-18          |    48.2    |     46.7     |     44.6     |    53.8    |    70.3      |     45.1    |    38.5  |     82.4     |      42.2     |     38.0    |    51.0    |
|    SimpleShot |   Resnet-18          |  **60.0**  |     54.2     |     55.9     |    78.6    |    77.8      |     57.4    | **49.2** |     90.3     |      49.6     |   **44.2**  |    61.7    |


|  Transductive methods  | Architecture|    ILSVRC  |    Omniglot  |   Aircraft   |    Birds   |   Textures   |  Quick Draw |   Fungi  |  VGG Flower  | Traffic Signs |    MSCOCO   |    Mean    |
|      ---      |      ---             |    ---     |       ---    |     ---      |     ---    |     ---      |    ---      |    ---   |     ---      |      ---      |     ---     |   ---      |
|     BD-CSPN   |   Resnet-18          |    60.5    |     54.4     |     55.2     |    80.9    |    77.9      |      57.3   |    50.0  |     91.7     |      47.8     |     43.9    |    62.0    |
|     TIM-GD    |   Resnet-18          |  **63.6**  |   **65.6**   |   **66.4**   |  **85.6**  |  **84.7**    |    **65.8** | **57.5** |   **95.6**   |    **65.2**   |   **50.9**  |  **70.1**  |

See Sect. 1.4 and 1.5 to reproduce these results.

### 1.4 Train models from scratch (optional)

In order to train you model from scratch, execute scripts/train.sh script:
```bash
bash scripts/train.sh <method> <architecture> <dataset>
```
`method` is to be chosen among all method specific config files in [config/](/config), `architecture` in ['resnet18', 'wideres2810'] and `dataset` among all datasets (as named by the META-DATASET converted folders). Note that the hierarchy of arguments passed to `src/train.py` and `src/eval.py` is the following: base_config < method_config < opts arguments.

**Mutiprocessing** : This code supports distributed training. To leverage this feature, set the `gpus` option accordingly (for instance `gpus: [0, 1, 2, 3]`).

### 1.5 Test your models

Once trained (or once pre-trained models downloaded), you can evaluate your model on the test split of each dataset by running:
```bash
bash scripts/test.sh <method> <architecture> <base_dataset> <test_dataset>
```
Results will be saved in `results/<method>/<exp_no>` where <exp_no> corresponds to a unique hash number of the config (you can only get the same result folder iff all hyperparameters are the same).

## 2. Visualization of results

### 2.1 Training metrics

During training, training loss and validation accuracy are recorded and saved as .npy files in the checkpoint folder. Then, you can use the src/plot.py to plot these metrics (even during training).

***Example 1***:  Plot the metrics of the standard (=non episodic) resnet-18 on ImageNet:
```bash
python -m src.plot --folder checkpoints/ilsvrc_2012/ilsvrc_2012/resnet18/standard/
```

***Example 2***: Plot the metrics of all Resnet-18 trained on ImageNet

```bash
python -m src.plot --folder checkpoints/ilsvrc_2012/ilsvrc_2012/resnet18/
```
<p align="center">
    <img src="github_figures/training_metric.png" width="500" height="500"/>
</p>

### 2.2 Inference metrics

For methods that perform **test-time optimization** (for instance MAML, TIM, Finetune, ...), method specific metrics are plotted in real-time (versus test iterations) and averaged over test epidodes, which can allow you to track unexpected behavior easily. Such metrics are implemented in `src/metrics/`, and the choice of which metric to plot is specificied through the  `eval_metrics` option in the method .yaml config file. An example with TIM method is provided below.

<p align="center">
    <img src="github_figures/inference_metrics.png" width="600" height="600"/>
</p>

### 2.3 Visualization of episodes

By setting the option `visu: True` at inference, you can visualize samples of episodes. An example of such visualization is given below:
<p align="center">
    <img src="github_figures/episode.png" width="600" height="600"/>
</p>

The samples will be saved in results/. 
All relevant optons can be found in the `base.yaml` file, in the EVAL-VISU section.

## 3. Incorporate your own method

This code was designed to allow easy incorporation of new methods. 

**Step 1**: Add your method .py file to `src/methods/` by following the template provided in src/methods/method.py. 

**Step 2**: Add import in `src/methods/__init__.py`

**Step 3**: Add your method .yaml config file including the required options `episodic_training` and `method` (name of the class corresponding to your method). Also make sure that if your method performs test-time optimization, you also properly set the option `iter` that specifies the number of optimization steps performed at inference (this argument is also used to plot the inference metrics, see section 2.2).


## 4. Contributions

Contributions are more than welcome. In particular, if you want to add methods/pre-trained models, do make a pull-request.


## 5. Citation

If you find this repo useful for your research, please consider citing the following papers:
```bibtex
@misc{boudiaf2021mutualinformation,
      title={Mutual-Information Based Few-Shot Classification}, 
      author={Malik Boudiaf and Ziko Imtiaz Masud and Jérôme Rony and Jose Dolz and Ismail Ben Ayed and Pablo Piantanida},
      year={2021},
      eprint={2106.12252},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
Additionally, do not hesitate to file issues if you encounter problems, or reach out directly to Malik Boudiaf (malik.boudiaf.1@etsmtl.net).


## 6. Acknowledgments

I thank the authors of [meta-dataset](https://github.com/google-research/meta-dataset) for releasing their code and the author of [open-source TFRecord reader](https://github.com/vahidk/tfrecord) for open sourcing an awesome Pytorch-compatible TFRecordReader ! Also big thanks to @hkervadec for his thorough code review !
