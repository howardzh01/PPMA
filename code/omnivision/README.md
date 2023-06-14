

Training pipeline for PPMA. We adapt this from the code base used for Omnivore and OmniMAE. 

## Installation
Omnivision requires Python 3.9. To install PyTorch 1.12 with CUDA 11.3 on Python 3.9 via conda, run the following instructions -

```bash
conda create --name ov python=3.9
conda activate ov
conda install pytorch=1.12 torchvision=0.13 cudatoolkit=11.3 -c pytorch
```

Install Omnivision in developer mode (where it runs code from your local checkout and picks up your changes) by running -
```bash
git clone https://github.com/howardzh01/omnivore.git
cd PPMA/code
pip install -e ".[dev]"
```

## Testing
Before running the tests, please ensure that you installed the necessary additional test dependencies.

Use the the following command to run the tests:
```bash
# run all tests
python -m unittest discover -v -s tests -t .
# run a specific test
python -m unittest tests.test_scheduler
```

## Data preparation
All our dataloaders rely on `.npy` numpy array files for the meta data.

For all datasets, please create two seperate `.npy` files for each split(train and val), such that there is one file consisting of a 1D arrays of image/video paths and another file consisting of the corresponding labels. 

Post that, update the `config/experiments/dataset_catalog.yaml` file with the paths to your newly created  `.npy` files for each dataset.

For instance, a sample numpy file for images or depth images or videos would look like this,
```
array(['/path_to_sample_1.JPEG', # .mp4, .avi, .png any such extensions are supported based on the data type.
       '/path_to_sample_2.JPEG',
       '/path_to_sample_3.JPEG',
       '/path_to_sample_4.JPEG',
       '/path_to_sample_5.JPEG',
       dtype='<U75')
```

And a sample numpy file for labels would look like this,
```
array([86, 2, 34, 48, 51]) # consisting of integer labels.
```

## Usage
All our given configs are designed to work on SLURM. We tested our configs with V100 and A100 GPUS with at least 32GB.
For locally running the configs and for quick debuging, append the following lines to your job commands.

```
submitit.use_cluster=false launcher.gpus_per_node=1 launcher.num_nodes=1
```

Additionally, update the SLURM config in `config/experiments/base.yaml` to reflect your enviroments partitions, constraints, etc.



### OmniMAE
```
python train_app_submitit.py hydra.job.chdir=False hydra.output_subdir=null \
+experiments=omnimae/omnimain_train_video.yaml \
++submitit.use_cluster=true ++launcher.gpus_per_node=1 \
++launcher.num_nodes=${NUM_NODES} ++num_workers=6 ++batch_size=$((BATCH_SIZE/NUM_NODES)) \
++lr_scale_factor=$((NOMINAL_BATCH/BATCH_SIZE)) ++epochs=200 trainer/model=vitbase +trainer=mae_k150_config.yaml \
++launcher.experiment_log_dir=${EXP_DIR}
```

Above is an example command. We must specify ```BATCH_SIZE, NUM_NODES, NOMINAL_BATCH```. 
- It is crucial to choose  `BATCH_SIZE` and `NUM_NODES` such that `batch_size`, the batch size per node, will fit in memory.
- We scale learning rate linearly with batch size through `lr_scale_factor`, and changing the `NOMINAL_BATCH` or `BATCH_SIZE` will change the scaling factor.
- Experiment configurations are found in `experiments=omnimae/omnimain_train_video.yaml.` This includes base learning rate.
- Model configurations are found in `trainer/model=vitbase.`
- The data configurations are found in `trainer=mae_k150_config.yaml.` One can specify different dataset configs found in `config/trainer`.


For stage 1 of MAE pretraining, use
```
sh utils/run_scripts/run_mae.sh
```

For stage 2 of Alignment pretraining, use
```
sh utils/run_scripts/run_align.sh
```

For evaluating pretrained models on the downstream tasks through finetuning, use
```
sh utils/run_scripts/run_downstream_ft.sh
```

For evaluating pretrained models on the downstream tasks through linear probing, use
```
sh utils/run_scripts/run_downstream_lp.sh
```

For evaluating averages of pretrained models on the downstream tasks through linear probing, use
```
sh utils/run_scripts/run_downstream_average_lp.sh
```


