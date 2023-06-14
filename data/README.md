# PPMA Benchmark
This repository contains instructions needed to recreate the benchmark detailed in the paper _Learning Human Action Recognition Representations
Without Real Humans_. Specifically, we discuss how to download the two pretraining datasets and six downstream evaluation datasets.

![](.github/main_figure.png "Main Figure")
*Fig 1: Schematic of Benchmark*

## Pretraining Datasets
### No-Human Kinetics
We run the HAT framework (https://github.com/princetonvisualai/HAT) to remove humans from the Kinetics dataset. The Kinetics-400 dataset can be downloaded [here](https://www.deepmind.com/open-source/kinetics). We run HAT on a subset of Kinetics that is 150 classes as indicated in ```splits/kinetics``` to created No-Human Kinetics, a dataset of 150 classes consisting of videos with humans removed from each frame through inpainting. We use No-Human Kinetics to pretraining action recognition models.

See this [google doc](https://docs.google.com/document/d/1aeKKv5ZcIvv3uXjd_rlbMYvC97KBq63rUlET5pjQ44M/edit?usp=sharing) for detailed instructions on running HAT.

The pretrained segmentation and inpainting models we use can be found in [google drive](https://drive.google.com/drive/folders/1sFgtiOJjRCRogtBM7b_NiPcGUyh0yvV4?usp=drive_link). 
### Synthetic Data
We use the same Synthetic dataset of 150 classes from SynAPT: https://github.com/mintjohnkim/SynAPT. Please follow instructions from that github.


## Downstream Evaluation Datasets
|   Downstream Dataset | Download Instructions                                                         |
|-----------|-------------------------------------------------------------------------------|
| UCF101    | https://www.crcv.ucf.edu/data/UCF101.php                                      |
| HMDB51    | https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/ |
| Mini-SSV2 | https://developer.qualcomm.com/software/ai-datasets/something-something       |
| Diving48  | http://www.svcl.ucsd.edu/projects/resound/dataset.html                        |
| IkeaFA    | https://tengdahan.github.io/ikea.html                                         |
| UAV-Human | https://github.com/sutdcv/UAV-Human                                           |

Note: Mini-SSV2 is a subset of the Something-Something V2 dataset found in the link above. After downloading, you can access the subset used for train and validation in ```splits/mini_ssv2```.


## More Information
- ```splits/``` folder contains train/val splits used to evaluate the models. Files are stored both as txt or csv for readability and npy file of either an array of video path or array of class labels. 


