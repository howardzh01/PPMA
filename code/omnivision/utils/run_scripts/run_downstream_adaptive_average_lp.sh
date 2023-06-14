# Perform Adaptive Average Downstream LP on ucf101.
# PRETRAINED_CHECKPOINT must be specified and should contain an array of two model weight dictionaries.

BATCH_SIZE=32 && NUM_NODES=4 && NOMINAL_BATCH=32 && PRETRAINED_CHECKPOINT=??? && EXP_DIR=??? &&
python train_app_submitit.py hydra.job.chdir=False hydra.output_subdir=null \
+experiments=omnimae/omnimain_train_video_ensemble_ft.yaml \
++trainer.val_epoch_freq=5 ++epochs=30 ++submitit.use_cluster=true ++launcher.num_nodes=${NUM_NODES} \
++batch_size=$((BATCH_SIZE/NUM_NODES)) ++lr_scale_factor=$((NOMINAL_BATCH/BATCH_SIZE)) ++num_workers=6 \
trainer/model=vitbase_average_ft ++ensemble_num_models=2 \
++ensemble_combine_mode=learned_average_softmax mixing_lr_factor=1 \
++pretrained_checkpoint_path=${PRETRAINED_CHECKPOINT} trainer/checkpoint=default_ensemble.yaml \
+trainer=mae_ft_ucf101_config.yaml ++video_labels=101 ++launcher.gpus_per_node=1 \
++launcher.experiment_log_dir=${EXP_DIR}


# To run on other 5 downstream tasks,
# change +trainer, ++video_labels