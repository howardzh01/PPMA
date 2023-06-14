# Perform Downstream LP on ucf101. PRETRAINED_CHECKPOINT must be specified

BATCH_SIZE=32 && NUM_NODES=4 && NOMINAL_BATCH=32 && PRETRAINED_CHECKPOINT=??? && EXP_DIR=??? &&
python train_app_submitit.py hydra.job.chdir=False +experiments=omnimae/omnimain_train_video_lp.yaml \
++submitit.use_cluster=true ++launcher.gpus_per_node=1 ++launcher.num_nodes=${NUM_NODES} \
++num_workers=1 ++batch_size=$((BATCH_SIZE/NUM_NODES)) \
++lr_scale_factor=$((NOMINAL_BATCH/BATCH_SIZE)) ++epochs=30 trainer/model=vitbase_ft \
++pretrained_checkpoint_path=${PRETRAINED_CHECKPOINT} \
trainer/checkpoint=no_heads_no_decoder.yaml +trainer=mae_ft_ucf101_config.yaml ++video_labels=101 \
++launcher.experiment_log_dir=${EXP_DIR}

# To run on other 5 downstream tasks,
# change +trainer, ++video_labels