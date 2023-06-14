# Perform Align pretraining on k150. PRETRAINED_CHECKPOINT can be specified or set to None.

BATCH_SIZE=128 && NUM_NODES=8 && NOMINAL_BATCH=256 && PRETRAINED_CHECKPOINT=??? && EXP_DIR=??? &&
python train_app_submitit.py hydra.job.chdir=False +experiments=omnimae/omnimain_train_video_ft.yaml \
++submitit.use_cluster=true ++launcher.gpus_per_node=1 ++launcher.num_nodes=${NUM_NODES} \
++num_workers=1 ++batch_size=$((BATCH_SIZE/NUM_NODES)) \
++lr_scale_factor=$((NOMINAL_BATCH/BATCH_SIZE)) ++epochs=50 trainer/model=vitbase_ft \
++pretrained_checkpoint_path=${PRETRAINED_CHECKPOINT} \
trainer/checkpoint=no_heads_no_decoder.yaml +trainer=mae_ft_k150_config.yaml ++video_labels=150 \
++launcher.experiment_log_dir=${EXP_DIR}

# To run on inpk150 (No-Human Kinetics), set trainer=mae_inpk150_config.yaml
# To run on sim150 (Synthetic), set trainer=mae_sim150_config.yaml
# To run on inpk150 + sim150 (cotraining), set trainer=mae_ft_inpk150_sim150_config.yaml