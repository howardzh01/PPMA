# Perform MAE pretraining on k150.
BATCH_SIZE=128 && NUM_NODES=8 && NOMINAL_BATCH=256 && EXP_DIR=??? &&
python train_app_submitit.py hydra.job.chdir=False hydra.output_subdir=null \
+experiments=omnimae/omnimain_train_video.yaml \
++submitit.use_cluster=true ++launcher.gpus_per_node=1 \
++launcher.num_nodes=${NUM_NODES} ++num_workers=6 ++batch_size=$((BATCH_SIZE/NUM_NODES)) \
++lr_scale_factor=$((NOMINAL_BATCH/BATCH_SIZE)) ++epochs=200 trainer/model=vitbase +trainer=mae_k150_config.yaml \
++launcher.experiment_log_dir=${EXP_DIR}

# To run on inpk150 (No-Human Kinetics), set trainer=mae_inpk150_config.yaml
# To run on sim150 (Synthetic), set trainer=mae_sim150_config.yaml