batch_size=150
val_batch_size=50
num_epochs=2000
# num_epochs=200
mse_mult=3e2
mae_mult=3e1
info_nce_mult=1e3
cos_mult=1e5
# mse_mult=2e1
# mae_mult=1e1
# nce_mult=1e-5
# info_nce_mult=1e0
rec_mult=1
cyc_mult=1
max_lr=5e-4
mixup_pct=0.2
n_blocks=4
h_size=2048
trainer_select="trainer_fmri_img_GIT"
model_name="MindBrige_image_GIT_ViT_msencecos_largencecos_2"
clip_variant="GIT-ViT"

cd src/
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
# export NCCL_NET=Socket
export WANDB_API_KEY='cd2737b4c9369aee69cbbfdbad6813d2b53448d6'

# # to use gloo backend
# export MASTER_ADDR='localhost'
# export MASTER_PORT='29502'

# CUDA_VISIBLE_DEVICES=4 python -W ignore \
accelerate launch --num_processes 1 --gpu_ids 0 --main_process_port 29502 \
main.py \
--wandb_project "MindBrige_image_GIT_ViT_token" --trainer_select $trainer_select --model_name $model_name --clip_variant $clip_variant \
--subj_list 1 2 5 7 --num_epochs $num_epochs --batch_size $batch_size --val_batch_size $val_batch_size \
--h_size $h_size --n_blocks $n_blocks --pool_type max --pool_num 8192 \
--cos_mult $cos_mult --mse_mult $mse_mult --mae_mult $mae_mult --info_nce_mult $info_nce_mult --rec_mult $rec_mult --cyc_mult $cyc_mult \
--eval_interval 1 --ckpt_interval 5 \
--max_lr $max_lr --num_workers 0 --mixup_pct $mixup_pct
