batch_size=31
val_batch_size=20
num_epochs=500
mae_mult=100
mse_mult=100
ssim_mult=300
rec_mult=10
cyc_mult=100
nce_mult=1
max_lr=5e-4
mixup_pct=0.2
trainer_select="trainer_fmri_vae"
model_name="MindBrige_vae"

cd src/
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_NET=Socket
export WANDB_API_KEY='cd2737b4c9369aee69cbbfdbad6813d2b53448d6'

accelerate launch --num_processes 1 --gpu_ids 0 --main_process_port 29502 \
main.py \
--wandb_project "MindBrige_vae" --trainer_select $trainer_select --model_name $model_name --subj_list 1 2 5 7 \
--num_epochs $num_epochs --batch_size $batch_size --val_batch_size $val_batch_size \
--h_size 4096 --n_blocks 4 --pool_type max --pool_num 8192 \
--mae_mult $mae_mult --mse_mult $mse_mult --ssim_mult $ssim_mult --rec_mult $rec_mult --cyc_mult $cyc_mult --eval_interval 1 --ckpt_interval 1 \
--max_lr $max_lr --num_workers 0 --nce_mult $nce_mult --mixup_pct $mixup_pct
