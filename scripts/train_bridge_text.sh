batch_size=250
val_batch_size=50
num_epochs=500
mse_mult=1e6
mae_mult=1e3
rec_mult=1
nce_mult=1e0
cyc_mult=1
max_lr=5e-4
mixup_pct=0.2
trainer_select="trainer_fmri_text"
model_name="MindBrige_text_infonce"

cd src/
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
# export NCCL_NET=Socket
export WANDB_API_KEY='cd2737b4c9369aee69cbbfdbad6813d2b53448d6'

# CUDA_VISIBLE_DEVICES=4 python -W ignore \
accelerate launch --num_processes 1 --gpu_ids 0 --main_process_port 29502 \
main.py \
--wandb_project "MindBrige_text_infonce" --trainer_select $trainer_select --model_name $model_name \
--subj_list 1 2 5 7 --num_epochs $num_epochs --batch_size $batch_size --val_batch_size $val_batch_size \
--h_size 2048 --n_blocks 4 --pool_type max --pool_num 8192 \
--mse_mult $mse_mult --mae_mult $mae_mult --rec_mult $rec_mult --cyc_mult $cyc_mult \
--eval_interval 1 --ckpt_interval 5 \
--max_lr $max_lr --num_workers 0 --nce_mult $nce_mult --mixup_pct $mixup_pct
