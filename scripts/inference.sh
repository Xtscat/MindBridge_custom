subj_load=1
subj_test=1
model_name="MindBrige_text_infonce_diva"
# model_name="MindBrige_text_only_mixco_loss_diffusers_vit_norm"
ckpt_from="last"
text_image_ratio=0.5
guidance=5
gpu_id=0

cd src/

# CUDA_VISIBLE_DEVICES=$gpu_id python -W ignore \
# recon.py \
# --model_name $model_name --ckpt_from $ckpt_from \
# --h_size 2048 --n_blocks 4 --pool_type max \
# --subj_load $subj_load --subj_test $subj_test \
# --text_image_ratio $text_image_ratio --guidance $guidance \
# --recons_per_sample 8 --pool_num 8192 --num_inference_steps 20

# --test_end 2 \
# --test_start 0 \


# results_path="/media/SSD_1_2T/xt/MindBridge/train_logs/VD_text_img_infonce_guidance5_ratio0.5/recon_on_subj1/"
# results_path="/media/SSD_1_2T/xt/MindBridge/train_logs/VD_text_img_infonce_multicoco_guidance5_ratio0.4/recon_on_subj1/"
# results_path="/media/SSD_1_2T/xt/MindBridge/train_logs/Postprocess_new_coco_caption_strength0.8_guidance5/"
# results_path="/media/SSD_1_2T/xt/MindBridge/train_logs/Postprocess_strength0.4_guidance5/"
results_path="/media/SSD_1_2T/xt/MindBridge/train_logs/VD_text_img_multicoco_multilayer_16_23/recon_on_subj1/"

CUDA_VISIBLE_DEVICES=$gpu_id python -W ignore \
eval.py --results_path $results_path
