accelerate launch train_text2image_vaecontrolnet.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --output_dir="models/output_text2image_prodigy_vae_experiment" \
 --resolution=512 \
 --train_batch_size=2 \
 --gradient_accumulation_steps 2 \
 --mixed_precision fp16 \
 --seed=42 \
 --resume_from_checkpoint latest \
 --num_validation_images 4 \
 --checkpoints_total_limit 3 \
 --optimizer="prodigy" \
 --learning_rate=1.0 \
 --prodigy_safeguard_warmup=True \
 --prodigy_use_bias_correction=True \
 --adam_beta1=0.9 \
 --adam_beta2=0.99 \
 --adam_weight_decay=0.01 \
 --snr_gamma=5.0 \
 --proportion_empty_prompts=0.5 \
 --proportion_patchworks=0.5 \
 --validation_steps 2500 \
 --checkpointing_steps 5000 \
 --max_train_steps=10000


 #--lr_scheduler "cosine_annealing" \
 #--proportion_empty_prompts=0.5 \
 #--proportion_patchworks=0.5 \
