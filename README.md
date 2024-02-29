# Dependecies
## Install PyTorch

https://pytorch.org/get-started/locally/
```
pip install torch torchvision torchaudio 
```

## Proceed with setting up the remaining dependencies
```
pip install -r requirements.txt
```
Or alternatively
```
pip install lightning tensorboard torch-tb-profiler opencv-python pandas  controlnet-aux tqdm  torchpack onnx onnxsim segment_anything monai  prodigyopt torchmetrics[multimodal]

pip install datasets transformers accelerate diffusers
```

# Segmentation
## Copy EfficientViT-L2-SAM checkpoint
from https://huggingface.co/han-cai/efficientvit-sam/blob/main/l2.pt or https://huggingface.co/andrei-ace/EdgeStyle/tree/main/sam to efficientvit/assets/checkpoints/sam/l2.pt

## Download pretrained segmentation models from 
https://huggingface.co/andrei-ace/EdgeStyle/tree/main/sam
```
efficientvit/assets/checkpoints/sam/trained_model_subject.pt
efficientvit/assets/checkpoints/sam/trained_model_body.pt
efficientvit/assets/checkpoints/sam/trained_model_clothes.pt
efficientvit/assets/checkpoints/sam/trained_model_head.pt
```
## Alternatively finetune 
### Finetune SAM model for subject extraction
```
python segmenter_training_subject.py
```
Optionally run tensorboard (ex)
```
tensorboard --logdir sam_subject/lightning_logs/
```

### Finetune SAM model for subject's head and neck extraction
```
python segmenter_training_head.py
```

### Finetune SAM model for subject's clothes extraction
```
python segmenter_training_clothes.py
```

### Finetune SAM model for subject's body extraction
```
python segmenter_training_body.py
```

# Dataset
## Create the dataset from videos (optional)
```
python extract_dataset.py
```

# Training
## Start training for 20k steps
```
accelerate launch train_text2image_pretrained_openpose.py \
--pretrained_model_name_or_path="SG161222/Realistic_Vision_V5.1_noVAE" \
--pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
--pretrained_openpose_name_or_path="lllyasviel/control_v11p_sd15_openpose" \
--output_dir="models/output_text2image_pretrained_openpose" \
--resolution=512 \
--train_batch_size=2 \
--gradient_accumulation_steps 32 \
--mixed_precision fp16 \
--controllora_use_vae \
--seed=42 \
--resume_from_checkpoint latest \
--num_validation_images 4 \
--checkpoints_total_limit 3 \
--dataloader_num_workers 2 \
--snr_gamma=5.0 \
--optimizer="prodigy" \
--learning_rate=1.0 \
--prodigy_safeguard_warmup=True \
--prodigy_use_bias_correction=True \
--adam_beta1=0.9 \
--adam_beta2=0.99 \
--adam_weight_decay=0.01 \
--proportion_empty_prompts=0.1 \
--proportion_empty_images=0.1 \
--proportion_cutout_images=0.1 \
--proportion_patchworked_images=0.1 \
--proportion_patchworks=0.1 \
--validation_steps 100 \
--checkpointing_steps 100 \
--max_train_steps=20000
```

## Check the training status using tensorboard

### training loss
![alt text](docs/train_loss.svg)

### training learning rate
![alt text](docs/train_lr.svg)

### Example image logged (after 15.7k steps)
Left corner the groud truth, followed by the images used by the controlnets:
* agnostic image
* first outfit image
* second outfit image
* The openpose images are ommited

![alt text](docs/1.png)
![alt text](docs/2.png)
![alt text](docs/3.png)
![alt text](docs/4.png)