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