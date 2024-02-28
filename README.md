# install PyTorch

https://pytorch.org/get-started/locally/
```
pip install torch torchvision torchaudio 
```

# proceed with setting up the remaining dependencies
```
pip install -r requirements.txt
```
Or alternatively
```
pip install lightning tensorboard torch-tb-profiler opencv-python pandas  controlnet-aux tqdm  torchpack onnx onnxsim segment_anything monai  prodigyopt torchmetrics[multimodal]

pip install datasets transformers accelerate diffusers
```


# copy EfficientViT-L2-SAM checkpoint
from https://huggingface.co/han-cai/efficientvit-sam/blob/main/l2.pt to efficientvit/assets/checkpoints/sam/l2.pt

# finetune SAM model for subject extraction
```
python segmenter_training_subject.py
```
Optionally run tensorboard (ex)
```
tensorboard --logdir sam_subject/lightning_logs/
```

# finetune SAM model for subject's head and neck extraction
```
python segmenter_training_head.py
```

# finetune SAM model for subject's clothes extraction
```
python segmenter_training_clothes.py
```

# finetune SAM model for subject's body extraction
```
python segmenter_training_body.py
```

# create the dataset from videos (optional
```
python extract_dataset.py
```