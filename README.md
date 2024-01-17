#

https://pytorch.org/get-started/locally/
```
pip install torch torchvision torchaudio 
```

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
Optionally run tensorboard
```
tensorboard --logdir sam_subject/lightning_logs/
```