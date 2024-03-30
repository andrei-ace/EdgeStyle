# misc
import numpy as np
import os
import argparse
from PIL import Image, ImageDraw, ImageFont

# torch
import torch
from torchvision import transforms
from diffusers import (
    AutoencoderKL,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
    ControlNetModel,
)
from diffusers.optimization import get_scheduler
from transformers import AutoTokenizer, CLIPTextModel, CLIPModel, CLIPProcessor

# local
from model.controllora import ControlLoRAModel
from model.utils import BestEmbeddings
from model.edgestyle_multicontrolnet import EdgeStyleMultiControlNetModel

RESOLUTION = 512

NUM_IMAGES = 6

IMAGES_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize(
            RESOLUTION, interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.CenterCrop(RESOLUTION),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

CONDITIONING_IMAGES_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize(
            RESOLUTION, interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.CenterCrop(RESOLUTION),
        transforms.ToTensor(),
    ]
)

CONTROLNET_PATTERN = [0, None, 1, None, 1, None]

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

best_embeddings = BestEmbeddings(model, processor)


def add_text_to_image(image, text):
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), text, (255, 255, 255))
    return image


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Simple example of a ControlNet training script."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_name_or_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained VAE model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--pretrained_openpose_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained openpose model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        required=False,
        help="Negative prompt to use.",
    )
    parser.add_argument(
        "--prompt_text_to_add",
        type=str,
        default="",
        required=False,
        help="Add text to prompt.",
    )

    parser.add_argument(
        "--source_path",
        type=str,
        default=None,
        required=True,
        help="Path to source image.",
    )
    parser.add_argument(
        "--source_image_name",
        type=str,
        default=None,
        required=True,
        help="Name of the source image.",
    )
    parser.add_argument(
        "--target_path",
        type=str,
        default=None,
        required=True,
        help="Path to target image.",
    )
    parser.add_argument(
        "--target_image_name",
        type=str,
        default=None,
        required=True,
        help="Name of the target image.",
    )
    parser.add_argument(
        "--target_path2",
        type=str,
        default=None,
        required=True,
        help="Path to second target image.",
    )
    parser.add_argument(
        "--target_image_name2",
        type=str,
        default=None,
        required=True,
        help="Name of the second target image.",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default=None,
        required=True,
        help="Path to save the generated image.",
    )

    parser.add_argument(
        "--image_result_name",
        type=str,
        default=None,
        required=True,
        help="Name of the generated image.",
    )
    parser.add_argument(
        "--use_agnostic_images",
        action="store_true",
        help=(
            "Feed agnostic images into the controlnet as input. In the absence of this setting, "
            "the controlnet defaults to utilizing images that exclusively feature the subject's head."
        ),
    )
    parser.add_argument(
        "--controllora_use_vae",
        action="store_true",
        default=False,
        help=("Whether to use the VAE in the controlnet."),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        use_fast=False,
    )

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
    )
    if args.pretrained_vae_name_or_path is not None:
        vae = AutoencoderKL.from_pretrained(args.pretrained_vae_name_or_path)
    else:
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
        )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=weight_dtype,
    )

    openpose = ControlNetModel.from_pretrained(
        args.pretrained_openpose_name_or_path,
        torch_dtype=weight_dtype,
    )

    controlnet = EdgeStyleMultiControlNetModel.from_pretrained(
        args.controlnet_model_name_or_path,
        vae=vae if args.controllora_use_vae else None,
        controlnet_class=ControlLoRAModel,
        load_pattern=CONTROLNET_PATTERN,
        static_controlnets=[None, openpose, None, openpose, None, openpose],
    )
    for net in controlnet.nets:
        if net is not openpose:
            net.tie_weights(unet)

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    generator = torch.Generator(device).manual_seed(42)
    pipeline = pipeline.to(device)

    # Load image
    subject = Image.open(
        os.path.join(args.source_path, "subject", args.source_image_name)
    )
    target = Image.open(
        os.path.join(args.target_path, "subject", args.target_image_name)
    )
    target2 = Image.open(
        os.path.join(args.target_path2, "subject", args.target_image_name2)
    )

    agnostic = Image.open(
        os.path.join(args.source_path, "agnostic", args.source_image_name)
    )

    head = Image.open(os.path.join(args.source_path, "head", args.source_image_name))

    agnostic_or_head = agnostic if args.use_agnostic_images else head

    original_openpose = Image.open(
        os.path.join(args.source_path, "openpose", args.source_image_name)
    )

    clothes = Image.open(
        os.path.join(args.target_path, "clothes", args.target_image_name)
    )

    clothes_openpose = Image.open(
        os.path.join(args.target_path, "openpose", args.target_image_name)
    )

    clothes2 = Image.open(
        os.path.join(args.target_path2, "clothes", args.target_image_name2)
    )

    clothes_openpose2 = Image.open(
        os.path.join(args.target_path2, "openpose", args.target_image_name2)
    )

    prompts = best_embeddings([clothes])

    guidance_scales = np.linspace(1.0, 7.0, NUM_IMAGES)

    images = [
        subject,
        target,
        target2,
    ]

    for i in range(NUM_IMAGES):
        with torch.autocast("cuda"):
            image = pipeline(
                prompt=prompts[0] + " " + args.prompt_text_to_add,
                guidance_scale=guidance_scales[i],
                image=[
                    (
                        IMAGES_TRANSFORMS(agnostic_or_head).unsqueeze(0)
                        if args.controllora_use_vae
                        else CONDITIONING_IMAGES_TRANSFORMS(agnostic_or_head).unsqueeze(
                            0
                        )
                    ),
                    CONDITIONING_IMAGES_TRANSFORMS(original_openpose).unsqueeze(0),
                    (
                        IMAGES_TRANSFORMS(clothes).unsqueeze(0)
                        if args.controllora_use_vae
                        else CONDITIONING_IMAGES_TRANSFORMS(clothes).unsqueeze(0)
                    ),
                    CONDITIONING_IMAGES_TRANSFORMS(clothes_openpose).unsqueeze(0),
                    (
                        IMAGES_TRANSFORMS(clothes2).unsqueeze(0)
                        if args.controllora_use_vae
                        else CONDITIONING_IMAGES_TRANSFORMS(clothes2).unsqueeze(0)
                    ),
                    CONDITIONING_IMAGES_TRANSFORMS(clothes_openpose2).unsqueeze(0),
                ],
                # controlnet_conditioning_scale=[1, 1, 1, 1],
                # control_guidance_start=0.0,
                # control_guidance_end=0.9,
                negative_prompt=args.negative_prompt,
                num_inference_steps=50,
                generator=generator,
            ).images[0]
        image = add_text_to_image(image, f"Guidance scale: {guidance_scales[i]:.2f}")
        images.append(image)

    image = image_grid(images, 3, len(images) // 3)

    image.save(os.path.join(args.result_path, args.image_result_name))


if __name__ == "__main__":
    args = parse_args()
    main(args)
