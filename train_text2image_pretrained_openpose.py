#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
import shutil
from pathlib import Path
import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision

import accelerate
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
    ControlNetModel
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version

from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from controllora import ControlLoRAModel

from diffusers.training_utils import compute_snr

from dataset_local import edgestyle_dataset, edgestyle_dataset_test
from utils import CollateFn, CONDITIONING_IMAGES_TRANSFORMS
from utils import InverseEmbeddings

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0")

logger = get_logger(__name__)

NEGATIVE_PROMPT = (
    "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime),"
    + "text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, "
    + "mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, "
    + "cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers,"
    + " too many fingers, long neck,"
)

PROMT_TO_ADD = ", RAW photo, subject, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"


def log_validation(
    vae,
    text_encoder,
    tokenizer,
    unet,
    controlnet,
    validation_batch,
    args,
    accelerator,
    weight_dtype,
    step,
):
    logger.info("Running validation... ")

    inverse_embeddings = InverseEmbeddings(tokenizer)

    controlnet = accelerator.unwrap_model(controlnet)

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    image_logs = []

    original_list = validation_batch["original"]

    agnostic_or_head_list = (
        validation_batch["agnostic"]
        if args.use_agnostic_images
        else validation_batch["head"]
    )
    original_openpose_list = validation_batch["original_openpose"]
    clothes_list = validation_batch["clothes"]
    clothes_openpose_list = validation_batch["clothes_openpose"]

    clothes2_list = validation_batch["clothes2"]
    clothes_openpose2_list = validation_batch["clothes_openpose2"]

    prompts = inverse_embeddings(validation_batch["input_ids"])

    i = 0
    for (
        original,
        agnostic_or_head,
        original_openpose,
        clothes,
        clothes_openpose,
        clothes2,
        clothes_openpose2,
        prompt,
    ) in zip(
        original_list,
        agnostic_or_head_list,
        original_openpose_list,
        clothes_list,
        clothes_openpose_list,
        clothes2_list,
        clothes_openpose2_list,
        prompts,
    ):
        images = []
        image_name = f"validation_image_{i}"
        # Generate increasing args.num_validation_images for each validation image
        guidance_scales = np.linspace(3.5, 7.0, args.num_validation_images)

        for _ in range(args.num_validation_images):
            with torch.autocast("cuda"):
                image = pipeline(
                    prompt=prompt + PROMT_TO_ADD,
                    guidance_scale=guidance_scales[_],
                    image=[
                        agnostic_or_head.unsqueeze(0),
                        original_openpose.unsqueeze(0),
                        clothes.unsqueeze(0),
                        clothes_openpose.unsqueeze(0),
                        clothes2.unsqueeze(0),
                        clothes_openpose2.unsqueeze(0),
                    ],
                    # guess_mode=True,
                    negative_prompt=NEGATIVE_PROMPT,
                    num_inference_steps=50,
                    generator=generator,
                ).images[0]

            images.append(image)

        image_logs.append(
            {
                "original": (original / 2 + 0.5).clamp(0, 1),
                "conditional_images": [
                    (
                        (agnostic_or_head / 2 + 0.5).clamp(0, 1)
                        if args.controllora_use_vae
                        else agnostic_or_head
                    ),
                    (
                        (clothes / 2 + 0.5).clamp(0, 1)
                        if args.controllora_use_vae
                        else clothes
                    ),
                    (
                        (clothes2 / 2 + 0.5).clamp(0, 1)
                        if args.controllora_use_vae
                        else clothes2
                    ),
                ],
                "images": [CONDITIONING_IMAGES_TRANSFORMS(image) for image in images],
                "validation_prompt": image_name,
            }
        )
        i += 1

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                original = log["original"]
                conditional_images = log["conditional_images"]
                images = log["images"]
                validation_prompt = log["validation_prompt"]

                formatted_images = []
                formatted_images.append(original)
                for image in conditional_images:
                    formatted_images.append(image.cpu())

                for image in images:
                    formatted_images.append(image.cpu())

                grid = torchvision.utils.make_grid(formatted_images, nrow=4)

                tracker.writer.add_image(
                    validation_prompt, grid, step, dataformats="CHW"
                )
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

        return image_logs


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    else:
        raise ValueError(f"{model_class} is not supported.")


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
        required=True,
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
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--controllora_linear_rank",
        type=int,
        default=32,
        help=("The dimension of the Linear Module LoRA update matrices."),
    )
    parser.add_argument(
        "--controllora_conv2d_rank",
        type=int,
        default=0,
        help=("The dimension of the Conv2d Module LoRA update matrices."),
    )
    parser.add_argument(
        "--controllora_use_vae",
        action="store_true",
        default=False,
        help=("Whether to use the VAE in the controlnet."),
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup", "cosine_annealing for prodigy"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodidy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_decouple",
        type=bool,
        default=True,
        help="Use AdamW style decoupled weight decay",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-04,
        help="Weight decay to use for unet params",
    )
    parser.add_argument(
        "--adam_weight_decay_text_encoder",
        type=float,
        default=None,
        help="Weight decay to use for text_encoder",
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )

    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
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

    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )

    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
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
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--proportion_empty_images",
        type=float,
        default=0,
        help="Proportion of empty agnostic/clothes images. Defaults to 0 (no empty image replacement).",
    )
    parser.add_argument(
        "--proportion_cutout_images",
        type=float,
        default=0,
        help="Proportion of cutout agnostic/clothes images (removing half of the image). Defaults to 0 (no cutout images).",
    )
    parser.add_argument(
        "--proportion_patchworks",
        type=float,
        default=0,
        help="Proportion of patchworks to color as background. Defaults to 0 (no patchworks will be colored).",
    )
    parser.add_argument(
        "--proportion_patchworked_images",
        type=float,
        default=0,
        help="Proportion of images to apply patchworks. Defaults to 0 (no image will be patched).",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")
    if args.proportion_empty_images < 0 or args.proportion_empty_images > 1:
        raise ValueError("`--proportion_empty_images` must be in the range [0, 1].")
    if args.proportion_cutout_images < 0 or args.proportion_cutout_images > 1:
        raise ValueError("`--proportion_cutout_images` must be in the range [0, 1].")
    if args.proportion_patchworks < 0 or args.proportion_patchworks > 1:
        raise ValueError("`--proportion_patchworks` must be in the range [0, 1].")
    if args.proportion_patchworked_images < 0 or args.proportion_patchworked_images > 1:
        raise ValueError(
            "`--proportion_patchworked_images` must be in the range [0, 1]."
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )
    return args


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, revision=args.revision, use_fast=False
        )
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
    )
    if args.pretrained_vae_name_or_path:
        vae = AutoencoderKL.from_pretrained(args.pretrained_vae_name_or_path)
    else:
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=args.revision,
            variant=args.variant,
        )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        variant=args.variant,
    )
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    openpose = ControlNetModel.from_pretrained(
        args.pretrained_openpose_name_or_path, torch_dtype=weight_dtype
    )
    openpose.requires_grad_(False)

    if args.controlnet_model_name_or_path:
        controlnet = MultiControlNetModel(
            [
                ControlLoRAModel.from_pretrained(
                    args.controlnet_model_name_or_path,
                    subfolder="controlnet-0",
                    vae=vae if args.controllora_use_vae else None,
                ),
                openpose,
                ControlLoRAModel.from_pretrained(
                    args.controlnet_model_name_or_path,
                    subfolder="controlnet-1",
                    vae=vae if args.controllora_use_vae else None,
                ),
                openpose,
                ControlLoRAModel.from_pretrained(
                    args.controlnet_model_name_or_path,
                    subfolder="controlnet-2",
                    vae=vae if args.controllora_use_vae else None,
                ),
                openpose,
            ]
        )
        for net in controlnet.nets:
            if net is not openpose:
                if net.uses_vae:
                    net.set_autoencoder(vae)
                net.tie_weights(unet)
    else:
        logger.info("Initializing controlnet weights from unet")
        controlnet = MultiControlNetModel(
            [
                ControlLoRAModel.from_unet(
                    unet,
                    lora_linear_rank=args.controllora_linear_rank,
                    lora_conv2d_rank=args.controllora_conv2d_rank,
                    autoencoder=vae if args.controllora_use_vae else None,
                ),
                openpose,
                ControlLoRAModel.from_unet(
                    unet,
                    lora_linear_rank=args.controllora_linear_rank,
                    lora_conv2d_rank=args.controllora_conv2d_rank,
                    autoencoder=vae if args.controllora_use_vae else None,
                ),
                openpose,
                ControlLoRAModel.from_unet(
                    unet,
                    lora_linear_rank=args.controllora_linear_rank,
                    lora_conv2d_rank=args.controllora_conv2d_rank,
                    autoencoder=vae if args.controllora_use_vae else None,
                ),
                openpose,
            ]
        )

    for net in controlnet.nets:
        if net is not openpose:
            net.train()

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]
                    j = 0
                    for net in model.nets:
                        if net is not openpose:
                            if net.uses_vae:
                                net.set_autoencoder(None)
                            net.save_pretrained(
                                os.path.join(output_dir, f"controlnet-{j}"),
                                save_checkpoint=True,
                            )
                            if net.uses_vae:
                                net.set_autoencoder(vae)
                            j += 1
                    i -= 1

        def load_model_hook(models, input_dir):
            i = len(models) - 1
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                j = 0
                for net in model.nets:
                    if net is not openpose:
                        load_model = ControlLoRAModel.from_pretrained(
                            input_dir,
                            subfolder=f"controlnet-{j}",
                            uses_vae=net.uses_vae,
                        )
                        if net.uses_vae:
                            load_model.set_autoencoder(vae)
                        net.load_state_dict(load_model.state_dict())
                        del load_model
                        j += 1

                for net in model.nets:
                    if net is not openpose:
                        net.tie_weights(unet)

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    for net in controlnet.nets:
        if net is not openpose:
            if accelerator.unwrap_model(net).dtype != torch.float32:
                raise ValueError(
                    f"Controlnet loaded as datatype {accelerator.unwrap_model(controlnet).dtype}. {low_precision_error_string}"
                )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    controlnet_parameters = list(
        filter(lambda p: p.requires_grad, controlnet.parameters())
    )

    # Optimization parameters
    controlnet_parameters_with_lr = {
        "params": controlnet_parameters,
        "lr": args.learning_rate,
    }

    params_to_optimize = [controlnet_parameters_with_lr]

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warn(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.optimizer.lower() == "adamw":
        optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError(
                "To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`"
            )

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warn(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    empty_prompt = tokenizer(
        "",
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids.squeeze()

    train_collate_fn = CollateFn(
        empty_prompt=empty_prompt,
        proportion_empty_prompts=args.proportion_empty_prompts,
        proportion_empty_images=args.proportion_empty_images,
        proportion_patchworked_images=args.proportion_patchworked_images,
        proportion_cutout_images=args.proportion_cutout_images,
        proportion_patchworks=args.proportion_patchworks,
        uses_vae=args.controllora_use_vae,
    )

    train_dataloader = torch.utils.data.DataLoader(
        edgestyle_dataset,
        shuffle=True,
        collate_fn=train_collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    test_collate_fn = CollateFn(
        empty_prompt=empty_prompt,
        uses_vae=args.controllora_use_vae,
    )

    test_dataloader = torch.utils.data.DataLoader(
        edgestyle_dataset_test,
        collate_fn=test_collate_fn,
        batch_size=len(edgestyle_dataset_test),
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if (
        args.optimizer.lower() == "prodigy"
        and args.lr_scheduler.lower() == "cosine_annealing"
    ):
        # n_epoch is the total number of epochs to train the network
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.max_train_steps * accelerator.num_processes,
            eta_min=1e-6,
        )
    else:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )

    # Prepare everything with our `accelerator`.
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    # vae.to(accelerator.device, dtype=torch.float32)  # vae should always be in fp32
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    openpose.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        # tracker_config.pop("validation_prompt")
        # tracker_config.pop("validation_image")

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(edgestyle_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    validation_batch = next(iter(test_dataloader))
    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                with torch.autocast("cuda"):
                    # Convert images to latent space
                    latents = vae.encode(batch["original"]).latent_dist.sample()
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                    latents = latents * vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (bsz,),
                        device=latents.device,
                    )
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    agnostic_or_head = (
                        batch["agnostic"] if args.use_agnostic_images else batch["head"]
                    )
                    original_openpose = batch["original_openpose"]
                    clothes = batch["clothes"]
                    clothes_openpose = batch["clothes_openpose"]
                    clothes2 = batch["clothes2"]
                    clothes_openpose2 = batch["clothes_openpose2"]

                    # randomly swap the clothes with clothes2 for each image and pose in the batch

                    for i in range(bsz):
                        if random.random() < 0.5:
                            clothes[i], clothes2[i] = clothes2[i], clothes[i]
                            clothes_openpose[i], clothes_openpose2[i] = (
                                clothes_openpose2[i],
                                clothes_openpose[i],
                            )
                        if random.random() < 0.5:
                            agnostic_or_head[i] = batch["head"][i]
                        else:
                            agnostic_or_head[i] = batch["agnostic"][i]

                    down_block_res_samples, mid_block_res_sample = controlnet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=[
                            agnostic_or_head,
                            original_openpose,
                            clothes,
                            clothes_openpose,
                            clothes2,
                            clothes_openpose2,
                        ],
                        conditioning_scale=[1.0] * len(controlnet.nets),
                        return_dict=False,
                    )

                    # Predict the noise residual
                    model_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                    ).sample

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(
                            f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                        )

                    # print(model_pred)
                    if args.snr_gamma is None:
                        loss = F.mse_loss(
                            model_pred.float(), target.float(), reduction="mean"
                        )
                    else:
                        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                        # This is discussed in Section 4.2 of the same paper.
                        snr = compute_snr(noise_scheduler, timesteps)
                        if noise_scheduler.config.prediction_type == "v_prediction":
                            # Velocity objective requires that we add one to SNR values before we divide by them.
                            snr = snr + 1
                        mse_loss_weights = (
                            torch.stack(
                                [snr, args.snr_gamma * torch.ones_like(timesteps)],
                                dim=1,
                            ).min(dim=1)[0]
                            / snr
                        )

                        loss = F.mse_loss(
                            model_pred.float(), target.float(), reduction="none"
                        )
                        loss = (
                            loss.mean(dim=list(range(1, len(loss.shape))))
                            * mse_loss_weights
                        )
                        loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = controlnet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if global_step % args.validation_steps == 0:
                        image_logs = log_validation(
                            vae,
                            text_encoder,
                            tokenizer,
                            unet,
                            controlnet,
                            validation_batch,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = accelerator.unwrap_model(controlnet)

        for j, net in enumerate(controlnet.nets):
            if net.uses_vae:
                net.set_autoencoder(None)
            net.save_pretrained(os.path.join(args.output_dir, f"controlnet-{j}"))
            if net.uses_vae:
                net.set_autoencoder(vae)
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
