import gradio as gr

# torch
import torch
from torchvision import transforms
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
    StableDiffusionControlNetPipeline,
)
from diffusers.optimization import get_scheduler
from transformers import AutoTokenizer, CLIPTextModel, CLIPModel, CLIPProcessor

from model.utils import BestEmbeddings
from model.edgestyle_multicontrolnet import EdgeStyleMultiControlNetModel

# local
from model.controllora import ControlLoRAModel, CachedControlNetModel
from model.utils import BestEmbeddings
from model.edgestyle_multicontrolnet import EdgeStyleMultiControlNetModel
from model.edgestyle_pipeline import EdgeStyleStableDiffusionControlNetPipeline
from extract_dataset import process_batch, create_sam_images_for_batch

RESOLUTION = 512

IMAGES_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

CONDITIONING_IMAGES_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

CONTROLNET_PATTERN = [0, None, 1, None, 1, None]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


PRETRAINED_MODEL_NAME_OR_PATH = "./models/Realistic_Vision_V5.1_noVAE"
PRETRAINED_VAE_NAME_OR_PATH = "./models/sd-vae-ft-mse"
PRETRAINED_OPENPOSE_NAME_OR_PATH = "./models/control_v11p_sd15_openpose"
CONTROLNET_MODEL_NAME_OR_PATH = "./models/EdgeStyle/controlnet"
CLIP_MODEL_NAME_OR_PATH = "./models/clip-vit-large-patch14"


NEGATIVE_PROMPT = (
    r"deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, "
    "cartoon, drawing, anime, mutated hands and fingers, deformed, distorted, "
    "disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, "
    "floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation"
)

PROMT_TO_ADD = (
    ", gray background, RAW photo, subject, 8k uhd, dslr, soft lighting, high quality"
)

model = CLIPModel.from_pretrained(CLIP_MODEL_NAME_OR_PATH).to(device)
processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME_OR_PATH)

best_embeddings = BestEmbeddings(model, processor)


tokenizer = AutoTokenizer.from_pretrained(
    PRETRAINED_MODEL_NAME_OR_PATH,
    subfolder="tokenizer",
    use_fast=False,
)

text_encoder = CLIPTextModel.from_pretrained(
    PRETRAINED_MODEL_NAME_OR_PATH,
    subfolder="text_encoder",
)
vae = AutoencoderKL.from_pretrained(PRETRAINED_VAE_NAME_OR_PATH)

unet = UNet2DConditionModel.from_pretrained(
    PRETRAINED_MODEL_NAME_OR_PATH,
    subfolder="unet",
)

openpose = CachedControlNetModel.from_pretrained(PRETRAINED_OPENPOSE_NAME_OR_PATH)

controlnet = EdgeStyleMultiControlNetModel.from_pretrained(
    CONTROLNET_MODEL_NAME_OR_PATH,
    vae=vae,
    controlnet_class=ControlLoRAModel,
    load_pattern=CONTROLNET_PATTERN,
    static_controlnets=[None, openpose, None, openpose, None, openpose],
)
for net in controlnet.nets:
    if net is not openpose:
        net.tie_weights(unet)

# pipeline = StableDiffusionControlNetPipeline.from_pretrained(
#     PRETRAINED_MODEL_NAME_OR_PATH,
#     vae=vae,
#     text_encoder=text_encoder,
#     tokenizer=tokenizer,
#     unet=unet,
#     controlnet=controlnet,
#     safety_checker=None,
# )

pipeline = EdgeStyleStableDiffusionControlNetPipeline.from_pretrained(
    PRETRAINED_MODEL_NAME_OR_PATH,
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    controlnet=controlnet,
    safety_checker=None,
)
pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
generator = torch.Generator(device).manual_seed(42)
# vae.enable_xformers_memory_efficient_attention(attention_op=None)
# pipeline.enable_xformers_memory_efficient_attention()
pipeline = pipeline.to(device)


def preprocess(image_subject, image_cloth1, image_cloth2):
    data = process_batch([image_subject, image_cloth1, image_cloth2])
    if data is None or len(data) < 3:
        # try again, sometimes first time fails
        print("Retrying")
        data = process_batch([image_subject, image_cloth1, image_cloth2])
    data = create_sam_images_for_batch(data)

    image_subject_head = data["head_image"].iloc[0]
    image_cloth1_clothes = data["clothes_image"].iloc[1]
    image_cloth2_clothes = data["clothes_image"].iloc[2]

    image_subject_openpose = data["openpose_image"].iloc[0]
    image_cloth1_openpose = data["openpose_image"].iloc[1]
    image_cloth2_openpose = data["openpose_image"].iloc[2]

    return (
        image_subject_head,
        image_subject_openpose,
        image_cloth1_clothes,
        image_cloth1_openpose,
        image_cloth2_clothes,
        image_cloth2_openpose,
    )


def try_on(
    image_subject_agnostic,
    image_subject_openpose,
    image_cloth1_clothes,
    image_cloth1_openpose,
    image_cloth2_clothes,
    image_cloth2_openpose,
    scale,
    steps,
):
    with torch.autocast("cuda"):
        generator.manual_seed(42)
        prompts = best_embeddings([image_cloth1_clothes])
        image = pipeline(
            prompt=prompts[0] + " " + PROMT_TO_ADD,
            # prompt=prompts[0],
            guidance_scale=scale,
            image=[
                IMAGES_TRANSFORMS(image_subject_agnostic).unsqueeze(0),
                CONDITIONING_IMAGES_TRANSFORMS(image_subject_openpose).unsqueeze(0),
                IMAGES_TRANSFORMS(image_cloth1_clothes).unsqueeze(0),
                CONDITIONING_IMAGES_TRANSFORMS(image_cloth1_openpose).unsqueeze(0),
                IMAGES_TRANSFORMS(image_cloth2_clothes).unsqueeze(0),
                CONDITIONING_IMAGES_TRANSFORMS(image_cloth2_openpose).unsqueeze(0),
            ],
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=steps,
            generator=generator,
            # control_guidance_start=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            # control_guidance_end=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ).images[0]
    return image


with gr.Blocks() as iface:
    with gr.Row():
        with gr.Column():
            image_subject = gr.Image(label="Subject")
        with gr.Column():
            image_cloth1 = gr.Image(label="Clothes 1")
        with gr.Column():
            image_cloth2 = gr.Image(label="Clothes 2")

    with gr.Row():
        with gr.Column():
            btn = gr.Button("Preprocess")
    with gr.Row():
        with gr.Column():
            image_subject_agnostic = gr.Image(height=RESOLUTION, width=RESOLUTION)
        with gr.Column():
            image_cloth1_clothes = gr.Image(height=RESOLUTION, width=RESOLUTION)
        with gr.Column():
            image_cloth2_clothes = gr.Image(height=RESOLUTION, width=RESOLUTION)

    with gr.Row():
        with gr.Column():
            image_subject_openpose = gr.Image(height=RESOLUTION, width=RESOLUTION)
        with gr.Column():
            image_cloth1_openpose = gr.Image(height=RESOLUTION, width=RESOLUTION)
        with gr.Column():
            image_cloth2_openpose = gr.Image(height=RESOLUTION, width=RESOLUTION)

    btn.click(
        preprocess,
        inputs=[image_subject, image_cloth1, image_cloth2],
        outputs=[
            image_subject_agnostic,
            image_subject_openpose,
            image_cloth1_clothes,
            image_cloth1_openpose,
            image_cloth2_clothes,
            image_cloth2_openpose,
        ],
    )

    with gr.Row():
        with gr.Column():
            sliderScale = gr.Slider(
                minimum=1.0, maximum=12.0, value=3.5, step=0.1, label="Guidance Scale"
            )
            sliderSteps = gr.Slider(
                minimum=20,
                maximum=100,
                value=20,
                step=1,
                label="Inference Steps",
            )
            btnTryOn = gr.Button("Try On")
    with gr.Row():
        with gr.Column():
            image_try_on = gr.Image(height=RESOLUTION, width=RESOLUTION)

    btnTryOn.click(
        try_on,
        inputs=[
            image_subject_agnostic,
            image_subject_openpose,
            image_cloth1_clothes,
            image_cloth1_openpose,
            image_cloth2_clothes,
            image_cloth2_openpose,
            sliderScale,
            sliderSteps,
        ],
        outputs=[image_try_on],
    )

iface.launch()
