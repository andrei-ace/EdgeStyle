import torch

from torchvision import transforms

from diffusers import (
    AutoencoderKL,
    StableDiffusionControlNetImg2ImgPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler

from transformers import AutoTokenizer, CLIPTextModel, CLIPModel, CLIPProcessor


from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from controllora import ControlLoRAModel

from utils import BestEmbeddings

from PIL import Image


CONTROLNET_DIR = "./models/output_text2image_prodigy_vae_experiment"
MODEL_NAME = "runwayml/stable-diffusion-v1-5"

NEGATIVE_PROMPT = "disfigured, ugly, bad, immature, cartoon, anime, 3d, painting, b&w"

RESOLUTION = 512

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

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

best_embeddings = BestEmbeddings(model, processor)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        subfolder="tokenizer",
        use_fast=False,
    )

    text_encoder = CLIPTextModel.from_pretrained(
        MODEL_NAME,
        subfolder="text_encoder",
    )
    vae = AutoencoderKL.from_pretrained(
        MODEL_NAME,
        subfolder="vae",
    )
    unet = UNet2DConditionModel.from_pretrained(
        MODEL_NAME,
        subfolder="unet",
    )

    controlnet = MultiControlNetModel(
        [
            ControlLoRAModel.from_pretrained(
                CONTROLNET_DIR,
                subfolder="controlnet-0",
                vae=vae,
            ),
            ControlLoRAModel.from_pretrained(
                CONTROLNET_DIR,
                subfolder="controlnet-1",
            ),
            ControlLoRAModel.from_pretrained(
                CONTROLNET_DIR,
                subfolder="controlnet-2",
                vae=vae,
            ),
            ControlLoRAModel.from_pretrained(
                CONTROLNET_DIR,
                subfolder="controlnet-3",
            ),
        ]
    )
    for net in controlnet.nets:
        if net.uses_vae:
            net.set_autoencoder(vae)
        net.tie_weights(unet)

    pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        MODEL_NAME,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16,
    )
    # pipeline.enable_freeu(s1=0.9, s2=0.2, b1=1.5, b2=1.6)
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    generator = torch.Generator(device).manual_seed(42)
    pipeline = pipeline.to(device)

    # Load image

    agnostic = Image.open("temp/test_data/original_agnostic.jpg")

    original_openpose = Image.open("temp/test_data/original_openpose.jpg")

    clothes = Image.open("temp/test_data/target_clothes.jpg")

    clothes_openpose = Image.open("temp/test_data/target_openpose.jpg")

    prompts = best_embeddings([clothes])

    with torch.autocast("cuda"):
        image = pipeline(
            # prompt=prompts[0],
            prompt=prompts[0] + "detailed, ultra quality, sharp focus, 8K UHD",
            # prompt="clear face, full body, ultra quality, sharp focus, 8K UHD",
            # prompt="",
            guidance_scale=4.5,
            # guess_mode=True,
            image=[
                IMAGES_TRANSFORMS(agnostic).unsqueeze(0),
            ],
            control_image=[
                IMAGES_TRANSFORMS(agnostic).unsqueeze(0),
                CONDITIONING_IMAGES_TRANSFORMS(original_openpose).unsqueeze(0),
                IMAGES_TRANSFORMS(clothes).unsqueeze(0),
                CONDITIONING_IMAGES_TRANSFORMS(clothes_openpose).unsqueeze(0),
            ],
            # controlnet_conditioning_scale=[1, 1, 1, 1],
            # control_guidance_start=0.0,
            # control_guidance_end=0.9,
            strength=0.99,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=50,
            generator=generator,
        ).images[0]

    image.save("temp/test_data/result_image2image_4nets.jpg")
