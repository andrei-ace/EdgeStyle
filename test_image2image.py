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


CONTROLNET_DIR = "./models/output_image2image_prodigy_vae_experiment"
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
            ),
            ControlLoRAModel.from_pretrained(
                CONTROLNET_DIR,
                subfolder="controlnet-1",
                vae=vae,
            ),
            ControlLoRAModel.from_pretrained(
                CONTROLNET_DIR,
                subfolder="controlnet-2",
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
        # torch_dtype=torch.float16,
    )
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
            prompt=prompts[0],
            # prompt="blue dress, clear face, full body ultra quality, sharp focus, 8K UHD",
            image=[IMAGES_TRANSFORMS(agnostic).unsqueeze(0)],
            control_image=[
                CONDITIONING_IMAGES_TRANSFORMS(original_openpose).unsqueeze(0),
                IMAGES_TRANSFORMS(clothes).unsqueeze(0),
                CONDITIONING_IMAGES_TRANSFORMS(clothes_openpose).unsqueeze(0),
            ],
            # strength=0.8,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=20,
            generator=generator,
        ).images[0]

    image.save("temp/test_data/result_image2image.jpg")
