# misc
import sys
import numpy as np
import os
import argparse
from PIL import Image, ImageDraw
import cv2

# torch
import torch
from torchvision import transforms
from diffusers import (
    AutoencoderKL,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from transformers import AutoTokenizer, CLIPTextModel, CLIPModel, CLIPProcessor

# controlnet_aux
from controlnet_aux import OpenposeDetector
from controlnet_aux.open_pose import draw_poses, resize_image
from controlnet_aux.util import HWC3

# local
from controllora import ControlLoRAModel
from utils import BestEmbeddings

# efficientvit
sys.path.insert(0, "efficientvit")

from efficientvit.models.efficientvit.sam import (
    EfficientViTSamPredictor,
)
from efficientvit.sam_model_zoo import create_sam_model


RESOLUTION = 512
BG_COLOR = (127, 127, 127)
NUM_IMAGES = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "l2"
MODEL_PATH = "efficientvit/assets/checkpoints/sam/l2.pt"
MODEL_PATH_SUBJECT = "efficientvit/assets/checkpoints/sam/trained_model_subject.pt"
MODEL_PATH_AGNOSTIC = "efficientvit/assets/checkpoints/sam/trained_model_body.pt"
MODEL_PATH_CLOTHES = "efficientvit/assets/checkpoints/sam/trained_model_clothes.pt"

# build model
efficientvit_sam = (
    create_sam_model(MODEL_NAME, True, MODEL_PATH).to(device=DEVICE).eval()
)
efficientvit_sam_predictor = EfficientViTSamPredictor(efficientvit_sam)

# agnostic
efficientvit_sam_subject = (
    create_sam_model(MODEL_NAME, True, MODEL_PATH_SUBJECT).to(device=DEVICE).eval()
)
efficientvit_sam_predictor_subject = EfficientViTSamPredictor(efficientvit_sam_subject)

# agnostic
efficientvit_sam_agnostic = (
    create_sam_model(MODEL_NAME, True, MODEL_PATH_AGNOSTIC).to(device=DEVICE).eval()
)
efficientvit_sam_predictor_agnostic = EfficientViTSamPredictor(
    efficientvit_sam_agnostic
)

# clothes
efficientvit_sam_clothes = (
    create_sam_model(MODEL_NAME, True, MODEL_PATH_CLOTHES).to(device=DEVICE).eval()
)
efficientvit_sam_predictor_clothes = EfficientViTSamPredictor(efficientvit_sam_clothes)

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

openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet").to(device=DEVICE)


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
        "--source",
        type=str,
        default=None,
        required=True,
        help="Path to source image.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        required=True,
        help="Path to target image.",
    )
    parser.add_argument(
        "--result",
        type=str,
        default=None,
        required=True,
        help="Path to save the generated image.",
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


def compute_area(keypoints):
    non_none_keypoints = [keypoint for keypoint in keypoints if keypoint is not None]
    keypoints = np.array(non_none_keypoints)[:, 0:2]
    min_x = np.min(keypoints[:, 0])
    max_x = np.max(keypoints[:, 0])
    min_y = np.min(keypoints[:, 1])
    max_y = np.max(keypoints[:, 1])
    return (max_x - min_x) * (max_y - min_y)


def extract_openpose(image: Image.Image):
    input_image = np.array(image)
    input_image = HWC3(input_image)
    input_image = resize_image(input_image, RESOLUTION)

    H, W, C = input_image.shape
    poses = openpose.detect_poses(input_image)

    # filter poses with low score
    poses = [pose for pose in poses if pose.body.total_score > 10]

    # filter poses with small number of total parts
    poses = [pose for pose in poses if pose.body.total_parts > 5]

    # remove poses that have no nose, neck, left eye or right eye or left ear or right ear
    poses = [
        pose
        for pose in poses
        if pose.body.keypoints[0] is not None
        or pose.body.keypoints[1] is not None
        or pose.body.keypoints[14] is not None
        or pose.body.keypoints[15] is not None
        or pose.body.keypoints[16] is not None
        or pose.body.keypoints[17] is not None
    ]

    # remove poses that have no shoulders
    poses = [
        pose
        for pose in poses
        if pose.body.keypoints[2] is not None or pose.body.keypoints[5] is not None
    ]

    # remove poses that have no hips
    poses = [
        pose
        for pose in poses
        if pose.body.keypoints[8] is not None or pose.body.keypoints[11] is not None
    ]

    poses = sorted(
        poses,
        key=lambda pose: compute_area(pose.body.keypoints),
        reverse=True,
    )

    # if no poses are detected, return None
    if len(poses) == 0:
        return None, None

    pose = poses[0].body

    # if pose has no hips, return None
    if pose.keypoints[8] is None and pose.keypoints[11] is None:
        return None, None

    posedict = {
        "keypoints": pose.keypoints,
        "total_score": pose.total_score,
        "total_parts": pose.total_parts,
    }

    canvas = draw_poses(
        [poses[0]],
        H,
        W,
        draw_body=True,
        draw_hand=False,
        draw_face=False,
    )

    detected_map = canvas
    detected_map = HWC3(detected_map)

    img = resize_image(input_image, RESOLUTION)
    H, W, C = img.shape

    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

    detected_map = Image.fromarray(detected_map)

    return detected_map, posedict["keypoints"]


def getBox(mask):
    # get bounding box from mask
    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return np.zeros(4)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    # bbox_rand = np.random.randint(0, 10, 4)
    H, W = mask.shape
    x_min = max(0, x_min - 20)
    x_max = min(W, x_max + 20)
    y_min = max(0, y_min - 20)
    y_max = min(H, y_max + 20)
    bbox = [x_min, y_min, x_max, y_max]
    return np.array(bbox)


def draw_binary_mask(raw_image, binary_mask, mask_color=(0, 0, 255)):
    # Ensure that binary_mask is a boolean array
    binary_mask = binary_mask.astype(bool)
    binary_mask = np.logical_not(binary_mask)

    # Create an output array with the same shape as raw_image
    masked_image = np.copy(raw_image)

    # Apply the mask_color to the locations where binary_mask is True
    # For each channel in the mask_color, apply it to the corresponding channel in the image
    for i in range(3):  # Assuming raw_image is in RGB format
        masked_image[:, :, i][binary_mask] = mask_color[i]

    # Convert the NumPy array back to a PIL Image and return
    return masked_image


def smooth_mask(mask, kernel_size=3, iterations=3):
    # Convert the mask from boolean to binary format (0 or 255)
    binary_mask = np.uint8(mask * 255)

    # Define the kernel for morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply closing (dilation followed by erosion) to fill gaps
    closed = cv2.dilate(binary_mask, kernel, iterations=iterations)
    closed = cv2.erode(closed, kernel, iterations=iterations)

    # Apply opening (erosion followed by dilation) to remove isolated pixels
    opened = cv2.erode(closed, kernel, iterations=iterations)
    smoothed_mask = cv2.dilate(opened, kernel, iterations=iterations)

    # Convert back to boolean format and return
    return smoothed_mask > 0


@torch.inference_mode()
def subject_image(image: Image.Image, openpose_keypoints):
    original_image = np.array(image)
    points = [
        point[0:2] * np.array([RESOLUTION, RESOLUTION])
        for point in openpose_keypoints
        if point is not None
    ]

    # openpose_image_mask = np.array(openpose_image) > 0
    # # merge into one channel
    # openpose_image_mask = openpose_image_mask.sum(axis=2) > 0

    efficientvit_sam_predictor.set_image(original_image)
    all_masks, _, _ = efficientvit_sam_predictor.predict(
        point_coords=np.array(points),
        point_labels=np.ones(len(points)),
    )
    all_masks = all_masks[0]
    box = getBox(all_masks)

    efficientvit_sam_predictor_subject.set_image(original_image)
    (
        subject_masks,
        subject_scores,
        _,
    ) = efficientvit_sam_predictor_subject.predict(
        box=box,
        multimask_output=False,
    )
    subject_masks = subject_masks[0]
    subject_scores = subject_scores[0]

    # all_masks = np.logical_or(all_masks, subject_masks)
    all_masks = smooth_mask(subject_masks)

    efficientvit_sam_predictor_agnostic.set_image(original_image)
    (
        predicted_agnostic_mask,
        _,
        _,
    ) = efficientvit_sam_predictor_agnostic.predict(
        box=box,
        multimask_output=False,
    )
    predicted_agnostic_mask = smooth_mask(predicted_agnostic_mask[0])

    efficientvit_sam_predictor_clothes.set_image(original_image)
    (
        predicted_clothes_mask,
        _,
        _,
    ) = efficientvit_sam_predictor_clothes.predict(
        box=box,
        multimask_output=False,
    )
    predicted_clothes_mask = smooth_mask(predicted_clothes_mask[0])

    all_masks = np.logical_or(
        all_masks, np.logical_or(predicted_agnostic_mask, predicted_clothes_mask)
    )  # type: ignore

    all_masks = smooth_mask(all_masks)

    unknown_mask = smooth_mask(
        np.logical_and(predicted_agnostic_mask, predicted_clothes_mask)
    )

    agnostic_mask = np.logical_and(
        predicted_agnostic_mask, np.logical_not(unknown_mask)
    )
    agnostic_mask = smooth_mask(agnostic_mask)

    clothes_mask = predicted_clothes_mask

    subject_image = Image.fromarray(
        draw_binary_mask(
            original_image,
            all_masks.squeeze(),
            mask_color=BG_COLOR,
        ),
        mode="RGB",
    )

    mask_image = Image.fromarray(
        draw_binary_mask(
            np.zeros_like(original_image),
            agnostic_mask.squeeze(),
            mask_color=(255, 255, 255),
        ),
        mode="RGB",
    )
    agnostic_image = Image.fromarray(
        draw_binary_mask(original_image, agnostic_mask.squeeze(), mask_color=BG_COLOR),
        mode="RGB",
    )

    clothes_image = Image.fromarray(
        draw_binary_mask(original_image, clothes_mask.squeeze(), mask_color=BG_COLOR),
        mode="RGB",
    )

    return (
        subject_image,
        agnostic_image,
        clothes_image,
    )


def resize_image_by_padding(image: Image.Image, size=RESOLUTION, color=(0, 0, 0)):
    old_size = image.size  # old_size[0] is in (width, height) format
    ratio = float(size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    image = image.resize(new_size)
    new_im = Image.new("RGB", (size, size), color)
    new_im.paste(image, ((size - new_size[0]) // 2, (size - new_size[1]) // 2))
    return new_im


def extract_images(source: str, target: str):
    source = resize_image_by_padding(Image.open(source))
    target = resize_image_by_padding(Image.open(target))
    original_openpose, original_keypoints = extract_openpose(source)
    clothes_openpose, target_keypoints = extract_openpose(target)
    source_subject, agnostic, _ = subject_image(source, original_keypoints)
    target_subject, _, clothes = subject_image(target, target_keypoints)
    return (
        source_subject,
        target_subject,
        agnostic,
        original_openpose,
        clothes,
        clothes_openpose,
    )


def main(args):
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
    )

    controlnet = MultiControlNetModel(
        [
            ControlLoRAModel.from_pretrained(
                args.controlnet_model_name_or_path,
                subfolder="controlnet-0",
                vae=vae,
            ),
            ControlLoRAModel.from_pretrained(
                args.controlnet_model_name_or_path,
                subfolder="controlnet-1",
            ),
            ControlLoRAModel.from_pretrained(
                args.controlnet_model_name_or_path,
                subfolder="controlnet-2",
                vae=vae,
            ),
            ControlLoRAModel.from_pretrained(
                args.controlnet_model_name_or_path,
                subfolder="controlnet-3",
            ),
        ]
    )
    for net in controlnet.nets:
        if net.uses_vae:
            net.set_autoencoder(vae)
        net.tie_weights(unet)

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
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
    generator = torch.Generator(DEVICE).manual_seed(42)
    pipeline = pipeline.to(DEVICE)

    (
        subject,
        target,
        agnostic,
        original_openpose,
        clothes,
        clothes_openpose,
    ) = extract_images(args.source, args.target)

    # # Load image
    # subject = Image.open(
    #     os.path.join(args.source_path, "subject", args.source_image_name)
    # )
    # target = Image.open(
    #     os.path.join(args.target_path, "subject", args.target_image_name)
    # )

    # agnostic = Image.open(
    #     os.path.join(args.source_path, "agnostic", args.source_image_name)
    # )

    # original_openpose = Image.open(
    #     os.path.join(args.source_path, "openpose", args.source_image_name)
    # )

    # clothes = Image.open(
    #     os.path.join(args.target_path, "clothes", args.target_image_name)
    # )

    # clothes_openpose = Image.open(
    #     os.path.join(args.target_path, "openpose", args.target_image_name)
    # )

    prompts = best_embeddings([clothes])

    guidance_scales = np.linspace(0.0, 15.0, NUM_IMAGES)

    images = [
        subject,
        target,
    ]

    for i in range(NUM_IMAGES):
        with torch.autocast("cuda"):
            image = pipeline(
                prompt=prompts[0] + " " + args.prompt_text_to_add,
                # prompt=prompts[0] + "detailed, ultra quality, sharp focus, 8K UHD",
                # prompt="clear face, full body, ultra quality, sharp focus, 8K UHD",
                # prompt="edgestyle",
                guidance_scale=guidance_scales[i],
                # guess_mode=True,
                image=[
                    IMAGES_TRANSFORMS(agnostic).unsqueeze(0),
                    CONDITIONING_IMAGES_TRANSFORMS(original_openpose).unsqueeze(0),
                    IMAGES_TRANSFORMS(clothes).unsqueeze(0),
                    CONDITIONING_IMAGES_TRANSFORMS(clothes_openpose).unsqueeze(0),
                ],
                # controlnet_conditioning_scale=[0.5, 0.5, 1, 1],
                # control_guidance_start=0.0,
                # control_guidance_end=0.9,
                negative_prompt=args.negative_prompt,
                num_inference_steps=50,
                generator=generator,
            ).images[0]
        image = add_text_to_image(image, f"Guidance scale: {guidance_scales[i]:.2f}")
        images.append(image)

    image = image_grid(images, 3, len(images) // 3)

    image.save(args.result)


if __name__ == "__main__":
    args = parse_args()
    main(args)
