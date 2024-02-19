import os
from PIL import Image
from dataset_local import edgestyle_dataset, edgestyle_dataset_test
import torch
from tqdm import tqdm
import math
import numpy as np

from skimage.measure import label, regionprops
from skimage.morphology import closing, square

import torchvision.transforms.functional as F

from model.utils import PairedTransform, PatchedTransform, IMAGES_TRANSFORMS

DATASET_PATH = "./temp/inspect_dataset"

BATCH_SIZE = 16
RESOLUTION = 512
RESOLUTION_PATCH = 32

BG_COLOR = (127, 127, 127)
BG_COLOR_CONTROLNET = (0, 0, 0)


def image_grid(imgs, rows, cols):
    if len(imgs) < rows * cols:
        imgs += [Image.new("RGB", imgs[0].size)] * (rows * cols - len(imgs))

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def collate_fn(examples):
    original = [example["original"] for example in examples]
    agnostic = [example["agnostic"] for example in examples]
    head = [example["head"] for example in examples]
    original_openpose = [example["original_openpose"] for example in examples]

    target = [example["target"] for example in examples]
    clothes = [example["clothes"] for example in examples]
    clothes_openpose = [example["clothes_openpose"] for example in examples]

    target2 = [example["target2"] for example in examples]
    clothes2 = [example["clothes2"] for example in examples]
    clothes_openpose2 = [example["clothes_openpose2"] for example in examples]

    input_ids = [example["input_ids"] for example in examples]

    target_transformed = []
    clothes_transformed = []
    clothes_openpose_transformed = []
    target_transformed2 = []
    clothes_transformed2 = []
    clothes_openpose_transformed2 = []
    transform = PairedTransform(RESOLUTION, (BG_COLOR, BG_COLOR, BG_COLOR_CONTROLNET))
    patched_transform = PatchedTransform([RESOLUTION_PATCH], 0.0, BG_COLOR)

    for (
        target_image,
        target_image2,
        clothes_image,
        clothes_image2,
        clothes_openpose_image,
        clothes_openpose_image2,
    ) in zip(target, target2, clothes, clothes2, clothes_openpose, clothes_openpose2):
        target_image, clothes_image, clothes_openpose_image = transform(
            [target_image, clothes_image, clothes_openpose_image]
        )
        target_image = patched_transform(target_image)
        target_transformed.append(target_image)
        clothes_transformed.append(clothes_image)
        clothes_openpose_transformed.append(clothes_openpose_image)

        target_image2 = patched_transform(target_image2)
        target_transformed2.append(target_image2)
        clothes_transformed2.append(clothes_image2)
        clothes_openpose_transformed2.append(clothes_openpose_image2)

    input_ids = [example["input_ids"] for example in examples]

    return {
        "original": original,
        "agnostic": agnostic,
        "head": head,
        "original_openpose": original_openpose,
        "target": target_transformed,
        "target2": target_transformed2,
        "clothes": clothes_openpose_transformed,
        "clothes2": clothes_openpose_transformed2,
        "clothes_openpose": clothes_openpose_transformed,
        "clothes_openpose2": clothes_openpose_transformed2,
        "input_ids": input_ids,
    }


def make_inpaint_condition(images: torch.Tensor):
    """
    Modify the background pixels of images to be -1 where the background is equal to target_value.

    Args:
    - images (torch.Tensor): Input tensor of images with shape (N, C, H, W) normalized in [-1, 1] range.
    - target_value (tuple): RGB values in the range [0, 255] to be considered as background.

    Returns:
    - torch.Tensor: Modified images with background pixels set to -1.
    """
    # Normalize the target RGB values to [-1, 1] range
    normalized_target = [(v / 255.0) * 2.0 - 1 for v in BG_COLOR]

    # Create a mask for pixels to be changed. Start with an empty mask that is False everywhere
    mask = torch.zeros_like(images, dtype=torch.bool)

    # For each channel in the image, mark the pixels that match the normalized target value for that channel
    for channel, target_val in enumerate(normalized_target):
        # Use a small epsilon to account for floating-point arithmetic issues
        epsilon = 0.1
        mask[:, channel, :, :] = (images[:, channel, :, :] > (target_val - epsilon)) & (
            images[:, channel, :, :] < (target_val + epsilon)
        )

    # Check if all channels of a pixel match the target value to consider it as background
    mask = mask.all(dim=1, keepdim=True)

    # for i in range(mask.shape[0]):
    #     np_mask = mask[i].detach().cpu().numpy().squeeze()
    #     labeled_array = label(np_mask)
    #     # Measure the properties of each component
    #     regions = regionprops(labeled_array)
    #     # Find the largest component by area
    #     largest_area = 0
    #     largest_component_label = 0
    #     for region in regions:
    #         if region.area > largest_area:
    #             largest_area = region.area
    #             largest_component_label = region.label

    #     # Keep only the largest component
    #     np_mask = labeled_array == largest_component_label
    #     mask[i] = torch.tensor(np_mask).unsqueeze(0).to(mask.device)

    # keep largest connected components

    # Set those pixels across all channels to -1
    images[mask.expand_as(images)] = -1

    return images


def redo_images(images: torch.Tensor):
    """
    Convert the input tensor of images to a single PIL image.

    Args:
    - images (torch.Tensor): Input tensor of images with shape (N, C, H, W) normalized in [-1, 1] range.

    Returns:
    - Image: A single PIL image containing the input images.
    """
    # Denormalize the images to [0, 1] range
    images = (images / 2 + 0.5).clamp(0, 1)

    # Convert the tensor to a numpy array and transpose the dimensions to (H, W, C)
    images = images.detach().cpu().numpy().transpose(0, 2, 3, 1)

    # Convert the numpy array to a list of PIL images
    return [Image.fromarray((img * 255).astype(np.uint8)) for img in images]


if __name__ == "__main__":
    os.makedirs(DATASET_PATH, exist_ok=True)
    train_dataloader = torch.utils.data.DataLoader(
        edgestyle_dataset,
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )
    test_dataloader = torch.utils.data.DataLoader(
        edgestyle_dataset_test,
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )
    for step, batch in tqdm(enumerate(test_dataloader)):
        images = []

        for original, target, target2 in zip(
            batch["original"], batch["target"], batch["target2"]
        ):
            images.append(image_grid([original, target, target2], 1, 3))

        image_grid(images, math.ceil(BATCH_SIZE / 2), 2).save(
            os.path.join(DATASET_PATH, f"test_{step:03d}.jpg")
        )

        for i, image in enumerate(batch["agnostic"]):
            redo_images(
                make_inpaint_condition(
                    F.normalize(F.to_tensor(image).unsqueeze(0), [0.5], [0.5])
                )
            )[0].save(os.path.join(DATASET_PATH, f"agnostic{i:03d}.jpg"))

    for step, batch in tqdm(enumerate(train_dataloader)):
        images = []

        for original, target, target2 in zip(
            batch["original"], batch["target"], batch["target2"]
        ):
            images.append(image_grid([original, target, target2], 1, 3))

        image_grid(images, math.ceil(BATCH_SIZE / 2), 2).save(
            os.path.join(DATASET_PATH, f"train_{step:03d}.jpg")
        )
