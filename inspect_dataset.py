import os
from PIL import Image
from dataset_local import edgestyle_dataset, edgestyle_dataset_test
import torch
from tqdm import tqdm
import math

from utils import PairedTransform, InverseEmbeddings, PatchedTransform

from transformers import AutoTokenizer

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
    mask = [example["mask"] for example in examples]
    original_openpose = [example["original_openpose"] for example in examples]

    target = [example["target"] for example in examples]
    clothes = [example["clothes"] for example in examples]
    clothes_openpose = [example["clothes_openpose"] for example in examples]

    input_ids = [example["input_ids"] for example in examples]

    target_transformed = []
    clothes_transformed = []
    clothes_openpose_transformed = []
    transform = PairedTransform(RESOLUTION, (BG_COLOR, BG_COLOR, BG_COLOR_CONTROLNET))
    patched_transform = PatchedTransform(RESOLUTION_PATCH, 0.5, BG_COLOR)

    for target_image, clothes_image, clothes_openpose_image in zip(
        target, clothes, clothes_openpose
    ):
        target_image, clothes_image, clothes_openpose_image = transform(
            [target_image, clothes_image, clothes_openpose_image]
        )
        target_image = patched_transform(target_image)
        target_transformed.append(target_image)
        clothes_transformed.append(clothes_image)
        clothes_openpose_transformed.append(clothes_openpose_image)

    input_ids = [example["input_ids"] for example in examples]

    return {
        "original": original,
        "agnostic": agnostic,
        "mask": mask,
        "original_openpose": original_openpose,
        "target": target_transformed,
        "clothes": clothes_openpose_transformed,
        "clothes_openpose": clothes_openpose_transformed,
        "input_ids": input_ids,
    }


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
        # print(step, batch)
        images = zip(
            batch["original"],
            batch["target"],
        )
        # flatten the list
        images_flatten = [img for pair in images for img in pair]

        image_grid(images_flatten, math.ceil(BATCH_SIZE / 4), 8).save(
            os.path.join(DATASET_PATH, f"test_{step:03d}.png")
        )
    for step, batch in tqdm(enumerate(train_dataloader)):
        # print(step, batch)
        images = zip(
            batch["original"],
            batch["target"],
        )
        # flatten the list
        images_flatten = [img for pair in images for img in pair]

        image_grid(images_flatten, math.ceil(BATCH_SIZE / 4), 8).save(
            os.path.join(DATASET_PATH, f"train_{step:03d}.png")
        )
