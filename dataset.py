import os

from itertools import permutations
import pandas as pd
import numpy as np

import datasets
from datasets import Dataset
from PIL import Image

import torch

from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    CLIPVisionModelWithProjection,
)

from tqdm import tqdm


IMAGES_PATH = "./data/image/"

DATASET_PATH = "./data/pairs_dataset/"

data_dirs = os.listdir(IMAGES_PATH)
# keep only directories
data_dirs = [
    os.path.join(IMAGES_PATH, data_dir)
    for data_dir in data_dirs
    if os.path.isdir(os.path.join(IMAGES_PATH, data_dir))
]

PROMPT = "human, ultra quality, sharp focus, 8K UHD"

RESOLUTION = 512
MAX_FRAMES = 8
BATCH_SIZE = 64
MAX_SCORE = 0.95
MIN_SCORE = 0.85

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="tokenizer",
    use_fast=False,
)

image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPVisionModelWithProjection.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(DEVICE)


def tokenize_captions(examples, is_train=True):
    captions = [PROMPT] * len(examples["original"])
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs.input_ids


def preprocess_train(examples):
    original = [
        Image.open(image).convert("RGB").resize((RESOLUTION, RESOLUTION))
        for image in examples["original"]
    ]

    agnostic = [
        Image.open(image).convert("RGB").resize((RESOLUTION, RESOLUTION))
        for image in examples["agnostic"]
    ]

    original_openpose = [
        Image.open(image).convert("RGB").resize((RESOLUTION, RESOLUTION))
        for image in examples["original_openpose"]
    ]

    clothes = [
        Image.open(image).convert("RGB").resize((RESOLUTION, RESOLUTION))
        for image in examples["clothes"]
    ]

    clothes_openpose = [
        Image.open(image).convert("RGB").resize((RESOLUTION, RESOLUTION))
        for image in examples["clothes_openpose"]
    ]

    mask = [
        Image.open(image).convert("RGB").resize((RESOLUTION, RESOLUTION))
        for image in examples["mask"]
    ]

    target = [
        Image.open(image).convert("RGB").resize((RESOLUTION, RESOLUTION))
        for image in examples["target"]
    ]

    examples["original"] = original
    examples["agnostic"] = agnostic
    examples["original_openpose"] = original_openpose
    examples["clothes"] = clothes
    examples["clothes_openpose"] = clothes_openpose
    examples["mask"] = mask
    examples["target"] = target
    examples["input_ids"] = tokenize_captions(examples)

    return examples


@torch.inference_mode()
def compute_scores_vectors(emb_one, emb_two):
    """Computes cosine similarity between two vectors."""
    scores = torch.nn.functional.cosine_similarity(emb_one, emb_two, dim=1)
    return scores


@torch.inference_mode()
def compute_scores(pd_image1: pd.Series, pd_image2: pd.Series) -> pd.Series:
    images1 = pd_image1.values
    images2 = pd_image2.values

    torch_data_1 = torch.stack(
        [
            image_processor(
                Image.open(image), return_tensors="pt"
            ).pixel_values.squeeze()
            for image in images1
        ]
    ).to(DEVICE)

    torch_data_2 = torch.stack(
        [
            image_processor(
                Image.open(image), return_tensors="pt"
            ).pixel_values.squeeze()
            for image in images2
        ]
    ).to(DEVICE)

    emb1 = model(torch_data_1).image_embeds
    emb2 = model(torch_data_2).image_embeds

    emb1 = torch.flatten(emb1, start_dim=1)
    emb2 = torch.flatten(emb2, start_dim=1)

    scores = compute_scores_vectors(emb1, emb2)

    return pd.Series(scores.cpu().numpy(), index=pd_image1.index)


# if DATASET_PATH exists, then load the dataset from DATASET_PATH
if not os.path.exists(DATASET_PATH):
    df_final = pd.DataFrame()
    # read each jpeg image file in the directory into a pandas dataframe
    for data_dir in tqdm(data_dirs):
        filenames = [
            filename
            for filename in os.listdir(os.path.join(data_dir, "subject"))
            if filename.endswith(".jpg")
        ]

        df = pd.DataFrame(permutations(filenames, 2), columns=["original", "clothes"])

        df = df.assign(agnostic=df["original"])
        df = df.assign(mask=df["original"])
        df = df.assign(original_openpose=df["original"])
        df = df.assign(clothes_openpose=df["clothes"])
        df = df.assign(target=df["clothes"])

        df["original"] = df["original"].apply(
            lambda x: os.path.join(data_dir, "subject", x)
        )
        df["target"] = df["target"].apply(
            lambda x: os.path.join(data_dir, "subject", x)
        )
        df["clothes"] = df["clothes"].apply(
            lambda x: os.path.join(data_dir, "clothes", x)
        )
        df["original_openpose"] = df["original_openpose"].apply(
            lambda x: os.path.join(data_dir, "openpose", x)
        )
        df["agnostic"] = df["agnostic"].apply(
            lambda x: os.path.join(data_dir, "agnostic", x)
        )
        df["clothes_openpose"] = df["clothes_openpose"].apply(
            lambda x: os.path.join(data_dir, "openpose", x)
        )

        df["mask"] = df["mask"].apply(lambda x: os.path.join(data_dir, "mask", x))

        df["folder_name"] = data_dir

        # split the dataframe into batches
        # compute the similarity score for each batch
        # remove rows where scores are too big or too small
        index_to_drop = []
        for i in range(0, len(df), BATCH_SIZE):
            batch = df.iloc[i : i + BATCH_SIZE]
            if batch.shape[0] == 0:
                break
            # compute similarity score for each batch
            batch_scores = compute_scores(batch["original"], batch["target"])
            # remove rows where scores are over or under a threshold

            index_to_drop += batch_scores[
                (batch_scores > MAX_SCORE) | (batch_scores < MIN_SCORE)
            ].index.tolist()

        # leave at least half of MAX_FRAMES rows
        if len(index_to_drop) > df.shape[0] - MAX_FRAMES // 2:
            index_to_drop = index_to_drop[: df.shape[0] - MAX_FRAMES // 2]
        df = df.drop(pd.Index(index_to_drop))

        # if df is empty, then skip this directory
        if df.shape[0] == 0:
            continue

        if df.shape[0] > MAX_FRAMES:
            # keep only MAX_FRAMES rows
            df = df.sample(n=MAX_FRAMES, random_state=42)

        df_final = pd.concat([df_final, df], ignore_index=True)

    dataset = Dataset.from_pandas(df_final)
    dataset.save_to_disk(DATASET_PATH)
else:
    dataset = datasets.load_from_disk(DATASET_PATH)

dataset.set_transform(preprocess_train)

dataset = dataset.train_test_split(test_size=4, shuffle=True, seed=42)

edgestyle_dataset = dataset["train"]
edgestyle_dataset_test = dataset["test"]


# print statistics about the dataset
print("Dataset train size: ", len(edgestyle_dataset))
print("Dataset test size: ", len(edgestyle_dataset_test))
