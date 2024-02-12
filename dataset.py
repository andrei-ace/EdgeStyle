import os
import io
from itertools import permutations
import pandas as pd
import numpy as np

import datasets
from datasets import Dataset
from PIL import Image

import torch


RESOLUTION = 512
MAX_FRAMES = 8
BATCH_SIZE = 64
MAX_SCORE = 0.90
MIN_SCORE = 0.80

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_dataset_back_to_images(examples):
    original = [
        Image.open(io.BytesIO(binary_data["bytes"]))
        for binary_data in examples["original"]
    ]

    agnostic = [
        Image.open(io.BytesIO(binary_data["bytes"]))
        for binary_data in examples["agnostic"]
    ]

    head = [
        Image.open(io.BytesIO(binary_data["bytes"])) for binary_data in examples["head"]
    ]

    original_openpose = [
        Image.open(io.BytesIO(binary_data["bytes"]))
        for binary_data in examples["original_openpose"]
    ]

    clothes = [
        Image.open(io.BytesIO(binary_data["bytes"]))
        for binary_data in examples["clothes"]
    ]

    clothes_openpose = [
        Image.open(io.BytesIO(binary_data["bytes"]))
        for binary_data in examples["clothes_openpose"]
    ]

    mask = [
        Image.open(io.BytesIO(binary_data["bytes"])) for binary_data in examples["mask"]
    ]

    target = [
        Image.open(io.BytesIO(binary_data["bytes"]))
        for binary_data in examples["target"]
    ]

    examples["original"] = original
    examples["agnostic"] = agnostic
    examples["head"] = head
    examples["original_openpose"] = original_openpose
    examples["clothes"] = clothes
    examples["clothes_openpose"] = clothes_openpose
    examples["mask"] = mask
    examples["target"] = target

    return examples


dataset = datasets.load_dataset("andrei-ace/EdgeStyle", split="train[:100%]")

# map back to images
dataset.set_transform(process_dataset_back_to_images)

dataset = dataset.train_test_split(test_size=4, shuffle=True, seed=42)

edgestyle_dataset = dataset["train"]
edgestyle_dataset_test = dataset["test"]


# print statistics about the dataset
print("Dataset train size: ", len(edgestyle_dataset))
print("Dataset test size: ", len(edgestyle_dataset_test))
