import os
import random
from tqdm import tqdm
import pandas as pd
from itertools import combinations
from PIL import Image

import torch
from transformers import AutoImageProcessor, CLIPVisionModelWithProjection

IMAGES_PATH = "./data/image"
SUBDIR = "subject"
BATCH_SIZE = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
model = CLIPVisionModelWithProjection.from_pretrained(
    "openai/clip-vit-large-patch14"
).to(DEVICE)

data_dirs = os.listdir(IMAGES_PATH)
# keep only directories
data_dirs = [
    os.path.join(IMAGES_PATH, data_dir)
    for data_dir in data_dirs
    if os.path.isdir(os.path.join(IMAGES_PATH, data_dir)) and "_skip_" not in data_dir
]

data_dirs = sorted(data_dirs)


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


if __name__ == "__main__":
    all_subjects = []
    for dir in data_dirs:
        subjects = os.listdir(os.path.join(dir, SUBDIR))
        # keep random subject
        subject = subjects[random.randint(0, len(subjects) - 1)]
        all_subjects.append(os.path.join(dir, SUBDIR, subject))

    # create dataframe
    df = pd.DataFrame(combinations(all_subjects, 2), columns=["original", "target"])

    # split the dataframe into batches
    # compute the similarity score for each batch
    all_batch_scores = []
    for i in tqdm(range(0, len(df), BATCH_SIZE)):
        batch = df.iloc[i : i + BATCH_SIZE]
        # compute similarity score for each batch
        batch_scores = compute_scores(batch["original"], batch["target"])
        # remove rows where scores are over or under a threshold

        all_batch_scores += batch_scores.tolist()

    # add the scores to the dataframe
    df["score"] = all_batch_scores

    # sort the dataframe by score
    df = df.sort_values("score", ascending=False)

    print(df.head())
