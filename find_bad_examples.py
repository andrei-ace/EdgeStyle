import os
import shutil
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torchmetrics.multimodal import CLIPImageQualityAssessment

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
IMAGES_PATH = "./data/image"
BAD_IMAGE_DIR = "./temp/bad_images"
RESOLUTION = 512
BAD_IMAGE_THRESHOLD = 0.5
NUMBER_OF_BAD_IMAGES = 64

SUBDIR = "subject"
# SUBDIR = "clothes"

prompts = (
    # "real",
    # "natural",
    # "sharpness",
    ("one", "two"),
    ("single", "multiple"),
    # ("high definition", "low resolution"),
    # ("Good photo of a person", "Pixelated photo with low resolution"),
    # ("", "umbrella"),
    # ("", "car"),
    # ("", "furniture"),
    # ("", "chair"),
    # ("", "pixelated"),
)


def get_prompt(prompt):
    if isinstance(prompt, str):
        return prompt
    else:
        return "_".join(prompt)


score_columns = ["score_" + get_prompt(prompt) for prompt in prompts]

image_quality_assessment = (
    CLIPImageQualityAssessment(prompts=prompts).to(device=DEVICE).eval()
)


data_dirs = os.listdir(IMAGES_PATH)
# keep only directories
data_dirs = [
    os.path.join(IMAGES_PATH, data_dir)
    for data_dir in data_dirs
    if os.path.isdir(os.path.join(IMAGES_PATH, data_dir))
]


def remove_files(dir, image_name):
    image_subject = os.path.join(dir, "subject", image_name)
    image_processed = os.path.join(dir, "processed", image_name)
    image_mask = os.path.join(dir, "mask", image_name)
    image_openpose = os.path.join(dir, "openpose", image_name)
    json_openpose = os.path.join(dir, "openpose", image_name.replace(".jpg", ".json"))
    clothes_image = os.path.join(dir, "clothes", image_name)
    agnostic_image = os.path.join(dir, "agnostic", image_name)

    # check if files exist
    if (
        os.path.exists(image_subject)
        and os.path.isfile(image_processed)
        and os.path.isfile(image_mask)
        and os.path.isfile(image_openpose)
        and os.path.isfile(json_openpose)
        and os.path.isfile(clothes_image)
        and os.path.isfile(agnostic_image)
    ):
        print(f"Removing {image_name} from {dir}")
        os.remove(image_subject)
        os.remove(image_processed)
        os.remove(image_mask)
        os.remove(image_openpose)
        os.remove(json_openpose)
        os.remove(clothes_image)
        os.remove(agnostic_image)


if __name__ == "__main__":
    all_filenames = []
    all_dirs = []
    for data_dir in tqdm(data_dirs):
        filenames = [
            filename
            for filename in os.listdir(os.path.join(data_dir, SUBDIR))
            if filename.endswith(".jpg")
        ]

        all_filenames += filenames
        all_dirs += [data_dir] * len(filenames)

    data = pd.DataFrame(
        columns=["directory", "image"] + score_columns,
        data=zip(
            all_dirs, all_filenames, *np.zeros((len(prompts), len(all_filenames)))
        ),
    )

    # split data dataframe into batches
    for i in tqdm(range(0, len(data), BATCH_SIZE)):
        batch = data.iloc[i : i + BATCH_SIZE]
        batch_images = []
        for image_name, directory in zip(batch["image"], batch["directory"]):
            image_path = os.path.join(directory, SUBDIR, image_name)
            image = Image.open(image_path).convert("RGB")
            image = transforms.Resize((RESOLUTION, RESOLUTION))(image)
            image = transforms.ToTensor()(image)
            batch_images.append(image)
        batch_images_tensors = torch.stack(batch_images).to(device=DEVICE)

        with torch.inference_mode():
            batch_scores = image_quality_assessment(batch_images_tensors)

        for i, score_column in enumerate(score_columns):
            if len(score_columns) == 1:
                batch.loc[:, score_column] = batch_scores.cpu().numpy()
            else:
                key = list(batch_scores.keys())[i]
                batch.loc[:, score_column] = batch_scores[key].cpu().numpy()

    # compute min and max and mean for each score to normalize
    min_max = {}
    for score_column in score_columns:
        min_max[score_column] = {
            "min": data[score_column].min(),
            "max": data[score_column].max(),
        }

    # normalize scores
    for score_column in score_columns:
        data[score_column] = (data[score_column] - min_max[score_column]["min"]) / (
            min_max[score_column]["max"] - min_max[score_column]["min"]
        )

    # compute mean score
    data["mean_score"] = data[score_columns].mean(axis=1)

    # keep only bad images
    # bad_images = data[data["mean_score"] < QUALITY_THRESHOLD]

    # sort by score
    data = data.sort_values(by="mean_score", ascending=True)

    bad_images = data[data["mean_score"] < BAD_IMAGE_THRESHOLD]
    bad_images = bad_images.head(NUMBER_OF_BAD_IMAGES)

    os.makedirs(BAD_IMAGE_DIR, exist_ok=True)
    # copy bad images to a new folder
    for index, row in bad_images.iterrows():
        # concat directory and image name
        directory = row["directory"].split("/")[-1]
        image_name = row["image"]
        scores = row[score_columns].values
        scores = [f"{score:.2f}" for score in scores]
        scores_strings = "_".join(scores)
        mean_score = row["mean_score"]
        image_name = f"{mean_score:.2f}_[{scores_strings}]_{directory}_{image_name}"
        image_path = os.path.join(row["directory"], SUBDIR, row["image"])
        # copy image to new folder
        shutil.copy(image_path, os.path.join(BAD_IMAGE_DIR, image_name))
        # remove files
        # remove_files(row["directory"], row["image"])

    # print(bad_images)
