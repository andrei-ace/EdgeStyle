import os
from tqdm import tqdm

IMAGES_PATH = "./data/image"
SUBDIR = "processed"

data_dirs = os.listdir(IMAGES_PATH)
# keep only directories
data_dirs = [
    os.path.join(IMAGES_PATH, data_dir)
    for data_dir in data_dirs
    if os.path.isdir(os.path.join(IMAGES_PATH, data_dir)) and "_skip_" not in data_dir
]

data_dirs = sorted(data_dirs)


def is_missing_data(dir, image_name):
    image_subject = os.path.join(dir, "subject", image_name)
    image_processed = os.path.join(dir, "processed", image_name)
    image_mask = os.path.join(dir, "mask", image_name)
    image_openpose = os.path.join(dir, "openpose", image_name)
    json_openpose = os.path.join(dir, "openpose", image_name.replace(".jpg", ".json"))
    clothes_image = os.path.join(dir, "clothes", image_name)
    agnostic_image = os.path.join(dir, "agnostic", image_name)
    head_image = os.path.join(dir, "head", image_name)

    # check if files exist
    return not (
        os.path.exists(image_subject)
        and os.path.isfile(image_processed)
        and os.path.isfile(image_mask)
        and os.path.isfile(image_openpose)
        and os.path.isfile(json_openpose)
        and os.path.isfile(clothes_image)
        and os.path.isfile(agnostic_image)
        and os.path.isfile(head_image)
    )


def remove_missing_data(dir, image_name):
    image_subject = os.path.join(dir, "subject", image_name)
    image_processed = os.path.join(dir, "processed", image_name)
    image_mask = os.path.join(dir, "mask", image_name)
    image_openpose = os.path.join(dir, "openpose", image_name)
    json_openpose = os.path.join(dir, "openpose", image_name.replace(".jpg", ".json"))
    clothes_image = os.path.join(dir, "clothes", image_name)
    agnostic_image = os.path.join(dir, "agnostic", image_name)
    head_image = os.path.join(dir, "head", image_name)

    # check if files exist
    if os.path.exists(image_subject):
        os.remove(image_subject)
    if os.path.isfile(image_processed):
        os.remove(image_processed)
    if os.path.isfile(image_mask):
        os.remove(image_mask)
    if os.path.isfile(image_openpose):
        os.remove(image_openpose)
    if os.path.isfile(json_openpose):
        os.remove(json_openpose)
    if os.path.isfile(clothes_image):
        os.remove(clothes_image)
    if os.path.isfile(agnostic_image):
        os.remove(agnostic_image)
    if os.path.isfile(head_image):
        os.remove(head_image)


if __name__ == "__main__":
    for data_dir in tqdm(data_dirs):
        filenames = [
            filename
            for filename in os.listdir(os.path.join(data_dir, SUBDIR))
            if filename.endswith(".jpg")
        ]

        for filename in filenames:
            if is_missing_data(data_dir, filename):
                print(f"Removing {filename} from {data_dir}")
                remove_missing_data(data_dir, filename)
