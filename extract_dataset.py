import sys
import json
import os
import cv2
import numpy as np

import pandas as pd
import torch
from PIL import Image

from controlnet_aux import OpenposeDetector
from controlnet_aux.open_pose import draw_poses, resize_image
from controlnet_aux.util import HWC3
from tqdm import tqdm

sys.path.insert(0, "efficientvit")

from efficientvit.models.efficientvit.sam import (
    EfficientViTSamPredictor,
)
from efficientvit.sam_model_zoo import create_sam_model


BATCH_SIZE = 64
CONFIDENCE_THRESHOLD = 0.90
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
BOX_MARGIN = 0.1
VIDEOS_PATH = "data/video"
IMAGES_PATH = "data/image"

MODEL_NAME = "l2"
MODEL_PATH = "efficientvit/assets/checkpoints/sam/l2.pt"
MODEL_PATH_AGNOSTIC = "efficientvit/assets/checkpoints/sam/trained_model_clothes.pt"
MODEL_PATH_CLOTHES = "efficientvit/assets/checkpoints/sam/trained_model_body.pt"

# Load the pre-trained YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)


openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

# build model
efficientvit_sam = create_sam_model(MODEL_NAME, True, MODEL_PATH).cuda().eval()
efficientvit_sam_predictor = EfficientViTSamPredictor(efficientvit_sam)

# agnostic
efficientvit_sam_agnostic = (
    create_sam_model(MODEL_NAME, True, MODEL_PATH_AGNOSTIC).cuda().eval()
)
efficientvit_sam_predictor_agnostic = EfficientViTSamPredictor(
    efficientvit_sam_agnostic
)

# clothes
efficientvit_sam_clothes = (
    create_sam_model(MODEL_NAME, True, MODEL_PATH_CLOTHES).cuda().eval()
)
efficientvit_sam_predictor_clothes = EfficientViTSamPredictor(efficientvit_sam_clothes)


def create_processed_image(row, final_width=IMAGE_WIDTH, final_height=IMAGE_HEIGHT):
    # Load the original image
    # convert image to np array
    original_image = row["image"]

    # Extract original bounding box coordinates
    xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]

    # Calculate the dimensions of the bounding box
    bbox_width = xmax - xmin
    bbox_height = ymax - ymin

    # Calculate the size of the margins (BOX_MARGIN of the bounding box size)
    margin_width = bbox_width * BOX_MARGIN
    margin_height = bbox_height * BOX_MARGIN

    # Adjust the bounding box to include margins and ensure it's within the image boundaries
    xmin = max(0, xmin - margin_width)
    xmax = min(original_image.width, xmax + margin_width)
    ymin = max(0, ymin - margin_height)
    ymax = min(original_image.height, ymax + margin_height)

    # Calculate the center of the adjusted bounding box
    bbox_center_x = (xmin + xmax) / 2
    bbox_center_y = (ymin + ymax) / 2

    # Determine scale to fit the bounding box (with margin) in the final image size
    scale_x = final_width / (xmax - xmin)
    scale_y = final_height / (ymax - ymin)
    scale = min(scale_x, scale_y)

    # Calculate new dimensions of the image
    new_width = int(original_image.width * scale)
    new_height = int(original_image.height * scale)

    # Resize the original image
    resized_image = original_image.resize(
        (new_width, new_height), Image.Resampling.LANCZOS
    )

    # Calculate the new bounding box center after resizing
    new_bbox_center_x = int(bbox_center_x * scale)
    new_bbox_center_y = int(bbox_center_y * scale)

    # Calculate the top left corner for cropping
    top_left_x = max(0, new_bbox_center_x - final_width // 2)
    top_left_y = max(0, new_bbox_center_y - final_height // 2)

    # Adjust crop coordinates to ensure the final image is 512x512
    if top_left_x + final_width > new_width:
        top_left_x = new_width - final_width
    if top_left_y + final_height > new_height:
        top_left_y = new_height - final_height

    # Crop the image to get the final 512x512 image
    final_image = resized_image.crop(
        (top_left_x, top_left_y, top_left_x + final_width, top_left_y + final_height)
    )

    return final_image


def compute_area(keypoints):
    non_none_keypoints = [keypoint for keypoint in keypoints if keypoint is not None]
    keypoints = np.array(non_none_keypoints)[:, 0:2]
    min_x = np.min(keypoints[:, 0])
    max_x = np.max(keypoints[:, 0])
    min_y = np.min(keypoints[:, 1])
    max_y = np.max(keypoints[:, 1])
    return (max_x - min_x) * (max_y - min_y)


def compute_distance_from_center(keypoints):
    center = (0.5, 0.5)
    non_none_keypoints = [keypoint for keypoint in keypoints if keypoint is not None]

    distances = [
        np.linalg.norm(np.array(keypoint[0:2]) - center)
        for keypoint in non_none_keypoints
    ]
    avg_distance = np.mean(distances)
    return avg_distance


# 0 - Nose
# 1 - Neck
# 2 - Right Shoulder
# 3 - Right Elbow
# 4 - Right Wrist
# 5 - Left Shoulder
# 6 - Left Elbow
# 7 - Left Wrist
# 8 - Right Hip
# 9 - Right Knee
# 10 - Right Ankle
# 11 - Left Hip
# 12 - Left Knee
# 13 - Left Ankle
# 14 - Right Eye
# 15 - Left Eye
# 16 - Right Ear
# 17 - Left Ear
def create_openpose(row, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    input_image = np.array(row["image"])
    input_image = HWC3(input_image)
    input_image = resize_image(input_image, IMAGE_WIDTH)

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

    img = resize_image(input_image, IMAGE_WIDTH)
    H, W, C = img.shape

    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

    detected_map = Image.fromarray(detected_map)

    return detected_map, posedict


def getBox(mask):
    # get bounding box from mask
    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return np.zeros(4)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = mask.shape
    x_min = max(0, x_min)
    x_max = min(W, x_max)
    y_min = max(0, y_min)
    y_max = min(H, y_max)
    bbox = [x_min, y_min, x_max, y_max]
    return np.array(bbox)


def draw_binary_mask(
    raw_image: np.ndarray, binary_mask: np.ndarray, mask_color=(0, 0, 255)
) -> np.ndarray:
    color_mask = np.zeros_like(raw_image, dtype=np.uint8)
    color_mask[binary_mask == 1] = mask_color
    binary_mask = np.expand_dims(binary_mask, axis=2)
    canvas = binary_mask * raw_image
    canvas = np.asarray(canvas, dtype=np.uint8)
    return Image.fromarray(canvas, mode="RGB")


def smooth_mask(mask, kernel_size=5, iterations=1):
    # Assuming 'mask' is your original mask with True/False values
    binary_mask = np.uint8(mask * 255)  # Convert to binary format (0/255)

    # Define the kernel size for morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply erosion followed by dilation
    eroded = cv2.erode(binary_mask, kernel, iterations=iterations)
    smoothed_mask = cv2.dilate(eroded, kernel, iterations=iterations)

    # Convert back to boolean format if needed
    return smoothed_mask > 0


def create_sam_images(row):
    original_image = np.array(row["image"])
    openpose_json = row["openpose_json"]
    if openpose_json is None:
        return None, None, None
    openpose_keypoints = openpose_json["keypoints"]

    points = [
        point[0:2] * np.array([IMAGE_WIDTH, IMAGE_HEIGHT])
        for point in openpose_keypoints
        if point is not None
    ]

    efficientvit_sam_predictor.set_image(original_image)
    all_masks, _, _ = efficientvit_sam_predictor.predict(
        point_coords=np.array(points),
        point_labels=np.ones(len(points)),
        multimask_output=False,
    )

    all_masks[0] = smooth_mask(all_masks[0], kernel_size=3, iterations=3)

    box = getBox(all_masks.squeeze())

    efficientvit_sam_predictor_agnostic.set_image(original_image)
    (
        predicted_agnostic_masks_inverse,
        _,
        _,
    ) = efficientvit_sam_predictor_agnostic.predict(
        box=box,
        multimask_output=False,
    )

    agnostic_masks = np.logical_and(
        all_masks, np.logical_not(predicted_agnostic_masks_inverse)
    )

    agnostic_masks[0] = smooth_mask(agnostic_masks[0], kernel_size=3, iterations=3)

    efficientvit_sam_predictor_clothes.set_image(original_image)
    predicted_clothes_masks_inverse, _, _ = efficientvit_sam_predictor_clothes.predict(
        box=box,
        multimask_output=False,
    )

    clothes_masks = np.logical_and(
        all_masks, np.logical_not(predicted_clothes_masks_inverse)
    )

    clothes_masks[0] = smooth_mask(clothes_masks[0], kernel_size=3, iterations=3)

    unknown_masks = np.logical_and(clothes_masks, agnostic_masks)

    agnostic_masks = np.logical_and(agnostic_masks, np.logical_not(unknown_masks))
    clothes_masks = np.logical_and(clothes_masks, np.logical_not(unknown_masks))

    subject_image = draw_binary_mask(original_image, all_masks.squeeze())
    agnostic_image = draw_binary_mask(original_image, agnostic_masks.squeeze())
    clothes_image = draw_binary_mask(original_image, clothes_masks.squeeze())

    return subject_image, agnostic_image, clothes_image


def get_largest_area(persons, area) -> pd.DataFrame:
    persons["area"] = 0.0
    alpha = 0.05
    # get largest area
    persons["area"] = (persons["xmax"] - persons["xmin"]) * (
        persons["ymax"] - persons["ymin"]
    )
    persons.query(f"area >= {area * alpha}", inplace=True)
    persons = persons.sort_values(by="area", ascending=False)
    # return the first row
    return persons.head(1)


def process_batch(batch) -> pd.DataFrame:
    data: pd.DataFrame = None
    batch_results = model(batch)
    for result, image in zip(batch_results.pandas().xyxy, batch):
        persons = result.query(
            f"name == 'person' and confidence >= {CONFIDENCE_THRESHOLD}"
        ).copy()
        persons = get_largest_area(persons, image.size).copy()
        # not empty
        if persons.shape[0] != 0:
            persons["image"] = None
            # reindex persons
            persons.reset_index(drop=True, inplace=True)
            persons.at[0, "image"] = Image.fromarray(image, mode="RGB")
            # merge with previous data
            data = pd.concat([data, persons], ignore_index=True)
    return data


def extract_frames(video_path, fps):
    # Load the video
    video = cv2.VideoCapture(video_path)

    frame_interval = int(video.get(cv2.CAP_PROP_FPS) / fps)

    frame_count = 0
    extracted_count = 0

    data = None
    IMAGES = []
    while True:
        success, frame = video.read()

        if not success:
            if len(IMAGES) > 0:
                data_from_batch = process_batch(IMAGES)
                if data_from_batch is not None:
                    data = pd.concat([data, data_from_batch], ignore_index=True)
            break

        if frame_count % frame_interval == 0:
            extracted_count += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            IMAGES.append(frame)

        frame_count += 1

        if len(IMAGES) > 0 and extracted_count % BATCH_SIZE == 0:
            data_from_batch = process_batch(IMAGES)
            if data_from_batch is not None:
                data = pd.concat([data, data_from_batch], ignore_index=True)
            IMAGES = []

    video.release()
    # apply lambda function to create processed images
    if data is not None:
        data["openpose_image"] = None
        data["openpose_json"] = None
        data["subject_image"] = None
        data["agnostic_image"] = None
        data["clothes_image"] = None
        for index in tqdm(range(data.shape[0])):
            row = data.iloc[index]
            image = create_processed_image(row)
            data.at[index, "image"] = image

            row = data.iloc[index]
            # create openpose image
            openpose_image, openpose_json = create_openpose(row)
            data.at[index, "openpose_image"] = openpose_image
            data.at[index, "openpose_json"] = openpose_json

            row = data.iloc[index]
            # create sam images
            subject_image, agnostic_image, clothes_image = create_sam_images(row)
            data.at[index, "subject_image"] = subject_image
            data.at[index, "agnostic_image"] = agnostic_image
            data.at[index, "clothes_image"] = clothes_image
    return data


if __name__ == "__main__":
    # get all videos from directory
    for filename in os.listdir(VIDEOS_PATH):
        # check if file and video
        if os.path.isfile(os.path.join(VIDEOS_PATH, filename)) and filename.endswith(
            ".mp4"
        ):
            image_path = os.path.join(IMAGES_PATH, filename.replace(".mp4", ""))
            os.makedirs(image_path, exist_ok=True)
            os.makedirs(os.path.join(image_path, "processed"), exist_ok=True)
            os.makedirs(os.path.join(image_path, "openpose"), exist_ok=True)
            os.makedirs(os.path.join(image_path, "subject"), exist_ok=True)
            os.makedirs(os.path.join(image_path, "agnostic"), exist_ok=True)
            os.makedirs(os.path.join(image_path, "clothes"), exist_ok=True)
            data = extract_frames(os.path.join(VIDEOS_PATH, filename), 4)
            if data is not None:
                # remove rows without image or openpose image (no person detected)
                data = data[data["image"].notna()]
                data = data[data["openpose_image"].notna()]
                data = data[data["subject_image"].notna()]
                data = data[data["agnostic_image"].notna()]
                data = data[data["clothes_image"].notna()]
                for index in tqdm(range(data.shape[0])):
                    row = data.iloc[index]
                    # save images
                    image = row["image"]
                    openpose_image = row["openpose_image"]
                    openpose_json = row["openpose_json"]
                    subject_image = row["subject_image"]
                    agnostic_image = row["agnostic_image"]
                    clothes_image = row["clothes_image"]
                    image.save(
                        os.path.join(image_path, "processed", f"{index}.jpg"), "JPEG"
                    )
                    openpose_image.save(
                        os.path.join(image_path, "openpose", f"{index}.jpg"), "JPEG"
                    )
                    subject_image.save(
                        os.path.join(image_path, "subject", f"{index}.jpg"), "JPEG"
                    )
                    agnostic_image.save(
                        os.path.join(image_path, "agnostic", f"{index}.jpg"), "JPEG"
                    )
                    clothes_image.save(
                        os.path.join(image_path, "clothes", f"{index}.jpg"), "JPEG"
                    )
                    json_file = os.path.join(image_path, "openpose", f"{index}.json")
                    json.dump(openpose_json, open(json_file, "w"))

                print(f"Extracted {data.shape[0]} frames for {filename}.")
