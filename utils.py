from typing import Any

import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import random
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np


RESOLUTION = 512
RESOLUTION_PATCH = 16

BG_COLOR = (127, 127, 127)
BG_COLOR_CONTROLNET = (0, 0, 0)

IMAGES_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize(RESOLUTION, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(RESOLUTION),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

CONDITIONING_IMAGES_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize(RESOLUTION, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(RESOLUTION),
        transforms.ToTensor(),
    ]
)


class PatchedTransform:
    def __init__(self, patch_size, color_percentage, color=(127, 127, 127)):
        self.patch_size = patch_size
        self.color_percentage = color_percentage
        self.color = torch.tensor(color, dtype=torch.float32) / 255.0

    def __call__(self, img):
        # Convert image to tensor
        img_tensor = TF.to_tensor(img)

        # Calculate number of patches
        patches_horizontal = img_tensor.size(2) // self.patch_size
        patches_vertical = img_tensor.size(1) // self.patch_size

        # Flatten the 2D grid of patches and decide which patches to color
        total_patches = patches_vertical * patches_horizontal
        num_to_color = int(total_patches * self.color_percentage)
        indices_to_color = np.random.choice(total_patches, num_to_color, replace=False)

        # Color the selected patches
        for idx in indices_to_color:
            row = (idx // patches_horizontal) * self.patch_size
            col = (idx % patches_horizontal) * self.patch_size
            img_tensor[
                :, row : row + self.patch_size, col : col + self.patch_size
            ] = self.color.view(3, 1, 1)

        return TF.to_pil_image(img_tensor)


class PairedTransform:
    def __init__(self, output_size, padding_colors):
        self.output_size = output_size
        self.padding_colors = padding_colors

    def cleanup_border(self, img, border_color, border_size=1):
        """Clean up the border by replacing black pixels with the border color."""
        pixels = img.load()

        for i in range(img.width):
            for j in range(border_size):  # Iterate over the border width
                if pixels[i, j] == (0, 0, 0):
                    pixels[i, j] = border_color
                if pixels[i, img.height - 1 - j] == (0, 0, 0):
                    pixels[i, img.height - 1 - j] = border_color

        for i in range(img.height):
            for j in range(border_size):  # Iterate over the border width
                if pixels[j, i] == (0, 0, 0):
                    pixels[j, i] = border_color
                if pixels[img.width - 1 - j, i] == (0, 0, 0):
                    pixels[img.width - 1 - j, i] = border_color

        return img

    def __call__(self, images):
        # Ensure that the number of images and padding colors are the same
        if len(images) != len(self.padding_colors):
            raise ValueError(
                "The number of images and padding colors must be the same."
            )

        # Random zoom
        scale = random.uniform(0.8, 1.2)
        new_size = int(self.output_size * scale)

        # Random shift
        dx, dy = random.randint(-20, 20), random.randint(-20, 20)

        # Apply resizing, padding, cropping, and shifting to all images
        transformed_images = []
        for img, color in zip(images, self.padding_colors):
            resized_img = TF.resize(
                img, new_size, interpolation=Image.NEAREST
            )  # Adjust interpolation method

            if scale < 1.0:
                padding = int((self.output_size - new_size) / 2)
                padding_tuple = (
                    padding,
                    padding,
                    padding,
                    padding,
                )  # Padding for left, top, right, bottom
                resized_img = TF.pad(
                    resized_img, padding_tuple, fill=color, padding_mode="constant"
                )  # Apply specified padding color
            elif scale > 1.0:
                top = random.randint(0, new_size - self.output_size)
                left = random.randint(0, new_size - self.output_size)
                resized_img = TF.crop(
                    resized_img, top, left, self.output_size, self.output_size
                )

            resized_img = TF.affine(
                resized_img,
                angle=0,
                translate=[dx, dy],
                scale=1,
                shear=0,
                fill=color,
            )

            # Post-transformation cleanup to fix the 1-pixel border issue
            cleaned_img = self.cleanup_border(resized_img, color)

            transformed_images.append(cleaned_img)

            final_images = []
            for i, img in enumerate(transformed_images):
                current_size = img.size
                if current_size != self.output_size:
                    # If the image is larger than the output size, crop it
                    if (
                        current_size[0] > self.output_size
                        or current_size[1] > self.output_size
                    ):
                        img = TF.center_crop(img, self.output_size)

                    # If the image is smaller than the output size, pad it
                    elif (
                        current_size[0] < self.output_size
                        or current_size[1] < self.output_size
                    ):
                        padding_needed = [
                            max(self.output_size - current_size[0], 0),
                            max(self.output_size - current_size[1], 0),
                        ]
                        padding = (
                            padding_needed[0] // 2,
                            padding_needed[1] // 2,
                            padding_needed[0] - padding_needed[0] // 2,
                            padding_needed[1] - padding_needed[1] // 2,
                        )
                        img = TF.pad(
                            img, padding, fill=self.padding_colors[i]
                        )  # Using the first padding color as default

                final_images.append(img)

        return final_images


COLORS = [
    "black",
    "white",
    "gray",
    "navy blue",
    "blue",
    "light blue",
    "red",
    "burgundy",
    "pink",
    "purple",
    "lavender",
    "green",
    "olive",
    "lime green",
    "yellow",
    "mustard",
    "orange",
    "brown",
    "beige",
    "tan",
    "khaki",
    "gold",
    "silver",
    "bronze",
    "teal",
    "turquoise",
    "aqua",
    "coral",
    "salmon",
    "peach",
    "magenta",
    "cream",
    "mint green",
]

CLOTHING_ITEMS = [
    "tshirt",
    "jeans",
    "blues jeans",
    "shorts",
    "pants",
    "sweater",
    "dress",
    "shorts",
    "skirt",
    "jacket",
    "coat",
    "parka",
    "suit",
    "blazer",
    "sweatshirt",
    "hoodie",
    "cardigan",
    "tank top",
    "blouse",
    "vest",
    "tunic",
    "romper",
    "jumpsuit",
    "cape",
    "gown",
    "robe",
    "kimono",
    "onesie",
    "pajamas",
    "leggings",
    "stockings",
    "tights",
]


class BestEmbeddings:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def __call__(self, images) -> Any:
        best_colors = self.find_best(COLORS, images)
        best_clothing_items = self.find_best(CLOTHING_ITEMS, images)

        best_prompts = []
        for best_color_image, best_clothing_item_image in zip(
            best_colors, best_clothing_items
        ):
            best_prompts.append(
                "edgestyle, " + ", ".join(best_color_image + best_clothing_item_image)
            )
        return best_prompts

    def find_best(self, items, images, n=2):
        inputs = self.processor(
            text=items, images=images, return_tensors="pt", padding=True
        ).to(self.model.device)
        outputs = self.model(**inputs)

        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        # get the best n predictions
        best = torch.argsort(probs, dim=1, descending=True)[:, :n]

        best_items = []
        for i in range(len(images)):
            best_items_per_image = []
            for j in range(n):
                best_items_per_image.append(items[best[i][j]])
            best_items.append(best_items_per_image)

        return best_items


class InverseEmbeddings:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids) -> Any:
        prompts = self.tokenizer.batch_decode(
            input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return prompts


class TextEmbeddings:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, prompts) -> Any:
        inputs = self.tokenizer(
            prompts,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids


class CollateFn:
    def __init__(self, proportion_patchworks=0.0, proportion_patchworks_images=0.0):
        self.proportion_patchworks = proportion_patchworks
        self.proportion_patchworks_images = proportion_patchworks_images

    def __call__(self, examples):
        # Initialize the transforms
        paired_transform = PairedTransform(
            RESOLUTION, (BG_COLOR, BG_COLOR, BG_COLOR_CONTROLNET)
        )
        patched_transform = PatchedTransform(
            RESOLUTION_PATCH, self.proportion_patchworks, BG_COLOR
        )

        # Apply the paired transform and collect the data
        paired_data = [
            paired_transform([ex["target"], ex["clothes"], ex["clothes_openpose"]])
            for ex in examples
        ]
        target_transformed, clothes_transformed, clothes_openpose_transformed = zip(
            *paired_data
        )

        # Apply patched transform to various image types
        patched_original = [
            patched_transform(ex["original"])
            if random.random() < self.proportion_patchworks_images
            else ex["original"]
            for ex in examples
        ]
        # patched_original = [ex["original"] for ex in examples]
        patched_agnostic = [
            patched_transform(ex["agnostic"])
            if random.random() < self.proportion_patchworks_images
            else ex["agnostic"]
            for ex in examples
        ]
        patched_clothes = [
            patched_transform(image)
            if random.random() < self.proportion_patchworks_images
            else image
            for image in clothes_transformed
        ]

        # Reassemble the examples with the transformed images
        transformed_examples = [
            {
                "original": po,
                "agnostic": pa,
                "mask": ex["mask"],
                "original_openpose": ex["original_openpose"],
                "target": tt,
                "clothes": pc,
                "clothes_openpose": co,
                "input_ids": ex["input_ids"],
            }
            for po, pa, tt, pc, co, ex in zip(
                patched_original,
                patched_agnostic,
                target_transformed,
                patched_clothes,
                clothes_openpose_transformed,
                examples,
            )
        ]

        # Create tensors for each image type and apply the appropriate transforms
        tensor_fields = {
            "original": IMAGES_TRANSFORMS,
            "agnostic": IMAGES_TRANSFORMS,
            "original_openpose": CONDITIONING_IMAGES_TRANSFORMS,
            "clothes": IMAGES_TRANSFORMS,
            "clothes_openpose": CONDITIONING_IMAGES_TRANSFORMS,
            "target": IMAGES_TRANSFORMS,
        }
        tensors = {
            field: self._stack_and_format(
                [ex[field] for ex in transformed_examples], transform
            )
            for field, transform in tensor_fields.items()
        }
        tensors["input_ids"] = torch.stack(
            [torch.from_numpy(np.array(ex["input_ids"])) for ex in examples]
        )

        return tensors

    def _stack_and_format(self, images, transform):
        stacked_images = torch.stack([transform(img) for img in images])
        return stacked_images.to(memory_format=torch.contiguous_format).float()


if __name__ == "__main__":
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    best_embeddings = BestEmbeddings(model, processor)

    best = best_embeddings(
        [
            Image.open("temp/test_data/original_clothes.jpg"),
            Image.open("temp/test_data/target_clothes.jpg"),
            Image.open("temp/test_data/original_clothes.jpg"),
            Image.open("temp/test_data/target_clothes.jpg"),
        ]
    )

    input_ids = TextEmbeddings(processor.tokenizer)(best)
    prompts = InverseEmbeddings(processor.tokenizer)(input_ids)

    print(prompts)
