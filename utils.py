from typing import Any
import torchvision.transforms.functional as TF
import random
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel


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
