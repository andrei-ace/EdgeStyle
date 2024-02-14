from typing import Any

import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import math
import random
from PIL import Image, ImageDraw
import torch
import numpy as np


RESOLUTION = 512
RESOLUTION_PATCH = [16, 32, 64]

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
    def __init__(self, patch_sizes, color_percentage, color=(127, 127, 127)):
        self.patch_sizes = patch_sizes
        self.color_percentage = color_percentage
        self.color = torch.tensor(color, dtype=torch.float32) / 255.0

    def __call__(self, img):
        # Convert image to tensor
        img_tensor = TF.to_tensor(img)

        patch_size = random.choice(self.patch_sizes)

        # Calculate number of patches
        patches_horizontal = img_tensor.size(2) // patch_size
        patches_vertical = img_tensor.size(1) // patch_size

        # Flatten the 2D grid of patches and decide which patches to color
        total_patches = patches_vertical * patches_horizontal
        num_to_color = int(total_patches * self.color_percentage)
        indices_to_color = np.random.choice(total_patches, num_to_color, replace=False)

        # Color the selected patches
        for idx in indices_to_color:
            row = (idx // patches_horizontal) * patch_size
            col = (idx % patches_horizontal) * patch_size
            img_tensor[:, row : row + patch_size, col : col + patch_size] = (
                self.color.view(3, 1, 1)
            )

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
        dx, dy = random.randint(-50, 50), random.randint(-50, 50)

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
    # New colors added below
    "charcoal",
    "ivory",
    "maroon",
    "navy",
    "ultramarine",
    "cyan",
    "electric blue",
    "ruby",
    "emerald",
    "sapphire",
    "amethyst",
    "periwinkle",
    "indigo",
    "jade",
    "citrus",
    "sunflower",
    "tangerine",
    "raspberry",
    "rose",
    "violet",
    "fuchsia",
    "caramel",
    "mocha",
    "espresso",
    "sky blue",
    "forest green",
    "sea green",
    "olive drab",
    "chartreuse",
    "saffron",
    "plum",
    "sienna",
    "ochre",
    "mahogany",
    "lemon",
    "flamingo",
    "lavender blush",
    "midnight blue",
    "neon green",
    "neon pink",
    "pastel blue",
    "pastel green",
    "pastel yellow",
    "pastel pink",
    "pastel purple",
    "metallic gold",
    "metallic silver",
    "metallic copper",
    "powder blue",
    "gunmetal",
    "ash gray",
    "electric lime",
    "bubblegum pink",
    "peach puff",
    "midnight green",
    "azure",
    "carmine",
    "cerulean",
    "burnt orange",
    "burnt sienna",
    "burnt umber",
    "champagne",
    "copper",
    "crimson",
    "denim",
    "eggplant",
    "fern green",
    "firebrick",
    "flax",
    "french blue",
    "frostbite",
    "gainsboro",
    "gamboge",
    "ghost white",
    "ginger",
    "glitter",
    "harlequin",
    "honeydew",
    "hot pink",
    "hunter green",
    "iceberg",
    "inchworm",
    "jazzberry jam",
    "jet",
    "jonquil",
    "kelly green",
    "lemon chiffon",
    "licorice",
    "lilac",
    "lime",
    "linen",
    "mango",
    "mauve",
    "midnight",
    "mint cream",
    "misty rose",
    "moss green",
    "mountbatten pink",
    "navajo white",
    "neon blue",
    "old gold",
    "old lace",
    "old rose",
    "olivine",
    "onyx",
    "opal",
    "orchid",
    "pale aqua",
    "pale cerulean",
    "pale pink",
    "pale taupe",
    "pansy purple",
    "papaya whip",
    "pastel orange",
    "pear",
    "periwinkle blue",
    "persimmon",
    "pine green",
    "pink lace",
    "pistachio",
    "platinum",
    "prussian blue",
    "pumpkin",
    "quartz",
    "queen blue",
    "quicksilver",
    "quince",
    "rackley",
    "raspberry pink",
    "raw umber",
    "razzmatazz",
    "red-violet",
    "rich black",
    "robin egg blue",
    "rocket metallic",
    "roman silver",
    "rose quartz",
    "rosewood",
    "royal blue",
    "royal purple",
    "ruby red",
    "russet",
    "rust",
    "safety orange",
    "sage",
    "sandy brown",
    "sangria",
    "sapphire blue",
    "scarlet",
    "school bus yellow",
    "sea blue",
    "seafoam green",
    "sepia",
    "shadow",
    "shamrock green",
    "shocking pink",
    "silver sand",
    "sinopia",
    "slate blue",
    "slate gray",
    "smalt",
    "smokey topaz",
    "snow",
    "soap",
    "soft amber",
    "soft peach",
    "solid pink",
    "spring bud",
    "spring green",
    "steel blue",
    "straw",
    "sunset",
    "tan hide",
    "tawny",
    "telemagenta",
    "thistle",
    "tiffany blue",
    "timberwolf",
    "titanium yellow",
    "tomato",
    "toolbox",
    "topaz",
    "true blue",
    "tufts blue",
    "tulip",
    "tumbleweed",
    "turkish rose",
    "turtle green",
    "tuscan",
    "tuscan brown",
    "tuscan red",
    "tuscan tan",
    "tuscany",
    "twilight lavender",
    "tyrian purple",
    "ube",
    "ucla blue",
    "ufo green",
    "ultra pink",
    "ultramarine blue",
    "umber",
    "unbleached silk",
    "unmellow yellow",
    "upsdell red",
    "urobilin",
    "van dyke brown",
    "vanilla",
    "vanilla ice",
    "vegas gold",
    "venetian red",
    "verdigris",
    "vermilion",
    "veronica",
    "violet-blue",
    "violet-red",
    "viridian",
    "warm black",
    "wheat",
    "white smoke",
    "wild blue yonder",
    "wild orchid",
    "wild strawberry",
    "wild watermelon",
    "willow green",
    "windsor tan",
    "wine",
    "wisteria",
    "xanadu",
    "yale blue",
    "yellow green",
    "yellow orange",
    "yellow sunshine",
    "zaffre",
    "zinnwaldite brown",
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
    "cargo pants",
    "chinos",
    "culottes",
    "palazzo pants",
    "capri pants",
    "harem pants",
    "sweatpants",
    "joggers",
    "leather pants",
    "slacks",
    "trousers",
    "bell-bottoms",
    "flared pants",
    "high-rise jeans",
    "low-rise jeans",
    "mid-rise jeans",
    "skinny jeans",
    "straight-leg jeans",
    "bootcut jeans",
    "boyfriend jeans",
    "mom jeans",
    "bikini",
    "swimsuit",
    "monokini",
    "tankini",
    "burkini",
    "sarong",
    "kaftan",
    "wrap dress",
    "shift dress",
    "sheath dress",
    "sundress",
    "maxi dress",
    "mini dress",
    "midi dress",
    "bodycon dress",
    "peplum dress",
    "A-line dress",
    "ball gown",
    "cocktail dress",
    "evening gown",
    "halter dress",
    "wrap skirt",
    "pencil skirt",
    "A-line skirt",
    "maxi skirt",
    "mini skirt",
    "midi skirt",
    "pleated skirt",
    "circle skirt",
    "bubble skirt",
    "tulle skirt",
    "leather skirt",
    "denim skirt",
    "cargo shorts",
    "bermuda shorts",
    "board shorts",
    "cycling shorts",
    "leather jacket",
    "denim jacket",
    "bomber jacket",
    "flight jacket",
    "motorcycle jacket",
    "varsity jacket",
    "puffer jacket",
    "rain jacket",
    "windbreaker",
    "pea coat",
    "trench coat",
    "duster coat",
    "fur coat",
    "long coat",
    "overcoat",
    "poncho",
    "shawl",
    "stole",
    "bolero",
    "shrug",
    "crop top",
    "halter top",
    "peplum top",
    "polo shirt",
    "henley shirt",
    "camisole",
    "tube top",
    "bodysuit",
    "corset",
    "bustier",
    "bralette",
    "bandeau",
    "crop jacket",
    "utility jacket",
    "double-breasted suit",
    "single-breasted suit",
    "tuxedo",
    "morning suit",
    "evening suit",
    "tailcoat",
    "waistcoat",
    "cummerbund",
    "ascot tie",
    "bow tie",
    "necktie",
    "cravat",
    "sari",
    "lehenga",
    "churidar",
    "salwar kameez",
    "sherwani",
    "kilt",
    "toga",
    "cheongsam",
    "qipao",
    "hanbok",
    "dashiki",
    "kaftan",
    "yukata",
    "hakama",
    "dirndl",
    "lederhosen",
    "bikeshorts",
    "fishing vest",
    "flak jacket",
    "flight suit",
    "wetsuit",
    "dry suit",
    "base layer",
    "compression shorts",
    "compression shirt",
    "rash guard",
    "ski jacket",
    "ski pants",
    "snowboard pants",
    "snowsuit",
    "thermal underwear",
    "track suit",
    "training pants",
    "trench coat",
    "waders",
    "wetsuit",
    "workout leggings",
    "yoga pants",
    "zoot suit",
    "band t-shirt",
    "graphic tee",
    "linen shirt",
    "silk blouse",
    "velvet blazer",
    "wool sweater",
    "mesh top",
    "sequin dress",
    "glitter top",
    "faux fur vest",
    "neoprene swimsuit",
    "spandex leggings",
    "vinyl jacket",
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


class Augmentations:
    def __init__(
        self,
        empty_prompt,
        proportion_empty_prompts=0.0,
        proportion_empty_images=0.0,
        proportion_patchworked_images=0.0,
        proportion_cutout_images=0.0,
        proportion_patchworks=0.0,
    ):
        self.proportions = [
            proportion_empty_prompts,
            proportion_empty_prompts + proportion_empty_images,
            proportion_empty_prompts
            + proportion_empty_images
            + proportion_patchworked_images,
            proportion_empty_prompts
            + proportion_empty_images
            + proportion_patchworked_images
            + proportion_cutout_images,
        ]
        # self.empty_prompt = empty_prompt
        self.proportion_patchworks = proportion_patchworks
        self.empty_prompt = empty_prompt

    def __call__(self, examples):
        # examples is an list of dictionaries
        for i, ex in enumerate(examples):
            original = ex["original"]
            agnostic = ex["agnostic"]
            head = ex["head"]
            mask = ex["mask"]
            original_openpose = ex["original_openpose"]
            target = ex["target"]
            clothes = ex["clothes"]
            clothes_openpose = ex["clothes_openpose"]
            target2 = ex["target2"]
            clothes2 = ex["clothes2"]
            clothes_openpose2 = ex["clothes_openpose2"]
            input_ids = ex["input_ids"]

            if random.random() < self.proportions[0]:
                # Empty prompt
                input_ids = self.empty_prompt
            elif random.random() < self.proportions[1]:
                if random.random() < 0.5:
                    # Empty images
                    agnostic = Image.new("RGB", (RESOLUTION, RESOLUTION), BG_COLOR)
                    head = Image.new("RGB", (RESOLUTION, RESOLUTION), BG_COLOR)
                else:
                    if random.random() < 0.5:
                        clothes = Image.new("RGB", (RESOLUTION, RESOLUTION), BG_COLOR)
                    else:
                        clothes2 = Image.new("RGB", (RESOLUTION, RESOLUTION), BG_COLOR)
            elif random.random() < self.proportions[2]:
                # Patchworked images
                patched_transform = PatchedTransform(
                    RESOLUTION_PATCH, self.proportion_patchworks, BG_COLOR
                )
                # chose one of three
                if random.random() < 0.3333:
                    agnostic = patched_transform(agnostic)
                    head = patched_transform(head)
                elif random.random() < 0.6666:
                    clothes = patched_transform(clothes)
                else:
                    clothes2 = patched_transform(clothes2)
            elif random.random() < self.proportions[3]:
                # Cutout images
                if random.random() < 0.333:
                    center_x, center_y = self.find_center(original_openpose)
                    agnostic = self.remove_half_image(agnostic, center_x, center_y)
                    head = self.remove_half_image(head, center_x, center_y)
                elif random.random() < 0.666:
                    center_x, center_y = self.find_center(clothes_openpose)
                    clothes = self.remove_half_image(clothes, center_x, center_y)
                else:
                    center_x, center_y = self.find_center(clothes_openpose2)
                    clothes2 = self.remove_half_image(clothes2, center_x, center_y)

            examples[i] = {
                "original": original,
                "agnostic": agnostic,
                "head": head,
                "mask": mask,
                "original_openpose": original_openpose,
                "clothes": clothes,
                "clothes_openpose": clothes_openpose,
                "target": target,
                "clothes2": clothes2,
                "clothes_openpose2": clothes_openpose2,
                "target2": target2,
                "input_ids": input_ids,
            }

        return examples

    def find_center(self, openpose_image: Image.Image):
        """
        Finds the center of non-zero pixels in an image.

        Parameters:
        - openpose_image: A PIL Image object.

        Returns:
        - A tuple (center_x, center_y) representing the center of the non-zero pixels in the image.
        """
        # Convert the image to a numpy array
        img_array = np.array(openpose_image)

        # Check if the image is grayscale or RGB by examining the shape of the array
        if len(img_array.shape) == 3:
            # If RGB, convert nonzero to True/False, considering a pixel "zero" if all its channels are 0
            non_zero_pixels = np.any(img_array != 0, axis=-1)
        else:
            # If grayscale, directly check for nonzero pixels
            non_zero_pixels = img_array != 0

        # Find indices of non-zero pixels
        non_zero_indices = np.argwhere(non_zero_pixels)

        # Calculate the mean of the indices to find the center
        center = non_zero_indices.mean(axis=0)

        # Return the center as (x, y), note that numpy's array gives row (y) then column (x)
        return center[::-1]

    def remove_half_image(self, image, center_x, center_y, color=BG_COLOR):
        """
        Removes half of the image by coloring it with the given color based on a random angle line
        passing through (center_x, center_y).

        Parameters:
        - image: PIL Image object.
        - center_x, center_y: Coordinates of the center point.
        - color: The color to use for removing half of the image. Default is black.

        Returns:
        - A new PIL Image object with half of the original image colored.
        """
        # Make a copy of the image to avoid modifying the original
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)

        # Image dimensions
        width, height = img_copy.size

        # Choose a random angle between 0 and 360 degrees
        angle = random.uniform(0, 360)
        angle_rad = math.radians(angle)

        # Calculate the slope (m) and intercept (b) of the line
        if angle != 90 and angle != 270:
            m = math.tan(angle_rad)
            b = center_y - (m * center_x)

            # Function to determine if a point is below or above the line
            def is_above_line(x, y):
                # Calculate y value on the line for x
                y_on_line = m * x + b
                return y > y_on_line

        # For vertical lines, where angle is 90 or 270 degrees
        else:

            def is_above_line(x, y):
                return x > center_x if angle == 90 else x < center_x

        # Determine which side to color
        side_to_color = is_above_line(0, 0)  # Using the origin as a reference

        # Iterate over all pixels
        for x in range(width):
            for y in range(height):
                if is_above_line(x, y) == side_to_color:
                    draw.point((x, y), fill=color)

        return img_copy


class CollateFn:
    def __init__(
        self,
        empty_prompt,
        proportion_empty_prompts=0.0,
        proportion_empty_images=0.0,
        proportion_patchworked_images=0.0,
        proportion_cutout_images=0.0,
        proportion_patchworks=0.0,
        uses_vae=False,
    ):
        self.augmentations = Augmentations(
            empty_prompt=empty_prompt,
            proportion_empty_prompts=proportion_empty_prompts,
            proportion_empty_images=proportion_empty_images,
            proportion_patchworked_images=proportion_patchworked_images,
            proportion_cutout_images=proportion_cutout_images,
            proportion_patchworks=proportion_patchworks,
        )
        self.paired_transform = PairedTransform(
            RESOLUTION, (BG_COLOR, BG_COLOR, BG_COLOR_CONTROLNET)
        )
        self.uses_vae = uses_vae

    def __call__(self, examples):
        # Initialize the transforms
        examples = self.augmentations(examples)

        # Apply the paired transform and collect the data
        paired_data = [
            self.paired_transform([ex["target"], ex["clothes"], ex["clothes_openpose"]])
            for ex in examples
        ]
        target_transformed, clothes_transformed, clothes_openpose_transformed = zip(
            *paired_data
        )

        # Apply the paired transform and collect the data
        paired_data2 = [
            self.paired_transform(
                [ex["target2"], ex["clothes2"], ex["clothes_openpose2"]]
            )
            for ex in examples
        ]
        target_transformed2, clothes_transformed2, clothes_openpose_transformed2 = zip(
            *paired_data2
        )

        for i, ex in enumerate(examples):
            ex["target"] = target_transformed[i]
            ex["clothes"] = clothes_transformed[i]
            ex["clothes_openpose"] = clothes_openpose_transformed[i]
            ex["target2"] = target_transformed2[i]
            ex["clothes2"] = clothes_transformed2[i]
            ex["clothes_openpose2"] = clothes_openpose_transformed2[i]

        # Create tensors for each image type and apply the appropriate transforms
        tensor_fields = {
            "original": IMAGES_TRANSFORMS,
            "agnostic": (
                IMAGES_TRANSFORMS if self.uses_vae else CONDITIONING_IMAGES_TRANSFORMS
            ),
            "head": (
                IMAGES_TRANSFORMS if self.uses_vae else CONDITIONING_IMAGES_TRANSFORMS
            ),
            "original_openpose": CONDITIONING_IMAGES_TRANSFORMS,
            "clothes": (
                IMAGES_TRANSFORMS if self.uses_vae else CONDITIONING_IMAGES_TRANSFORMS
            ),
            "clothes_openpose": CONDITIONING_IMAGES_TRANSFORMS,
            "target": IMAGES_TRANSFORMS,
            "clothes2": (
                IMAGES_TRANSFORMS if self.uses_vae else CONDITIONING_IMAGES_TRANSFORMS
            ),
            "clothes_openpose2": CONDITIONING_IMAGES_TRANSFORMS,
            "target2": IMAGES_TRANSFORMS,
        }
        tensors = {
            field: self._stack_and_format([ex[field] for ex in examples], transform)
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
    from dataset_local import edgestyle_dataset_test

    example = edgestyle_dataset_test[0:2]

    # example is a dictionary of lists, transform it to a list of dictionaries
    example = [dict(zip(example.keys(), values)) for values in zip(*example.values())]

    augmenations = Augmentations(empty_prompt=None, proportion_cutout_images=1.0)
    # augmenations = Augmentations(
    #     empty_prompt=None, proportion_patchworked_images=1.0, proportion_patchworks=0.1
    # )
    example = augmenations(example)

    # save images
    for i, ex in enumerate(example):
        ex["agnostic"].save(f"./temp/agnostic_{i}.png")
        ex["clothes"].save(f"./temp/clothes_{i}.png")
        ex["clothes2"].save(f"./temp/clothes2_{i}.png")
