import copy
import sys
from tqdm import tqdm
import numpy as np
import datasets
from PIL import Image
from datasets import Dataset

import torch
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim

import torchvision
from torchvision import transforms
from torchvision.transforms import functional as TVF
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from monai.losses import DiceCELoss, DiceLoss
from segment_anything.modeling.prompt_encoder import PromptEncoder
from segment_anything.utils.transforms import ResizeLongestSide

sys.path.append("efficientvit")

from efficientvit.sam_model_zoo import create_sam_model, EfficientViTSam
from efficientvit.models.efficientvit.sam import (
    MaskDecoder,
    EfficientViTSamImageEncoder,
    SamPad,
)

torch.set_float32_matmul_precision("high")

WEIGHT_URL = "efficientvit/assets/checkpoints/sam/l2.pt"
MASK_WEIGHT_URL = "efficientvit/assets/checkpoints/sam/mask_decoder.pt"
TRAINED_WEIGHT_URL = "efficientvit/assets/checkpoints/sam/trained_model.pt"
ORIGINAL_SIZE = (600, 400)

# background     0
# hat            1
# hair           2
# sunglass       3
# upper-clothes  4
# skirt          5
# pants          6
# dress          7
# belt           8
# left-shoe      9
# right-shoe     10
# face           11
# left-leg       12
# right-leg      13
# left-arm       14
# right-arm      15
# bag            16
# scarf          17

# KEEP_CATEGORIES = [4, 5, 6, 7, 8, 17]
KEEP_CATEGORIES = [1, 2, 3, 9, 10, 11, 12, 13, 14, 15, 16]


def resize(example: dict, height: int, width: int) -> dict:
    image = example["image"]
    mask = example["mask"]

    image_width, image_height = image.size
    if image_width != width or image_height != height:
        new_image = Image.new("RGB", (width, height), (255, 255, 255))
        new_image.paste(image.convert("RGB"), (0, 0))

        new_mask = Image.new("L", (width, height), (0))
        new_mask.paste(mask.convert("L"), (0, 0))
        example["image"] = new_image
        example["mask"] = new_mask

    return example


class SamResize:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        _, h, w = image.shape
        long_side = max(h, w)
        if long_side != self.size:
            return self.apply_image(image)
        else:
            return image

    def apply_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(
            image.shape[1], image.shape[2], self.size
        )
        image = TVF.resize(image, target_size)
        # convert tensor to float
        image = image.float()
        return image

    @staticmethod
    def get_preprocess_shape(
        oldh: int, oldw: int, long_side_length: int
    ) -> tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self.size})"


class TrainableModel(L.LightningModule):
    def __init__(self, model: EfficientViTSam, original_size=ORIGINAL_SIZE):
        super().__init__()
        self.loss = DiceCELoss(
            sigmoid=True,
            squared_pred=True,
            reduction="mean",
        )
        self.model = model
        self.prompt_encoder: PromptEncoder = model.prompt_encoder
        self.image_encoder: EfficientViTSamImageEncoder = model.image_encoder
        self.mask_decoder: MaskDecoder = model.mask_decoder
        self.transform = transforms.Compose(
            [
                SamResize(self.model.image_size[1]),
                transforms.Normalize(
                    mean=[123.675 / 255, 116.28 / 255, 103.53 / 255],
                    std=[58.395 / 255, 57.12 / 255, 57.375 / 255],
                ),
                SamPad(self.model.image_size[1]),
            ]
        )
        for name, parameter in model.named_parameters():
            if name.startswith("prompt_encoder") or name.startswith("image_encoder"):
                parameter.requires_grad_(False)

        self.original_size = original_size
        self.input_size = ResizeLongestSide.get_preprocess_shape(
            *self.original_size, long_side_length=self.model.image_size[0]
        )
        self.first_batch = None

    def apply_coords(self, coords: torch.Tensor) -> torch.Tensor:
        old_h, old_w = self.original_size
        new_h, new_w = self.input_size
        coords = coords.clone()
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2))
        return boxes.reshape(-1, 4)

    def getBox(self, mask: torch.Tensor) -> torch.Tensor:
        # get bounding box from mask
        y_indices, x_indices = torch.where(mask > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return torch.zeros(4, device=mask.device).float()
        x_min, x_max = torch.min(x_indices), torch.max(x_indices)
        y_min, y_max = torch.min(y_indices), torch.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = mask.shape
        x_min = torch.max(torch.as_tensor(0), x_min - np.random.randint(0, 20)).float()
        x_max = torch.min(torch.as_tensor(W), x_max + np.random.randint(0, 20)).float()
        y_min = torch.max(torch.as_tensor(0), y_min - np.random.randint(0, 20)).float()
        y_max = torch.min(torch.as_tensor(H), y_max + np.random.randint(0, 20)).float()
        bbox = [x_min, y_min, x_max, y_max]
        return torch.stack(bbox, dim=0)

    def predict_masks(self, features, sparse_embeddings, dense_embeddings, masks):
        # Preallocate tensor for predicted_masks
        predicted_masks = torch.empty_like(masks, dtype=torch.float)

        for i, (feature, sparse_embedding, dense_embedding) in enumerate(
            zip(features, sparse_embeddings, dense_embeddings)
        ):
            predicted_mask = self.predict_mask(
                feature, sparse_embedding, dense_embedding
            )
            predicted_masks[i] = predicted_mask.squeeze()

        return predicted_masks

    def predict_mask(self, feature, sparse_embedding, dense_embedding):
        dense_pe = self.prompt_encoder.get_dense_pe()
        # Predict masks for each feature
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=feature.unsqueeze(0),
            image_pe=dense_pe,
            sparse_prompt_embeddings=sparse_embedding.unsqueeze(0),
            dense_prompt_embeddings=dense_embedding.unsqueeze(0),
            multimask_output=False,
        )
        return self.model.postprocess_masks(
            low_res_masks, self.input_size, self.original_size
        )

    def apply_conditions(self, masks):
        # Create a condition tensor
        condition = torch.any(
            torch.stack([masks == category for category in KEEP_CATEGORIES]),
            dim=0,
        )
        return (masks * condition) > 0

    @torch.inference_mode()
    def training_step_frozen(self, batch, batch_idx):
        images = batch["image"]
        masks = batch["mask"]

        boxes = []
        transformed_images = []
        for image, mask in zip(images, masks):
            box = self.getBox(mask)
            box = self.apply_boxes(box.unsqueeze(0))
            boxes.append(box)
            image = image.permute(2, 0, 1) / 255.0
            transformed_image = self.transform(image)
            transformed_images.append(transformed_image)

        boxes = torch.stack(boxes, dim=0)

        transformed_images = torch.stack(transformed_images, dim=0)

        features = self.image_encoder(transformed_images)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None,
        )

        return features, sparse_embeddings, dense_embeddings

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.first_batch = batch
        masks = batch["mask"]

        features, sparse_embeddings, dense_embeddings = self.training_step_frozen(
            batch, batch_idx
        )

        # Call the new function
        predicted_masks = self.predict_masks(
            features, sparse_embeddings, dense_embeddings, masks
        )

        # Apply conditions to masks
        masks = self.apply_conditions(masks)

        loss = self.loss(predicted_masks.unsqueeze(1), masks.unsqueeze(1))

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    # epoch end
    def on_train_epoch_end(self):
        # Check if first_batch is not None and log to tensorboard
        if self.first_batch is not None:
            images = self.first_batch["image"]
            masks = self.first_batch["mask"]
            N = images.shape[0]

            features, sparse_embeddings, dense_embeddings = self.training_step_frozen(
                self.first_batch, 0
            )

            predicted_masks = self.predict_masks(
                features, sparse_embeddings, dense_embeddings, masks
            )
            predicted_masks = predicted_masks > self.model.mask_threshold
            predicted_masks = predicted_masks.bool()

            # Apply conditions to masks
            masks = self.apply_conditions(masks)
            masks = masks.bool()

            # NCWH -> NHWC
            images = images.permute(0, 3, 1, 2)

            images_with_mask_ground_truth = []
            images_with_mask = []
            for image, mask, predicted_mask in zip(images, masks, predicted_masks):
                images_with_mask.append(
                    torchvision.utils.draw_segmentation_masks(
                        image, predicted_mask, alpha=0.9, colors="blue"
                    )
                )

                images_with_mask_ground_truth.append(
                    torchvision.utils.draw_segmentation_masks(
                        image, mask, alpha=0.9, colors="blue"
                    )
                )

            grid_size = int(np.sqrt(N))

            image_grid = torchvision.utils.make_grid(images_with_mask, nrow=grid_size)

            image_grid_groud_truth = torchvision.utils.make_grid(
                images_with_mask_ground_truth, nrow=grid_size
            )
            # Log image and mask
            self.logger.experiment.add_image(
                "image",
                image_grid,
                self.current_epoch,
                dataformats="CHW",
            )
            # Log image and mask
            self.logger.experiment.add_image(
                "image_groud_truth",
                image_grid_groud_truth,
                self.current_epoch,
                dataformats="CHW",
            )

        self.first_batch = None

    # save image to tensorboard

    def validation_step(self, batch, batch_idx):
        masks = batch["mask"]

        features, sparse_embeddings, dense_embeddings = self.training_step_frozen(
            batch, batch_idx
        )

        # Call the new function
        predicted_masks = self.predict_masks(
            features, sparse_embeddings, dense_embeddings, masks
        )

        # Apply conditions to masks
        masks = self.apply_conditions(masks)

        loss = self.loss(predicted_masks.unsqueeze(1), masks.unsqueeze(1))

        self.log("validation_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def main() -> None:
    # dataset = datasets.load_dataset(
    #     "mattmdjaga/human_parsing_dataset", split="train[:100%]"
    # )

    # dataset = dataset.map(
    #     lambda example: resize(example, 600, 400), writer_batch_size=100
    # )

    # dataset.save_to_disk("dataset_resized")

    L.seed_everything(42)

    dataset = Dataset.load_from_disk("dataset_resized")
    dataset = dataset.with_format("torch", columns=["image", "mask"])

    train, val = random_split(dataset, [0.9, 0.1])

    train_loader = DataLoader(train, batch_size=16, shuffle=False, num_workers=4)
    val_loader = DataLoader(val, batch_size=16, shuffle=False, num_workers=4)

    model = TrainableModel(
        create_sam_model("l2", pretrained=True, weight_url=WEIGHT_URL)
    )

    model_checkpoint_callback = ModelCheckpoint(
        monitor="validation_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    trainer = L.Trainer(
        devices="auto",
        max_epochs=30,
        overfit_batches=0,
        fast_dev_run=False,
        callbacks=[model_checkpoint_callback],
    )
    trainer.fit(
        model,
        train_loader,
        val_loader,
        # ckpt_path="lightning_logs/version_1/checkpoints/epoch=17-step=17928.ckpt",
    )

    # take the best model
    model = TrainableModel.load_from_checkpoint(
        model_checkpoint_callback.best_model_path,
        original_size=ORIGINAL_SIZE,
        model=model.model,
    )

    torch.save(model.mask_decoder.state_dict(), MASK_WEIGHT_URL)
    torch.save(model.model.state_dict(), TRAINED_WEIGHT_URL)


if __name__ == "__main__":
    main()
