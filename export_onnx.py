from typing import Dict, Any, Tuple, Union, Optional

import os
import torch
import onnx

from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers.models.unet_2d_condition import UNet2DConditionOutput

UNET_PRETRAINED_MODEL_NAME_OR_PATH = "./models/Realistic_Vision_V5.1_noVAE"
VAE_PRETRAINED_MODEL_NAME_OR_PATH = "./models/sd-vae-ft-mse"


class OnnxUNet2DConditionModel(UNet2DConditionModel):
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals_0: Optional[torch.Tensor] = None,
        down_block_additional_residuals_1: Optional[torch.Tensor] = None,
        down_block_additional_residuals_2: Optional[torch.Tensor] = None,
        down_block_additional_residuals_3: Optional[torch.Tensor] = None,
        down_block_additional_residuals_4: Optional[torch.Tensor] = None,
        down_block_additional_residuals_5: Optional[torch.Tensor] = None,
        down_block_additional_residuals_6: Optional[torch.Tensor] = None,
        down_block_additional_residuals_7: Optional[torch.Tensor] = None,
        down_block_additional_residuals_8: Optional[torch.Tensor] = None,
        down_block_additional_residuals_9: Optional[torch.Tensor] = None,
        down_block_additional_residuals_10: Optional[torch.Tensor] = None,
        down_block_additional_residuals_11: Optional[torch.Tensor] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        if (
            down_block_additional_residuals_0 is not None
            or down_block_additional_residuals_1 is not None
            or down_block_additional_residuals_2 is not None
            or down_block_additional_residuals_3 is not None
            or down_block_additional_residuals_4 is not None
            or down_block_additional_residuals_5 is not None
            or down_block_additional_residuals_6 is not None
            or down_block_additional_residuals_7 is not None
            or down_block_additional_residuals_8 is not None
            or down_block_additional_residuals_9 is not None
            or down_block_additional_residuals_10 is not None
            or down_block_additional_residuals_11 is not None
        ):

            down_block_additional_residuals = [
                down_block_additional_residuals_0,
                down_block_additional_residuals_1,
                down_block_additional_residuals_2,
                down_block_additional_residuals_3,
                down_block_additional_residuals_4,
                down_block_additional_residuals_5,
                down_block_additional_residuals_6,
                down_block_additional_residuals_7,
                down_block_additional_residuals_8,
                down_block_additional_residuals_9,
                down_block_additional_residuals_10,
                down_block_additional_residuals_11,
            ]
        else:
            down_block_additional_residuals = None
        return super().forward(
            sample,
            timestep,
            encoder_hidden_states,
            class_labels,
            timestep_cond,
            attention_mask,
            cross_attention_kwargs,
            added_cond_kwargs,
            down_block_additional_residuals,
            mid_block_additional_residual,
            down_intrablock_additional_residuals,
            encoder_attention_mask,
            return_dict,
        )


@torch.no_grad()
def export_unet():
    unet = OnnxUNet2DConditionModel.from_pretrained(
        UNET_PRETRAINED_MODEL_NAME_OR_PATH,
        subfolder="unet",
    )

    latent_model_input = torch.randn(2, 4, 64, 64)
    timesteps = torch.randint(0, 1, (2,)).long()
    prompt_embeds = torch.randn(2, 77, 768)
    down_block_res_samples = [
        torch.randn(2, 320, 64, 64),
        torch.randn(2, 320, 64, 64),
        torch.randn(2, 320, 64, 64),
        torch.randn(2, 320, 32, 32),
        torch.randn(2, 640, 32, 32),
        torch.randn(2, 640, 32, 32),
        torch.randn(2, 640, 16, 16),
        torch.randn(2, 1280, 16, 16),
        torch.randn(2, 1280, 16, 16),
        torch.randn(2, 1280, 8, 8),
        torch.randn(2, 1280, 8, 8),
        torch.randn(2, 1280, 8, 8),
    ]
    mid_block_res_sample = torch.randn(2, 1280, 8, 8)

    dummy_input = (
        latent_model_input,
        timesteps,
        prompt_embeds,
        None,
        None,
        None,
        None,
        None,
        *down_block_res_samples,
        mid_block_res_sample,
        None,
        None,
        False,
    )

    onnx_model_path = os.path.join(
        UNET_PRETRAINED_MODEL_NAME_OR_PATH + "-onnx", "unet", "model.onnx"
    )

    os.makedirs(onnx_model_path.replace("model.onnx", ""), exist_ok=True)

    # Conversion to ONNX
    torch.onnx.export(
        unet,
        dummy_input,
        onnx_model_path,
        export_params=True,
        input_names=[
            "sample",
            "timestep",
            "encoder_hidden_states",
            "down_block_additional_residuals_0",
            "down_block_additional_residuals_1",
            "down_block_additional_residuals_2",
            "down_block_additional_residuals_3",
            "down_block_additional_residuals_4",
            "down_block_additional_residuals_5",
            "down_block_additional_residuals_6",
            "down_block_additional_residuals_7",
            "down_block_additional_residuals_8",
            "down_block_additional_residuals_9",
            "down_block_additional_residuals_10",
            "down_block_additional_residuals_11",
            "mid_block_additional_residual",
        ],
        output_names=["output"],
        dynamic_axes={
            "sample": {0: "batch_size"},  # variable length axes
            "timestep": {0: "batch_size"},
            "encoder_hidden_states": {0: "batch_size"},
            "down_block_additional_residuals_0": {0: "batch_size"},
            "down_block_additional_residuals_1": {0: "batch_size"},
            "down_block_additional_residuals_2": {0: "batch_size"},
            "down_block_additional_residuals_3": {0: "batch_size"},
            "down_block_additional_residuals_4": {0: "batch_size"},
            "down_block_additional_residuals_5": {0: "batch_size"},
            "down_block_additional_residuals_6": {0: "batch_size"},
            "down_block_additional_residuals_7": {0: "batch_size"},
            "down_block_additional_residuals_8": {0: "batch_size"},
            "down_block_additional_residuals_9": {0: "batch_size"},
            "down_block_additional_residuals_10": {0: "batch_size"},
            "down_block_additional_residuals_11": {0: "batch_size"},
            "mid_block_additional_residual": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    onnx.checker.check_model(onnx_model_path)


@torch.no_grad()
def export_vae_encoder_decoder():

    vae = AutoencoderKL.from_pretrained(VAE_PRETRAINED_MODEL_NAME_OR_PATH)

    vae_encoder = vae.encoder
    vae_decoder = vae.decoder

    latent_model_input = torch.randn(2, 3, 512, 512)
    dummy_input = latent_model_input

    onnx_model_path = os.path.join(
        VAE_PRETRAINED_MODEL_NAME_OR_PATH + "-onnx", "encoder", "model.onnx"
    )

    os.makedirs(onnx_model_path.replace("model.onnx", ""), exist_ok=True)

    torch.onnx.export(
        vae_encoder,
        dummy_input,
        onnx_model_path,
        export_params=True,
        input_names=[
            "image",
        ],
        output_names=["output"],
        dynamic_axes={
            "image": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )
    # check the output has correct shape

    onnx.checker.check_model(onnx_model_path)

    latent_model_input = torch.randn(2, 4, 64, 64)
    dummy_input = latent_model_input

    onnx_model_path = os.path.join(
        VAE_PRETRAINED_MODEL_NAME_OR_PATH + "-onnx", "decoder", "model.onnx"
    )

    os.makedirs(onnx_model_path.replace("model.onnx", ""), exist_ok=True)

    torch.onnx.export(
        vae_decoder,
        dummy_input,
        onnx_model_path,
        export_params=True,
        input_names=[
            "latent_sample",
        ],
        output_names=["output"],
        dynamic_axes={
            "latent_sample": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )
    onnx.checker.check_model(onnx_model_path)


if __name__ == "__main__":

    export_unet()
    export_vae_encoder_decoder()
