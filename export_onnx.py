from typing import Dict, Any, Tuple, Union, Optional

import os
import torch
import onnx

import numpy as np

from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers.models.modeling_utils import ModelMixin
from diffusers.pipelines.onnx_utils import ORT_TO_NP_TYPE, OnnxRuntimeModel

from model.edgestyle_multicontrolnet import EdgeStyleMultiControlNetModel
from model.controllora import ControlLoRAModel, CachedControlNetModel

PRETRAINED_MODEL_NAME_OR_PATH = "./models/Realistic_Vision_V5.1_noVAE"
PRETRAINED_VAE_NAME_OR_PATH = "./models/sd-vae-ft-mse"
PRETRAINED_OPENPOSE_NAME_OR_PATH = "./models/control_v11p_sd15_openpose"
CONTROLNET_MODEL_NAME_OR_PATH = "./models/EdgeStyle/controlnet"
ONNX_MODEL_NAME_OR_PATH = "./models/EdgeStyle/unet"
CONTROLNET_PATTERN = [0, None, 1, None, 1, None]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OnnxUNetAndControlnets(ModelMixin):
    def __init__(
        self, unet: UNet2DConditionModel, controlnet: EdgeStyleMultiControlNetModel
    ):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        conditioning_scale: torch.Tensor,
        image_0: torch.FloatTensor,
        image_1: torch.FloatTensor,
        image_2: torch.FloatTensor,
        image_3: torch.FloatTensor,
        image_4: torch.FloatTensor,
        image_5: torch.FloatTensor,
    ) -> torch.FloatTensor:
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=[image_0, image_1, image_2, image_3, image_4, image_5],
            conditioning_scale=conditioning_scale,
            return_dict=False,
        )

        noise_pred = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )[0]

        return noise_pred


@torch.no_grad()
def export_unet():
    vae = AutoencoderKL.from_pretrained(PRETRAINED_VAE_NAME_OR_PATH)

    unet = UNet2DConditionModel.from_pretrained(
        PRETRAINED_MODEL_NAME_OR_PATH,
        subfolder="unet",
    )

    openpose = CachedControlNetModel.from_pretrained(PRETRAINED_OPENPOSE_NAME_OR_PATH)

    controlnet = EdgeStyleMultiControlNetModel.from_pretrained(
        CONTROLNET_MODEL_NAME_OR_PATH,
        vae=vae,
        controlnet_class=ControlLoRAModel,
        load_pattern=CONTROLNET_PATTERN,
        static_controlnets=[None, openpose, None, openpose, None, openpose],
    )
    for net in controlnet.nets:
        if net is not openpose:
            net.tie_weights(unet)
            # net.fuse_lora()

    model = OnnxUNetAndControlnets(unet, controlnet)

    # set all parameters to not require gradients
    for param in model.parameters():
        param.requires_grad = False

    onnx_model_path = os.path.join(ONNX_MODEL_NAME_OR_PATH, "unet", "model.onnx")
    os.makedirs(onnx_model_path.replace("model.onnx", ""), exist_ok=True)

    latent_model_input = torch.randn(2, 4, 64, 64)
    timesteps = torch.randint(0, 1, (1,)).long()
    prompt_embeds = torch.randn(2, 77, 768)

    conditioning_scale = torch.randn(6)

    image_0 = torch.randn(2, 320, 64, 64)
    image_1 = torch.randn(2, 3, 512, 512)
    image_2 = torch.randn(2, 320, 64, 64)
    image_3 = torch.randn(2, 3, 512, 512)
    image_4 = torch.randn(2, 320, 64, 64)
    image_5 = torch.randn(2, 3, 512, 512)

    dummy_input = (
        latent_model_input,
        timesteps,
        prompt_embeds,
        conditioning_scale,
        image_0,
        image_1,
        image_2,
        image_3,
        image_4,
        image_5,
    )

    model = model.to(DEVICE)
    dummy_input = tuple(x.to(DEVICE) for x in dummy_input)

    predicted_noise_torch = model(*dummy_input)

    # Conversion to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        export_params=True,
        input_names=[
            "sample",
            "timestep",
            "encoder_hidden_states",
            "conditioning_scale",
            "image_0",
            "image_1",
            "image_2",
            "image_3",
            "image_4",
            "image_5",
        ],
        output_names=["output"],
        training=torch.onnx.TrainingMode.EVAL,
        # verbose=True,
        opset_version=17,
    )

    onnx_unet = OnnxRuntimeModel.from_pretrained(
        onnx_model_path.replace("model.onnx", "")
    )
    predicted_noise_onnx = onnx_unet(
        sample=latent_model_input,
        timestep=timesteps,
        encoder_hidden_states=prompt_embeds,
        conditioning_scale=conditioning_scale,
        image_0=image_0,
        image_1=image_1,
        image_2=image_2,
        image_3=image_3,
        image_4=image_4,
        image_5=image_5,
    )[0]

    # compare the output error predicted_noise_torch is FloatTensor and predicted_noise_onnx is numpy array
    np.testing.assert_allclose(
        predicted_noise_torch.detach().cpu().numpy(),
        predicted_noise_onnx,
        rtol=1e-03,
        atol=1e-05,
    )

    # onnx.checker.check_model(onnx_model_path)

    # latent_model_input = torch.randn(2, 4, 64, 64)
    # timesteps = torch.randint(0, 1, (1,)).long()
    # prompt_embeds = torch.randn(2, 77, 768)
    # down_block_res_samples = [
    #     torch.randn(2, 320, 64, 64),
    #     torch.randn(2, 320, 64, 64),
    #     torch.randn(2, 320, 64, 64),
    #     torch.randn(2, 320, 32, 32),
    #     torch.randn(2, 640, 32, 32),
    #     torch.randn(2, 640, 32, 32),
    #     torch.randn(2, 640, 16, 16),
    #     torch.randn(2, 1280, 16, 16),
    #     torch.randn(2, 1280, 16, 16),
    #     torch.randn(2, 1280, 8, 8),
    #     torch.randn(2, 1280, 8, 8),
    #     torch.randn(2, 1280, 8, 8),
    # ]
    # mid_block_res_sample = torch.randn(2, 1280, 8, 8)

    # dummy_input = (
    #     latent_model_input,
    #     timesteps,
    #     prompt_embeds,
    #     None,
    #     None,
    #     None,
    #     None,
    #     None,
    #     *down_block_res_samples,
    #     mid_block_res_sample,
    #     None,
    #     None,
    #     False,
    # )

    # onnx_model_path = os.path.join(
    #     UNET_PRETRAINED_MODEL_NAME_OR_PATH + "-onnx", "unet", "model.onnx"
    # )

    # os.makedirs(onnx_model_path.replace("model.onnx", ""), exist_ok=True)

    # # Conversion to ONNX
    # torch.onnx.export(
    #     unet,
    #     dummy_input,
    #     onnx_model_path,
    #     export_params=True,
    #     input_names=[
    #         "sample",
    #         "timestep",
    #         "encoder_hidden_states",
    #         "down_block_additional_residuals_0",
    #         "down_block_additional_residuals_1",
    #         "down_block_additional_residuals_2",
    #         "down_block_additional_residuals_3",
    #         "down_block_additional_residuals_4",
    #         "down_block_additional_residuals_5",
    #         "down_block_additional_residuals_6",
    #         "down_block_additional_residuals_7",
    #         "down_block_additional_residuals_8",
    #         "down_block_additional_residuals_9",
    #         "down_block_additional_residuals_10",
    #         "down_block_additional_residuals_11",
    #         "mid_block_additional_residual",
    #     ],
    #     output_names=["output"],
    #     # dynamic_axes={
    #     #     "sample": {0: "batch_size"},  # variable length axes
    #     #     "timestep": {0: "batch_size"},
    #     #     "encoder_hidden_states": {0: "batch_size"},
    #     #     "down_block_additional_residuals_0": {0: "batch_size"},
    #     #     "down_block_additional_residuals_1": {0: "batch_size"},
    #     #     "down_block_additional_residuals_2": {0: "batch_size"},
    #     #     "down_block_additional_residuals_3": {0: "batch_size"},
    #     #     "down_block_additional_residuals_4": {0: "batch_size"},
    #     #     "down_block_additional_residuals_5": {0: "batch_size"},
    #     #     "down_block_additional_residuals_6": {0: "batch_size"},
    #     #     "down_block_additional_residuals_7": {0: "batch_size"},
    #     #     "down_block_additional_residuals_8": {0: "batch_size"},
    #     #     "down_block_additional_residuals_9": {0: "batch_size"},
    #     #     "down_block_additional_residuals_10": {0: "batch_size"},
    #     #     "down_block_additional_residuals_11": {0: "batch_size"},
    #     #     "mid_block_additional_residual": {0: "batch_size"},
    #     #     "output": {0: "batch_size"},
    #     # },
    # )

    # onnx.checker.check_model(onnx_model_path)


@torch.no_grad()
def export_vae_encoder_decoder():

    vae = AutoencoderKL.from_pretrained(PRETRAINED_VAE_NAME_OR_PATH)

    vae_encoder = vae.encoder
    vae_decoder = vae.decoder

    latent_model_input = torch.randn(2, 3, 512, 512)
    dummy_input = latent_model_input

    onnx_model_path = os.path.join(
        PRETRAINED_VAE_NAME_OR_PATH + "-onnx", "encoder", "model.onnx"
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
        PRETRAINED_VAE_NAME_OR_PATH + "-onnx", "decoder", "model.onnx"
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
    # export_vae_encoder_decoder()
