from typing import Dict, Any, Tuple, Union, Optional

import gc
import os
import torch
import onnx

import numpy as np

from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers.models.modeling_utils import ModelMixin
from diffusers.pipelines.onnx_utils import ORT_TO_NP_TYPE, OnnxRuntimeModel

from model.edgestyle_multicontrolnet import EdgeStyleMultiControlNetModel
from model.controllora import ControlLoRAModel, CachedControlNetModel

from optimum.onnx.utils import check_model_uses_external_data

import onnx_graphsurgeon as gs

PRETRAINED_MODEL_NAME_OR_PATH = "./models/Realistic_Vision_V5.1_noVAE"
PRETRAINED_VAE_NAME_OR_PATH = "./models/sd-vae-ft-mse"
PRETRAINED_OPENPOSE_NAME_OR_PATH = "./models/control_v11p_sd15_openpose"
CONTROLNET_MODEL_NAME_OR_PATH = "./models/EdgeStyle/controlnet"
ONNX_MODEL_NAME_OR_PATH = "./models/EdgeStyle/"
CONTROLNET_PATTERN = [0, None, 1, None, 1, None]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    model = model.eval()

    model = model.to(device)

    latent_model_input = torch.randn(2, 4, 64, 64)
    timesteps = torch.randint(0, 1, (1,))
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
    dummy_input = tuple(x.to(device) for x in dummy_input)

    predicted_noise_torch = model(*dummy_input)

    onnx_model_path = os.path.join(ONNX_MODEL_NAME_OR_PATH, 'unet', "model.onnx")
    onnx_model_dir = os.path.join(onnx_model_path.replace("model.onnx", ""))
                                                  
    os.makedirs(onnx_model_dir, exist_ok=True)

    print('exporting onnx model for unet and controlnets...') 

    # Conversion to ONNX
    # torch.onnx.export(
    #     model,
    #     dummy_input,
    #     onnx_model_path,
    #     export_params=True,
    #     input_names=[
    #         "sample",
    #         "timestep",
    #         "encoder_hidden_states",
    #         "conditioning_scale",
    #         "image_0",
    #         "image_1",
    #         "image_2",
    #         "image_3",
    #         "image_4",
    #         "image_5",
    #     ],
    #     output_names=["output"],
    #     dynamic_axes={
    #         "sample": {0: "batch_size"},  # variable length axes
    #         "encoder_hidden_states": {0: "batch_size"},            
    #         "image_0": {0: "batch_size"},
    #         "image_1": {0: "batch_size"},
    #         "image_2": {0: "batch_size"},
    #         "image_3": {0: "batch_size"},
    #         "image_4": {0: "batch_size"},
    #         "image_5": {0: "batch_size"},
    #         "output": {0: "batch_size"},
    #     },
    #     training=torch.onnx.TrainingMode.EVAL,
    #     do_constant_folding=True,
    #     # verbose=True,
    #     opset_version=17,
    # )

    # # check if external data was exported
    # onnx_model = onnx.load(onnx_model_path, load_external_data=False)
    # model_uses_external_data = check_model_uses_external_data(onnx_model)

    # if model_uses_external_data:
    #     # try free model memory
    #     del model
    #     del onnx_model
    #     gc.collect()
    #     if device.type == "cuda" and torch.cuda.is_available():
    #         torch.cuda.empty_cache()

    #     onnx_model = onnx.load(
    #         str(onnx_model_path), load_external_data=True
    #     )  # this will probably be too memory heavy for large models
    #     onnx.save(
    #         onnx_model,
    #         str(onnx_model_path),
    #         save_as_external_data=True,
    #         all_tensors_to_one_file=True,
    #         location='model.onnx' + "_data",
    #         size_threshold=1024,
    #         convert_attribute=True,
    #     )

    #     del onnx_model
    #     gc.collect()import onnx_graphsurgeon as gs
    #     if device.type == "cuda" and torch.cuda.is_available():
    #         torch.cuda.empty_cache()

    #     # delete all files except the model.onnx and onnx external data
    #     for file in os.listdir(onnx_model_dir):
    #         if file != "model.onnx" and file != "model.onnx" + "_data":
    #             os.remove(os.path.join(onnx_model_dir, file))
        
    print('running shape inference script...')

    graph = gs.import_onnx(onnx.load(onnx_model_path))
    tensors = graph.tensors()

    for tensor_key in tensors.keys():
        if '/controlnet/multi_controlnet_down_blocks' in tensor_key:
            print(tensors[tensor_key])
    

    # os.rename(f"{onnx_model_path}", f"{onnx_model_path}-no-infer")

    # run shape inference script
    # os.system(
    #     f"python -m onnxruntime.tools.symbolic_shape_infer "
    #     f"--input {onnx_model_path} "
    #     f"--output {onnx_model_path}-infer "
    #     f"--auto_merge "
    #     # f"--save_as_external_data "
    #     # f"--all_tensors_to_one_file "
    #     # f"--external_data_location model.onnx-infer_data "
    #     # f"--external_data_size_threshold 1024 "
    #     f"--verbose 3"
    # )
    
    # # delete old onnx model and data and rename new onnx model
    # # os.remove(onnx_model_path)
    # # os.remove(onnx_model_path + "_data")
    # os.rename(f"{onnx_model_path}-infer", onnx_model_path)
    
    # print('shape inference script completed...')

    print('exported onnx model for unet and controlnets...')
    print('checking onnx model...')    

    onnx_unet = OnnxRuntimeModel.from_pretrained(onnx_model_dir, provider="TensorrtExecutionProvider")
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
    print('checking how close the output is...')

    # compare the output error predicted_noise_torch is FloatTensor and predicted_noise_onnx is numpy array
    np.testing.assert_allclose(
        predicted_noise_torch.detach().cpu().numpy(),
        predicted_noise_onnx,
        rtol=1e-03,
        atol=1e-05,
    )
    print('exported onnx model for unet and controlnets is correct...')


@torch.no_grad()
def export_vae_encoder_decoder():

    vae = AutoencoderKL.from_pretrained(PRETRAINED_VAE_NAME_OR_PATH)

    vae_encoder = vae.encoder
    vae_decoder = vae.decoder

    latent_model_input = torch.randn(2, 3, 512, 512)
    dummy_input = latent_model_input

    onnx_model_path = os.path.join(
        ONNX_MODEL_NAME_OR_PATH, "encoder", "model.onnx"
    )
    onnx_model_dir = os.path.join(onnx_model_path.replace("model.onnx", ""))

    os.makedirs(onnx_model_dir, exist_ok=True)

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
        ONNX_MODEL_NAME_OR_PATH, "decoder", "model.onnx"
    )
    onnx_model_dir = os.path.join(onnx_model_path.replace("model.onnx", ""))

    os.makedirs(onnx_model_dir, exist_ok=True)

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
