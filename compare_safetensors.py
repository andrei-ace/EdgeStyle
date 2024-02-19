import torch

from diffusers import ControlNetModel, AutoencoderKL

from model.edgestyle_multicontrolnet import EdgeStyleMultiControlNetModel
from model.controllora import ControlLoRAModel

CONTROLNET_PATTERN = [False, False, True, False, True, False]

DIR1 = "models/output_text2image_pretrained_openpose_inpaint"
DIR2 = "models/output_text2image_pretrained_openpose_inpaint/checkpoint-1000"

weight_dtype = torch.float16

pretrained_openpose_name_or_path = "lllyasviel/control_v11p_sd15_openpose"
pretrained_inpaint_name_or_path = "lllyasviel/control_v11p_sd15_inpaint"
pretrained_vae_name_or_path = "stabilityai/sd-vae-ft-mse"
vae = AutoencoderKL.from_pretrained(pretrained_vae_name_or_path)
controllora_use_vae = True

openpose = ControlNetModel.from_pretrained(
    pretrained_openpose_name_or_path, torch_dtype=weight_dtype
)
openpose.requires_grad_(False)

inpaint = ControlNetModel.from_pretrained(
    pretrained_inpaint_name_or_path, torch_dtype=weight_dtype
)
inpaint.requires_grad_(False)

load_model1 = EdgeStyleMultiControlNetModel.from_pretrained(
    DIR1,
    load_pattern=CONTROLNET_PATTERN,
    filler_controlnet=[
        inpaint,
        openpose,
        None,
        openpose,
        None,
        openpose,
    ],
    vae=vae if controllora_use_vae else None,
    controlnet_class=ControlLoRAModel,
    only_one_model=True,
)

load_model2 = EdgeStyleMultiControlNetModel.from_pretrained(
    DIR2,
    load_pattern=CONTROLNET_PATTERN,
    filler_controlnet=[
        inpaint,
        openpose,
        None,
        openpose,
        None,
        openpose,
    ],
    vae=vae if controllora_use_vae else None,
    controlnet_class=ControlLoRAModel,
    only_one_model=True,
)

state_dict1 = load_model1.state_dict()
state_dict2 = load_model2.state_dict()
for state1, state2 in zip(state_dict1, state_dict2):
    state1_value = state_dict1[state1]
    state2_value = state_dict2[state2]
    if (state1_value != state2_value).any():
        raise Exception("State is not equal")

state_dict1 = load_model1.nets[2].state_dict()
state_dict2 = load_model2.nets[2].state_dict()
for state1, state2 in zip(state_dict1, state_dict2):
    state1_value = state_dict1[state1]
    state2_value = state_dict2[state2]
    if (state1_value != state2_value).any():
        raise Exception("State is not equal")


state_dict1 = load_model1.nets[4].state_dict()
state_dict2 = load_model2.nets[4].state_dict()
for state1, state2 in zip(state_dict1, state_dict2):
    state1_value = state_dict1[state1]
    state2_value = state_dict2[state2]
    if (state1_value != state2_value).any():
        raise Exception("State is not equal")


print("All states are equal")
