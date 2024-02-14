import os
from typing import List, Tuple, Union, Optional, Dict, Any, Callable

import torch
from torch import nn

import safetensors
import diffusers
from diffusers.models.controlnet import ControlNetModel, ControlNetOutput
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.models.modeling_utils import (
    SAFETENSORS_WEIGHTS_NAME,
    SAFETENSORS_FILE_EXTENSION,
    WEIGHTS_NAME,
    _add_variant,
    _get_model_file,
)


class EdgeStyleMultiControlNetModel(MultiControlNetModel):
    def __init__(
        self, controlnets: Union[List[ControlNetModel], Tuple[ControlNetModel]]
    ):
        super().__init__(controlnets=controlnets)
        self.multi_controlnet_down_blocks = nn.ModuleList([])
        # only works for default unet architecture for SD 1.5
        self.down_output_channels = [
            320,
            320,
            320,
            320,
            640,
            640,
            640,
            1280,
            1280,
            1280,
            1280,
            1280,
        ]
        self.mid_output_channels = 1280
        for output_channel in self.down_output_channels:
            controlnet_block = nn.Sequential(
                nn.Conv2d(
                    output_channel * len(controlnets),
                    output_channel,
                    kernel_size=1,
                    groups=output_channel,
                ),
                nn.SiLU(),
                nn.Dropout2d(0.5),
                nn.Conv2d(
                    output_channel,
                    output_channel,
                    kernel_size=1,
                    groups=output_channel,
                ),
            )
            # controlnet_block = zero_module(controlnet_block)
            # controlnet_block = ones_module(controlnet_block)
            self.multi_controlnet_down_blocks.append(controlnet_block)

        output_channel = self.mid_output_channels
        controlnet_block = nn.Sequential(
            nn.Conv2d(
                output_channel * len(controlnets),
                output_channel,
                kernel_size=1,
                groups=output_channel,
            ),
            nn.SiLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(
                output_channel,
                output_channel,
                kernel_size=1,
                groups=output_channel,
            ),
        )  # type: ignore
        # controlnet_block = zero_module(controlnet_block)
        # controlnet_block = ones_module(controlnet_block)
        self.multi_controlnet_mid_block = controlnet_block

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: List[torch.tensor],
        conditioning_scale: List[float],
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guess_mode: bool = False,
        return_dict: bool = True,
    ) -> Union[ControlNetOutput, Tuple]:
        down_block_res_samples = []
        mid_block_res_sample = []
        for i, (image, scale, controlnet) in enumerate(
            zip(controlnet_cond, conditioning_scale, self.nets)
        ):
            down_samples, mid_sample = controlnet(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=image,
                conditioning_scale=scale,
                class_labels=class_labels,
                timestep_cond=timestep_cond,
                attention_mask=attention_mask,
                added_cond_kwargs=added_cond_kwargs,
                cross_attention_kwargs=cross_attention_kwargs,
                guess_mode=guess_mode,
                return_dict=return_dict,
            )

            down_block_res_samples.append(down_samples)
            mid_block_res_sample.append(mid_sample)

        # concatenate the samples by channel
        # interleave the channels from the controlnets
        # down_block_res_samples = [
        #     torch.cat(down_block_res_sample, dim=1)
        #     for down_block_res_sample in zip(*down_block_res_samples)
        # ]
        down_block_res_samples = interleave_tensors_from_list_of_lists(
            zip(*down_block_res_samples)
        )
        # mid_block_res_sample = torch.cat(mid_block_res_sample, dim=1)  # type: ignore
        mid_block_res_sample = interleave_tensors(mid_block_res_sample)

        # apply controlnet blocks
        for i, controlnet_block in enumerate(self.multi_controlnet_down_blocks):
            down_block_res_samples[i] = controlnet_block(down_block_res_samples[i])
        mid_block_res_sample = self.multi_controlnet_mid_block(mid_block_res_sample)

        return down_block_res_samples, mid_block_res_sample

    def state_dict(self, *args, **kwargs):
        # only save self.multi_controlnet_down_blocks and self.multi_controlnet_mid_block
        multi_controlnet_down_blocks_dict = (
            self.multi_controlnet_down_blocks.state_dict()
        )
        multi_controlnet_mid_block_dict = self.multi_controlnet_mid_block.state_dict()
        # append the keys with the prefix
        multi_controlnet_down_blocks_dict = {
            f"multi_controlnet_down_blocks.{k}": v
            for k, v in multi_controlnet_down_blocks_dict.items()
        }
        multi_controlnet_mid_block_dict = {
            f"multi_controlnet_mid_block.{k}": v
            for k, v in multi_controlnet_mid_block_dict.items()
        }
        # merge the state_dicts keys
        state_dict = {
            **multi_controlnet_down_blocks_dict,
            **multi_controlnet_mid_block_dict,
        }
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        # remove the prefix from the keys
        multi_controlnet_down_blocks_dict = {
            k.split("multi_controlnet_down_blocks.")[-1]: v
            for k, v in state_dict.items()
            if k.startswith("multi_controlnet_down_blocks.")
        }
        multi_controlnet_mid_block_dict = {
            k.split("multi_controlnet_mid_block.")[-1]: v
            for k, v in state_dict.items()
            if k.startswith("multi_controlnet_mid_block.")
        }
        # load the state_dicts
        self.multi_controlnet_down_blocks.load_state_dict(
            multi_controlnet_down_blocks_dict
        )
        self.multi_controlnet_mid_block.load_state_dict(multi_controlnet_mid_block_dict)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        save_function: Optional[Callable] = None,
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        push_to_hub: bool = False,
        **kwargs,
    ):
        if os.path.isfile(save_directory):
            raise ValueError(
                f"Provided path ({save_directory}) should be a directory, not a file"
            )
            return

        os.makedirs(save_directory, exist_ok=True)

        # Only save the model itself if we are using distributed training
        model_to_save = self

        # # Attach architecture to the config
        # # Save the config
        # if is_main_process:
        #     model_to_save.save_config(save_directory)

        # Save the model
        state_dict = model_to_save.state_dict()

        weights_name = SAFETENSORS_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
        weights_name = _add_variant(weights_name, variant)

        # Save the model
        if safe_serialization:
            safetensors.torch.save_file(
                state_dict,
                os.path.join(save_directory, weights_name),
                metadata={"format": "pt"},
            )
        else:
            torch.save(state_dict, os.path.join(save_directory, weights_name))

        idx = 0
        model_path_to_save = save_directory
        if "save_pattern" in kwargs:
            save_pattern = kwargs["save_pattern"]
        else:
            save_pattern = [True] * len(self.nets)
        for i, controlnet in enumerate(self.nets):
            # get the save_pattern from kwargs
            if save_pattern[i]:
                controlnet.save_pretrained(
                    os.path.join(model_path_to_save, f"controlnet_{idx}"),
                    is_main_process=is_main_process,
                    save_function=save_function,
                    safe_serialization=safe_serialization,
                    variant=variant,
                )
                idx += 1

    @classmethod
    def from_pretrained(
        cls, pretrained_model_path: Optional[Union[str, os.PathLike]], **kwargs
    ):
        controlnet_class = kwargs.pop("controlnet_class", ControlNetModel)
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        subfolder = kwargs.pop("subfolder", None)
        variant = kwargs.pop("variant", None)
        use_safetensors = kwargs.pop("use_safetensors", None)

        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

        user_agent = {
            "diffusers": diffusers.__version__,
            "file_type": "model",
            "framework": "pytorch",
        }

        if use_safetensors:
            try:
                model_file = _get_model_file(
                    pretrained_model_path,
                    weights_name=_add_variant(SAFETENSORS_WEIGHTS_NAME, variant),
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                    # commit_hash=commit_hash,
                )
            except IOError as e:
                if not allow_pickle:
                    raise e
                pass
        if model_file is None:
            model_file = _get_model_file(
                pretrained_model_path,
                weights_name=_add_variant(WEIGHTS_NAME, variant),
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
                # commit_hash=commit_hash,
            )

        idx = 0
        controlnets = []
        model_path_to_load = pretrained_model_path
        if not os.path.isdir(model_path_to_load):  # type: ignore
            raise ValueError(
                f"Provided path ({pretrained_model_path}) should be a directory"
            )

        if "load_pattern" in kwargs:
            load_pattern = kwargs["load_pattern"]
        else:
            raise ValueError("load_pattern must be provided")

        if "filler_controlnet" in kwargs:
            filler_controlnet = kwargs["filler_controlnet"]
        else:
            filler_controlnet = None

        if "vae" in kwargs:
            vae = kwargs["vae"]
        else:
            vae = None

        for i in range(len(load_pattern)):
            if load_pattern[i]:
                controlnet = controlnet_class.from_pretrained(
                    os.path.join(model_path_to_load, f"controlnet_{idx}"), **kwargs  # type: ignore
                )
                if controlnet.uses_vae:
                    if vae is None:
                        raise ValueError(
                            "vae must be provided if any of the controlnets uses a vae"
                        )
                    controlnet.set_autoencoder(vae)
                controlnets.append(controlnet)
                idx += 1
            else:
                controlnets.append(filler_controlnet)

        model = cls(controlnets)
        state_dict = load_state_dict(model_file, variant=variant)
        model._convert_deprecated_attention_blocks(state_dict)

        if torch_dtype is not None and not isinstance(torch_dtype, torch.dtype):
            raise ValueError(
                f"{torch_dtype} needs to be of type `torch.dtype`, e.g. `torch.float16`, but is {type(torch_dtype)}."
            )
        elif torch_dtype is not None:
            model = model.to(torch_dtype)

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()
        return model


def load_state_dict(
    checkpoint_file: Union[str, os.PathLike], variant: Optional[str] = None
):
    """
    Reads a checkpoint file, returning properly formatted errors if they arise.
    """
    try:
        file_extension = os.path.basename(checkpoint_file).split(".")[-1]
        if file_extension == SAFETENSORS_FILE_EXTENSION:
            return safetensors.torch.load_file(checkpoint_file, device="cpu")
        else:
            return torch.load(checkpoint_file, map_location="cpu")
    except Exception as e:
        try:
            with open(checkpoint_file) as f:
                if f.read().startswith("version"):
                    raise OSError(
                        "You seem to have cloned a repository without having git-lfs installed. Please install "
                        "git-lfs and run `git lfs install` followed by `git lfs pull` in the folder "
                        "you cloned."
                    )
                else:
                    raise ValueError(
                        f"Unable to locate the file {checkpoint_file} which is necessary to load this pretrained "
                        "model. Make sure you have saved the model properly."
                    ) from e
        except (UnicodeDecodeError, ValueError):
            raise OSError(
                f"Unable to load weights from checkpoint file for '{checkpoint_file}' "
                f"at '{checkpoint_file}'. "
                "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True."
            )


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def ones_module(module):
    for p in module.parameters():
        nn.init.ones_(p)
    return module


def interleave_tensors(tensors):
    """
    Interleaves the channels of a list of tensors.

    Args:
    - tensors (list of Tensor): A list of tensors of the same shape [batch_size, channels, height, width].

    Returns:
    - Tensor: A tensor with channels from the input tensors interleaved.
    """
    assert all(
        t.size() == tensors[0].size() for t in tensors
    ), "All tensors must have the same shape, they have: " + str(
        [t.size() for t in tensors]
    )
    stacked = torch.stack(
        tensors, dim=1
    )  # Shape: [batch_size, num_tensors, channels, height, width]
    batch_size, num_tensors, channels, height, width = stacked.shape
    interleaved = (
        stacked.permute(0, 2, 1, 3, 4).contiguous().view(batch_size, -1, height, width)
    )
    return interleaved


def interleave_tensors_from_list_of_lists(tensor_lists):
    """
    For each list of tensors, interleave their channels and return a list of interleaved tensors.

    Args:
    - tensor_lists (list of list of Tensor): A list where each element is a list of tensors of the same shape.

    Returns:
    - list of Tensor: A list where each element is a tensor with channels from the input tensors interleaved.
    """
    return [interleave_tensors(tensors) for tensors in tensor_lists]
