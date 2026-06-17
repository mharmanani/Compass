from collections import defaultdict
import copy
import inspect
from logging import getLogger, warn
import warnings
from omegaconf import OmegaConf
import torch
from torch import nn
import logging
from typing import Tuple, Dict, Any, Callable
from src import registry as global_registry


logger = getLogger("model_registry")
logger.setLevel(logging.INFO)


def register_model(func_or_name):
    if callable(func_or_name):
        _register_model(func_or_name.__name__, func_or_name)
        return func_or_name
    else:

        def wrapper(inner_func):
            _register_model(func_or_name, inner_func)
            return inner_func

        return wrapper


def _register_model(name, model_entrypoint):
    return global_registry.register("model", name)(model_entrypoint)


def list_models():
    return global_registry.list_names("model")


def model_help(name):
    return help(global_registry.get_entrypoint("model", name))


def create_model(
    name,
    checkpoint=None,
    load_checkpoint_kw=dict(strict=False),
    state_dict_getter=None,
    state_dict_prefix=None,
    **kwargs,
):
    model_entrypoint = global_registry.get_entrypoint("model", name)

    # If the model has a pretrained_cfg parameter, use it
    if checkpoint and "checkpoint" in inspect.signature(model_entrypoint).parameters:
        kwargs["checkpoint"] = checkpoint
        checkpoint = None
    model = model_entrypoint(**kwargs)

    if checkpoint:
        logging.info(f"Loading checkpoint {checkpoint}")
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)

        if "model" in state_dict:
            state_dict = state_dict["model"]

        if state_dict_getter is not None:
            state_dict = state_dict_getter(state_dict)

        if state_dict_prefix is not None:
            from tensordict import TensorDict
            sd_tensor_dict = TensorDict(state_dict).flatten_keys()
            sd = {
                k.replace(state_dict_prefix, "", 1): v
                for k, v in sd_tensor_dict.items()
                if k.startswith(state_dict_prefix)
            }
            state_dict = sd

        message = model.load_state_dict(state_dict, **load_checkpoint_kw)
        logging.info(f"Load message: {message}")

    model.model_kwargs = kwargs
    model.name = name
    model._medAI_registry_config = {"name": name, **kwargs}

    return model
