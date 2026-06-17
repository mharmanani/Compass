import argparse
from collections import defaultdict
from pydantic import BaseModel, Field
from typing import Literal, Mapping, Optional
import torch
from torch.optim.lr_scheduler import LambdaLR
from traitlets import default
from src.utils.lr_scheduler import LinearWarmupCosineAnnealing
from src.utils.distributed import get_world_size
from src.utils.cosine_scheduler import cosine_scheduler
from src.optimizer.lars import LARS
import logging


__all__ = [
    "build_optimizer_v0",
    "build_optimizer_v1",
    "OptimizerCfg",
    "OptimGroupCfg",
]


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_optimizer_v0(
    model,
    niter_per_ep,
    warmup_epochs=0,
    lr=1e-4,
    wd=0,
    batch_size_per_gpu=32,
    optimizer_type: Literal["adamw", "sgd"] = "adamw",
    scheduler_type: Literal["cosine", "constant"] = "cosine",
    scheduler_epochs=100,
    do_batch_size_scaling=True,
):
    """
    Build a simple optimizer and scheduler without parameter groups.
    """

    lr = lr * batch_size_per_gpu / 256 if do_batch_size_scaling else lr

    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=wd
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    if scheduler_type == "cosine":
        scheduler = LambdaLR(
            optimizer,
            LinearWarmupCosineAnnealing(
                0, warmup_epochs, scheduler_epochs, niter_per_ep
            ),
        )
    else:
        scheduler = LambdaLR(optimizer, lambda it: 1)

    return optimizer, scheduler


class OptimGroupCfg(BaseModel):
    base_lr: float = 1e-4
    other_pg_settings: dict = Field(default_factory=dict)
    min_lr: float = 0.
    warmup_epochs: int | None = None
    frozen_epochs: int | None = None
    weight_decay: float = 0.04
    weight_decay_end: float | None = None
    num_annealing_phases: int = 1


class OptimizerCfg(BaseModel):

    optimizer_type: Literal["adamw", "sgd", "lars"] = "adamw"
    momentum_teacher: float | None = None
    param_groups_config: OptimGroupCfg | dict[str, OptimGroupCfg] | None = None
    scheduler_type: Literal["cosine", "constant"] = "cosine"


def _get_param_group_dict_by_attribute(model): 
    pg_dict = defaultdict(list)
    for parameter in model.parameters(): 
        if not parameter.requires_grad:
            continue
        if not hasattr(parameter, "param_group"): 
            pg_dict['default'].append(parameter)
        else:
            pg_dict[parameter.param_group].append(parameter)

    return pg_dict


def build_optimizer_v1(
    model_dict, num_iters_per_epoch, conf: OptimizerCfg, batch_size_per_gpu, epochs
):
    """
    Build optimizer with parameter groups, each having its own learning rate and weight decay schedules.
    Also builds a momentum schedule, lr schedule, and weight decay schedule.
    """

    def compute_true_lr_from_base_lr(base_lr):
        return base_lr * batch_size_per_gpu * get_world_size() / 256.0

    params_groups = []
    lr_schedulers = []
    wd_schedulers = []

    named_parameter_groups = {k: v.named_parameters() for k, v in model_dict.items()}

    for group_name, named_parameters in named_parameter_groups.items():

        named_parameters = list(named_parameters)
        logging.info(
            f"Creating optimizer group for {group_name}: {len(named_parameters)} params, config: {conf.param_groups_config.get(group_name)}"
        )

        opt_conf_for_group = conf.param_groups_config.get(group_name)
        if opt_conf_for_group is None:
            raise ValueError(
                f"Trying to configure optimizer for group {group_name}, but did not find it in config (keys {list(conf.param_groups_config.keys())})"
            )

        # get regularized vs. non-regularized parameter groups
        regularized = []
        not_regularized = []
        for name, param in named_parameters:
            if not param.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)

        regularized = {"params": regularized}
        params_groups.append(regularized)
        not_regularized = {"params": not_regularized, "weight_decay": 0.0}
        params_groups.append(not_regularized)
        for _ in range(
            2
        ):  # lr has the same schedule for regularized vs. non_regularized
            lr_schedulers.append(
                cosine_scheduler(
                    compute_true_lr_from_base_lr(
                        opt_conf_for_group.base_lr
                    ),  # linear scaling rule
                    opt_conf_for_group.min_lr,
                    epochs,
                    num_iters_per_epoch,
                    warmup_epochs=opt_conf_for_group.warmup_epochs,
                    frozen_epochs=opt_conf_for_group.frozen_epochs,
                    num_annealing_phases=opt_conf_for_group.num_annealing_phases,
                )
            )
        wd_schedulers.append(
            cosine_scheduler(
                opt_conf_for_group.weight_decay,
                opt_conf_for_group.weight_decay_end,
                epochs,
                num_iters_per_epoch,
            )
        )
        # second group receives 0 wd
        wd_schedulers.append(cosine_scheduler(0, 0, epochs, num_iters_per_epoch))

    # ============ preparing optimizer ... ============
    if conf.optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif conf.optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            params_groups, lr=0, momentum=0.9
        )  # lr is set by scheduler
    elif conf.optimizer_type == "lars":
        optimizer = LARS(params_groups)  # to use with convnet and large batches
    else:
        raise ValueError(f"Unknown optimizer type: {conf.optimizer_type}")

    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = cosine_scheduler(
        conf.momentum_teacher, 1, epochs, num_iters_per_epoch
    )

    return optimizer, lr_schedulers, wd_schedulers, momentum_schedule


def build_optimizer_v2(model, niter_per_epoch, epochs, config: OptimizerCfg): 

    parameter_groups_dict = _get_param_group_dict_by_attribute(model)
    msg = ""
    for k, v in parameter_groups_dict.items():
        msg += f"Parameter group '{k}': {len(v)} parameters.\n"
    logging.info(f"Found parameter groups:\n{msg}")

    parameter_groups_config_dict = config.param_groups_config
    if not isinstance(parameter_groups_config_dict, dict): 
        logging.info("Using single parameter group config for all parameters because param_groups_config is not a dict.")
        parameter_groups_config_dict = {'default': parameter_groups_config_dict}
        parameter_groups_dict_merged = {}
        for k in parameter_groups_dict.keys():
            parameter_groups_dict_merged['default'].extend(parameter_groups_dict[k])
        parameter_groups_dict = parameter_groups_dict_merged

    parameter_groups_config_dict: dict[str, OptimGroupCfg] = parameter_groups_config_dict

    class LRSchedule: 
        def __init__(self, optim_config: OptimizerCfg, optimizer_group_config: OptimGroupCfg): 
            self.optim_config = optim_config
            self.optimizer_group_config = optimizer_group_config
            if self.optim_config.scheduler_type == "cosine": 
                self.scheduler = cosine_scheduler(
                    base_value=optimizer_group_config.base_lr,
                    final_value=optimizer_group_config.min_lr,
                    epochs=epochs,
                    niter_per_ep=niter_per_epoch,  # placeholder, will be set properly in __call__
                    warmup_epochs=optimizer_group_config.warmup_epochs or 0,
                    frozen_epochs=optimizer_group_config.frozen_epochs or 0,
                    num_annealing_phases=optimizer_group_config.num_annealing_phases,
                )
            else: 
                self.scheduler

        def __call__(self, iter): 
            if self.optim_config.scheduler_type == "constant": 
                # implement cosine schedule here
                return 1.0
            else: 
                return self.scheduler[iter]
                
    parameter_groups = []
    schedules = []
    for group_name, parameters in parameter_groups_dict.items():

        logging.info(f"Parameter group '{group_name}': {sum(p.numel() for p in parameters)} parameters.")

        if group_name not in parameter_groups_config_dict:
            raise ValueError(f"Parameter group '{group_name}' not found in config.")

        parameter_groups_i = {}
        parameter_groups_i["params"] = parameters
        parameter_groups_i["lr"] = parameter_groups_config_dict[group_name].base_lr
        parameter_groups_i["weight_decay"] = parameter_groups_config_dict[group_name].weight_decay
        parameter_groups_i.update(
            parameter_groups_config_dict[group_name].other_pg_settings
        )
        parameter_groups.append(parameter_groups_i)

        schedules.append(
            LRSchedule(
                optim_config=config,
                optimizer_group_config=parameter_groups_config_dict[group_name],
            )
        )
    
    if config.optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(parameter_groups)
    elif config.optimizer_type == "sgd":
        optimizer = torch.optim.SGD(parameter_groups)
    elif config.optimizer_type == "lars":
        raise NotImplementedError("LARS optimizer not yet implemented in v2.")
        #optimizer = LARS(parameter_groups)
    else:
        raise ValueError(f"Unknown optimizer type: {config.optimizer_type}")

    scheduler = LambdaLR(optimizer, lr_lambda=schedules)

    return optimizer, scheduler

    # def _compute_warmup_cosine_lr(base_lr, min_lr, epochs, iters_per_epoch, warmup_epochs): 
    #     return cosine_scheduler(
    #         base_lr,
    #         min_lr,
    #         epochs,
    #         iters_per_epoch,
    #         warmup_epochs=warmup_epochs,
    #     )