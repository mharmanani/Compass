from logging import getLogger
from torch import distributed as dist
import os
import torch


logger = getLogger()


def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1


def init_distributed(port=40112, rank_and_world_size=(None, None)):

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    rank, world_size = rank_and_world_size
    os.environ["MASTER_ADDR"] = "localhost"

    if (rank is None) or (world_size is None):
        try:
            world_size = int(os.environ.get("SLURM_NTASKS", "1"))
            rank = int(os.environ["SLURM_PROCID"])
            os.environ["MASTER_ADDR"] = os.environ.get("HOSTNAME", 'localhost')
        except Exception:
            logger.info("SLURM vars not set (distributed training not available)")
            world_size, rank = 1, 0
            return world_size, rank

    try:
        os.environ["MASTER_PORT"] = str(port)
        torch.distributed.init_process_group(
            backend="nccl", world_size=world_size, rank=rank
        )
    
        if len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) > 1:
            torch.cuda.set_device(dist.get_rank())
        else:
            torch.cuda.set_device(0)
            
    except Exception as e:
        world_size, rank = 1, 0
        logger.info(f"distributed training not available {e}")

    return world_size, rank


def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if not is_dist_avail_and_initialized():
        return tensor
    
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True