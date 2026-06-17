import json
from typing import Iterable, Mapping
import torch 


def convert_batch(batch, device): 
    primitives = (bool, str, int, float, type(None))
    if type(batch) in primitives:
        return batch
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, Mapping): 
        return {k: convert_batch(v, device) for k, v in batch.items()}
    elif isinstance(batch, Iterable):
        return [convert_batch(v, device) for v in batch]
    else:
        return batch


def add_prefix(d, prefix, sep="/"): 
    return {prefix + sep + k: v for k, v in d.items()}


def load_splits_from_file(splits_filepath, split_id=None, split='train'): 
    with open(splits_filepath, 'r') as f:
        splits = json.load(f)
    if split_id is not None:
        splits = splits[split_id]
    else: 
        splits = next(iter(splits.values()))
    return splits[split]