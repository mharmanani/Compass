"""Generic training script for case-level models on the OPTIMUM dataset.

Supports:
    - Biopsy-only MIL (like prostnfound_mil)
    - Sweep-only models
    - Joint sweep + biopsy models
    - Case-level and biopsy-level (token) losses
    - Mixed-precision training
    - Gradient clipping
    - WandB logging

Usage:
    python train.py --config configs/experiment.yaml
    python train.py --config configs/experiment.yaml -o train.epochs=50 -o train.lr=1e-5
"""

import logging
import os
import random
import time
from argparse import ArgumentParser
from dataclasses import dataclass, field

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src.datasets.OPTIMUM.cases_with_sweeps_and_biopsies_dataset import (
    OPTIMUMCasesWithSweepsAndBiopsiesJSONDataset,
)
from src.evaluation.case_and_biopsy_performance_tracker import (
    CaseAndBiopsyPerformanceTracker,
)
from src.factories.optimizer import build_optimizer_v0
from src.losses.case_level_loss import CaseLevelLoss
from src.modeling import create_model
from src.modeling.utils import collate_mil_batch
from src.transforms.optimum_case_transform import OPTIMUMCaseTransform
from src.utils.logger import Logger
from src.utils.move_to_device import move_to_device
from src.utils.metric_logger import MetricLogger


def build_dataloaders(cfg):
    """Build train and val dataloaders from config."""

    ds_cfg = cfg.dataset
    json_path = ds_cfg.json_path
    splits_file = ds_cfg.get("splits_file")
    split_id = ds_cfg.get("split_id")
    load_sweeps = ds_cfg.get("load_sweeps", True)
    load_biopsies = ds_cfg.get("load_biopsies", True)
    embeddings_h5_path = ds_cfg.get("embeddings_h5_path")

    transform_cfg = OmegaConf.to_object(cfg.get("transform", {}))
    train_transform = OPTIMUMCaseTransform(**transform_cfg, train=True)
    val_transform = OPTIMUMCaseTransform(**transform_cfg, train=False)

    train_ds = OPTIMUMCasesWithSweepsAndBiopsiesJSONDataset(
        json_path,
        splits_file=splits_file,
        split_id=split_id,
        split="train",
        transform=train_transform,
        load_sweeps=load_sweeps,
        load_biopsies=load_biopsies,
        embeddings_h5_path=embeddings_h5_path,
    )
    val_ds = OPTIMUMCasesWithSweepsAndBiopsiesJSONDataset(
        json_path,
        splits_file=splits_file,
        split_id=split_id,
        split="val",
        transform=val_transform,
        load_sweeps=load_sweeps,
        load_biopsies=load_biopsies,
        embeddings_h5_path=embeddings_h5_path,
    )

    overfit_samples = cfg.train.get("overfit_samples")
    if overfit_samples:
        train_ds = torch.utils.data.Subset(train_ds, list(range(overfit_samples)))
        val_ds = torch.utils.data.Subset(train_ds, list(range(overfit_samples)))

    batch_size = cfg.train.get("batch_size", 4)
    num_workers = cfg.train.get("num_workers", 0)

    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_mil_batch,
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_mil_batch,
    )

    return train_dl, val_dl


# =============================================================================
# Main
# =============================================================================


def main(cfg):
    logging.basicConfig(level=logging.INFO)

    # --- seed ---
    seed = cfg.train.get("seed", 0)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # --- logger ---
    logger = Logger(**cfg.logger, config=cfg)

    # --- data ---
    train_loader, val_loader = build_dataloaders(cfg)

    # --- model ---
    device = cfg.train.get("device", "cuda")
    model = create_model(**cfg.model).to(device)

    # --- optimizer ---
    optimizer, scheduler = build_optimizer_v0(
        model,
        niter_per_ep=len(train_loader),
        do_batch_size_scaling=False,
        **cfg.optimizer,
    )

    criterion = CaseLevelLoss(**cfg.get("loss", {}))
    tracker = CaseAndBiopsyPerformanceTracker(
        cfg.logger.output_dir, logger, **cfg.get("tracker", {})
    )

    # --- amp ---
    use_amp = cfg.train.get("use_amp", True)
    scaler = torch.GradScaler(enabled=use_amp)
    autocast_ctx = torch.autocast(torch.device(device).type, enabled=use_amp)

    clip_grad = cfg.train.get("clip_grad")
    epochs = cfg.train.get("epochs", 100)

    save_best_model = cfg.train.get("save_best_model", False)
    tracked_metric = cfg.train.get("tracked_metric", "val/auroc")
    # Lower-is-better metrics (loss terms); everything else treated as higher-is-better.
    _lower_is_better = "loss" in tracked_metric
    best_metric_val = float("inf") if _lower_is_better else float("-inf")

    # --- training loop ---
    for epoch in range(epochs):
        # ---- train ----------------------------------------------------------
        tracker.reset()
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} Train")
        data_time = 0.0
        step_start = time.perf_counter()
        for data in pbar:
            data_elapsed = time.perf_counter() - step_start
            data_time += data_elapsed

            with autocast_ctx:
                data = move_to_device(data, cfg.train.device)
                data.update(model(**data))
                data = criterion(data)
                loss = data["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.get_total_norm(
                (p.grad for p in model.parameters() if p.grad is not None)
            )
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            logger(
                {
                    "loss": loss.item(),
                    "epoch": epoch,
                    "lr": scheduler.get_last_lr()[0],
                    "grad_norm": grad_norm.item(),
                    "data_time": data_elapsed,
                }
            )
            tracker.update(data)
            pbar.set_postfix(loss=f"{loss.item():.4f}", data=f"{data_elapsed:.2f}s")
            step_start = time.perf_counter()

        logging.info(f"Epoch {epoch} train data loading time: {data_time:.2f}s")
        logger({"epoch": epoch, "train/data_time": data_time})
        tracker.complete_epoch("train", epoch)

        # ---- val ------------------------------------------------------------
        tracker.reset()
        model.eval()
        data_time = 0.0
        step_start = time.perf_counter()
        with torch.no_grad():
            for data in tqdm(val_loader, desc=f"Epoch {epoch} Val"):
                data_elapsed = time.perf_counter() - step_start
                data_time += data_elapsed
                with autocast_ctx:
                    data = move_to_device(data, cfg.train.device)
                    data.update(model(**data))
                    data = criterion(data)
                    loss = data["loss"]
                tracker.update(data)
                step_start = time.perf_counter()

        logging.info(f"Epoch {epoch} val data loading time: {data_time:.2f}s")
        logger({"epoch": epoch, "val/data_time": data_time})
        val_metrics = tracker.complete_epoch("val", epoch)

        # ---- checkpoint -----------------------------------------------------
        if cfg.train.get("save_checkpoint"):
            ckpt_dir = os.path.join(cfg.logger.output_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                os.path.join(ckpt_dir, "latest.pth"),
            )

        if save_best_model and tracked_metric in val_metrics:
            current = val_metrics[tracked_metric]
            is_best = current < best_metric_val if _lower_is_better else current > best_metric_val
            if is_best:
                best_metric_val = current
                ckpt_dir = os.path.join(cfg.logger.output_dir, "checkpoints")
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        f"best_{tracked_metric}": best_metric_val,
                    },
                    os.path.join(ckpt_dir, "best.pth"),
                )
                logging.info(f"Saved best model (epoch {epoch}, {tracked_metric}={best_metric_val:.4f})")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--config", "-c", required=True)
    p.add_argument("-o", "-ov", "--override", action="append", default=[])
    args = p.parse_args()
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.override))
    main(cfg)
