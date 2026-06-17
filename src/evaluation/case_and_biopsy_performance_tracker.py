from src.utils.accumulators import DataFrameCollector


import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from src.layers.masked_prediction_module import get_bags_of_predictions


import os
from collections import defaultdict


class CaseAndBiopsyPerformanceTracker:
    """Accumulates per-case and optional per-biopsy predictions each epoch.

    Args:
        output_dir: directory for per-epoch CSV files.
        logger: callable for logging metrics (``print`` or ``Logger``).
        case_label_key: data dict key for case-level binary labels.
        biopsy_logits_key: data dict key for per-biopsy logit lists.
        biopsy_label_key: data dict key for per-biopsy binary labels.
        biopsy_grade_group_key: data dict key for per-biopsy grade groups.
        biopsy_id_key: data dict key for per-biopsy identifiers (e.g. cine_id).
    """

    def __init__(
        self,
        output_dir,
        logger=print,
        *,
        case_label_key: str = "label",
        biopsy_logits_key: str = "biopsy_logits",
        biopsy_label_key: str = "biopsy_label",
        biopsy_grade_group_key: str = "biopsy_grade_group",
        biopsy_id_key: str = "biopsy_cine_id",
        heatmap_logits_key: str | None = None,
        heatmap_needle_mask_key: str = "biopsy_needle_mask",
    ):
        self.output_dir = output_dir
        self.logger = logger
        self.case_label_key = case_label_key
        self.biopsy_logits_key = biopsy_logits_key
        self.biopsy_label_key = biopsy_label_key
        self.biopsy_grade_group_key = biopsy_grade_group_key
        self.biopsy_id_key = biopsy_id_key
        self.heatmap_logits_key = heatmap_logits_key
        self.heatmap_needle_mask_key = heatmap_needle_mask_key
        self.history = defaultdict(list)
        self.core_history = DataFrameCollector()

    def update(self, data):
        number_of_cases = len(data["case"])

        self.history["case"].extend(data["case"])
        self.history["label"].extend(data[self.case_label_key])
        self.history["logits"].append(data["logits"].detach().cpu())
        self.history["loss"].append(data["loss"].item())

        if self.biopsy_logits_key in data:
            for i in range(number_of_cases):
                core_history_data = {
                    "logits_core": data[self.biopsy_logits_key][i],
                    "biopsy_label": data[self.biopsy_label_key][i],
                    "grade_group": data[self.biopsy_grade_group_key][i],
                    "biopsy_id": data[self.biopsy_id_key][i],
                }

                if self.heatmap_logits_key and self.heatmap_logits_key in data:
                    heatmap_list = data[self.heatmap_logits_key]
                    mask_list = data.get(self.heatmap_needle_mask_key, [])

                    hmap = heatmap_list[i] if i < len(heatmap_list) else None
                    nmask = mask_list[i] if i < len(mask_list) else None

                    if hmap is not None and nmask is not None:
                        bags = get_bags_of_predictions(hmap, nmask)
                        core_history_data["avg_needle_heatmap"] = torch.tensor(
                            [b.sigmoid().mean().item() for b in bags]
                        )

                self.core_history(core_history_data)

    def complete_epoch(self, stage, epoch):
        metrics = {}
        case = self.history["case"]
        labels = torch.stack(self.history["label"]).cpu().numpy()
        logits = torch.cat(self.history["logits"])[..., 1].cpu().numpy()
        total_loss = torch.tensor(self.history["loss"]).mean().item()
        auroc = roc_auc_score(labels, logits)
        metrics[f"{stage}/total_loss"] = total_loss
        metrics[f"{stage}/auroc"] = auroc
        self.logger(
            {"epoch": epoch, f"{stage}/total_loss": total_loss, f"{stage}/auroc": auroc}
        )
        pd.DataFrame({"case": case, "logits": logits, "label": labels}).to_csv(
            os.path.join(self.output_dir, f"{stage}_epoch_{epoch}.csv")
        )

        # biopsy-level results
        if len(self.core_history.compute()) > 0:
            core_level_df = self.core_history.compute()
            core_level_df.to_csv(
                os.path.join(
                    self.output_dir, f"{stage}_epoch_{epoch}_core_results.csv"
                ),
                index=False,
            )
            core_level_cspca_auc = roc_auc_score(
                core_level_df["biopsy_label"], core_level_df["logits_core_1"]
            )
            metrics[f"{stage}/core_cspca_auroc"] = core_level_cspca_auc
            self.logger(
                {"epoch": epoch, f"{stage}/core_cspca_auroc": core_level_cspca_auc}
            )

            if "avg_needle_heatmap" in core_level_df.columns:
                avg_heatmap = core_level_df["avg_needle_heatmap"].values
                gg = core_level_df["grade_group"].values

                for task, task_labels in [
                    ("pca", (gg >= 1).astype(int)),
                    ("cspca", (gg >= 2).astype(int)),
                ]:
                    try:
                        auroc = roc_auc_score(task_labels, avg_heatmap)
                        fpr, tpr, _ = roc_curve(task_labels, avg_heatmap)
                        sens60 = float(tpr[np.argmax(fpr > 0.40)])
                        metrics[f"{stage}/heatmap_{task}_auroc"] = auroc
                        metrics[f"{stage}/heatmap_{task}_sens_at_60spe"] = sens60
                        self.logger(
                            {
                                "epoch": epoch,
                                f"{stage}/heatmap_{task}_auroc": auroc,
                                f"{stage}/heatmap_{task}_sens_at_60spe": sens60,
                            }
                        )
                    except Exception:
                        pass

        return metrics

    def reset(self):
        self.history = defaultdict(list)
        self.core_history.reset()
