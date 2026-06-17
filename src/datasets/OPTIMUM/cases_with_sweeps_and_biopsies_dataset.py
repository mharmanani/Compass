from collections import defaultdict
import json
import math

import h5py
import numpy as np
from src.utils.misc import load_splits_from_file
from copy import deepcopy
import PIL


class OPTIMUMCasesWithSweepsAndBiopsiesJSONDataset:
    def __init__(
        self,
        json_path,
        splits_file=None,
        split_id=None,
        split: str | None = "train",
        transform=None,
        load_sweeps: bool = True,
        load_biopsies: bool = True,
        embeddings_h5_path: str | None = None,
        sweep_key: str = "sweep", 
        needle_mask_key: str = "needle_mask_path",
    ):
        self.transform = transform
        self.load_sweeps = load_sweeps
        self.load_biopsies = load_biopsies
        self.embeddings_h5_path = embeddings_h5_path
        self.sweep_key = sweep_key
        self.needle_mask_key = needle_mask_key
        self._embeddings_h5 = None  # lazy-opened h5py.File handle
        # When pre-extracted embeddings are available, skip loading raw images
        # (the heavy I/O) but still load all metadata.
        self._skip_images = embeddings_h5_path is not None

        with open(json_path, "r") as f:
            self.samples = json.load(f)

        if splits_file:
            case_ids = load_splits_from_file(splits_file, split_id, split)
            self.samples = {k: v for k, v in self.samples.items() if k in case_ids}

        self.case_ids = sorted(list((self.samples.keys())))

    @property
    def _h5(self):
        """Lazy-open the embeddings HDF5 file (one handle per worker process)."""
        if self._embeddings_h5 is None and self.embeddings_h5_path is not None:
            self._embeddings_h5 = h5py.File(self.embeddings_h5_path, "r")
        return self._embeddings_h5

    def __getitem__(self, idx):
        outputs = {}

        sample = deepcopy(self.samples[self.case_ids[idx]])
        case_id = self.case_ids[idx]

        # --- Pre-extracted backbone embeddings (optional) ---
        if self._h5 is not None and case_id in self._h5:
            grp = self._h5[case_id]

            # Biopsy embeddings: (N_biopsies, D) → list of (D,) arrays
            if "biopsy_embeddings" in grp:
                biopsy_emb_all = grp["biopsy_embeddings"][:]  # (N, D)
                outputs["biopsy_backbone_embeddings"] = [
                    biopsy_emb_all[i] for i in range(biopsy_emb_all.shape[0])
                ]

            # Biopsy image feature maps: (N, 256, H, W) float16 → list of (256, H, W) arrays
            if "biopsy_image_feats" in grp:
                raw = grp["biopsy_image_feats"][:]           # (N, 256, H, W) float16 numpy
                outputs["biopsy_image_feats"] = list(raw)   # list of (256, H, W) arrays

            # Sweep embeddings: sweep_0/embeddings, sweep_1/embeddings, …
            num_sweeps = int(grp["num_sweeps"][()]) if "num_sweeps" in grp else 0
            if num_sweeps > 0 and self.load_sweeps:
                sweep_embs = []
                for j in range(num_sweeps):
                    ds_name = f"sweep_{j}/embeddings"
                    if ds_name in grp:
                        sweep_embs.append(grp[ds_name][:])  # (T_j, D)
                outputs["sweep_backbone_embeddings"] = sweep_embs

        if self.load_biopsies:
            for biopsy_data_i in sample.pop("cines"):
                roll_path = biopsy_data_i["roll_path"]
                roll_series = np.load(roll_path)["roll"]
                roll = roll_series[-1]
                pct_cancer = biopsy_data_i["% Cancer"]

                gg = biopsy_data_i["GG"]
                if math.isnan(gg):
                    gg = 0

                try:
                    primus = int(biopsy_data_i["PRI-MUS"])
                except:
                    primus = None
                cine_id = biopsy_data_i["cine_id"]

                if not self._skip_images:
                    image_paths = biopsy_data_i["image_path"]
                    image = np.array(PIL.Image.open(image_paths[-1]))
                    outputs.setdefault("biopsy_images", []).append(image)

                needle_mask_path = biopsy_data_i.get(self.needle_mask_key)
                if needle_mask_path:
                    mask = (np.array(PIL.Image.open(needle_mask_path).convert("L")) * 255).astype('int')
                    outputs.setdefault("biopsy_needle_masks", []).append(mask)

                outputs.setdefault("biopsy_roll_angle", []).append(roll)
                outputs.setdefault("biopsy_grade_group", []).append(gg)
                outputs.setdefault("biopsy_primus", []).append(primus)
                outputs.setdefault("biopsy_cine_id", []).append(cine_id)
                outputs.setdefault("biopsy_pct_cancer", []).append(pct_cancer)
        else:
            sample.pop("cines", None)

        if self.load_sweeps:
            for sweep_data_i in sample.pop("sweeps"):
                outputs.setdefault("sweep_id", []).append(sweep_data_i["sweep_id"])
                roll_path = sweep_data_i["roll_path"]
                roll_series = np.load(roll_path)["roll"]
                outputs.setdefault("sweep_roll_series", []).append(roll_series)
                if not self._skip_images:
                    with h5py.File(sweep_data_i["sweep_path"]) as f:
                        outputs.setdefault("sweep", []).append(f[self.sweep_key][:])
        else:
            sample.pop("sweeps", None)
        outputs.update(sample)

        if self.transform:
            outputs = self.transform(outputs)

        return outputs

    def __len__(self):
        return len(self.case_ids)

    def load_roll_npz(self, fpath):
        return np.load(fpath)["roll"]

    def load_sweep(self, fpath):
        with h5py.File(fpath) as f:
            return f["sweep"][:]
