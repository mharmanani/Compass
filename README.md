# Compass: Prostate Cancer Detection Needs Multi-View Context

Compass is a multi-instance learning model for case-level prostate cancer detection
from micro-ultrasound sweeps and systematic biopsy cores.

## Installation

```bash
git clone https://github.com/mharmanani/Compass
cd Compass
pip install -e .
```

### Requirements

- Python >= 3.9
- PyTorch >= 2.0
- See `setup.py` for full dependency list

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `PNF_PLUS_CHECKPOINT` | **Yes** | Path to the ProstNFound+ pretrained backbone checkpoint (MedSAM-based) |
| `WANDB_API_KEY` | No | WandB API key for experiment logging |

The ProstNFound+ checkpoint provides the image encoder backbone. It is a MedSAM
checkpoint fine-tuned on micro-ultrasound data and is available upon request from
the authors.

## Data

Compass was trained on the **OPTIMUM micro-ultrasound dataset**, which is a
proprietary clinical dataset and is **not included** in this repository.

The dataset is specified via a `samples.json` file. The top level is a JSON object
keyed by case ID. Each case has the following structure:

```json
{
  "UA-004": {
    "case": "UA-004",
    "center": "UA",
    "age": 67,
    "psa": 6.5,
    "max_grade_group": 2.0,
    "study_arm": "MicroUS",
    "cines": [
      {
        "cine_id": "UA-004-001",
        "image_path": ["/path/to/core_001.png"],
        "roll_path": "/path/to/core_001_roll.npz",
        "needle_mask_path": "/path/to/core_001_mask.png",
        "GG": 2.0,
        "% Cancer": 40.0,
        "PRI-MUS": "4"
      }
    ],
    "sweeps": [
      {
        "sweep_id": "UA-004-S001",
        "sweep_path": "/path/to/sweep_001.h5",
        "roll_path": "/path/to/sweep_001_roll.npz"
      }
    ]
  }
}
```

**Case fields:**
- `case` (str): case identifier (matches the top-level key)
- `center` (str): imaging site
- `age` (int): patient age at biopsy
- `psa` (float): PSA value (ng/mL)
- `max_grade_group` (float): case-level Gleason grade group (0 = benign)
- `study_arm` (str): study protocol arm

**Biopsy core fields** (`cines` list):
- `cine_id` (str): unique core identifier
- `image_path` (list of str): path(s) to biopsy B-mode PNG; the last path is used
- `roll_path` (str): path to `.npz` file containing a `roll` key — a 1-D array of probe roll angles (degrees) captured during the biopsy; the last frame is the canonical angle
- `needle_mask_path` (str, optional): path to PNG needle segmentation mask
- `GG` (float): Gleason grade group (NaN or 0 = benign)
- `% Cancer` (float): percentage of core involved by cancer (NaN = benign)
- `PRI-MUS` (str/int, optional): PRIMUS micro-ultrasound suspicion score (1–5)

**Sweep fields** (`sweeps` list):
- `sweep_id` (str): unique sweep identifier
- `sweep_path` (str): path to HDF5 file; must contain a `sweep` dataset of shape `(T, H, W)` or `(T, H, W, C)` — a sequence of B-mode frames
- `roll_path` (str): path to `.npz` file containing a `roll` key — a 1-D array of probe roll angles (degrees) over the sweep

The case-level cancer label (`label`) is derived automatically: a case is positive
(label = 1) if any biopsy core has GG ≥ 2 (clinically significant cancer).

The patient splits file (`resources/splits.json`) defines cross-validation folds
using case IDs.

## Training

```bash
python train.py --config configs/compass.yaml \
    -o runtime.fold=0 -o runtime.group_id=run1
```

To run all 5 folds:

```bash
for fold in 0 1 2 3 4; do
    python train.py --config configs/compass.yaml \
        -o runtime.fold=$fold -o runtime.group_id=run1
done
```

Configuration options can be overridden with `-o key=value`.

## Model Architecture

Compass uses:
- **ProstNFound+** as a frozen image encoder backbone (MedSAM-based)
- A **BERT-based MIL aggregator** to integrate biopsy core features
- A sweep video encoder for multi-view temporal context
- Clinical prompt conditioning (PSA, age)
- Sinusoidal roll positional embeddings

## Citation

```bibtex
@inproceedings{wilson2026compass,
  title   = {Compass: Prostate Cancer Detection Needs Multi-View Context},
  author  = {Wilson, Paul F. R. and Harmanani, Mohamed and Guo, Zhuoxin and Dzikunu, Obed K. and Cash, Hannes and Kinnaird, Adam and Wodlinger, Brian and Abolmaesumi, Purang and Mousavi, Parvin},
  booktitle = {International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year    = {2026},
}
```

## License

MIT License

Copyright (c) 2026 Paul F. R. Wilson, Mohamed Harmanani, Zhuoxin Guo,
Obed K. Dzikunu, Hannes Cash, Adam Kinnaird, Brian Wodlinger,
Purang Abolmaesumi, and Parvin Mousavi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
