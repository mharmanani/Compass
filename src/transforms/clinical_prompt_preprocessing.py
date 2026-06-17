"""Clinical prompt preprocessing for ProstNFound-style models.

Normalizes clinical variables (PSA, age, PSA density, family history, etc.)
to model-ready float tensors.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


# ============================================================================
# Default normalization constants (from EXACTVU_NCT2013 + OPTIMUM cohorts)
# ============================================================================

PSA_MIN = 0.2
PSA_MAX = 32.95
PSA_AVG = 6.821426488456866

AGE_MIN = 0
AGE_MAX = 79
AGE_AVG = 62.5816

APPROX_PSA_DENSITY_MIN = 4.615739672282483e-06
APPROX_PSA_DENSITY_MAX = 0.000837278201784
APPROX_PSA_DENSITY_AVG = 0.000175347951594383


# ============================================================================
# Individual prompt processors
# ============================================================================


def normalize_scalar(
    value: float,
    vmin: float,
    vmax: float,
    fallback: float,
) -> float:
    """Min-max normalize a scalar, replacing NaN with *fallback* first."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        value = fallback
    return (value - vmin) / (vmax - vmin)


def process_psa(
    psa: float,
    psa_min: float = PSA_MIN,
    psa_max: float = PSA_MAX,
    psa_avg: float = PSA_AVG,
) -> float:
    """Normalize PSA to [0, 1].  NaN → population average."""
    return normalize_scalar(psa, psa_min, psa_max, psa_avg)


def process_age(
    age: float,
    age_min: float = AGE_MIN,
    age_max: float = AGE_MAX,
    age_avg: float = AGE_AVG,
) -> float:
    """Normalize age to [0, 1].  NaN → population average."""
    return normalize_scalar(age, age_min, age_max, age_avg)


def process_approx_psa_density(
    value: float,
    vmin: float = APPROX_PSA_DENSITY_MIN,
    vmax: float = APPROX_PSA_DENSITY_MAX,
    avg: float = APPROX_PSA_DENSITY_AVG,
) -> float:
    """Normalize approximate PSA density to [0, 1].  NaN → average."""
    return normalize_scalar(value, vmin, vmax, avg)


def process_family_history(value) -> float:
    """Encode family history as {1, -1, 0} for {True, False, unknown/NaN}."""
    if value is True:
        return 1.0
    elif value is False:
        return -1.0
    elif value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    return 0.0


# ============================================================================
# Batch / sample-level processor
# ============================================================================


@dataclass
class PreprocessClinicalPrompts:
    """Process clinical prompts from a dict sample into model-ready tensors.

    For each prompt key present in the input dict, the corresponding value is
    normalised and stored as a ``(1,)`` float tensor.  Missing keys are silently
    skipped unless ``fill_missing=True``, in which case default (average) values
    are used.

    This class is stateless — the normalization constants are read from class-
    level defaults (or overridden via ``__init__``).

    Args:
        psa_key: Key for PSA in the input dict.
        age_key: Key for age in the input dict.
        approx_psa_density_key: Key for approximate PSA density.
        family_history_key: Key for family history.
        fill_missing: If True, fill missing keys with population averages.
        psa_min / psa_max / psa_avg: Override PSA normalization constants.
        age_min / age_max / age_avg: Override age normalization constants.
    """

    psa_key: str = "psa"
    age_key: str = "age"
    approx_psa_density_key: str = "approx_psa_density"
    family_history_key: str = "family_history"
    fill_missing: bool = False

    # normalization constants (override-able)
    psa_min: float = PSA_MIN
    psa_max: float = PSA_MAX
    psa_avg: float = PSA_AVG
    age_min: float = AGE_MIN
    age_max: float = AGE_MAX
    age_avg: float = AGE_AVG
    approx_psa_density_min: float = APPROX_PSA_DENSITY_MIN
    approx_psa_density_max: float = APPROX_PSA_DENSITY_MAX
    approx_psa_density_avg: float = APPROX_PSA_DENSITY_AVG

    def __call__(self, sample: dict) -> dict:
        """Process prompts in-place and return the sample dict."""
        out = sample.copy()

        # PSA
        if self.psa_key in out or self.fill_missing:
            raw = out.get(self.psa_key, self.psa_avg)
            if isinstance(raw, list):
                out[self.psa_key] = torch.tensor(
                    [process_psa(v, self.psa_min, self.psa_max, self.psa_avg) for v in raw]
                ).float().unsqueeze(-1)
            else:
                out[self.psa_key] = torch.tensor(
                    [process_psa(raw, self.psa_min, self.psa_max, self.psa_avg)]
                ).float()

        # Age
        if self.age_key in out or self.fill_missing:
            raw = out.get(self.age_key, self.age_avg)
            if isinstance(raw, list):
                out[self.age_key] = torch.tensor(
                    [process_age(v, self.age_min, self.age_max, self.age_avg) for v in raw]
                ).float().unsqueeze(-1)
            else:
                out[self.age_key] = torch.tensor(
                    [process_age(raw, self.age_min, self.age_max, self.age_avg)]
                ).float()

        # Approx PSA density
        if self.approx_psa_density_key in out or self.fill_missing:
            raw = out.get(self.approx_psa_density_key, self.approx_psa_density_avg)
            if isinstance(raw, list):
                out[self.approx_psa_density_key] = torch.tensor(
                    [
                        process_approx_psa_density(
                            v, self.approx_psa_density_min,
                            self.approx_psa_density_max,
                            self.approx_psa_density_avg,
                        )
                        for v in raw
                    ]
                ).float().unsqueeze(-1)
            else:
                out[self.approx_psa_density_key] = torch.tensor(
                    [
                        process_approx_psa_density(
                            raw, self.approx_psa_density_min,
                            self.approx_psa_density_max,
                            self.approx_psa_density_avg,
                        )
                    ]
                ).float()

        # Family history
        if self.family_history_key in out or self.fill_missing:
            raw = out.get(self.family_history_key, None)
            if isinstance(raw, list):
                out[self.family_history_key] = torch.tensor(
                    [process_family_history(v) for v in raw]
                ).float()
            else:
                out[self.family_history_key] = torch.tensor(
                    process_family_history(raw)
                ).float()

        return out
