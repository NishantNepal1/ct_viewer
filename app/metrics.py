"""Dice similarity coefficient between a model mask and the reference."""

import numpy as np

from app.geometry import same_grid
from app.nifti_io import MaskImage


def dice(a: np.ndarray, b: np.ndarray) -> float | None:
    """2|A∩B| / (|A|+|B|) on boolean arrays. None when both are empty (undefined)."""
    total = int(np.count_nonzero(a)) + int(np.count_nonzero(b))
    if total == 0:
        return None
    return 2.0 * int(np.count_nonzero(a & b)) / total


def dice_report(model: MaskImage, reference: MaskImage) -> dict:
    """Dice plus voxel counts and volumes. Computed on the NIfTI grids, only if they match."""
    ok, reason = same_grid(model, reference)
    report = {
        "structure": model.name,
        "dice": None,
        "grid_ok": ok,
        "grid_reason": reason,
        "model_voxels": model.voxel_count,
        "reference_voxels": reference.voxel_count,
        "model_ml": model.volume_ml,
        "reference_ml": reference.volume_ml,
        "message": "",
    }
    if not ok:
        report["message"] = f"The model mask and the reference are on different grids ({reason}), so Dice isn't computed."
        return report
    report["dice"] = dice(model.data, reference.data)
    if report["dice"] is None:
        report["message"] = "Both masks are empty, so Dice is undefined."
    return report
