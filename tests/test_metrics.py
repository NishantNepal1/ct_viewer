import numpy as np
import pytest

from app.metrics import dice, dice_report
from app.nifti_io import MaskImage
from tests.helpers import expected_ras_affine


def mask(voxels, affine=None, shape=(6, 5, 4)) -> MaskImage:
    data = np.zeros(shape, dtype=bool)
    for v in voxels:
        data[v] = True
    return MaskImage("lesions", data, expected_ras_affine() if affine is None else affine)


def test_identical():
    a = mask([(0, 0, 0), (1, 1, 1)]).data
    assert dice(a, a) == 1.0


def test_disjoint():
    assert dice(mask([(0, 0, 0)]).data, mask([(1, 1, 1)]).data) == 0.0


def test_partial_overlap_by_hand():
    # |A| = 3, |B| = 2, |A∩B| = 1  ->  2*1 / (3+2) = 0.4
    a = mask([(0, 0, 0), (1, 0, 0), (2, 0, 0)]).data
    b = mask([(2, 0, 0), (3, 0, 0)]).data
    assert dice(a, b) == pytest.approx(0.4)


def test_both_empty_is_undefined():
    report = dice_report(mask([]), mask([]))
    assert report["dice"] is None
    assert "undefined" in report["message"]


def test_grid_mismatch_gives_reason_not_number():
    shifted = expected_ras_affine().copy()
    shifted[0, 3] += 5.0
    report = dice_report(mask([(0, 0, 0)]), mask([(0, 0, 0)], affine=shifted))
    assert report["dice"] is None and report["grid_ok"] is False
    assert "different grids" in report["message"]


def test_volumes_in_ml():
    report = dice_report(mask([(0, 0, 0), (1, 0, 0)]), mask([(0, 0, 0)]))
    voxel_ml = 0.6 * 0.8 * 2.5 / 1000
    assert report["model_voxels"] == 2 and report["reference_voxels"] == 1
    assert report["model_ml"] == pytest.approx(2 * voxel_ml)
    assert report["dice"] == pytest.approx(2 / 3)
