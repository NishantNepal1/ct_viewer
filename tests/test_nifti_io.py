import gzip

import numpy as np
import pytest

from app.errors import AppError
from app.nifti_io import gunzip_bounded, load_mask_bytes
from tests.helpers import empty_mask, expected_ras_affine, nifti_gz


def test_valid_mask_round_trip():
    data = empty_mask()
    data[1, 2, 3] = 1
    mask = load_mask_bytes(nifti_gz(data, expected_ras_affine()), "lesions")
    assert mask.shape == (6, 5, 4)
    assert mask.data.dtype == bool and mask.voxel_count == 1
    assert mask.warnings == []
    np.testing.assert_allclose(mask.affine, expected_ras_affine(), atol=1e-5)


def test_not_gzip():
    with pytest.raises(AppError, match="gzip"):
        load_mask_bytes(b"<html>gateway error</html>", "m")


def test_corrupt_gzip():
    with pytest.raises(AppError, match="damaged"):
        load_mask_bytes(b"\x1f\x8b\x08\x00" + b"\xde\xad\xbe\xef" * 20, "m")


def test_truncated_gzip():
    good = nifti_gz(empty_mask(), expected_ras_affine())
    with pytest.raises(AppError, match="incomplete"):
        load_mask_bytes(good[: len(good) // 2], "m")


def test_gzip_of_non_nifti_bytes():
    with pytest.raises(AppError, match="isn't a valid NIfTI"):
        load_mask_bytes(gzip.compress(b"definitely not a nifti header" * 20), "m")


def test_decompression_cap():
    bomb = gzip.compress(b"\0" * 2_000_000)
    with pytest.raises(AppError, match="too large"):
        gunzip_bounded(bomb, max_bytes=1_000_000)


def test_non_binary_values_are_binarised_with_warning():
    data = np.zeros((6, 5, 4), dtype=np.float32)
    data[0, 0, 0], data[1, 0, 0], data[2, 0, 0] = 0.3, 0.7, 2.0
    mask = load_mask_bytes(nifti_gz(data, expected_ras_affine(), dtype=np.float32), "m")
    assert mask.voxel_count == 2           # 0.7 and 2.0 are > 0.5
    assert mask.warnings and "0/1" in mask.warnings[0]


def test_nan_values_are_refused():
    data = np.zeros((6, 5, 4), dtype=np.float32)
    data[0, 0, 0] = np.nan
    with pytest.raises(AppError, match="NaN"):
        load_mask_bytes(nifti_gz(data, expected_ras_affine(), dtype=np.float32), "m")


def test_four_dimensional_mask_is_refused():
    with pytest.raises(AppError, match="3-D"):
        load_mask_bytes(nifti_gz(np.zeros((6, 5, 4, 2)), expected_ras_affine()), "m")
