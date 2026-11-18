import numpy as np
import pytest

from app.dicom_io import load_dicom_zip
from app.errors import AppError
from tests.helpers import (COL_SPACING, ROW_SPACING, SLICE_STEP, SLICES, dicom_slice_bytes, expected_ras_affine,
                           make_zip, series_files, series_zip, slice_pixels)


def test_volume_layout_and_affine():
    ct = load_dicom_zip(series_zip())
    assert ct.volume.shape == (SLICES, 5, 6)          # (k, j, i) = (slices, rows, columns)
    assert ct.shape == (6, 5, SLICES)                 # (i, j, k)
    for k in range(SLICES):
        np.testing.assert_array_equal(ct.volume[k], slice_pixels(k))
    np.testing.assert_allclose(ct.affine, expected_ras_affine(), atol=1e-6)
    assert ct.spacing == pytest.approx((COL_SPACING, ROW_SPACING, SLICE_STEP))
    assert ct.labels == {"right": "L", "left": "R", "bottom": "P", "top": "A"}


def test_sorting_ignores_filenames_and_instance_numbers():
    ct = load_dicom_zip(series_zip(shuffled=True))
    for k in range(SLICES):
        np.testing.assert_array_equal(ct.volume[k], slice_pixels(k))
    assert ct.slice_positions == sorted(ct.slice_positions)


def test_rescale_slope_and_intercept():
    raw = np.full((5, 6), 100, dtype=np.int16)
    files = [(f"{k}.dcm", dicom_slice_bytes(k, "1.2.9", k, slope=2.0, intercept=-1024.0, pixels=raw))
             for k in range(SLICES)]
    ct = load_dicom_zip(make_zip(files))
    assert np.all(ct.volume == 100 * 2 - 1024)


def test_non_dicom_files_are_reported():
    files = series_files() + [("series/notes.txt", b"hello"), ("series/x.bin", b"\0" * 300)]
    with pytest.raises(AppError, match="2 files in the zip aren't readable DICOM"):
        load_dicom_zip(make_zip(files))


def test_zip_without_dicom():
    with pytest.raises(AppError, match="doesn't contain any readable DICOM"):
        load_dicom_zip(make_zip([("readme.txt", b"just text")]))


def test_two_series():
    files = series_files(series_uid="1.2.1") + [("other/0001.dcm", dicom_slice_bytes(0, "1.2.2", 1))]
    with pytest.raises(AppError, match="2 different scans"):
        load_dicom_zip(make_zip(files))


def test_truncated_dicom():
    files = series_files()
    name, data = files[1]
    files[1] = (name, data[:-25])
    with pytest.raises(AppError):
        load_dicom_zip(make_zip(files))


def test_duplicate_slice_position():
    uid = "1.2.5"
    files = [(f"{n}.dcm", dicom_slice_bytes(k, uid, n)) for n, k in enumerate([0, 1, 1, 2])]
    with pytest.raises(AppError, match="same position"):
        load_dicom_zip(make_zip(files))


def test_missing_slice_is_refused():
    uid = "1.2.6"
    files = [(f"{n}.dcm", dicom_slice_bytes(k, uid, n)) for n, k in enumerate([0, 1, 3, 4])]
    with pytest.raises(AppError, match="evenly spaced"):
        load_dicom_zip(make_zip(files))
