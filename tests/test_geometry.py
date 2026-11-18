import numpy as np
import pytest

from app.dicom_io import load_dicom_zip
from app.errors import AppError
from app.geometry import lps_to_ras, map_mask_to_ct, same_grid
from app.nifti_io import load_mask_bytes
from tests.helpers import empty_mask, expected_ras_affine, nifti_gz, series_zip


@pytest.fixture(scope="module")
def ct():
    return load_dicom_zip(series_zip())


def test_lps_to_ras_flips_x_and_y():
    lps = np.eye(4)
    lps[:3, 3] = [1, 2, 3]
    np.testing.assert_array_equal(lps_to_ras(lps)[:3, 3], [-1, -2, 3])


def test_dicom_affine_equals_nifti_affine(ct):
    mask = load_mask_bytes(nifti_gz(empty_mask(), expected_ras_affine()), "m")
    assert same_grid(ct, mask) == (True, "same shape and affine")


def test_single_voxel_lands_at_display_k_j_i(ct):
    i, j, k = 4, 1, 2
    data = empty_mask()
    data[i, j, k] = 1
    display = map_mask_to_ct(data.astype(bool), expected_ras_affine(), ct.shape, ct.affine)
    assert display.shape == ct.volume.shape
    assert np.argwhere(display).tolist() == [[k, j, i]]
    # The CT voxel under it is DICOM slice k, row j, column i (value 1000k + 10j + i).
    assert ct.volume[k, j, i] == 1000 * k + 10 * j + i


def test_flipped_axis_with_matching_affine_maps_to_same_place(ct):
    """Same physical voxels stored with i reversed: the affine says so, so it still lines up."""
    i, j, k = 4, 1, 2
    data = empty_mask()
    data[i, j, k] = 1
    n_i = data.shape[0]
    flip_i = np.eye(4)
    flip_i[0, 0], flip_i[0, 3] = -1, n_i - 1          # stored index i' -> original n_i-1-i'
    flipped_affine = expected_ras_affine() @ flip_i
    display = map_mask_to_ct(data[::-1].astype(bool), flipped_affine, ct.shape, ct.affine)
    assert np.argwhere(display).tolist() == [[k, j, i]]


def test_permuted_axes_with_matching_affine_maps_to_same_place(ct):
    i, j, k = 4, 1, 2
    data = empty_mask()
    data[i, j, k] = 1
    swap_ij = np.eye(4)[[1, 0, 2, 3]]                  # stored (j, i, k)
    display = map_mask_to_ct(data.transpose(1, 0, 2).astype(bool),
                             expected_ras_affine() @ swap_ij, ct.shape, ct.affine)
    assert np.argwhere(display).tolist() == [[k, j, i]]


def test_flipped_affine_pointing_outside_ct_is_refused(ct):
    affine = expected_ras_affine().copy()
    affine[0, 0] *= -1                                  # flip i without moving the origin
    with pytest.raises(AppError, match="outside the CT"):
        map_mask_to_ct(empty_mask().astype(bool), affine, ct.shape, ct.affine)


@pytest.mark.parametrize("change", ["spacing", "rotation", "half_voxel_shift"])
def test_mismatched_affine_is_refused(ct, change):
    affine = expected_ras_affine().copy()
    if change == "spacing":
        affine[0, 0] *= 1.5
    elif change == "rotation":
        c, s = np.cos(np.radians(30)), np.sin(np.radians(30))
        affine[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ affine[:3, :3]
    else:
        affine[0, 3] += 0.3
    with pytest.raises(AppError, match="different grid"):
        map_mask_to_ct(empty_mask().astype(bool), affine, ct.shape, ct.affine)


def test_same_grid_reports_why(ct):
    other_shape = load_mask_bytes(nifti_gz(np.zeros((6, 5, 3)), expected_ras_affine()), "m")
    ok, reason = same_grid(ct, other_shape)
    assert not ok and "shapes differ" in reason

    moved = expected_ras_affine().copy()
    moved[2, 3] += 1.0
    other_affine = load_mask_bytes(nifti_gz(empty_mask(), moved), "m")
    ok, reason = same_grid(ct, other_affine)
    assert not ok and "affines differ" in reason
