"""Affines, LPS<->RAS, grid comparison and exact mask -> CT-grid mapping.

All grids here use voxel index order (i, j, k) and 4x4 affines mapping
(i, j, k, 1) -> world mm. DICOM world space is LPS; NIfTI world space is RAS.
No automatic reorientation (e.g. nib.as_closest_canonical) and no reshape:
axes are only ever permuted / flipped as dictated by the affines.
"""

import numpy as np

from app.errors import AppError

# LPS and RAS differ only in the sign of x and y.
LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])

AFFINE_ATOL_MM = 1e-3   # absolute tolerance on affine entries (mm)
AFFINE_RTOL = 1e-4      # relative tolerance on affine entries
MAPPING_ATOL = 1e-3     # tolerance on the voxel->voxel matrix (CT voxel units)


def lps_to_ras(affine_lps: np.ndarray) -> np.ndarray:
    return LPS_TO_RAS @ affine_lps


def same_grid(a, b) -> tuple[bool, str]:
    """Do two images share a voxel grid? `a`/`b` need `.shape` (i, j, k) and `.affine` (RAS).

    Works for nibabel images, our MaskImage and our CtSeries.
    """
    if tuple(a.shape) != tuple(b.shape):
        return False, f"shapes differ: {tuple(a.shape)} vs {tuple(b.shape)}"
    if not np.allclose(a.affine, b.affine, rtol=AFFINE_RTOL, atol=AFFINE_ATOL_MM):
        worst = float(np.max(np.abs(np.asarray(a.affine) - np.asarray(b.affine))))
        return False, (f"affines differ by up to {worst:.4g} mm "
                       f"(tolerance {AFFINE_ATOL_MM} mm + {AFFINE_RTOL} relative)")
    return True, "same shape and affine"


def map_mask_to_ct(mask_data: np.ndarray, mask_affine: np.ndarray,
                   ct_shape: tuple, ct_affine: np.ndarray) -> np.ndarray:
    """Place a boolean (i, j, k) mask onto the CT grid; return display layout (k, j, i).

    Only exact matches are allowed: M must be a signed permutation with an
    integer translation, i.e. the mask voxels sit exactly on CT voxels (maybe
    with axes swapped or flipped). Anything else is refused, not resampled.
    """
    m = np.linalg.inv(ct_affine) @ mask_affine  # mask voxel index -> CT voxel index
    rot, shift = m[:3, :3], m[:3, 3]
    rounded_rot = np.round(rot)
    is_signed_permutation = (
        np.allclose(rot, rounded_rot, atol=MAPPING_ATOL)
        and np.all(np.abs(rounded_rot).sum(axis=0) == 1)
        and np.all(np.abs(rounded_rot).sum(axis=1) == 1)
    )
    if not is_signed_permutation or not np.allclose(shift, np.round(shift), atol=MAPPING_ATOL):
        raise AppError("geometry", "The returned mask is on a different grid from the CT, "
                                   "so it can't be overlaid safely.",
                       detail=f"mask->CT voxel matrix:\n{np.array2string(m[:3], precision=4)}")

    # For each CT axis b: which mask axis feeds it, and with which sign.
    ct_ijk = np.zeros(tuple(ct_shape), dtype=bool)
    source_axis = [int(np.argmax(np.abs(rounded_rot[b]))) for b in range(3)]
    signs = [int(rounded_rot[b, source_axis[b]]) for b in range(3)]
    moved = np.transpose(mask_data, source_axis)  # axis b of `moved` = mask axis source_axis[b]

    slices = []
    for b in range(3):
        n = moved.shape[b]
        start = int(round(shift[b]))
        if signs[b] < 0:
            # CT index = shift - mask index. After flipping this axis, mask index
            # f maps to CT index shift - (n - 1) + f, so the slab starts there.
            moved = np.flip(moved, axis=b)
            start = start - (n - 1)
        if start < 0 or start + n > ct_shape[b]:
            raise AppError("geometry", "The returned mask extends outside the CT volume, "
                                       "so it can't be overlaid safely.",
                           detail=f"axis {b}: mask covers CT indices {start}..{start + n - 1}, "
                                  f"CT has 0..{ct_shape[b] - 1}")
        slices.append(slice(start, start + n))
    ct_ijk[tuple(slices)] = moved

    # (i, j, k) -> display (k, j, i)
    return np.ascontiguousarray(ct_ijk.transpose(2, 1, 0))


def orientation_labels(row_cos: np.ndarray, col_cos: np.ndarray) -> dict:
    """Patient-direction letters for the edges of an axial slice shown as stored (radiological)."""
    return {
        "right": _direction_letter(row_cos),   # columns increase to the right
        "left": _direction_letter(-row_cos),
        "bottom": _direction_letter(col_cos),  # rows increase downwards
        "top": _direction_letter(-col_cos),
    }


def _direction_letter(vector_lps: np.ndarray) -> str:
    """Dominant LPS direction of a vector: L/R, P/A or H/F."""
    axis = int(np.argmax(np.abs(vector_lps)))
    positive = vector_lps[axis] > 0
    return [("L", "R"), ("P", "A"), ("H", "F")][axis][0 if positive else 1]
