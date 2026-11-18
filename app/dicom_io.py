"""Parse a DICOM series zip into a CT volume plus its geometry.

Index convention (same as NIfTI):  i = column, j = row, k = sorted slice.
The display array is stored as volume[k, j, i] = (slice, row, column), so one
axial slice volume[k] is exactly the DICOM pixel array for that slice.
"""

import io
from dataclasses import dataclass, field

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

from app.errors import AppError
from app.geometry import lps_to_ras, orientation_labels
from app.zip_safety import read_zip_members

REQUIRED_TAGS = ("SeriesInstanceUID", "ImagePositionPatient", "ImageOrientationPatient",
                 "PixelSpacing", "Rows", "Columns", "PixelData")
GEOMETRY_ATOL = 1e-4              # direction cosines / pixel spacing agreement
DUPLICATE_POSITION_MM = 1e-3
SPACING_WARN_FRACTION = 0.01      # warn if a gap differs from the median by >1 %
SPACING_FAIL_FRACTION = 0.10      # refuse if >10 % (e.g. a missing slice)


@dataclass
class CtSeries:
    volume: np.ndarray          # int16 HU, shape (k, j, i) = (slices, rows, columns)
    affine: np.ndarray          # 4x4, voxel (i, j, k, 1) -> RAS mm (comparable with NIfTI)
    spacing: tuple              # mm along (i, j, k)
    slice_positions: list       # z (mm, IPP) of each sorted slice
    labels: dict                # orientation labels for the edges of an axial slice
    warnings: list = field(default_factory=list)

    @property
    def shape(self) -> tuple:
        """Grid shape in (i, j, k) order, like a NIfTI image."""
        k, j, i = self.volume.shape
        return (i, j, k)


def load_dicom_zip(data: bytes) -> CtSeries:
    members = read_zip_members(data)
    datasets = _parse_members(members)
    _check_single_series(datasets)
    _check_required_tags(datasets)
    _check_consistent_geometry(datasets)

    row_cos, col_cos, normal = _direction_cosines(datasets[0])
    datasets.sort(key=lambda ds: float(np.dot(_ipp(ds), normal)))
    warnings = _check_slice_spacing(datasets, normal)

    volume = np.stack([_hu_pixels(ds) for ds in datasets])  # (k, j, i)
    affine_lps = _affine_lps(datasets, row_cos, col_cos)
    spacing = tuple(float(np.linalg.norm(affine_lps[:3, axis])) for axis in range(3))
    return CtSeries(
        volume=volume,
        affine=lps_to_ras(affine_lps),
        spacing=spacing,
        slice_positions=[float(_ipp(ds)[2]) for ds in datasets],
        labels=orientation_labels(row_cos, col_cos),
        warnings=warnings,
    )


def _parse_members(members: list) -> list:
    datasets, bad = [], []
    for name, raw in members:
        try:
            datasets.append(pydicom.dcmread(io.BytesIO(raw), force=False))
        except InvalidDicomError:
            bad.append(name)
        except Exception:  # pydicom raises many other types on damaged bytes
            bad.append(name)
    if not datasets:
        raise AppError("input", "The zip doesn't contain any readable DICOM images.",
                       detail=f"{len(bad)} unreadable file(s)")
    if bad:
        noun = "file in the zip isn't a" if len(bad) == 1 else "files in the zip aren't"
        raise AppError("input", f"{len(bad)} {noun} readable DICOM image{'s' if len(bad) > 1 else ''}.",
                       detail="e.g. " + ", ".join(bad[:5]))
    return datasets


def _check_single_series(datasets: list) -> None:
    uids = {str(ds.get("SeriesInstanceUID", "")) for ds in datasets}
    if len(uids) > 1:
        raise AppError("input", f"The zip contains {len(uids)} different scans (series). "
                                "Please upload one series at a time.")


def _check_required_tags(datasets: list) -> None:
    missing = sorted({tag for ds in datasets for tag in REQUIRED_TAGS if tag not in ds})
    if missing:
        raise AppError("input", "The DICOM files are missing information needed to place the images.",
                       detail="missing tags: " + ", ".join(missing))
    if len(datasets) < 2:
        raise AppError("input", "The scan has only one slice; a 3-D series is needed.")


def _check_consistent_geometry(datasets: list) -> None:
    first = datasets[0]
    for ds in datasets[1:]:
        same = (
            np.allclose(_floats(ds.ImageOrientationPatient, 6), _floats(first.ImageOrientationPatient, 6), atol=GEOMETRY_ATOL)
            and np.allclose(_floats(ds.PixelSpacing, 2), _floats(first.PixelSpacing, 2), atol=GEOMETRY_ATOL)
            and (ds.Rows, ds.Columns) == (first.Rows, first.Columns)
        )
        if not same:
            raise AppError("input", "The slices in the scan don't all have the same size, spacing or orientation.")


def _direction_cosines(ds) -> tuple:
    iop = _floats(ds.ImageOrientationPatient, 6)
    row_cos = iop[:3]   # direction of increasing column index i (along a row)
    col_cos = iop[3:]   # direction of increasing row index j (down a column)
    normal = np.cross(row_cos, col_cos)
    if not np.isclose(np.linalg.norm(normal), 1.0, atol=1e-3):
        raise AppError("input", "The scan's orientation information is invalid.",
                       detail=f"ImageOrientationPatient={list(iop)}")
    return row_cos, col_cos, normal


def _check_slice_spacing(sorted_datasets: list, normal: np.ndarray) -> list:
    positions = np.array([np.dot(_ipp(ds), normal) for ds in sorted_datasets])
    gaps = np.diff(positions)
    if np.any(gaps < DUPLICATE_POSITION_MM):
        raise AppError("input", "The scan contains two or more slices at the same position.")
    median = float(np.median(gaps))
    worst = float(np.max(np.abs(gaps - median)))
    if worst > SPACING_FAIL_FRACTION * median:
        raise AppError("input", "The slices aren't evenly spaced (a slice may be missing).",
                       detail=f"median gap {median:.3f} mm, worst deviation {worst:.3f} mm")
    if worst > SPACING_WARN_FRACTION * median:
        return [f"Slice spacing varies slightly (up to {worst:.3f} mm from {median:.3f} mm)."]
    return []


def _hu_pixels(ds) -> np.ndarray:
    try:
        # pixel_array already honours PixelRepresentation (signed vs unsigned).
        pixels = ds.pixel_array
    except Exception as exc:  # truncated data, compressed syntax without a decoder, ...
        raise AppError("input", "One or more DICOM images in the zip are damaged or incomplete.",
                       detail=f"{type(exc).__name__}: {exc}") from exc
    if pixels.shape != (ds.Rows, ds.Columns):
        raise AppError("input", "Multi-frame or colour DICOM images aren't supported.",
                       detail=f"pixel array shape {pixels.shape}")
    slope = float(ds.get("RescaleSlope", 1) or 1)
    intercept = float(ds.get("RescaleIntercept", 0) or 0)
    hu = pixels.astype(np.float32) * slope + intercept
    # Whole HU are plenty for display; int16 halves memory and transfer size.
    return np.clip(np.rint(hu), -32768, 32767).astype(np.int16)


def _affine_lps(sorted_datasets: list, row_cos: np.ndarray, col_cos: np.ndarray) -> np.ndarray:
    # PixelSpacing is [row spacing, column spacing]:
    #   row spacing    = distance between rows    = step along j (col_cos)
    #   column spacing = distance between columns = step along i (row_cos)
    row_spacing, col_spacing = _floats(sorted_datasets[0].PixelSpacing, 2)
    first, last = _ipp(sorted_datasets[0]), _ipp(sorted_datasets[-1])
    affine = np.eye(4)
    affine[:3, 0] = row_cos * col_spacing
    affine[:3, 1] = col_cos * row_spacing
    affine[:3, 2] = (last - first) / (len(sorted_datasets) - 1)  # mean slice step vector
    affine[:3, 3] = first
    return affine


def _ipp(ds) -> np.ndarray:
    return _floats(ds.ImagePositionPatient, 3)


def _floats(values, count: int) -> np.ndarray:
    """A numeric multi-valued tag as floats; must have `count` finite values."""
    try:
        arr = np.array([float(v) for v in values], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise AppError("input", "The scan's position information is invalid.", detail=str(exc)) from exc
    if arr.shape != (count,) or not np.all(np.isfinite(arr)):
        raise AppError("input", "The scan's position information is invalid.", detail=f"values={list(values)}")
    return arr
