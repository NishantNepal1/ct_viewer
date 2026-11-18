"""Load a .nii.gz mask from bytes safely, validate it and binarise it."""

import zlib
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np

from app.config import MAX_MASK_DECOMPRESSED_BYTES, MAX_MASK_DIM
from app.errors import AppError

GZIP_MAGIC = b"\x1f\x8b"


@dataclass
class MaskImage:
    name: str
    data: np.ndarray            # bool, shape (i, j, k) exactly as stored in the file
    affine: np.ndarray          # 4x4 voxel (i, j, k, 1) -> RAS mm
    warnings: list = field(default_factory=list)

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def voxel_count(self) -> int:
        return int(np.count_nonzero(self.data))

    @property
    def volume_ml(self) -> float:
        voxel_mm3 = abs(float(np.linalg.det(self.affine[:3, :3])))
        return self.voxel_count * voxel_mm3 / 1000.0  # mm^3 -> mL


def gunzip_bounded(data: bytes, max_bytes: int = MAX_MASK_DECOMPRESSED_BYTES) -> bytes:
    """Decompress gzip data, aborting as soon as the output would exceed max_bytes."""
    if not data.startswith(GZIP_MAGIC):
        raise AppError("mask", "The mask file isn't gzip-compressed NIfTI data.")
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)  # 16+ = expect a gzip header
    out = bytearray()
    pending = data
    try:
        while not decompressor.eof:
            piece = decompressor.decompress(pending, max_bytes + 1 - len(out))
            out += piece
            if len(out) > max_bytes:
                raise AppError("mask", "The mask file is too large once decompressed.",
                               detail=f"limit {max_bytes} bytes")
            pending = decompressor.unconsumed_tail
            if not pending and not piece:
                raise AppError("mask", "The mask file is incomplete (cut off).", detail="gzip stream truncated")
    except zlib.error as exc:
        raise AppError("mask", "The mask file is damaged (corrupt gzip data).", detail=str(exc)) from exc
    return bytes(out)


def load_mask_bytes(data: bytes, name: str) -> MaskImage:
    raw = gunzip_bounded(data)
    try:
        img = nib.Nifti1Image.from_bytes(raw)
        values = np.asanyarray(img.dataobj)  # applies scl_slope / scl_inter
    except Exception as exc:  # nibabel raises several types on bad headers / short data
        raise AppError("mask", "The mask file isn't a valid NIfTI image.",
                       detail=f"{type(exc).__name__}: {exc}") from exc

    if values.ndim != 3:
        raise AppError("mask", "The mask isn't a 3-D image.", detail=f"shape {values.shape}")
    if any(n < 1 or n > MAX_MASK_DIM for n in values.shape):
        raise AppError("mask", "The mask has an implausible size.", detail=f"shape {values.shape}")
    affine = np.asarray(img.affine, dtype=np.float64)
    if not np.all(np.isfinite(affine)) or abs(np.linalg.det(affine[:3, :3])) < 1e-9:
        raise AppError("mask", "The mask's position information is invalid.", detail=f"affine {affine.tolist()}")
    if values.dtype.kind == "f" and not np.all(np.isfinite(values)):
        raise AppError("mask", "The mask contains invalid values (NaN or infinity).")

    warnings = []
    if np.any((values != 0) & (values != 1)):
        warnings.append("Mask values weren't strictly 0/1; values above 0.5 were treated as inside.")
    return MaskImage(name=name, data=values > 0.5, affine=affine, warnings=warnings)


def load_mask_file(path: Path, name: str) -> MaskImage:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise AppError("mask", "The reference mask file couldn't be read.", detail=f"{path.name}: {exc.strerror}") from exc
    return load_mask_bytes(data, name)
