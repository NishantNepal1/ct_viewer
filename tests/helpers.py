"""Synthetic DICOM series and NIfTI masks for tests. Nothing here reads data/."""

import gzip
import io
import zipfile

import nibabel as nib
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

# Deliberately non-square and anisotropic so swapped axes or spacings get caught.
ROWS, COLS, SLICES = 5, 6, 4
ROW_SPACING, COL_SPACING, SLICE_STEP = 0.8, 0.6, 2.5
ORIGIN_LPS = np.array([-20.0, -30.0, 10.0])


def slice_pixels(k: int) -> np.ndarray:
    """Unique value per voxel: 1000*slice + 10*row + column."""
    rows, cols = np.mgrid[0:ROWS, 0:COLS]
    return (1000 * k + 10 * rows + cols).astype(np.int16)


def dicom_slice_bytes(k: int, series_uid: str, instance_number: int,
                      slope: float = 1.0, intercept: float = 0.0, pixels=None) -> bytes:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = "1.2.3"
    ds.SeriesInstanceUID = series_uid
    ds.Modality = "CT"
    ds.InstanceNumber = instance_number
    ipp = ORIGIN_LPS + np.array([0.0, 0.0, SLICE_STEP * k])
    ds.ImagePositionPatient = [f"{v:.4f}" for v in ipp]
    ds.ImageOrientationPatient = ["1", "0", "0", "0", "1", "0"]
    ds.PixelSpacing = [str(ROW_SPACING), str(COL_SPACING)]  # [row, column]
    ds.Rows, ds.Columns = ROWS, COLS
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
    ds.PixelRepresentation = 1
    ds.RescaleSlope, ds.RescaleIntercept = str(slope), str(intercept)
    data = slice_pixels(k) if pixels is None else pixels
    ds.PixelData = data.astype("<i2").tobytes()

    buf = io.BytesIO()
    pydicom.dcmwrite(buf, ds, enforce_file_format=True)
    return buf.getvalue()


def series_files(series_uid: str | None = None, shuffled: bool = False) -> list:
    """[(filename, bytes)]. With shuffled=True, filenames and InstanceNumbers don't follow z."""
    uid = series_uid or generate_uid()
    order = [2, 0, 3, 1] if shuffled else list(range(SLICES))  # file n holds slice order[n]
    return [(f"series/{n:04d}.dcm", dicom_slice_bytes(k, uid, instance_number=SLICES - n))
            for n, k in enumerate(order)]


def make_zip(files: list) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files:
            zf.writestr(name, data)
    return buf.getvalue()


def series_zip(**kwargs) -> bytes:
    return make_zip(series_files(**kwargs))


def expected_ras_affine() -> np.ndarray:
    """What the synthetic series' affine must be: LPS->RAS flips x and y."""
    return np.array([
        [-COL_SPACING, 0, 0, -ORIGIN_LPS[0]],
        [0, -ROW_SPACING, 0, -ORIGIN_LPS[1]],
        [0, 0, SLICE_STEP, ORIGIN_LPS[2]],
        [0, 0, 0, 1],
    ])


def nifti_gz(data: np.ndarray, affine: np.ndarray, dtype=np.uint8) -> bytes:
    img = nib.Nifti1Image(np.asarray(data).astype(dtype), affine)
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    return gzip.compress(img.to_bytes())


def empty_mask() -> np.ndarray:
    """A mask array on the synthetic CT grid, in (i, j, k) order."""
    return np.zeros((COLS, ROWS, SLICES), dtype=np.uint8)
