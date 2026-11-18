"""

In-memory series and job stores, and the background task that runs one job.

Every result lives on its Job (keyed by a random job id), never in a cache 
keyed by mask URL or structure name: two uploads of the same case get the
same mask URL, so a URL-keyed cache would mix their results.

"""

import asyncio
import itertools
import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from app.api_client import NucleoClient, Structure
from app.config import REFERENCE_CASE_ID
from app.dicom_io import CtSeries
from app.errors import AppError
from app.geometry import map_mask_to_ct
from app.metrics import dice_report
from app.nifti_io import MaskImage, load_mask_bytes

logger = logging.getLogger(__name__)
_job_numbers = itertools.count(1)  # start order, used to pick the job shown by default


class BoundedStore:
    """A dict that forgets its oldest entries beyond max_items (keeps RAM bounded)."""

    def __init__(self, max_items: int):
        self._items = OrderedDict()
        self._max_items = max_items

    def add(self, key: str, value) -> None:
        self._items[key] = value
        while len(self._items) > self._max_items:
            self._items.popitem(last=False)

    def get(self, key: str):
        return self._items.get(key)

    def values(self) -> list:
        return list(self._items.values())


@dataclass
class SeriesEntry:
    id: str
    zip_bytes: bytes
    ct: CtSeries
    reference_packed: bytes | None = None   # reference mask on this CT grid, np.packbits of (k, j, i)
    reference_message: str = ""


@dataclass
class StructureResult:
    name: str
    status: str                             # "ok" or "failed"
    message: str = ""
    detail: str = ""
    mask: MaskImage | None = None           # on its own NIfTI grid, used for Dice
    display_packed: bytes | None = None     # on the CT display grid, np.packbits of (k, j, i)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "detail": self.detail,
            "has_overlay": self.display_packed is not None,
            "voxel_count": self.mask.voxel_count if self.mask else None,
            "volume_ml": self.mask.volume_ml if self.mask else None,
            "warnings": self.mask.warnings if self.mask else [],
        }


@dataclass
class Job:
    series_id: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    number: int = field(default_factory=lambda: next(_job_numbers))
    status: str = "queued"
    message: str = "Queued"
    error: dict | None = None
    case_id: str | None = None
    structures: dict = field(default_factory=dict)   # name -> StructureResult, in response order
    dice: dict | None = None
    finished_at: float | None = None

    def report(self, status: str, message: str) -> None:
        self.status, self.message = status, message

    def finish(self, status: str, message: str, error: AppError | None = None) -> None:
        self.status, self.message = status, message
        self.error = error.to_dict() if error else None
        self.finished_at = time.time()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "series_id": self.series_id,
            "number": self.number,
            "status": self.status,
            "message": self.message,
            "error": self.error,
            "case_id": self.case_id,
            "structures": [s.to_dict() for s in self.structures.values()],
            "dice": self.dice,
            "finished_at": self.finished_at,
        }


def pick_default_job(jobs: list) -> Job | None:
    """The most recently *started* job that finished successfully (not the last to finish)."""
    done = [job for job in jobs if job.status == "done"]
    return max(done, key=lambda job: job.number, default=None)


async def run_job(job: Job, series: SeriesEntry, client: NucleoClient,
                  reference: MaskImage | None, lesion_name: str) -> None:
    """Upload, download every structure, map onto the CT grid, compute Dice. Never raises."""
    try:
        result = await client.upload(series.zip_bytes, job.report)
        job.case_id = result.case_id
        for structure in result.structures:
            if structure.name in job.structures:
                continue  # duplicate name in the response; keep the first
            job.report("downloading", f"Downloading mask '{structure.name}'")
            job.structures[structure.name] = await fetch_structure(client, structure, series.ct, job.report)
        job.dice = await asyncio.to_thread(job_dice, job, reference, lesion_name)
        job.finish("done", "Done")
    except AppError as exc:
        job.finish("failed", exc.user_message, exc)
    except Exception:
        error_id = uuid.uuid4().hex[:8]
        logger.exception("Job %s failed unexpectedly (error id %s)", job.id, error_id)
        job.finish("failed", "Something went wrong on our side. Please try again.",
                   AppError("internal", "Something went wrong on our side. Please try again.",
                            detail=f"error id {error_id}"))


async def fetch_structure(client: NucleoClient, structure: Structure, ct: CtSeries, report) -> StructureResult:
    """One structure; a failure here marks only this structure as failed."""
    result = StructureResult(structure.name, "failed")
    try:
        data = await client.download(structure.url, report)
        result.mask = await asyncio.to_thread(load_mask_bytes, data, structure.name)
        result.display_packed = await asyncio.to_thread(pack_on_ct_grid, result.mask, ct)
        result.status = "ok"
    except AppError as exc:
        result.message, result.detail = exc.user_message, exc.detail
    return result


def pack_on_ct_grid(mask: MaskImage, ct: CtSeries) -> bytes:
    """Map a mask onto the CT display grid (k, j, i) and pack 8 voxels per byte."""
    display = map_mask_to_ct(mask.data, mask.affine, ct.shape, ct.affine)
    return np.packbits(display).tobytes()


def job_dice(job: Job, reference: MaskImage | None, structure_name: str) -> dict:
    """Dice for one structure of a job against the reference, or a message saying why not."""
    choices = [s.name for s in job.structures.values() if s.mask is not None]
    base = {"structure": structure_name, "choices": choices, "dice": None}
    if reference is None:
        return {**base, "message": "The reference mask isn't available, so Dice can't be computed."}
    if job.case_id != REFERENCE_CASE_ID:
        return {**base, "message": f"The reference belongs to {REFERENCE_CASE_ID}, but the service identified "
                                   f"this scan as '{job.case_id}', so Dice isn't computed."}
    structure = job.structures.get(structure_name)
    if structure is None:
        return {**base, "message": f"The model didn't return a '{structure_name}' mask. "
                                   "Choose a returned structure to compare instead."}
    if structure.mask is None:
        return {**base, "message": f"The '{structure_name}' mask couldn't be loaded, so Dice can't be computed."}
    return {**base, **dice_report(structure.mask, reference)}
