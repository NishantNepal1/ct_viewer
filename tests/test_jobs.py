"""Job runner tests: failed uploads, and concurrent jobs that must not mix results."""

import asyncio

import httpx
import numpy as np

from app.api_client import NucleoClient
from app.dicom_io import load_dicom_zip
from app.jobs import Job, SeriesEntry, pick_default_job, run_job
from app.nifti_io import load_mask_bytes
from tests.helpers import empty_mask, expected_ras_affine, nifti_gz, series_zip

LESIONS_URL = "/v1/cases/case-001/structures/lesions"


def make_series(uid: str) -> SeriesEntry:
    data = series_zip(series_uid=uid)
    return SeriesEntry(id=uid, zip_bytes=data, ct=load_dicom_zip(data))


def mask_with_voxel(i, j, k) -> bytes:
    data = empty_mask()
    data[i, j, k] = 1
    return nifti_gz(data, expected_ras_affine())


def unpack(job: Job, name: str, ct_shape) -> np.ndarray:
    packed = np.frombuffer(job.structures[name].display_packed, dtype=np.uint8)
    return np.unpackbits(packed)[: int(np.prod(ct_shape))].reshape(ct_shape)


async def test_failed_upload_marks_job_failed_with_plain_message():
    def handler(request):
        return httpx.Response(400, json={"error": {"code": "bad_request", "message": "x",
                                                   "details": {"reason": "corrupt_zip"}}})

    series = make_series("1.2.10")
    client = NucleoClient("https://api.test", "tid_test.secret", transport=httpx.MockTransport(handler))
    job = Job(series_id=series.id)
    await run_job(job, series, client, reference=None, lesion_name="lesions")
    state = job.to_dict()
    assert state["status"] == "failed"
    assert state["message"] == "The file is not a valid zip archive. Please check you selected the right file."
    assert "Traceback" not in str(state)
    assert state["structures"] == []


async def test_concurrent_jobs_do_not_mix_results():
    """Job A starts first but its upload answers last. Both get the SAME lesions URL,
    and the stub serves different bytes on each download of that URL."""
    series_a, series_b = make_series("1.2.21"), make_series("1.2.22")
    b_finished_downloading = asyncio.Event()
    lesion_downloads = []

    async def handler(request: httpx.Request):
        path = request.url.path
        if path == "/v1/segmentations":
            if request.content == series_a.zip_bytes:
                await b_finished_downloading.wait()
                return httpx.Response(200, json={"case_id": "case-001", "structures": [
                    {"name": "lesions", "url": LESIONS_URL}]})
            return httpx.Response(200, json={"case_id": "case-001", "structures": [
                {"name": "lesions", "url": LESIONS_URL},
                {"name": "liver", "url": "/v1/cases/case-001/structures/liver"}]})
        if path == LESIONS_URL:
            lesion_downloads.append(path)
            voxel = (0, 0, 0) if len(lesion_downloads) == 1 else (1, 1, 1)
            return httpx.Response(200, content=mask_with_voxel(*voxel))
        if path.endswith("/liver"):
            b_finished_downloading.set()
            return httpx.Response(200, content=mask_with_voxel(2, 2, 2))
        return httpx.Response(404)

    client = NucleoClient("https://api.test", "tid_test.secret", transport=httpx.MockTransport(handler))
    reference = load_mask_bytes(mask_with_voxel(1, 1, 1), "reference")
    job_a = Job(series_id=series_a.id)
    job_b = Job(series_id=series_b.id)
    await asyncio.gather(run_job(job_a, series_a, client, reference, "lesions"),
                         run_job(job_b, series_b, client, reference, "lesions"))

    assert job_a.status == job_b.status == "done"
    assert job_b.finished_at <= job_a.finished_at          # B really did finish first
    assert list(job_a.structures) == ["lesions"]
    assert list(job_b.structures) == ["lesions", "liver"]

    shape = series_a.ct.volume.shape
    assert np.argwhere(unpack(job_b, "lesions", shape)).tolist() == [[0, 0, 0]]   # first download
    assert np.argwhere(unpack(job_a, "lesions", shape)).tolist() == [[1, 1, 1]]   # second download
    assert np.argwhere(unpack(job_b, "liver", shape)).tolist() == [[2, 2, 2]]     # display (k, j, i)
    assert job_a.dice["dice"] == 1.0 and job_b.dice["dice"] == 0.0

    # The default job is the most recently *started* finished job (B), even though A finished last.
    assert pick_default_job([job_a, job_b]) is job_b
    assert pick_default_job([job_a]) is job_a


def test_default_job_ignores_unfinished_and_failed():
    done, failed, running = Job(series_id="s"), Job(series_id="s"), Job(series_id="s")
    done.status, failed.status, running.status = "done", "failed", "uploading"
    assert pick_default_job([done, failed, running]) is done
    assert pick_default_job([failed, running]) is None


async def test_one_bad_mask_does_not_fail_the_job():
    series = make_series("1.2.30")

    def handler(request):
        if request.url.path == "/v1/segmentations":
            return httpx.Response(200, json={"case_id": "case-001", "structures": [
                {"name": "lesions", "url": LESIONS_URL}, {"name": "broken", "url": "/v1/x/broken"}]})
        if request.url.path == LESIONS_URL:
            return httpx.Response(200, content=mask_with_voxel(0, 0, 0))
        return httpx.Response(200, content=b"\x1f\x8b\x08\x00garbage-not-gzip")

    client = NucleoClient("https://api.test", "tid_test.secret", transport=httpx.MockTransport(handler))
    job = Job(series_id=series.id)
    await run_job(job, series, client, reference=None, lesion_name="lesions")
    assert job.status == "done"
    assert job.structures["lesions"].status == "ok"
    assert job.structures["broken"].status == "failed"
    assert "damaged" in job.structures["broken"].message
    assert "reference mask isn't available" in job.dice["message"]


async def test_missing_lesion_structure_offers_choices():
    series = make_series("1.2.31")

    def handler(request):
        if request.url.path == "/v1/segmentations":
            return httpx.Response(200, json={"case_id": "case-001", "structures": [
                {"name": "liver", "url": "/v1/x/liver"}]})
        return httpx.Response(200, content=mask_with_voxel(0, 0, 0))

    client = NucleoClient("https://api.test", "tid_test.secret", transport=httpx.MockTransport(handler))
    reference = load_mask_bytes(mask_with_voxel(0, 0, 0), "reference")
    job = Job(series_id=series.id)
    await run_job(job, series, client, reference, lesion_name="lesions")
    assert job.dice["dice"] is None
    assert "didn't return a 'lesions' mask" in job.dice["message"]
    assert job.dice["choices"] == ["liver"]
