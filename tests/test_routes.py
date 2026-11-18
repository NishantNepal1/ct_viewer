"""End-to-end tests of our own HTTP routes, with the Nucleo API stubbed."""

import asyncio
import time

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import Settings
from tests.helpers import (SLICES, dicom_slice_bytes, empty_mask, expected_ras_affine, make_zip, nifti_gz,
                           series_files, series_zip, slice_pixels)

LESION_VOXEL = (4, 1, 2)  # (i, j, k)


def lesion_mask_bytes() -> bytes:
    data = empty_mask()
    data[LESION_VOXEL] = 1
    return nifti_gz(data, expected_ras_affine())


def api_stub(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/v1/segmentations":
        if not request.content.startswith(b"PK"):
            return httpx.Response(400, json={"error": {"code": "bad_request", "details": {"reason": "not_zip"}}})
        return httpx.Response(200, json={"case_id": "case-001", "structures": [
            {"name": "lesions", "url": "/v1/cases/case-001/structures/lesions"}]})
    return httpx.Response(200, content=lesion_mask_bytes())


@pytest.fixture
def client(tmp_path):
    ref_path = tmp_path / "reference.nii.gz"
    ref_path.write_bytes(lesion_mask_bytes())
    settings = Settings("https://api.test", "tid_test.secret", ref_path, "lesions")
    app = main.create_app(settings, transport=httpx.MockTransport(api_stub))
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def upload_series(client, data: bytes):
    return client.post("/api/series", content=data, headers={"Content-Type": "application/zip"})


def wait_for_job(client, job_id: str) -> dict:
    for _ in range(100):
        state = client.get(f"/api/jobs/{job_id}").json()
        if state["status"] in ("done", "failed"):
            return state
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_full_flow(client):
    meta = upload_series(client, series_zip()).json()
    assert meta["shape"] == {"slices": SLICES, "rows": 5, "columns": 6}
    assert meta["reference"]["available"] is True

    volume = client.get(f"/api/series/{meta['id']}/volume")
    assert volume.headers["X-Volume-Shape"] == "4,5,6"
    hu = np.frombuffer(volume.content, dtype="<i2").reshape(4, 5, 6)
    np.testing.assert_array_equal(hu[2], slice_pixels(2))

    job_id = client.post(f"/api/series/{meta['id']}/jobs").json()["job_id"]
    state = wait_for_job(client, job_id)
    assert state["status"] == "done", state
    assert [s["name"] for s in state["structures"]] == ["lesions"]
    assert state["dice"]["dice"] == 1.0

    packed = client.get(f"/api/jobs/{job_id}/masks/lesions").content
    display = np.unpackbits(np.frombuffer(packed, np.uint8))[: 4 * 5 * 6].reshape(4, 5, 6)
    i, j, k = LESION_VOXEL
    assert np.argwhere(display).tolist() == [[k, j, i]]
    assert client.get(f"/api/series/{meta['id']}/reference").content == packed

    listing = client.get("/api/jobs", params={"series_id": meta["id"]}).json()
    assert listing["default_job_id"] == job_id


def test_job_without_token_fails_clearly(tmp_path):
    settings = Settings("https://api.test", None, tmp_path / "missing.nii.gz", "lesions")
    app = main.create_app(settings, transport=httpx.MockTransport(api_stub))
    with TestClient(app) as client:
        health = client.get("/api/health").json()
        assert health["token_configured"] is False and health["reference_available"] is False
        meta = upload_series(client, series_zip()).json()
        assert meta["reference"]["available"] is False
        state = wait_for_job(client, client.post(f"/api/series/{meta['id']}/jobs").json()["job_id"])
    assert state["status"] == "failed"
    assert "API token not configured" in state["message"]


def two_series_zip():
    return make_zip(series_files(series_uid="1.2.1") + [("b/1.dcm", dicom_slice_bytes(0, "1.2.2", 1))])


def truncated_dicom_zip():
    files = series_files()
    files[0] = (files[0][0], files[0][1][:-40])
    return make_zip(files)


@pytest.mark.parametrize("make_body,expected", [
    (lambda: b"", "empty"),
    (lambda: b"%PDF-1.4 not a zip", "not a zip"),
    (lambda: make_zip([("notes.txt", b"hello")]), "readable DICOM"),
    (two_series_zip, "different scans"),
    (lambda: make_zip([(f"{n}.dcm", b"") for n in range(5001)]), "too many files"),
    (truncated_dicom_zip, "damaged or incomplete"),
])
def test_bad_local_input_gives_friendly_error_and_server_keeps_serving(client, make_body, expected):
    resp = upload_series(client, make_body())
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert expected in error["message"]
    assert "Traceback" not in resp.text
    assert client.get("/api/health").status_code == 200


def test_upload_over_size_limit(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 1000)
    resp = upload_series(client, b"PK\x03\x04" + b"\0" * 2000)
    assert resp.status_code == 413
    assert "larger than" in resp.json()["error"]["message"]


def test_unexpected_exception_is_generic_500(client, monkeypatch):
    def explode(data):
        raise RuntimeError("secret internals")
    monkeypatch.setattr(main, "load_dicom_zip", explode)
    resp = upload_series(client, series_zip())
    assert resp.status_code == 500
    assert resp.json()["error"]["message"] == main.GENERIC_ERROR
    assert "secret internals" not in resp.text and "Traceback" not in resp.text


def test_unknown_ids_are_friendly_404s(client):
    assert client.get("/api/series/nope/volume").json()["error"]["message"].startswith("That scan")
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.get("/api/no-such-route").json()["error"]["message"] == "Not found."


async def test_concurrent_uploads_through_the_server(tmp_path):
    """Good and bad uploads hit our server at the same time, then two jobs run at once.
    Job A's API response is slower, so A finishes after B. Results must not mix."""
    zip_a, zip_b = series_zip(series_uid="1.2.71"), series_zip(series_uid="1.2.72")

    def mask_at(i, j, k):
        data = empty_mask()
        data[i, j, k] = 1
        return nifti_gz(data, expected_ras_affine())

    async def stub(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/segmentations":
            if request.content == zip_a:
                await asyncio.sleep(0.3)
                return httpx.Response(200, json={"case_id": "case-001", "structures": [
                    {"name": "lesions", "url": "/a/lesions"}]})
            return httpx.Response(200, json={"case_id": "case-001", "structures": [
                {"name": "lesions", "url": "/b/lesions"}, {"name": "liver", "url": "/b/liver"}]})
        voxel = {"/a/lesions": (1, 1, 1), "/b/lesions": (2, 2, 2), "/b/liver": (3, 3, 3)}[path]
        return httpx.Response(200, content=mask_at(*voxel))

    ref_path = tmp_path / "reference.nii.gz"
    ref_path.write_bytes(mask_at(1, 1, 1))
    app = main.create_app(Settings("https://api.test", "tid_test.secret", ref_path, "lesions"),
                          transport=httpx.MockTransport(stub))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local") as client:
            uploads = await asyncio.gather(*[
                client.post("/api/series", content=body, headers={"Content-Type": "application/zip"})
                for body in (zip_a, b"", zip_b, b"PK\x03\x04 corrupt zip", truncated_dicom_zip())])
            assert [r.status_code for r in uploads] == [200, 400, 200, 400, 400]
            assert all("Traceback" not in r.text for r in uploads)
            series_a, series_b = uploads[0].json()["id"], uploads[2].json()["id"]

            started = await asyncio.gather(client.post(f"/api/series/{series_a}/jobs"),
                                           client.post(f"/api/series/{series_b}/jobs"))
            job_a, job_b = (r.json()["job_id"] for r in started)
            for _ in range(200):
                states = [(await client.get(f"/api/jobs/{j}")).json() for j in (job_a, job_b)]
                if all(s["status"] in ("done", "failed") for s in states):
                    break
                await asyncio.sleep(0.02)
            state_a, state_b = states
            assert state_a["status"] == state_b["status"] == "done"
            assert state_b["finished_at"] <= state_a["finished_at"]
            assert [s["name"] for s in state_a["structures"]] == ["lesions"]
            assert [s["name"] for s in state_b["structures"]] == ["lesions", "liver"]

            async def voxels(job_id, name):
                packed = (await client.get(f"/api/jobs/{job_id}/masks/{name}")).content
                bits = np.unpackbits(np.frombuffer(packed, np.uint8))[: 4 * 5 * 6].reshape(4, 5, 6)
                return np.argwhere(bits).tolist()

            assert await voxels(job_a, "lesions") == [[1, 1, 1]]
            assert await voxels(job_b, "lesions") == [[2, 2, 2]]
            assert await voxels(job_b, "liver") == [[3, 3, 3]]
            assert state_a["dice"]["dice"] == 1.0 and state_b["dice"]["dice"] == 0.0

            listing = (await client.get("/api/jobs", params={"series_id": series_a})).json()
            assert [j["id"] for j in listing["jobs"]] == [job_a]      # jobs stay with their own scan
            assert (await client.get("/api/health")).status_code == 200
