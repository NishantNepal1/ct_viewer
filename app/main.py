"""FastAPI routes, static files and error handlers. Runs locally only."""

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api_client import NucleoClient
from app.config import (MAX_JOBS_IN_MEMORY, MAX_SERIES_IN_MEMORY, MAX_UPLOAD_BYTES, PROJECT_ROOT, Settings,
                        load_settings)
from app.dicom_io import load_dicom_zip
from app.errors import AppError
from app.jobs import BoundedStore, Job, SeriesEntry, job_dice, pack_on_ct_grid, pick_default_job, run_job
from app.nifti_io import MaskImage, load_mask_file

logger = logging.getLogger("ctviewer")
GENERIC_ERROR = "Something went wrong on our side. Please try again."
STATIC_DIR = PROJECT_ROOT / "static"


def create_app(settings: Settings | None = None, transport=None, sleep=asyncio.sleep) -> FastAPI:
    """Build the app. Tests pass a stub `transport` (and `sleep`) so no real API call is made."""
    settings = settings or load_settings()
    series_store = BoundedStore(MAX_SERIES_IN_MEMORY)
    job_store = BoundedStore(MAX_JOBS_IN_MEMORY)
    running_tasks = set()   # keep references so background tasks aren't garbage-collected
    state = {}              # "client", "reference", "reference_message"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["client"] = NucleoClient(settings.api_base_url, settings.api_token, transport=transport, sleep=sleep)
        state["reference"], state["reference_message"] = await asyncio.to_thread(load_reference, settings)
        if settings.api_token is None:
            logger.warning("NUCLEO_API_TOKEN is not set; uploads will fail with a clear message.")
        yield
        for task in running_tasks:
            task.cancel()
        await state["client"].aclose()

    app = FastAPI(title="CT segmentation viewer", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # ---- error handlers: every error body is {"error": {"kind", "message", "detail"}} ----

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        return JSONResponse({"error": exc.to_dict()}, status_code=exc.http_status)

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, exc: StarletteHTTPException):
        message = "Not found." if exc.status_code == 404 else "The request couldn't be handled."
        return JSONResponse({"error": {"kind": "http", "message": message, "detail": f"HTTP {exc.status_code}"}},
                            status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception):
        detail_id = uuid.uuid4().hex[:8]
        logger.error("Unhandled error %s on %s %s", detail_id, request.method, request.url.path, exc_info=exc)
        return JSONResponse({"error": {"kind": "internal", "message": GENERIC_ERROR,
                                       "detail": f"error id {detail_id}", "detail_id": detail_id}},
                            status_code=500)

    # ---- helpers that look things up or 404 ----

    def get_series(series_id: str) -> SeriesEntry:
        series = series_store.get(series_id)
        if series is None:
            raise AppError("not_found", "That scan is no longer loaded. Please open the file again.", http_status=404)
        return series

    def get_job(job_id: str) -> Job:
        job = job_store.get(job_id)
        if job is None:
            raise AppError("not_found", "That job is no longer available.", http_status=404)
        return job

    # ---- routes ----

    @app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    async def health():
        return {
            "status": "ok",
            "token_configured": settings.api_token is not None,
            "reference_available": state.get("reference") is not None,
            "reference_message": state.get("reference_message", ""),
        }

    @app.post("/api/series")
    async def create_series(request: Request):
        data = await read_body_capped(request, MAX_UPLOAD_BYTES)
        ct = await run_in_threadpool(load_dicom_zip, data)
        series = SeriesEntry(id=uuid.uuid4().hex, zip_bytes=data, ct=ct)
        series.reference_packed, series.reference_message = await run_in_threadpool(
            reference_on_grid, state["reference"], state["reference_message"], ct)
        series_store.add(series.id, series)
        return series_metadata(series)

    @app.get("/api/series/{series_id}/volume")
    async def series_volume(series_id: str):
        ct = get_series(series_id).ct
        body = await run_in_threadpool(lambda: ct.volume.astype("<i2").tobytes())  # (k, j, i), little-endian
        k, j, i = ct.volume.shape
        return Response(body, media_type="application/octet-stream", headers={"X-Volume-Shape": f"{k},{j},{i}"})

    @app.get("/api/series/{series_id}/reference")
    async def series_reference(series_id: str):
        series = get_series(series_id)
        if series.reference_packed is None:
            raise AppError("reference", series.reference_message, http_status=404)
        return Response(series.reference_packed, media_type="application/octet-stream")

    @app.post("/api/series/{series_id}/jobs")
    async def create_job(series_id: str):
        series = get_series(series_id)
        job = Job(series_id=series.id)
        job_store.add(job.id, job)
        task = asyncio.create_task(run_job(job, series, state["client"], state["reference"],
                                           settings.lesion_structure_name))
        running_tasks.add(task)
        task.add_done_callback(running_tasks.discard)
        return {"job_id": job.id, "number": job.number}

    @app.get("/api/jobs")
    async def list_jobs(series_id: str | None = None):
        jobs = [job for job in job_store.values() if series_id is None or job.series_id == series_id]
        jobs.sort(key=lambda job: job.number)
        default = pick_default_job(jobs)
        return {"jobs": [job.to_dict() for job in jobs], "default_job_id": default.id if default else None}

    @app.get("/api/jobs/{job_id}")
    async def job_status(job_id: str):
        return get_job(job_id).to_dict()

    @app.get("/api/jobs/{job_id}/masks/{name:path}")
    async def job_mask(job_id: str, name: str):
        structure = get_job(job_id).structures.get(name)
        if structure is None or structure.display_packed is None:
            raise AppError("not_found", "That mask isn't available for this job.", http_status=404)
        return Response(structure.display_packed, media_type="application/octet-stream")

    @app.get("/api/jobs/{job_id}/dice")
    async def job_dice_for(job_id: str, structure: str):
        job = get_job(job_id)
        if job.status != "done":
            raise AppError("not_ready", "This job hasn't finished yet.", http_status=409)
        return await run_in_threadpool(job_dice, job, state["reference"], structure)

    return app


async def read_body_capped(request: Request, max_bytes: int) -> bytes:
    """Read the upload body, refusing it as soon as it passes max_bytes."""
    too_large = AppError("input", f"The file is larger than the {max_bytes // (1024 * 1024)} MB limit.",
                         http_status=413)
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > max_bytes:
        raise too_large
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > max_bytes:
            raise too_large
    if not body:
        raise AppError("input", "The file is empty.")
    return bytes(body)


def load_reference(settings: Settings) -> tuple[MaskImage | None, str]:
    """Load the reference mask once at startup. Missing or broken -> (None, reason)."""
    path = settings.reference_mask_path
    if not path.is_file():
        return None, f"Reference mask not found ({path.name}), so Dice and the reference overlay aren't available."
    try:
        return load_mask_file(path, "reference"), ""
    except AppError as exc:
        return None, f"Reference mask couldn't be loaded: {exc.user_message}"


def reference_on_grid(reference: MaskImage | None, message: str, ct) -> tuple[bytes | None, str]:
    if reference is None:
        return None, message
    try:
        return pack_on_ct_grid(reference, ct), ""
    except AppError as exc:
        return None, f"The reference can't be shown on this scan: {exc.user_message}"


def series_metadata(series: SeriesEntry) -> dict:
    ct = series.ct
    k, j, i = ct.volume.shape
    return {
        "id": series.id,
        "shape": {"slices": k, "rows": j, "columns": i},
        "spacing_mm": {"column": ct.spacing[0], "row": ct.spacing[1], "slice": ct.spacing[2]},
        "slice_positions_mm": ct.slice_positions,
        "labels": ct.labels,
        "warnings": ct.warnings,
        "reference": {"available": series.reference_packed is not None, "message": series.reference_message},
    }


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = create_app()
