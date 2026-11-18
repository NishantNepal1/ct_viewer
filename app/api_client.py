"""Async client for the Nucleo segmentation API: upload, mask download, retries."""

import asyncio
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Awaitable, Callable
from urllib.parse import urlsplit

import httpx

from app.config import (BACKOFF_BASE_S, HTTP_CONNECT_TIMEOUT_S, HTTP_TOTAL_TIMEOUT_S, MAX_ATTEMPTS,
                        MAX_CONCURRENT_UPLOADS, MAX_MASK_DOWNLOAD_BYTES, MAX_RETRY_WAIT_S, MAX_TRANSIENT_ATTEMPTS)
from app.errors import AppError, api_error, parse_api_error

# report(status, message): lets a job show "uploading", "retrying in 4 s", ...
Reporter = Callable[[str, str], None]
MAX_UPLOAD_RESPONSE_BYTES = 1024 * 1024
TRANSIENT_STATUSES = (502, 504)


@dataclass
class Structure:
    name: str
    url: str


@dataclass
class UploadResult:
    case_id: str
    structures: list  # [Structure]


def parse_retry_after(value: str | None) -> float | None:
    """Retry-After as seconds: an integer, or an HTTP-date. None if absent or unparseable."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


def parse_upload_response(body: bytes) -> UploadResult:
    """Validate the 200 body: {"case_id": str, "structures": [{"name": str, "url": str}, ...]}."""
    bad = AppError("api", "The analysis service sent a response the app couldn't understand.",
                   detail="malformed 200 response body")
    try:
        data = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        raise bad from None
    if not isinstance(data, dict) or not isinstance(data.get("case_id"), str):
        raise bad
    items = data.get("structures")
    if not isinstance(items, list):
        raise bad
    if not items:
        raise AppError("api", "The model returned no structures.", detail="structures: []")
    structures = []
    for item in items:
        if not (isinstance(item, dict) and isinstance(item.get("name"), str) and isinstance(item.get("url"), str)
                and item["name"] and item["url"]):
            raise bad
        structures.append(Structure(item["name"], item["url"]))
    return UploadResult(case_id=data["case_id"], structures=structures)


class NucleoClient:
    def __init__(self, base_url: str, token: str | None, transport: httpx.AsyncBaseTransport | None = None,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                 jitter: Callable[[], float] = random.random):
        self.base_url = base_url.rstrip("/")
        self._token = token
        self._sleep = sleep      # injectable so tests don't wait
        self._jitter = jitter    # returns a float in [0, 1)
        self._upload_slots = asyncio.Semaphore(MAX_CONCURRENT_UPLOADS)
        # follow_redirects=False (the default) so the token is never sent to a redirect target.
        self._http = httpx.AsyncClient(
            transport=transport,
            timeout=httpx.Timeout(HTTP_TOTAL_TIMEOUT_S, connect=HTTP_CONNECT_TIMEOUT_S),
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def upload(self, zip_bytes: bytes, report: Reporter) -> UploadResult:
        if self._upload_slots.locked():
            report("waiting", "Waiting for a free upload slot")
        async with self._upload_slots:
            report("uploading", "Uploading scan and running the model (usually 5–10 s)")
            body = await self._request_with_retries(
                "POST", self.base_url + "/v1/segmentations", report, content=zip_bytes,
                headers={"Content-Type": "application/zip"}, max_body=MAX_UPLOAD_RESPONSE_BYTES,
            )
        return parse_upload_response(body)

    async def download(self, url: str, report: Reporter) -> bytes:
        return await self._request_with_retries("GET", self.resolve_url(url), report,
                                                max_body=MAX_MASK_DOWNLOAD_BYTES)

    def resolve_url(self, url: str) -> str:
        """Join a relative URL onto the base URL. Absolute URLs must be on the same host."""
        parts = urlsplit(url)
        if parts.scheme or parts.netloc:
            base = urlsplit(self.base_url)
            if (parts.scheme, parts.netloc) != (base.scheme, base.netloc):
                raise AppError("api", "The analysis service pointed to a mask on a different server, "
                                      "so it wasn't downloaded (to keep the API token safe).",
                               detail=f"refused host: {parts.netloc or '(none)'}")
            return url
        # Plain string join keeps any path prefix in the base URL (urljoin would drop it).
        return self.base_url + "/" + url.lstrip("/")

    async def _request_with_retries(self, method: str, url: str, report: Reporter, *,
                                    max_body: int, content: bytes | None = None, headers: dict | None = None) -> bytes:
        if not self._token:
            raise AppError("config", "API token not configured. Set NUCLEO_API_TOKEN in .env and restart the app.")
        all_headers = {"Authorization": f"Bearer {self._token}", **(headers or {})}
        transient_failures = 0
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                status, resp_headers, body = await self._send(method, url, all_headers, content, max_body)
            except httpx.TransportError as exc:  # connection refused/reset, DNS, read timeout, ...
                transient_failures += 1
                if transient_failures >= MAX_TRANSIENT_ATTEMPTS:
                    raise AppError("network", "Couldn't reach the analysis service. "
                                              "Check your internet connection and try again.",
                                   detail=type(exc).__name__) from exc
                await self._wait(self._backoff(attempt), "Connection problem", report)
                continue

            if status == 200:
                return body
            if status in TRANSIENT_STATUSES:
                transient_failures += 1
                if transient_failures >= MAX_TRANSIENT_ATTEMPTS:
                    raise api_error(status, body)
                await self._wait(self._backoff(attempt), "Gateway problem", report)
                continue
            if not self._is_retryable(status, body) or attempt == MAX_ATTEMPTS:
                raise api_error(status, body)
            wait = self._retry_wait(resp_headers, body, attempt)
            if wait is None:
                raise AppError("api", "The analysis service is busy and asked us to wait more than a minute. "
                                      "Please try again later.", detail=f"HTTP {status}, Retry-After too long")
            await self._wait(wait, "Server busy", report)
        raise AssertionError("unreachable")  # the loop always returns or raises

    async def _send(self, method, url, headers, content, max_body) -> tuple[int, httpx.Headers, bytes]:
        """One HTTP request with an overall deadline and a cap on the response size."""
        try:
            async with asyncio.timeout(HTTP_TOTAL_TIMEOUT_S):
                async with self._http.stream(method, url, headers=headers, content=content) as resp:
                    body = bytearray()
                    async for chunk in resp.aiter_bytes():
                        body += chunk
                        if len(body) > max_body:
                            raise AppError("api", "The analysis service sent a response that is too large.",
                                           detail=f"limit {max_body} bytes")
                    return resp.status_code, resp.headers, bytes(body)
        except TimeoutError as exc:
            raise AppError("network", "The analysis service took too long to respond. Please try again.",
                           detail=f"no complete response within {HTTP_TOTAL_TIMEOUT_S:.0f} s") from exc

    @staticmethod
    def _is_retryable(status: int, body: bytes) -> bool:
        if status == 503:
            return True
        if status == 429:
            _, details = parse_api_error(body)
            return details.get("limit") != "failed_auth_per_ip"  # a bad token won't fix itself
        return False

    def _retry_wait(self, headers: httpx.Headers, body: bytes, attempt: int) -> float | None:
        """Seconds to wait: at least the server's hint, or jittered backoff. None if hint > cap."""
        hint = parse_retry_after(headers.get("retry-after"))
        if hint is None:
            _, details = parse_api_error(body)
            value = details.get("retry_after_s")
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
                hint = float(value)
        if hint is not None and hint > MAX_RETRY_WAIT_S:
            return None
        return max(hint or 0.0, self._backoff(attempt))

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff with full jitter: random in [0, base * 2^(attempt-1)), capped."""
        return self._jitter() * min(MAX_RETRY_WAIT_S, BACKOFF_BASE_S * 2 ** (attempt - 1))

    async def _wait(self, seconds: float, why: str, report: Reporter) -> None:
        report("retrying", f"{why}, retrying in {max(1, round(seconds))} s")
        await self._sleep(seconds)
        report("retrying", f"{why}, retrying now")
