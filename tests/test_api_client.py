"""API client tests. The API is always stubbed with httpx.MockTransport."""

import asyncio
import json

import httpx
import pytest

from app.api_client import NucleoClient, parse_retry_after
from app.errors import AppError

BASE = "https://api.test"
OK_BODY = {"case_id": "case-001", "structures": [{"name": "lesions", "url": "/v1/cases/case-001/structures/lesions"}]}


def error_response(status, code, headers=None, **details):
    body = {"error": {"code": code, "message": "server text we must not branch on", "details": details}}
    return httpx.Response(status, json=body, headers=headers or {})


class Stub:
    """Replies with the given responses in order and records every request."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.sleeps = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def sleep(self, seconds):
        self.sleeps.append(seconds)

    def client(self, base=BASE, token="tid_test.secret", jitter=lambda: 0.5) -> NucleoClient:
        return NucleoClient(base, token, transport=httpx.MockTransport(self.handler),
                            sleep=self.sleep, jitter=jitter)


def no_report(status, message):
    pass


async def test_upload_sends_raw_zip_with_token():
    stub = Stub(httpx.Response(200, json=OK_BODY))
    result = await stub.client().upload(b"PK\x03\x04zipbytes", no_report)
    request = stub.requests[0]
    assert request.method == "POST" and str(request.url) == BASE + "/v1/segmentations"
    assert request.headers["Content-Type"] == "application/zip"
    assert request.headers["Authorization"] == "Bearer tid_test.secret"
    assert request.content == b"PK\x03\x04zipbytes"   # raw body, not multipart
    assert result.case_id == "case-001" and result.structures[0].name == "lesions"


async def test_failed_upload_gives_plain_message():
    stub = Stub(error_response(400, "bad_request", reason="corrupt_zip"))
    with pytest.raises(AppError) as info:
        await stub.client().upload(b"junk", no_report)
    assert info.value.user_message.startswith("The file is not a valid zip archive")
    assert "reason=corrupt_zip" in info.value.detail
    assert len(stub.requests) == 1


async def test_retry_after_is_honoured_on_429():
    stub = Stub(error_response(429, "rate_limited", headers={"Retry-After": "2"}, limit="uploads_per_token"),
                httpx.Response(200, json=OK_BODY))
    messages = []
    await stub.client().upload(b"zip", lambda s, m: messages.append(m))
    assert len(stub.requests) == 2           # retried exactly once
    assert len(stub.sleeps) == 1 and stub.sleeps[0] >= 2
    assert any("retrying in 2 s" in m for m in messages)


async def test_retry_after_from_body_when_header_missing():
    stub = Stub(error_response(503, "unavailable", reason="uploads_in_flight", retry_after_s=3),
                httpx.Response(200, json=OK_BODY))
    await stub.client().upload(b"zip", no_report)
    assert stub.sleeps[0] >= 3


async def test_repeated_503_backs_off_exponentially():
    busy = lambda: error_response(503, "unavailable", headers={"Retry-After": "1"}, reason="uploads_in_flight")
    stub = Stub(busy(), busy(), busy(), httpx.Response(200, json=OK_BODY))
    await stub.client(jitter=lambda: 0.99).upload(b"zip", no_report)
    # attempt n waits max(Retry-After, 0.99 * 2^(n-1)) -> 1, 1.98, 3.96
    assert stub.sleeps == pytest.approx([1.0, 1.98, 3.96])


async def test_gives_up_after_max_attempts():
    stub = Stub(*[error_response(503, "unavailable", headers={"Retry-After": "1"}) for _ in range(5)])
    with pytest.raises(AppError, match="busy"):
        await stub.client().upload(b"zip", no_report)
    assert len(stub.requests) == 5 and len(stub.sleeps) == 4


async def test_retry_after_longer_than_cap_gives_up_without_waiting():
    stub = Stub(error_response(429, "rate_limited", headers={"Retry-After": "600"}, limit="uploads_per_token"))
    with pytest.raises(AppError, match="more than a minute"):
        await stub.client().upload(b"zip", no_report)
    assert stub.sleeps == []


@pytest.mark.parametrize("status,code,reason", [
    (400, "bad_request", "not_dicom"),
    (401, "unauthorized", "invalid"),
    (422, "unprocessable", "incomplete_series"),
    (404, "not_found", "unknown_series"),
    (415, "unsupported_media_type", None),
])
async def test_client_errors_are_not_retried(status, code, reason):
    stub = Stub(error_response(status, code, **({"reason": reason} if reason else {})))
    with pytest.raises(AppError):
        await stub.client().upload(b"zip", no_report)
    assert len(stub.requests) == 1 and stub.sleeps == []


async def test_failed_auth_rate_limit_is_not_retried():
    stub = Stub(error_response(429, "rate_limited", headers={"Retry-After": "5"}, limit="failed_auth_per_ip"))
    with pytest.raises(AppError, match="check the API token"):
        await stub.client().upload(b"zip", no_report)
    assert len(stub.requests) == 1 and stub.sleeps == []


async def test_gateway_html_502_gives_friendly_error():
    html = lambda: httpx.Response(502, text="<html><body>Bad Gateway</body></html>")
    stub = Stub(html(), html(), html())
    with pytest.raises(AppError) as info:
        await stub.client().upload(b"zip", no_report)
    assert "temporarily unreachable" in info.value.user_message
    assert "<html" not in info.value.user_message
    assert len(stub.requests) == 3           # small retry cap for gateway errors


async def test_network_error_is_retried_then_reported():
    stub = Stub(*[httpx.ConnectError("refused") for _ in range(3)])
    with pytest.raises(AppError, match="Couldn't reach"):
        await stub.client().upload(b"zip", no_report)
    assert len(stub.requests) == 3


@pytest.mark.parametrize("body", [
    b"not json at all",
    json.dumps({"case_id": "case-001"}).encode(),                       # missing structures
    json.dumps({"case_id": 7, "structures": []}).encode(),              # wrong type
    json.dumps({"case_id": "c", "structures": [{"name": "x"}]}).encode(),  # missing url
])
async def test_malformed_200_gives_friendly_error(body):
    stub = Stub(httpx.Response(200, content=body))
    with pytest.raises(AppError, match="couldn't understand"):
        await stub.client().upload(b"zip", no_report)


async def test_empty_structure_list():
    stub = Stub(httpx.Response(200, json={"case_id": "case-001", "structures": []}))
    with pytest.raises(AppError, match="no structures"):
        await stub.client().upload(b"zip", no_report)


@pytest.mark.parametrize("url", ["https://evil.test/steal", "//evil.test/steal", "http://api.test/x"])
async def test_mask_url_on_other_host_is_refused_and_not_requested(url):
    stub = Stub()
    with pytest.raises(AppError, match="different server"):
        await stub.client().download(url, no_report)
    assert stub.requests == []               # the token never left


async def test_relative_url_keeps_base_path():
    stub = Stub(httpx.Response(200, content=b"mask"))
    client = stub.client(base="https://api.test/prefix/")
    assert await client.download("/v1/cases/case-001/structures/lesions", no_report) == b"mask"
    assert str(stub.requests[0].url) == "https://api.test/prefix/v1/cases/case-001/structures/lesions"
    assert stub.requests[0].headers["Authorization"] == "Bearer tid_test.secret"


async def test_missing_token_sends_nothing():
    stub = Stub()
    with pytest.raises(AppError, match="token not configured"):
        await stub.client(token=None).upload(b"zip", no_report)
    assert stub.requests == []


def test_parse_retry_after():
    assert parse_retry_after("7") == 7.0
    assert parse_retry_after(None) is None
    assert parse_retry_after("soon") is None
    assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0   # in the past -> no wait


async def test_at_most_two_uploads_in_flight():
    in_flight, peak = 0, 0

    async def handler(request):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        return httpx.Response(200, json=OK_BODY)

    client = NucleoClient(BASE, "tid_test.secret", transport=httpx.MockTransport(handler))
    statuses = []
    await asyncio.gather(*[client.upload(b"zip", lambda s, m: statuses.append(s)) for _ in range(4)])
    assert peak == 2
    assert "waiting" in statuses


async def test_backend_that_never_answers_times_out(monkeypatch):
    monkeypatch.setattr("app.api_client.HTTP_TOTAL_TIMEOUT_S", 0.1)
    requests = []

    async def hang(request):
        requests.append(request)
        await asyncio.sleep(10)

    client = NucleoClient(BASE, "tid_test.secret", transport=httpx.MockTransport(hang))
    with pytest.raises(AppError, match="took too long"):
        await client.upload(b"zip", no_report)
    assert len(requests) == 1               # a full timeout is not retried


async def test_oversized_response_is_refused(monkeypatch):
    monkeypatch.setattr("app.api_client.MAX_MASK_DOWNLOAD_BYTES", 100)
    stub = Stub(httpx.Response(200, content=b"\0" * 1000))
    with pytest.raises(AppError, match="too large"):
        await stub.client().download("/v1/cases/case-001/structures/lesions", no_report)
