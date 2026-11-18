"""AppError, and the mapping from Nucleo API errors to plain-language messages."""

import json


class AppError(Exception):
    """An error we expect and can explain to the user.

    kind:         short category, e.g. "input", "api", "geometry", "network".
    user_message: plain language, shown in the UI.
    detail:       technical details (codes, reasons), shown collapsed. Never a traceback.
    http_status:  status code used when this error is returned by our own routes.
    """

    def __init__(self, kind: str, user_message: str, detail: str = "", http_status: int = 400):
        super().__init__(user_message)
        self.kind = kind
        self.user_message = user_message
        self.detail = detail
        self.http_status = http_status

    def to_dict(self) -> dict:
        return {"kind": self.kind, "message": self.user_message, "detail": self.detail}


# Messages per (code, reason). Branch on code and reason, never on the API's message.
REASON_MESSAGES = {
    ("bad_request", "empty_body"): "The uploaded file is empty.",
    ("bad_request", "not_zip"): "The file is not a zip archive. Please select the DICOM zip.",
    ("bad_request", "empty_archive"): "The zip archive is empty.",
    ("bad_request", "corrupt_zip"): "The file is not a valid zip archive. Please check you selected the right file.",
    ("bad_request", "not_dicom"): "The zip contains files that aren't DICOM images.",
    ("bad_request", "corrupt_dicom"): "One or more DICOM files in the zip are damaged.",
    ("bad_request", "truncated_pixel_data"): "One or more DICOM images in the zip are incomplete (cut off).",
    ("bad_request", "too_many_entries"): "The zip has too many files in it (the limit is 5,000).",
    ("bad_request", "uncompressed_too_large"): "The zip is too large once unpacked (the limit is 1 GB).",
    ("bad_request", "member_too_large"): "A file inside the zip is too large (the limit is 64 MB per file).",
    ("unprocessable", "missing_tags"): "The DICOM files are missing information needed to place the images.",
    ("unprocessable", "multiple_series"): "The zip contains more than one scan. Please upload a single series.",
    ("unprocessable", "inconsistent_geometry"): "The slices in the scan don't all have the same size or orientation.",
    ("unprocessable", "incomplete_series"): "The scan is missing some slices, so it can't be analysed.",
    ("unprocessable", "inconsistent_spacing"): "The slices in the scan aren't evenly spaced, so it can't be analysed.",
    ("unprocessable", "duplicate_position"): "The scan contains duplicate slices at the same position.",
    ("unprocessable", "duplicate_instance"): "The scan contains the same image more than once.",
    ("unprocessable", "unsupported_transfer_syntax"): "The DICOM files use a compression format the service doesn't support.",
    ("unprocessable", "bad_geometry"): "The scan's position and orientation information is invalid.",
    ("not_found", "unknown_series"): "The server doesn't recognise this scan. Only the provided case can be analysed.",
    ("unauthorized", "missing"): "The app couldn't sign in to the analysis service (token missing).",
    ("unauthorized", "invalid"): "The app couldn't sign in to the analysis service (token invalid).",
    ("unauthorized", "expired"): "The app couldn't sign in to the analysis service (token expired).",
    ("unauthorized", "revoked"): "The app couldn't sign in to the analysis service (token revoked).",
    ("unavailable", "uploads_in_flight"): "The analysis service is busy right now. Please try again shortly.",
    ("unavailable", "upload_processing_saturated"): "The analysis service is busy right now. Please try again shortly.",
}

# 429 responses describe themselves with details.limit rather than details.reason.
RATE_LIMIT_MESSAGES = {
    "failed_auth_per_ip": "Too many failed sign-ins to the analysis service. Please check the API token.",
    "requests_per_token": "Too many requests to the analysis service. Please wait a minute and try again.",
    "uploads_per_token": "Too many uploads in the last minute. Please wait a minute and try again.",
    "concurrent_uploads_per_client": "Too many uploads at once. Please wait for one to finish.",
}

CODE_MESSAGES = {
    "bad_request": "The file couldn't be read as a DICOM zip.",
    "unauthorized": "The app couldn't sign in to the analysis service (token missing, invalid or expired).",
    "not_found": "The analysis service couldn't find what was requested.",
    "method_not_allowed": "The app sent a request the analysis service doesn't accept. This is a bug in the app.",
    "timeout": "The analysis service took too long to receive or process the scan. Please try again.",
    "too_large": "The file is larger than the 150 MB limit.",
    "unsupported_media_type": "The app sent the file in a format the service doesn't accept. This is a bug in the app.",
    "unprocessable": "The scan was readable, but the analysis service can't use this series.",
    "rate_limited": "Too many requests to the analysis service. Please wait a minute and try again.",
    "unavailable": "The analysis service is busy right now. Please try again shortly.",
    "internal": "The analysis service had an internal error. Please try again later.",
}

UNKNOWN_MESSAGE = "The analysis service returned an unexpected error. Please try again later."


def parse_api_error(body: bytes) -> tuple[str | None, dict]:
    """Return (code, details) from an error body. Tolerates non-JSON, empty or partial bodies."""
    try:
        data = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None, {}
    error = data.get("error") if isinstance(data, dict) else None
    if not isinstance(error, dict):
        return None, {}
    code = error.get("code") if isinstance(error.get("code"), str) else None
    details = error.get("details") if isinstance(error.get("details"), dict) else {}
    return code, details


def api_error_message(code: str | None, details: dict) -> str:
    reason = details.get("reason")
    if (code, reason) in REASON_MESSAGES:
        return REASON_MESSAGES[(code, reason)]
    if code == "rate_limited" and details.get("limit") in RATE_LIMIT_MESSAGES:
        return RATE_LIMIT_MESSAGES[details["limit"]]
    return CODE_MESSAGES.get(code, UNKNOWN_MESSAGE)


def api_error(status: int, body: bytes) -> AppError:
    """Build an AppError from a non-200 API response."""
    code, details = parse_api_error(body)
    if code is None and status in (502, 503, 504):
        message = "The analysis service is temporarily unreachable. Please try again shortly."
    else:
        message = api_error_message(code, details)
    parts = [f"HTTP {status}", f"code={code or 'none'}"]
    for key in ("reason", "limit"):
        if key in details:
            parts.append(f"{key}={details[key]}")
    return AppError("api", message, detail=", ".join(parts), http_status=502)
