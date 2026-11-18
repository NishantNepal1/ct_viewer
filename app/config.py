"""Settings from the environment (.env is loaded here) and every limit in one place."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

MiB = 1024 * 1024

# Local upload and zip limits (same values the remote API enforces).
MAX_UPLOAD_BYTES = 150 * MiB
MAX_ZIP_ENTRIES = 5000
MAX_ZIP_MEMBER_BYTES = 64 * MiB
MAX_ZIP_TOTAL_BYTES = 1024 * MiB

# Masks: compressed download size, and size after gunzip. The real mask is
# 416*350*180 voxels (~26 MB as uint8), so 256 MiB leaves room for float types.
MAX_MASK_DOWNLOAD_BYTES = 64 * MiB
MAX_MASK_DECOMPRESSED_BYTES = 256 * MiB
MAX_MASK_DIM = 4096

# Outbound HTTP.
HTTP_CONNECT_TIMEOUT_S = 10.0
HTTP_TOTAL_TIMEOUT_S = 90.0         # whole request, including the body
MAX_ATTEMPTS = 5                    # per request, for 429 / 503
MAX_TRANSIENT_ATTEMPTS = 3          # per request, for network errors / 502 / 504
MAX_RETRY_WAIT_S = 60.0             # a longer Retry-After makes us give up instead
BACKOFF_BASE_S = 1.0
MAX_CONCURRENT_UPLOADS = 2          # the token allows 2 concurrent uploads

# In-memory stores (oldest entries are evicted first).
MAX_SERIES_IN_MEMORY = 3
MAX_JOBS_IN_MEMORY = 10

DEFAULT_API_BASE_URL = "https://bnj2vb4zpt.us-east-1.awsapprunner.com"
DEFAULT_REFERENCE_PATH = "data/case-001-reference-lesions.nii.gz"
REFERENCE_CASE_ID = "case-001"  # the case the reference mask belongs to


@dataclass(frozen=True)
class Settings:
    api_base_url: str
    api_token: str | None
    reference_mask_path: Path
    lesion_structure_name: str


def load_settings() -> Settings:
    base_url = os.environ.get("NUCLEO_API_BASE_URL", "").strip() or DEFAULT_API_BASE_URL
    token = os.environ.get("NUCLEO_API_TOKEN", "").strip() or None
    ref_path = Path(os.environ.get("REFERENCE_MASK_PATH", "").strip() or DEFAULT_REFERENCE_PATH)
    if not ref_path.is_absolute():
        ref_path = PROJECT_ROOT / ref_path
    lesion_name = os.environ.get("LESION_STRUCTURE_NAME", "").strip() or "lesions"
    return Settings(
        api_base_url=base_url.rstrip("/"),
        api_token=token,
        reference_mask_path=ref_path,
        lesion_structure_name=lesion_name,
    )
