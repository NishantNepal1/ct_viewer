"""Read a zip archive in memory with explicit limits.

Nothing is extracted to disk, so path traversal in member names can't write files.
"""

import io
import zipfile
import zlib

from app.config import MAX_ZIP_ENTRIES, MAX_ZIP_MEMBER_BYTES, MAX_ZIP_TOTAL_BYTES
from app.errors import AppError

CHUNK_BYTES = 1024 * 1024


def is_ignored(info: zipfile.ZipInfo) -> bool:
    """Directories, macOS resource forks, dot-files and DICOMDIR index files are skipped."""
    name = info.filename
    parts = [p for p in name.split("/") if p]
    if info.is_dir() or not parts:
        return True
    if parts[0] == "__MACOSX" or any(p.startswith(".") for p in parts):
        return True
    return parts[-1].upper() == "DICOMDIR"


def read_zip_members(
    data: bytes,
    max_entries: int = MAX_ZIP_ENTRIES,
    max_member: int = MAX_ZIP_MEMBER_BYTES,
    max_total: int = MAX_ZIP_TOTAL_BYTES,
) -> list[tuple[str, bytes]]:
    """Return [(name, bytes)] for every non-ignored file in the zip."""
    if not data:
        raise AppError("input", "The file is empty.")
    stream = io.BytesIO(data)
    if not zipfile.is_zipfile(stream):
        raise AppError("input", "The file is not a zip archive. Please select the DICOM zip.")
    try:
        archive = zipfile.ZipFile(stream)
    except (zipfile.BadZipFile, OSError, ValueError) as exc:
        raise AppError("input", "The file is not a valid zip archive. Please check you selected the right file.",
                       detail=f"{type(exc).__name__}: {exc}") from exc

    with archive:
        infos = archive.infolist()
        # Check the central directory before reading anything.
        if len(infos) > max_entries:
            raise AppError("input", f"The zip has too many files in it (the limit is {max_entries:,}).",
                           detail=f"entries={len(infos)}")
        wanted = [info for info in infos if not is_ignored(info)]
        if not wanted:
            raise AppError("input", "The zip archive has no files in it.")
        if any(info.file_size > max_member for info in wanted):
            raise AppError("input", "A file inside the zip is too large.", detail=f"max_member={max_member}")
        if sum(info.file_size for info in wanted) > max_total:
            raise AppError("input", "The zip is too large once unpacked.", detail=f"max_total={max_total}")

        # Declared sizes can lie, so cap again while actually decompressing.
        members = []
        total = 0
        for info in wanted:
            content = _read_member(archive, info, min(max_member, max_total - total))
            total += len(content)
            members.append((info.filename, content))
        return members


def _read_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, limit: int) -> bytes:
    buf = bytearray()
    try:
        with archive.open(info) as member:
            while chunk := member.read(CHUNK_BYTES):
                buf += chunk
                if len(buf) > limit:
                    raise AppError("input", "The zip is too large once unpacked.",
                                   detail=f"member {info.filename!r} exceeded {limit} bytes")
    except AppError:
        raise
    except (zipfile.BadZipFile, zlib.error, EOFError, OSError) as exc:
        raise AppError("input", "The zip archive is damaged and couldn't be unpacked.",
                       detail=f"{type(exc).__name__} in {info.filename!r}") from exc
    except (NotImplementedError, RuntimeError) as exc:
        # Unsupported compression method, or an encrypted (password-protected) member.
        raise AppError("input", "The zip uses encryption or a compression method that isn't supported.",
                       detail=f"{type(exc).__name__} in {info.filename!r}") from exc
    return bytes(buf)
