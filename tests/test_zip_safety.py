import pytest

from app.errors import AppError
from app.zip_safety import read_zip_members
from tests.helpers import make_zip


def test_empty_file():
    with pytest.raises(AppError, match="empty"):
        read_zip_members(b"")


def test_not_a_zip():
    with pytest.raises(AppError, match="not a zip"):
        read_zip_members(b"this is plain text, not a zip archive")


def test_truncated_zip_is_rejected():
    data = make_zip([("a.dcm", b"x" * 1000)])
    with pytest.raises(AppError):
        read_zip_members(data[: len(data) // 2])


def test_ignored_entries_are_skipped():
    data = make_zip([
        ("case/0001.dcm", b"one"),
        ("__MACOSX/case/._0001.dcm", b"resource fork"),
        ("case/.DS_Store", b"finder"),
        ("case/DICOMDIR", b"index"),
        ("case/sub/", b""),
    ])
    assert read_zip_members(data) == [("case/0001.dcm", b"one")]


def test_only_ignored_entries_means_no_files():
    with pytest.raises(AppError, match="no files"):
        read_zip_members(make_zip([("__MACOSX/x", b"x")]))


def test_too_many_entries():
    data = make_zip([(f"f{n}.dcm", b"") for n in range(5001)])
    with pytest.raises(AppError, match="too many files"):
        read_zip_members(data)


def test_expanded_size_cap():
    # 3 MB of zeros compresses to a few KB; the cap is set low to keep the test fast.
    data = make_zip([("big.dcm", b"\0" * 3_000_000)])
    with pytest.raises(AppError, match="too large"):
        read_zip_members(data, max_total=1_000_000)


def test_member_size_cap():
    data = make_zip([("big.dcm", b"\0" * 2_000_000)])
    with pytest.raises(AppError, match="too large"):
        read_zip_members(data, max_member=1_000_000)


def test_corrupt_member_data():
    payload = b"HELLO-DICOM-PAYLOAD" * 50
    data = bytearray(make_zip([("a.dcm", payload)]))
    # Deflated content isn't findable, so corrupt the middle of the compressed stream.
    header_end = data.index(b"a.dcm") + len("a.dcm")
    data[header_end + 5: header_end + 15] = b"\xff" * 10
    with pytest.raises(AppError):
        read_zip_members(bytes(data))
