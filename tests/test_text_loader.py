from pathlib import Path

import pytest

from backend.app.loaders.text import TextLoader, decode_text

SAMPLE = "Slab temperature shall be 20°F minimum at placement."


def test_utf8_plain() -> None:
    assert decode_text(SAMPLE.encode("utf-8")) == SAMPLE


def test_utf8_bom_is_stripped() -> None:
    assert decode_text(b"\xef\xbb\xbf" + SAMPLE.encode("utf-8")) == SAMPLE


@pytest.mark.parametrize("encoding", ["utf-16-le", "utf-16-be"])
def test_utf16_with_bom(encoding: str) -> None:
    bom = b"\xff\xfe" if encoding.endswith("le") else b"\xfe\xff"
    assert decode_text(bom + SAMPLE.encode(encoding)) == SAMPLE


@pytest.mark.parametrize("padding", ["", "!"])
def test_cp1252_is_not_mistaken_for_utf16(padding: str) -> None:
    """Regression: an even-length cp1252 file used to decode as UTF-16 CJK mojibake.

    Both parities are exercised because only even-length input fooled the old decoder.
    """
    raw = (SAMPLE + padding).encode("cp1252")
    assert decode_text(raw) == SAMPLE + padding


def test_cp1252_even_length_regression_input_really_is_even() -> None:
    assert len(SAMPLE.encode("cp1252")) % 2 == 0  # guards the test above against edits


def test_undecodable_bytes_raise() -> None:
    # 0x81 is undefined in cp1252 and invalid as a UTF-8 lead byte.
    with pytest.raises(UnicodeDecodeError):
        decode_text(b"abc\x81def")


def test_loader_reads_file(tmp_path: Path) -> None:
    path = tmp_path / "spec.txt"
    path.write_bytes(SAMPLE.encode("cp1252"))
    docs = TextLoader().load(path)
    assert [d.page_content for d in docs] == [SAMPLE]
    assert docs[0].metadata == {}


@pytest.mark.parametrize(
    ("head", "expected"),
    [
        (b"plain ascii", True),
        (b"\xff\xfeA\x00B\x00", True),  # UTF-16 text legitimately contains NULs
        (b"\xfe\xff\x00A\x00B", True),
        (b"MZ\x90\x00\x03\x00", False),  # binary
    ],
)
def test_validate_magic(head: bytes, expected: bool) -> None:
    assert TextLoader().validate_magic(head) is expected
