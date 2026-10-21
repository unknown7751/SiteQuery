"""Durable file primitives: a rename is only crash-safe if the data was flushed first."""

import os
from pathlib import Path


def fsync_file(path: Path) -> None:
    """Flush `path`'s contents to disk. Opens its own handle, so it works on files written
    by libraries (FAISS, pickle) that give us no file object."""
    fd = os.open(path, os.O_RDWR | getattr(os, "O_BINARY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_dir(path: Path) -> None:
    """Flush a directory entry (new file name / rename) to disk.

    POSIX only: Windows cannot open a directory, and NTFS journals metadata itself.
    """
    if os.name != "posix":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Replace `path` with `data` so readers see the old or the new content, never a mix.

    Write to a sibling temp file, fsync it, rename over the target, then fsync the directory
    so the rename itself survives power loss.
    """
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)
