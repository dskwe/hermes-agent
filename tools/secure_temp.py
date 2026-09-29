"""Secure ownership checks for directories under shared temporary roots."""

import errno
import os
import stat
from pathlib import Path


class SecureTempDirError(RuntimeError):
    """Raised when a shared-temp directory is not safe to use."""


def user_scoped_name(prefix: str) -> str:
    """Return a temp-directory name that cannot collide across Unix users."""
    uid = os.getuid() if hasattr(os, "getuid") else 0
    return f"{prefix}-{uid}"


def ensure_private_dir(path: str | Path) -> str:
    """Create and validate a private directory without accepting symlinks."""
    path = os.fspath(path)
    if os.name == "nt":
        os.makedirs(path, mode=0o700, exist_ok=True)
        return path
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        pass
    except OSError as exc:
        raise SecureTempDirError(f"Could not create secure temp directory {path}: {exc}") from exc
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise SecureTempDirError(f"Refusing symlink temp directory: {path}") from exc
        raise SecureTempDirError(f"Could not open secure temp directory {path}: {exc}") from exc
    try:
        st = os.fstat(fd)
        uid = os.getuid()
        if not stat.S_ISDIR(st.st_mode):
            raise SecureTempDirError(f"Refusing non-directory temp path: {path}")
        if st.st_uid != uid:
            raise SecureTempDirError(f"Temp directory {path} is owned by uid {st.st_uid}, not {uid}")
        if st.st_mode & 0o077:
            os.fchmod(fd, 0o700)
    finally:
        os.close(fd)
    return path


def write_private_file(path: str | Path, content: str) -> None:
    """Write a file without following a symlink on Unix."""
    path = os.fspath(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if os.name != "nt":
        flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise SecureTempDirError(f"Could not securely write temp file {path}: {exc}") from exc
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(content)
