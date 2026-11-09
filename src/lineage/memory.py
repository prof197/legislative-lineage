"""The process's memory, current and peak, counted the way the operating system charges it.

On macOS, resident set size undercounts: the kernel compresses idle pages, and compressed
pages are not resident. The physical footprint, which Activity Monitor shows as Memory,
includes them, so it is what this module reports there. Linux does not compress anonymous
memory by default, and its resident set size serves.
"""

import ctypes
import os
import resource
import sys
from pathlib import Path

# proc_pid_rusage's RUSAGE_INFO_V4: a 16-byte UUID followed by 64-bit counters, of which the
# 8th is the current physical footprint and the 29th the lifetime maximum (sys/resource.h).
_RUSAGE_INFO_V4 = 4
_FOOTPRINT = 8
_LIFETIME_MAX_FOOTPRINT = 29
_COUNTERS = 35


class _RusageInfoV4(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16), ("counters", ctypes.c_uint64 * _COUNTERS)]


def _darwin_counter(index: int) -> int:
    libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
    info = _RusageInfoV4()
    if libproc.proc_pid_rusage(os.getpid(), _RUSAGE_INFO_V4, ctypes.byref(info)) != 0:
        raise OSError(ctypes.get_errno(), "proc_pid_rusage failed")
    counters: ctypes.Array[ctypes.c_uint64] = info.counters
    return int(counters[index - 1])


def _linux_status(field: str) -> int:
    with Path("/proc/self/status").open() as status:
        for line in status:
            if line.startswith(field + ":"):
                return int(line.split()[1]) * 1024
    raise OSError(f"{field} missing from /proc/self/status")


def current_bytes() -> int:
    if sys.platform == "darwin":
        return _darwin_counter(_FOOTPRINT)
    return _linux_status("VmRSS")


def peak_bytes() -> int:
    if sys.platform == "darwin":
        return _darwin_counter(_LIFETIME_MAX_FOOTPRINT)
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


# DR-3's bound: a build stage within half the reference laptop's 32 GiB.
BOUND_BYTES = 16 * 1024**3


class MemoryBoundError(Exception):
    pass


def require_within(bound: int, what: str) -> None:
    """Stop when the process's footprint has crossed `bound`, before it pushes into swap."""
    now = current_bytes()
    if now > bound:
        raise MemoryBoundError(
            f"{what}: the process holds {now / 1024**3:.2f} GiB, over the "
            f"{bound / 1024**3:.0f} GiB bound; stopped"
        )
