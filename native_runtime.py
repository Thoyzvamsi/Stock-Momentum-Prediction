import ctypes
import sys
from pathlib import Path


_OPENMP_RUNTIME = None


if sys.platform.startswith("linux"):
    bundled_libgomp = (
        Path(__file__).resolve().parent / "vendor" / "lib" / "libgomp.so.1"
    )
    if bundled_libgomp.is_file():
        _OPENMP_RUNTIME = ctypes.CDLL(
            str(bundled_libgomp), mode=ctypes.RTLD_GLOBAL
        )
