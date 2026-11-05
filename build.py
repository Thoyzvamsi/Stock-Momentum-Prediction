import hashlib
import platform
import tarfile
from pathlib import Path
import tempfile
from urllib.request import urlopen


LIBGOMP_ARCHIVES = {
    "x86_64": (
        "https://conda.anaconda.org/conda-forge/linux-64/"
        "libgomp-12.2.0-h65d4601_19.tar.bz2",
        "81a76d20cfdee9fe0728b93ef057ba93494fd1450d42bc3717af4e468235661e",
    ),
    "aarch64": (
        "https://conda.anaconda.org/conda-forge/linux-aarch64/"
        "libgomp-12.2.0-h607ecd0_19.tar.bz2",
        "d802eaceaf6f77fb8cb2e990aacfb3eaa83361b16369a760fc1585841d7885c",
    ),
}
LIBGOMP_MEMBER = "lib/libgomp.so.1.0.0"
OUTPUT = Path(__file__).resolve().parent / "vendor" / "lib" / "libgomp.so.1"


def main():
    if platform.system() != "Linux":
        print("Skipping bundled libgomp; Vercel's Linux build will add it.")
        return

    machine = platform.machine().lower()
    try:
        url, expected_sha256 = LIBGOMP_ARCHIVES[machine]
    except KeyError as error:
        raise RuntimeError(
            f"No bundled libgomp package is configured for {machine}."
        ) from error

    with tempfile.TemporaryDirectory() as temporary_directory:
        archive_path = Path(temporary_directory) / "libgomp.tar.bz2"
        with urlopen(url, timeout=60) as response, archive_path.open("wb") as archive:
            while chunk := response.read(1024 * 1024):
                archive.write(chunk)

        actual_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError("Downloaded libgomp package failed SHA-256 verification.")

        with tarfile.open(archive_path, mode="r:bz2") as package:
            member = package.getmember(LIBGOMP_MEMBER)
            source = package.extractfile(member)
            if source is None:
                raise RuntimeError("libgomp package does not contain its shared library.")
            OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            OUTPUT.write_bytes(source.read())

    OUTPUT.chmod(0o755)
    print(f"Bundled verified libgomp runtime at {OUTPUT.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
