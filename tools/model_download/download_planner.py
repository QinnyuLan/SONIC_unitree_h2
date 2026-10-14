#!/usr/bin/env python3
"""Download and verify the shared SONIC G1 planner used by the independent planner service."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

REPO_ID = "nvidia/GEAR-SONIC"
PLANNER_FILE = ("planner_sonic.onnx", "motion_planner_g1/planner_sonic.onnx")
PLANNER_REVISION = "72935a37d3860204617ebaaf6c855a1c5aff7b8c"
PLANNER_SHA256 = "39b553e197f62f077975ba38512bc04781a3fc37c2af7c6756e04629f760edea"


def _sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    hf_hub_download,
    repo_id,
    hf_filename,
    local_dest,
    token=None,
    revision=None,
    expected_sha256=None,
):
    """Download hf_filename from the Hub and place it at local_dest."""
    local_dest = Path(local_dest)
    if expected_sha256 is not None and local_dest.is_file():
        digest = _sha256_file(local_dest)
        if digest == expected_sha256:
            print(f"  Already verified: {local_dest}")
            return

    print(f"  Downloading {hf_filename} ...", flush=True)
    cached = hf_hub_download(
        repo_id=repo_id,
        filename=hf_filename,
        token=token,
        revision=revision,
    )
    local_dest.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{local_dest.name}.",
        suffix=".part",
        dir=local_dest.parent,
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        shutil.copy2(cached, temporary_path)
        if expected_sha256 is not None:
            digest = _sha256_file(temporary_path)
            if digest != expected_sha256:
                raise RuntimeError(
                    f"SHA-256 mismatch for {hf_filename}: expected {expected_sha256}, got {digest}"
                )
            print(f"  SHA-256 verified: {digest}")
        os.replace(temporary_path, local_dest)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"  -> {local_dest}")



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[2] / "models",
                        help="model root; planner is stored under motion_planner_g1/")
    args = parser.parse_args()
    destination = args.output_dir / PLANNER_FILE[1]
    if destination.is_file() and _sha256_file(destination) == PLANNER_SHA256:
        print(f"Already verified G1 planner: {destination}")
        return
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        parser.exit(1, "Install download dependencies: python -m pip install -r requirements/download.txt\n")
    download_file(hf_hub_download, REPO_ID, PLANNER_FILE[0], destination,
                  revision=PLANNER_REVISION, expected_sha256=PLANNER_SHA256)


if __name__ == "__main__":
    main()
