#!/usr/bin/env python3
"""Prepare a local XRoboToolkit build without modifying its source submodule."""
import hashlib
import json
import platform
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "third_party/xrobotoolkit_pybind"
SDK = ROOT / "third_party/xrobotoolkit_sdk"
BUILD = ROOT / "build/third_party/xrobotoolkit_pybind"
SOURCE_FILES = (
    "CMakeLists.txt", "setup.py", "README.md", "LICENSE",
    "bindings/py_bindings.cpp", "include/PXREARobotSDK.h",
    "include/nlohmann/json.hpp", "include/nlohmann/json_fwd.hpp",
)


def prepare() -> Path:
    architecture = platform.machine()
    if platform.system() != "Linux" or architecture not in {"x86_64", "aarch64"}:
        raise RuntimeError("The bundled XRoboToolkit SDK supports Linux x86_64 and aarch64.")
    for name in SOURCE_FILES:
        if not (SOURCE / name).is_file():
            raise RuntimeError(
                "XRoboToolkit sources are missing. Run: git submodule update --init --recursive"
            )

    # Check every supplied native artifact before preparing a build.
    manifest = json.loads((SDK / "manifest.json").read_text())
    for entry in manifest["files"]:
        path = SDK / entry["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"XRoboToolkit SDK checksum mismatch: {path}")

    for name in SOURCE_FILES:
        target = BUILD / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / name, target)
    library = SDK / "lib" / architecture / "libPXREARobotSDK.so"
    target_dir = BUILD / "lib"
    if architecture == "aarch64":
        target_dir /= "aarch64"
        shutil.copytree(SDK / "include/aarch64", BUILD / "include/aarch64", dirs_exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(library, target_dir / library.name)
    return BUILD


if __name__ == "__main__":
    print(prepare())
