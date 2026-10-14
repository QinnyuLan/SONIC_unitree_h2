# XRoboToolkit Native SDK

These are the native SDK artifacts already used by the deployment release:

- `lib/x86_64/libPXREARobotSDK.so`
- `lib/aarch64/libPXREARobotSDK.so`
- `include/aarch64/`: headers paired with the ARM64 library

The x86_64/common headers come from the pinned `xrobotoolkit_pybind` submodule. The SDK artifacts are preserved byte for byte; their sizes, SHA256 hashes and source release are recorded in [manifest.json](manifest.json).

`./install.sh teleop` combines the appropriate artifacts with the upstream binding in `build/third_party/xrobotoolkit_pybind/` and builds the Python extension there. These libraries provide the SDK client; the workstation's [XRoboToolkit PC Service](https://github.com/XR-Robotics/XRoboToolkit-PC-Service) is installed separately.
