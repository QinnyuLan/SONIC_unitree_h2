# Inspire Hand Service on H2

[中文](INSPIRE_HANDS_CN.md) · [Deployment](DEPLOYMENT.md) · [Tools](../tools/README.md)

The service in [`tools/robot_services/inspire_hands/`](../tools/robot_services/inspire_hands/) runs on H2 PC2, where the Inspire RH56 hands connect through RS-485. It receives DDS targets from the workstation's C++ runtime and publishes measured hand positions. This is the Inspire backend; Unitree Dex3 uses separate DDS topics and does not use this serial service.

```text
Python teleoperation / hand commands
  → C++ policy runtime (HAND_BACKEND=inspire)
  → DDS over the robot network
  → H2 PC2: inspire_hand_service
  → independent right/left RS-485 ports
```

Only the hand service starts automatically on PC2. The body policy and VR processes are started separately by the operator. The service has no dependency on policy weights, CUDA, TensorRT or a Python environment.

The wire contract is `rt/inspire/cmd` (`unitree_go::msg::dds_::MotorCmds_`) and `rt/inspire/state` (`MotorStates_`). Both messages contain 12 entries: right hand first, then left. Each side is ordered `[pinky, ring, middle, index, thumb_bend, thumb_rotate]`; positions are normalized to `[0, 1]`, with `0` closed and `1` open. State `lost>0` marks unavailable/stale measurements. Runtime converts the upstream seven-value hand representation to this six-value-per-hand contract.

## Build and transfer

Run on the development workstation, from the repository root. This creates a source package without models or Python environments:

```bash
mkdir -p outputs
tar -czf outputs/inspire-hands-source.tar.gz \
  tools/robot_services/inspire_hands \
  src/runtime/third_party/unitree_sdk2 \
  tests/tools/robot_services/inspire_hands \
  licenses/inspire_hands docs/INSPIRE_HANDS.md docs/INSPIRE_HANDS_CN.md
scp outputs/inspire-hands-source.tar.gz unitree@192.168.123.162:/home/unitree/
```

`192.168.123.162` is the PC2 address used in the original deployment; substitute the address and login for your robot. On PC2:

```bash
mkdir -p ~/SONIC_unitree_h2
tar -xzf ~/inspire-hands-source.tar.gz -C ~/SONIC_unitree_h2
cd ~/SONIC_unitree_h2
sudo apt-get update
sudo apt-get install build-essential cmake libboost-program-options-dev libeigen3-dev
cmake -S tools/robot_services/inspire_hands -B build/inspire_hands \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/inspire_hands -j2
```

Build on PC2 for its architecture and OS libraries; do not copy an x86_64 workstation binary to an ARM64 robot. The bundled SDK contains both architectures and requires no download. An existing SDK installation can instead be selected at configuration time with `-DUNITREE_SDK_ROOT=/unitree/opt`. Use a separate build directory when switching SDKs. If PC2 is offline, prepare the four build packages and their dependencies for its Ubuntu release/architecture beforehand, or use the dependencies already installed there.

The two binaries are `build/inspire_hands/bin/inspire_hand_service` and `inspire_hand_client`. The client only reads DDS state unless `--position` or `--cycle` is explicitly supplied.

## Serial mapping and initial check

Stop hand-command senders and the previous hand daemon before probing or replacing the installation. A probe queries positions without initializing DDS or sending position targets, but it still needs exclusive access to each serial port. Keep the hands stationary and clear during setup.

On PC2, inspect the existing service and port names:

```bash
systemctl cat h2-inspire-hands.service
sudo systemctl stop h2-inspire-hands.service
pgrep -af 'example_h2_native485|inspire_hand_service|inspire_g1|inspire_h1'
ls -l /dev/serial/by-id/
ip -br link
```

On a first installation the service may not exist yet. Stop any older manually launched daemon separately. The previous G1-style FTDI wiring uses **right `if01`, left `if02`, device ID 1 on each port**, at 115200 baud, 8N1. Verify the physical mapping on your robot. Prefer stable `/dev/serial/by-id/` links; `/dev/ttyUSB*` numbering can change. Native wiring can use `/dev/ttyUN0` and `/dev/ttyUN1` after verification.

Install the binaries and create the configuration only if it does not already exist:

```bash
sudo cmake --install build/inspire_hands --prefix /opt/SONIC_unitree_h2/inspire_hands
sudo install -d -m 0755 /etc/sonic
if [ ! -e /etc/sonic/inspire-hands.env ]; then
  sudo install -m 0644 tools/robot_services/inspire_hands/systemd/inspire-hands.env \
    /etc/sonic/inspire-hands.env
fi
sudoedit /etc/sonic/inspire-hands.env
```

Set `NETWORK_INTERFACE` to PC2's robot-facing interface (typically `eth0`), `DDS_DOMAIN` to the runtime's domain (normally `0`), and replace **both** `REPLACE_WITH_*` port placeholders. The workstation may use `eno1`; the interface names need not match across computers. Keep values as simple shell-compatible assignments. The installed binaries carry the selected SDK's DDS shared libraries under `lib/`, so startup does not need an interactive shell or `LD_LIBRARY_PATH`.

Probe as the service user, with its `dialout` group:

```bash
set -a
source /etc/sonic/inspire-hands.env
set +a
sudo -u unitree -g dialout /opt/SONIC_unitree_h2/inspire_hands/bin/inspire_hand_service \
  --probe-only --hands both --require-both \
  --right-port "$RIGHT_PORT" --left-port "$LEFT_PORT" \
  --right-id "$RIGHT_ID" --left-id "$LEFT_ID"
```

Both sides should report `ONLINE`. `--require-both` is for this acceptance probe only. The normal daemon deliberately keeps running with missing hands. Check port ownership and permissions if opening a device fails; the systemd unit grants `dialout` membership without requiring a root daemon.

## Install and enable the service

Preserve the old unit before replacing an existing installation:

```bash
if [ -f /etc/systemd/system/h2-inspire-hands.service ]; then
  sudo cp -an /etc/systemd/system/h2-inspire-hands.service \
    /etc/systemd/system/h2-inspire-hands.service.before-sonic
fi
sudo install -m 0644 tools/robot_services/inspire_hands/systemd/h2-inspire-hands.service \
  /etc/systemd/system/h2-inspire-hands.service
sudo systemd-analyze verify /etc/systemd/system/h2-inspire-hands.service
sudo systemctl daemon-reload
sudo systemctl enable --now h2-inspire-hands.service
systemctl is-enabled h2-inspire-hands.service
systemctl status h2-inspire-hands.service --no-pager
journalctl -u h2-inspire-hands.service -b -n 50 -o cat
```

The supplied unit runs as `unitree:unitree`. Adjust `User` and `Group` in the installed unit if the robot uses another account. Readiness means DDS and worker threads have started; it does not mean both hands are online. Check measured DDS state as well:

```bash
/opt/SONIC_unitree_h2/inspire_hands/bin/inspire_hand_client \
  --network "$NETWORK_INTERFACE" --domain "$DDS_DOMAIN" --duration-ms 5000
```

Both sides should have `lost=0` and finite positions. To test an open target, first stop other publishers on `rt/inspire/cmd`, then run the same client with `--position 1.0`. A target of `0` means fully closed, `1` fully open. The client writes both hands when an explicit position is requested.

On the workstation, set `HAND_BACKEND=inspire` in [`config/h2_deploy.env`](../config/h2_deploy.env), use the same DDS domain, and follow [Sim2Real](DEPLOYMENT.md#sim2real). Stop the test publisher before starting runtime hand control. Keep exactly one command publisher and one serial daemon for these hands.

## Autostart, watchdog and recovery

| Condition | Behavior |
|---|---|
| PC2 boots | `multi-user.target` starts the enabled service after the network-online target |
| Process exits or crashes | systemd restarts it after 2 seconds |
| Main DDS loop stalls | Heartbeats cease; the 10-second systemd watchdog kills and restarts the process |
| A serial worker stalls | Its progress check suppresses heartbeats after its normal cycle budget plus a 2-second margin; systemd then times out |
| USB port is missing or unplugged | DDS stays alive; each side retries opening its port once per second, independently |
| RS-485 hand stops responding | After 3 failures, that side polls at 2 Hz; a successful read restores the normal 50 Hz loop |
| DDS commands stop | No new serial position targets are written; commands queued for over 500 ms are discarded |
| Operator runs `systemctl stop` | The service stays stopped despite `Restart=always` |

The service sends `READY=1` and `WATCHDOG=1` from its main loop and checks progress of both enabled serial workers. An unplugged hand is an expected device state, so it does not cause a process restart loop. The state publisher normally runs at 20 Hz, and measurements older than 250 ms are marked lost. These are a process watchdog and device recovery mechanisms; they do not reset a frozen PC2 OS or remove motor power. See the systemd references for [service restart/watchdog settings](https://www.freedesktop.org/software/systemd/man/systemd.service.html) and the [notification protocol](https://www.freedesktop.org/software/systemd/man/sd_notify.html).

Startup, recovery and shutdown do not generate an open/close pose. USB loss clears queued commands, and commands arriving while that side is unavailable are rejected. After reconnection the daemon accepts new commands; the runtime's Inspire adapter additionally requires Trigger release before closing again. A powered hand can retain its last target when communication stops.

Inspect supervision without moving the hands:

```bash
systemctl show h2-inspire-hands.service \
  -p ActiveState -p MainPID -p NRestarts -p WatchdogUSec -p WatchdogTimestamp
journalctl -u h2-inspire-hands.service -f
```

For a deliberate watchdog acceptance test, stop all hand-command senders and keep the hands clear. This pauses the service and interrupts hand communication. Within approximately one 10-second watchdog interval plus the 2-second restart delay, `MainPID` should change and `NRestarts` should increase:

```bash
sudo systemctl kill --kill-who=main --signal=SIGSTOP h2-inspire-hands.service
systemctl show h2-inspire-hands.service -p MainPID -p NRestarts
```

Inspect again after the watchdog interval and check the journal for `watchdog` followed by a new startup. After a planned reboot, confirm `is-enabled`, `is-active`, and `lost=0` on both hands. Running this repository's validation does not reboot or update a robot.

## Maintenance and rollback

After editing `/etc/sonic/inspire-hands.env`, restart the service. After editing the unit itself, run `daemon-reload` first. Use `sudo systemctl disable --now h2-inspire-hands.service` to stop it and remove autostart. For troubleshooting, inspect the journal, configured interface/domain, serial permissions and `by-id` links; `active` with `lost>0` indicates a device/serial problem rather than a daemon crash.

Before upgrading binaries, stop hand-command senders and the service, and copy `/opt/SONIC_unitree_h2/inspire_hands/` and `/etc/sonic/inspire-hands.env` to a versioned backup. Build and install the new version, then repeat the probe and read-only DDS checks. To restore the original pre-migration service, stop the new one, restore `h2-inspire-hands.service.before-sonic` as the unit, run `daemon-reload`, and start it. Keep the old `/home/unitree/dfx_inspire_service/` installation until the new deployment has been accepted; the old unit may reference it.

## Local validation

On a development Linux host with Python 3.10 or newer:

```bash
cmake -S tools/robot_services/inspire_hands -B build/inspire_hands -DBUILD_TESTING=ON
cmake --build build/inspire_hands -j2
ctest --test-dir build/inspire_hands --output-on-failure
python3 tools/validation/check_inspire_watchdog.py
```

CTest uses pseudo terminals and loopback DDS domain 218. It checks fragmented serial responses, missing devices, reconnect/backoff, DDS-to-serial commands, rejection of commands during disconnection, systemd notifications, and an injected serial-worker stall. The last command additionally requires a running systemd **user** manager: it creates a temporary service with nonexistent serial paths on loopback, verifies watchdog/crash recovery and explicit stop, and removes it afterward. It does not install a system service. Results go to `outputs/validation/inspire_watchdog.json`. These checks do not replace acceptance on the actual robot.
