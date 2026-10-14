"""Check recorded inference, gains and real-time validity after the smoke run."""
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE / "smoke_configured_g1"
report = json.loads((RUN / "process_result.json").read_text())
assert report["deployment_exit_code"] == 0
sim = json.loads((RUN / "sim/metadata.json").read_text())
assert sim["realtime"]["valid"] is True
assert sim["dropped_rows"] == 0 and sim["writer_error"] is None
report["realtime"] = sim["realtime"]
report["signals"] = {}
for name, width, header in (("encoder_input", 1751, False), ("policy_input", 994, False),
                            ("action", 34, True), ("q", 36, True),
                            ("dq", 36, True), ("base_quat", 9, True)):
    with (RUN / "deploy" / (name + ".csv")).open() as stream:
        rows = csv.reader(stream)
        if header:
            next(rows)
        values = np.array([[float(v) for v in row if v != ""] for row in rows])
    assert values.shape[0] > 400 and values.shape[1] == width, (name, values.shape)
    assert np.isfinite(values).all(), name
    report["signals"][name] = dict(rows=len(values), columns=width, finite=True)
    if name == "encoder_input":
        assert np.all(values[:, 0] == 3)
        report["signals"][name]["selectors"] = [3]
with (RUN / "deploy/lowcmd_command.csv").open() as stream:
    commands = list(csv.DictReader(stream))
active = [r for r in commands if float(r["kp_5"]) > 0]
assert active
for r in active:
    for i, kp in ((5, 56), (11, 64), (3, 260), (9, 260)):
        assert abs(float(r[f"kp_{i}"]) - kp) < 1e-4
        assert float(r[f"kd_{i}"]) == (2 if i in (5,11) else 4)
    for i in (13, 14):
        assert float(r[f"kp_{i}"]) == 150 and float(r[f"kd_{i}"]) == 3
log = (RUN / "deploy.log").read_text()
assert " completed." in log
assert "Found 7 motion folders" in log
report.update(all_g1_demos_loaded=7, played_motions=[line for line in log.splitlines() if " completed." in line],
              configured_ankle_knee_gains_checked=True)
(HERE / "local_configured_sim_validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: report[k] for k in ("signals", "realtime", "played_motions")}, indent=2))
