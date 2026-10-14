#!/usr/bin/env python3
"""Safely sync an active SwanLab run without marking it crashed.

The regular ``swanlab sync`` command updates the experiment state at the end.
For an in-progress run, ``backup.swanlab`` has no footer yet, so the regular
sync path marks the cloud experiment as CRASHED.  This helper uploads the local
backup to the existing run id while skipping that state update.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="SwanLab local run directory.")
    parser.add_argument("--run-id", required=True, help="Existing SwanLab cloud run id.")
    parser.add_argument("--project", default="sonic-h2lora", help="SwanLab project name.")
    parser.add_argument("--workspace", default=None, help="Optional SwanLab workspace.")
    parser.add_argument("--api-key", default=None, help="Optional SwanLab API key.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    backup_file = run_dir / "backup.swanlab"
    if not backup_file.is_file():
        raise FileNotFoundError(backup_file)

    import swanlab.data.porter as porter_mod
    from swanlab.package import get_key
    from swanlab.sync import sync

    def _skip_update_experiment_state(*_args, **kwargs) -> None:
        print(
            "[swanlab-safe-active-sync] skip update_experiment_state for active run: "
            f"{kwargs}",
            flush=True,
        )

    porter_mod.update_experiment_state = _skip_update_experiment_state
    api_key = args.api_key or get_key()

    print(f"[swanlab-safe-active-sync] run_dir={run_dir}", flush=True)
    print(f"[swanlab-safe-active-sync] run_id={args.run_id}", flush=True)
    sync(
        str(run_dir),
        workspace=args.workspace,
        project=args.project,
        id=args.run_id,
        api_key=api_key,
        raise_error=True,
    )
    print("[swanlab-safe-active-sync] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
