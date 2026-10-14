#!/usr/bin/env python3
"""Trim a line-delimited XRoboToolkit full-body recording by timestamp."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def trim_recording(source: Path, output: Path, duration_seconds: float) -> tuple[int, float]:
    if duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be positive")

    first_timestamp_ns: int | None = None
    last_timestamp_ns: int | None = None
    frame_count = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open(encoding="utf-8") as input_stream, output.open(
        "w", encoding="utf-8"
    ) as output_stream:
        for line_number, line in enumerate(input_stream, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc

            is_body_frame = isinstance(entry, dict) and "Body" in entry
            if not is_body_frame:
                if first_timestamp_ns is None:
                    output_stream.write(line if line.endswith("\n") else line + "\n")
                continue

            timestamp_value = entry.get("timeStampNs", entry.get("timestampNs"))
            if timestamp_value is None:
                raise ValueError(f"body frame on line {line_number} has no timestamp")
            timestamp_ns = int(timestamp_value)
            if first_timestamp_ns is None:
                first_timestamp_ns = timestamp_ns
            elapsed_seconds = (timestamp_ns - first_timestamp_ns) / 1e9
            if elapsed_seconds >= duration_seconds:
                break

            output_stream.write(line if line.endswith("\n") else line + "\n")
            last_timestamp_ns = timestamp_ns
            frame_count += 1

    if first_timestamp_ns is None or last_timestamp_ns is None:
        raise ValueError(f"{source} contains no full-body frames")
    return frame_count, (last_timestamp_ns - first_timestamp_ns) / 1e9


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    args = parser.parse_args()
    frame_count, duration = trim_recording(
        args.source, args.output, args.duration_seconds
    )
    print(
        f"Wrote {args.output}: {frame_count} body frames, "
        f"timestamps 0.000000..{duration:.6f} s"
    )


if __name__ == "__main__":
    main()
