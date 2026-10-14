"""Bounded, non-blocking CSV diagnostics, separate from control decisions."""

import csv
import json
from pathlib import Path
import queue
import threading
import time


class DiagnosticLog:
    def __init__(self, directory, headers, metadata, capacity=4096):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.headers = headers
        self.metadata = metadata
        self.queue = queue.Queue(maxsize=capacity)
        self.closed = False
        self.error = None
        self.dropped = 0
        self.counts = dict.fromkeys(headers, 0)
        self.files = {}
        self.writers = {}
        try:
            for name, header in headers.items():
                f = (self.directory / (name + ".csv")).open("x", newline="")
                self.files[name] = f
                self.writers[name] = csv.writer(f)
                self.writers[name].writerow(header)
            self._metadata("recording")
        except BaseException:
            for f in self.files.values():
                f.close()
            raise
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def _metadata(self, status):
        data = dict(self.metadata, status=status, rows=self.counts,
                    dropped_rows=self.dropped, writer_error=self.error)
        (self.directory / "metadata.json").write_text(json.dumps(data, indent=2) + "\n")

    def put(self, name, row):
        if self.closed or self.error:
            self.dropped += 1
            return
        try:
            self.queue.put_nowait((name, tuple(row)))
        except queue.Full:
            self.dropped += 1

    def _write(self):
        last_flush = time.monotonic()
        try:
            while not self.closed or not self.queue.empty():
                try:
                    name, row = self.queue.get(timeout=.1)
                except queue.Empty:
                    pass
                else:
                    if len(row) != len(self.headers[name]):
                        raise ValueError(f"{name}: row/header length mismatch")
                    self.writers[name].writerow(row)
                    self.counts[name] += 1
                if time.monotonic() - last_flush >= 1:
                    for f in self.files.values():
                        f.flush()
                    self._metadata("recording")
                    last_flush = time.monotonic()
        except Exception as exc:
            self.error = str(exc)
        finally:
            for f in self.files.values():
                f.close()

    def close(self, *, incomplete=False):
        if self.closed:
            return
        self.closed = True
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            self.error = "writer did not flush within 10 seconds"
        self._metadata("incomplete" if incomplete or self.error or self.dropped else "completed")
