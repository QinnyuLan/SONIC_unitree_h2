"""A small producer client; no robot SDK, planner, or model dependency."""
from __future__ import annotations

import uuid
import zmq
from .protocol import envelope


class RuntimeClient:
    def __init__(self, source: str, endpoint: str = "tcp://127.0.0.1:5558", *,
                 ttl_ms: int = 250, takeover: bool = False):
        self.source, self.ttl_ms = source, ttl_ms
        self.session, self.seq = uuid.uuid4().hex, 0
        self.takeover = takeover
        self.socket = zmq.Context.instance().socket(zmq.PUSH)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.IMMEDIATE, 1)
        self.socket.setsockopt(zmq.SNDHWM, 2)
        self.socket.setsockopt(zmq.SNDTIMEO, 100)
        self.socket.connect(endpoint)

    def send(self, kind: str, data: dict | None = None) -> bool:
        message = envelope(kind, data or {}, source=self.source, session=self.session,
                           seq=self.seq, ttl_ms=self.ttl_ms, takeover=self.takeover)
        self.seq += 1
        try:
            self.socket.send_json(message, allow_nan=False)
        except zmq.Again:
            return False
        self.takeover = False
        return True

    def stand(self):
        return self.send("stand")

    def close(self):
        self.socket.close(linger=0)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stand()
        self.close()
