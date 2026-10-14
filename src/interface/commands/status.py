"""Read fresh policy status without connecting to DDS or controlling the runtime."""
import time

import zmq


class RuntimeStatus:
    def __init__(self, endpoint):
        self.socket = zmq.Context.instance().socket(zmq.SUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt(zmq.MAXMSGSIZE, 131072)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.socket.connect(endpoint)
        self.state = None

    def poll(self):
        try:
            state = self.socket.recv_json(flags=zmq.NOBLOCK)
            sent = state.get("sent_at_ms") if isinstance(state, dict) else None
            self.state = state if (type(sent) is int and state.get("schema") == "sonic.h2.state.v1") else None
        except zmq.Again:
            pass
        except (ValueError, UnicodeError):
            self.state = None
        # Clear keyboard intent before the planner's 500 ms state timeout, so
        # reconnecting cannot reuse a request from before a state interruption.
        if self.state and not -250 <= time.time_ns() // 1_000_000 - self.state["sent_at_ms"] < 250:
            self.state = None
        return self.state

    @property
    def ready(self):
        return bool(self.state and self.state.get("phase") == "running")

    @property
    def label(self):
        if not self.state:
            return "Runtime: waiting for fresh state"
        return f"Runtime: {self.state.get('phase')} | {self.state.get('body_kind')} | source: {self.state.get('source') or 'none'}"

    def close(self):
        self.socket.close()
