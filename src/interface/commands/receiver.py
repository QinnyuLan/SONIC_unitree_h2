"""Bounded command reception and producer ownership for Python services."""
from __future__ import annotations

import time
import zmq
from .protocol import SCHEMA, envelope


class CommandInbox:
    def __init__(self, kinds):
        self.kinds = set(kinds)
        self.command = None
        self.expires = 0
        self.sequences = {}
        self.rejected = 0
        self.last_error = ""

    def active(self, now=None):
        now = time.monotonic() if now is None else now
        return self.command if now < self.expires else None

    def accept(self, message, *, now=None, wall_ms=None):
        now = time.monotonic() if now is None else now
        wall_ms = time.time_ns() // 1_000_000 if wall_ms is None else wall_ms
        try:
            if message["schema"] != SCHEMA or message["kind"] not in self.kinds:
                raise ValueError("command on wrong channel or unsupported schema")
            for field in ("source", "session"):
                if not isinstance(message[field], str):
                    raise ValueError(f"{field} must be a string")
            if not isinstance(message.get("takeover", False), bool):
                raise ValueError("takeover must be boolean")
            sent = message["sent_at_ms"]
            if type(sent) is not int or not 0 <= sent <= 2**63 - 1:
                raise ValueError("invalid timestamp")
            command = envelope(message["kind"], message["data"], source=message["source"],
                session=message["session"], seq=message["seq"], ttl_ms=message["ttl_ms"],
                takeover=message.get("takeover", False))
            if command["seq"] > 2**64 - 1 or "head" in message or "hands" in message:
                raise ValueError("invalid command envelope")
            age = wall_ms - sent
            if age < -250 or age >= command["ttl_ms"]:
                raise ValueError("expired command or clock skew")
            key = (command["source"], command["session"])
            self.sequences = {k: v for k, v in self.sequences.items() if now - v[1] <= 60}
            if key in self.sequences and command["seq"] <= self.sequences[key][0]:
                raise ValueError("out-of-order command")
            active = self.active(now)
            if active and (active["source"], active["session"]) != key and not command["takeover"]:
                raise ValueError("another producer owns the input")
            if key not in self.sequences and len(self.sequences) >= 128:
                raise ValueError("too many producer sessions")
            self.sequences[key] = command["seq"], now
            command["sent_at_ms"] = sent
            self.command = None if command["kind"] == "stand" else command
            self.expires = now + (command["ttl_ms"] - max(0, age)) / 1000
            return True
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            self.rejected += 1
            self.last_error = str(error)
            return False


class Receiver:
    def __init__(self, context, endpoint, kinds):
        self.socket = context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVHWM, 8)
        self.socket.setsockopt(zmq.MAXMSGSIZE, 131072)
        self.socket.bind(endpoint)
        self.inbox = CommandInbox(kinds)

    def poll(self):
        for _ in range(16):
            try:
                message = self.socket.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            except (ValueError, UnicodeError) as error:
                self.inbox.rejected += 1
                self.inbox.last_error = str(error)
                continue
            self.inbox.accept(message)

    def close(self):
        self.socket.close()
