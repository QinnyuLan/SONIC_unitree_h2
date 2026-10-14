"""Bound manager telemetry independently of the tracking/control loop cadence."""

class ManagerStateRateLimiter:
    def __init__(self, hz=50):
        if hz <= 0:
            raise ValueError('hz must be positive')
        self.period = 1.0 / hz
        self.last_sent = None

    def due(self, now, force=False):
        # Button edges bypass the heartbeat limit; legacy boolean-edge consumers
        # must see every start/save/discard even within one heartbeat interval.
        if force or self.last_sent is None or now-self.last_sent >= self.period:
            self.last_sent = now
            return True
        return False
