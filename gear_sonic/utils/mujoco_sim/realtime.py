"""Wall-clock pacing for a simulator driven by an external realtime policy."""

import math
import time


class RealtimeOverrun(RuntimeError):
    pass


class RealtimePacer:
    def __init__(self, dt, max_lag_s=0.02, *, clock=time.monotonic, sleep=time.sleep):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("physics timestep must be finite and positive")
        if not math.isfinite(max_lag_s) or max_lag_s <= 0:
            raise ValueError("realtime maximum lag must be finite and positive")
        self.dt = dt
        self.max_lag_s = max_lag_s
        self.clock, self.sleep = clock, sleep
        self.start = self.now = clock()
        self.steps = 0
        self.max_lag = 0.0
        self.lag = 0.0
        self.failure = None

    def finish_step(self):
        self.steps += 1
        # Keep the original epoch: oversleep and short stalls must not accumulate.
        # Every catch-up iteration still performs one real physics step.
        deadline = self.start + self.steps * self.dt
        self.now = self.clock()
        while self.now < deadline:
            self.sleep(deadline - self.now)
            self.now = self.clock()
        self.lag = max(0.0, self.now - deadline)
        self.max_lag = max(self.max_lag, self.lag)
        if self.lag > self.max_lag_s:
            self.failure = (
                f"simulation fell behind wall clock by {self.lag * 1000:.1f} ms "
                f"(limit {self.max_lag_s * 1000:g} ms); stopping simulation. "
                "This run is not valid for realtime policy comparison."
            )
            raise RealtimeOverrun(self.failure)

    def snapshot(self):
        wall = self.now - self.start
        simulated = self.steps * self.dt
        return dict(time_monotonic_ms=self.now * 1000, wall_elapsed_s=wall,
                    physics_elapsed_s=simulated, physics_steps=self.steps,
                    realtime_factor=simulated / wall if wall > 0 else 1.0,
                    lag_ms=self.lag * 1000, max_lag_ms=self.max_lag * 1000)
