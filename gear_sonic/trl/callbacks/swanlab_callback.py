from __future__ import annotations

import math
import os

from transformers import TrainerCallback


class SwanLabCallback(TrainerCallback):
    """Rank-0 SwanLab logger for PPO training metrics."""

    @staticmethod
    def _is_disabled() -> bool:
        return os.environ.get("SWANLAB_MODE", "").lower() in {"disabled", "off", "none"}

    @staticmethod
    def _as_scalar(value):
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int | float):
            scalar = float(value)
        elif hasattr(value, "detach") and hasattr(value, "numel") and value.numel() == 1:
            scalar = float(value.detach().cpu().item())
        elif hasattr(value, "item"):
            try:
                scalar = float(value.item())
            except (TypeError, ValueError):
                return None
        else:
            return None
        if not math.isfinite(scalar):
            return None
        return scalar

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ARG002
        if logs is None or not state.is_world_process_zero or self._is_disabled():
            return
        try:
            import swanlab
        except ImportError:
            return

        clean_logs = {}
        for key, value in logs.items():
            scalar = self._as_scalar(value)
            if scalar is not None:
                clean_logs[key] = scalar
        clean_logs["global_step"] = state.global_step
        try:
            swanlab.log(clean_logs, step=state.global_step)
        except RuntimeError as exc:
            if "swanlab.init" in str(exc) or "must call swanlab.init" in str(exc):
                return
            raise

    def on_train_end(self, args, state, control, **kwargs):  # noqa: ARG002
        if not state.is_world_process_zero or self._is_disabled():
            return
        try:
            import swanlab
        except ImportError:
            return
        finish = getattr(swanlab, "finish", None)
        if finish is not None:
            try:
                finish()
            except RuntimeError as exc:
                if "swanlab.init" in str(exc) or "must call swanlab.init" in str(exc):
                    return
                raise
