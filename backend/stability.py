"""Lightweight laptop-side IMU stability indicators and personal-baseline logic.

These signals are experimental engineering indicators, not clinical measures.
They do not diagnose falls or predict future fall risk.
"""

from collections import deque
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional


class StabilityEngine:
    """Compute inexpensive IMU-derived motion and orientation indicators."""

    DEFAULT_THRESHOLDS = {
        "movement_accel": 0.08,
        "movement_gyro": 8.0,
        "unusual_jerk": 4.0,
        "unusual_gyro": 180.0,
        "unusual_orientation_change": 25.0,
    }

    def __init__(
        self,
        sample_rate_hz: float = 10.0,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        self.dt = 1.0 / sample_rate_hz
        self.thresholds = dict(self.DEFAULT_THRESHOLDS)
        if thresholds:
            self.thresholds.update(thresholds)

        self.accel_history = deque(maxlen=10)
        self.gyro_history = deque(maxlen=10)
        self.previous_accel_magnitude: Optional[float] = None
        self.previous_roll: Optional[float] = None
        self.previous_pitch: Optional[float] = None
        self.roll = 0.0
        self.pitch = 0.0

    def update(
        self,
        ax: float,
        ay: float,
        az: float,
        gx: float,
        gy: float,
        gz: float,
    ) -> Dict[str, Any]:
        """Return lightweight IMU-derived motion stability indicators."""
        accel_magnitude = math.sqrt(ax * ax + ay * ay + az * az)
        gyro_magnitude = math.sqrt(gx * gx + gy * gy + gz * gz)

        self.accel_history.append(accel_magnitude)
        self.gyro_history.append(gyro_magnitude)
        acceleration_variability = self._standard_deviation(self.accel_history)
        gyro_variability = self._standard_deviation(self.gyro_history)

        if self.previous_accel_magnitude is None:
            jerk = 0.0
        else:
            jerk = abs(accel_magnitude - self.previous_accel_magnitude) / self.dt
        self.previous_accel_magnitude = accel_magnitude

        accel_roll = math.degrees(math.atan2(ay, az))
        accel_pitch = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az)))
        previous_roll = self.roll
        previous_pitch = self.pitch
        self.roll = 0.98 * (self.roll + gx * self.dt) + 0.02 * accel_roll
        self.pitch = 0.98 * (self.pitch + gy * self.dt) + 0.02 * accel_pitch
        orientation_change = math.hypot(
            self.roll - previous_roll,
            self.pitch - previous_pitch,
        )

        movement_status = self._movement_status(
            acceleration_variability,
            gyro_magnitude,
        )
        unusual_movement = (
            jerk >= self.thresholds["unusual_jerk"]
            or gyro_magnitude >= self.thresholds["unusual_gyro"]
            or orientation_change >= self.thresholds["unusual_orientation_change"]
        )

        return {
            "movement_status": movement_status,
            "unusual_movement": unusual_movement,
            "accel_magnitude": accel_magnitude,
            "gyro_magnitude": gyro_magnitude,
            "acceleration_variability": acceleration_variability,
            "gyro_variability": gyro_variability,
            "jerk": jerk,
            "orientation_change": orientation_change,
            "roll": self.roll,
            "pitch": self.pitch,
            "thresholds_experimental_unvalidated": True,
        }

    def _movement_status(self, acceleration_variability: float, gyro_magnitude: float) -> str:
        if (
            acceleration_variability >= self.thresholds["movement_accel"]
            or gyro_magnitude >= self.thresholds["movement_gyro"]
        ):
            return "MOVING"
        return "STILL"

    @staticmethod
    def _standard_deviation(values: deque) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


class PersonalBaseline:
    """Learn a user's normal IMU-derived movement pattern for local demo feedback only."""

    FEATURE_NAMES = (
        "accel_magnitude",
        "gyro_magnitude",
        "acceleration_variability",
        "gyro_variability",
        "jerk",
        "orientation_change",
    )

    def __init__(
        self,
        sample_count: int = 60,
        deviation_threshold: float = 2.5,
        baseline_path: Optional[Path] = None,
    ) -> None:
        self.sample_count = sample_count
        self.deviation_threshold = deviation_threshold
        self.baseline_path = baseline_path or Path(__file__).resolve().parent.parent / "data" / "personal_baseline.json"
        self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
        self.samples = []
        self.mean = None
        self.std = None
        self._load()

    @property
    def ready(self) -> bool:
        return self.mean is not None

    def start(self) -> Dict[str, Any]:
        self.samples = []
        self.mean = None
        self.std = None
        if self.baseline_path.exists():
            self.baseline_path.unlink()
        return self.status()

    def update(self, stability: Dict[str, Any]) -> Dict[str, Any]:
        values = [float(stability[name]) for name in self.FEATURE_NAMES]

        if not self.ready:
            self.samples.append(values)
            if len(self.samples) >= self.sample_count:
                self._fit()
                self._save()
            else:
                return self.status()

        deviations = [
            abs(value - baseline_mean) / baseline_std
            for value, baseline_mean, baseline_std in zip(values, self.mean, self.std)
        ]
        deviation = sum(deviations) / len(deviations)
        unusual = deviation > self.deviation_threshold
        return {
            "baseline_ready": True,
            "baseline_status": "READY",
            "baseline_deviation": round(float(deviation), 3),
            "movement_status": "Movement within baseline" if not unusual else "Deviation from personal baseline",
            "unusual_movement": unusual,
            "samples_collected": len(self.samples) if self.samples else self.sample_count,
            "threshold_experimental_unvalidated": True,
        }

    def status(self) -> Dict[str, Any]:
        if self.ready:
            return {
                "baseline_ready": True,
                "baseline_status": "READY",
                "baseline_deviation": 0.0,
                "movement_status": "Movement within baseline",
                "unusual_movement": False,
                "samples_collected": self.sample_count,
                "threshold_experimental_unvalidated": True,
            }
        if self.samples:
            return {
                "baseline_ready": False,
                "baseline_status": "COLLECTING",
                "baseline_deviation": None,
                "movement_status": "Collecting personal baseline",
                "unusual_movement": False,
                "samples_collected": len(self.samples),
                "samples_required": self.sample_count,
                "threshold_experimental_unvalidated": True,
            }
        return {
            "baseline_ready": False,
            "baseline_status": "NOT READY",
            "baseline_deviation": None,
            "movement_status": "Baseline not started",
            "unusual_movement": False,
            "samples_collected": 0,
            "samples_required": self.sample_count,
            "threshold_experimental_unvalidated": True,
        }

    def _fit(self) -> None:
        columns = list(zip(*self.samples))
        self.mean = [sum(column) / len(column) for column in columns]
        self.std = [max(self._standard_deviation(column), 1e-6) for column in columns]
        self.samples = []

    def _save(self) -> None:
        if not self.mean or not self.std:
            return
        payload = {
            "mean": self.mean,
            "std": self.std,
            "sample_count": self.sample_count,
            "deviation_threshold": self.deviation_threshold,
        }
        with self.baseline_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _load(self) -> None:
        if not self.baseline_path.exists():
            return
        try:
            with self.baseline_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.mean = payload.get("mean")
            self.std = payload.get("std")
            if payload.get("sample_count"):
                self.sample_count = int(payload["sample_count"])
            if payload.get("deviation_threshold"):
                self.deviation_threshold = float(payload["deviation_threshold"])
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            self.mean = None
            self.std = None

    @staticmethod
    def _standard_deviation(values) -> float:
        mean = sum(values) / len(values)
        return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
