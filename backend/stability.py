"""Lightweight laptop-side IMU stability indicators.

These indicators are experimental engineering signals, not clinical measures.
They do not estimate fall risk, fall probability, or future falls.
"""

from collections import deque
import math
from typing import Any, Dict, Optional


class StabilityEngine:
    """Compute inexpensive IMU-derived motion and orientation indicators."""

    # Experimental, unvalidated demo thresholds. Tune only with measured data.
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
        """Process one IMU sample and return serializable indicator values.

        Accelerometer and gyroscope units follow the existing M5 telemetry.
        The complementary filter assumes gyro values are degrees per second.
        """
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
    """Learn one user's normal IMU-derived pattern for demo feedback only."""

    FEATURE_NAMES = (
        "accel_magnitude",
        "gyro_magnitude",
        "acceleration_variability",
        "gyro_variability",
        "jerk",
        "orientation_change",
    )

    def __init__(self, sample_count: int = 300, deviation_threshold: float = 3.0) -> None:
        self.sample_count = sample_count
        self.deviation_threshold = deviation_threshold
        self.samples = []
        self.mean = None
        self.std = None

    @property
    def ready(self) -> bool:
        return self.mean is not None

    def start(self) -> Dict[str, Any]:
        self.samples = []
        self.mean = None
        self.std = None
        return self.status()

    def update(self, stability: Dict[str, Any]) -> Dict[str, Any]:
        values = [float(stability[name]) for name in self.FEATURE_NAMES]

        if not self.ready:
            self.samples.append(values)
            if len(self.samples) >= self.sample_count:
                self._fit()
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
            "baseline_deviation": deviation,
            "movement_status": "Unusual movement" if unusual else "Within baseline",
            "unusual_movement": unusual,
            "samples_collected": self.sample_count,
            "threshold_experimental_unvalidated": True,
        }

    def status(self) -> Dict[str, Any]:
        if self.ready:
            return {
                "baseline_ready": True,
                "baseline_deviation": 0.0,
                "movement_status": "Within baseline",
                "unusual_movement": False,
                "samples_collected": self.sample_count,
                "threshold_experimental_unvalidated": True,
            }
        if self.samples:
            return {
                "baseline_ready": False,
                "baseline_deviation": None,
                "movement_status": "Collecting baseline",
                "unusual_movement": False,
                "samples_collected": len(self.samples),
                "samples_required": self.sample_count,
                "threshold_experimental_unvalidated": True,
            }
        return {
            "baseline_ready": False,
            "baseline_deviation": None,
            "movement_status": "Not calibrated",
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

    @staticmethod
    def _standard_deviation(values) -> float:
        mean = sum(values) / len(values)
        return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
