"""Shared, deterministic IMU feature extraction for training and live inference."""

import numpy as np


CHANNEL_NAMES = ("ax", "ay", "az", "gx", "gy", "gz")
REFERENCE_FEATURE_NAMES = ("ax_mean", "ay_mean", "az_mean", "ax_std", "ay_std", "az_std", "ax_min", "ax_max", "ay_min", "ay_max", "az_min", "az_max", "acc_mag_mean", "acc_mag_std", "acc_mag_max", "gx_mean", "gy_mean", "gz_mean", "gx_std", "gy_std", "gz_std", "gyro_mag_mean", "gyro_mag_std", "gyro_mag_max", "jerk_mean", "jerk_std")
FEATURE_NAMES = []
for prefix in ("raw", "delta"):
    for channel in CHANNEL_NAMES:
        FEATURE_NAMES.extend([
            f"{prefix}_{channel}_mean", f"{prefix}_{channel}_std", f"{prefix}_{channel}_min", f"{prefix}_{channel}_max",
            f"{prefix}_{channel}_median", f"{prefix}_{channel}_iqr", f"{prefix}_{channel}_rms", f"{prefix}_{channel}_energy",
        ])
for magnitude in ("acc_mag", "gyro_mag"):
    FEATURE_NAMES.extend([f"{magnitude}_{stat}" for stat in ("mean", "std", "min", "max", "median", "iqr", "rms", "energy")])
for left, right in (("ax", "ay"), ("ax", "az"), ("ay", "az"), ("gx", "gy"), ("gx", "gz"), ("gy", "gz")):
    FEATURE_NAMES.append(f"corr_{left}_{right}")
for channel in CHANNEL_NAMES:
    FEATURE_NAMES.extend([f"{channel}_spectral_peak_ratio", f"{channel}_dominant_bin"])


def _summary_features(values):
    q25, median, q75 = np.percentile(values, [25, 50, 75])
    return [values.mean(), values.std(), values.min(), values.max(), median, q75 - q25,
            np.sqrt(np.mean(values ** 2)), np.mean(values ** 2)]


def extract_features(window):
    """Return 130 features from an ``(n_samples >= 2, 6)`` IMU window."""
    window = np.asarray(window, dtype=np.float32)
    if window.ndim != 2 or window.shape[1] != 6 or len(window) < 2:
        raise ValueError("window must have shape (n_samples >= 2, 6)")
    features = []
    for signal in (window, np.diff(window, axis=0)):
        for axis in range(6):
            features.extend(_summary_features(signal[:, axis]))
    for magnitude in (np.linalg.norm(window[:, :3], axis=1), np.linalg.norm(window[:, 3:], axis=1)):
        features.extend(_summary_features(magnitude))
    for left, right in ((0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)):
        x, y = window[:, left], window[:, right]
        features.append(0.0 if x.std() == 0 or y.std() == 0 else np.corrcoef(x, y)[0, 1])
    for axis in range(6):
        power = np.abs(np.fft.rfft(window[:, axis] - window[:, axis].mean())) ** 2
        non_dc = power[1:]
        features.extend([non_dc.max() / (non_dc.sum() + 1e-8), np.argmax(non_dc) + 1])
    return np.nan_to_num(np.asarray(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def extract_reference_features(window):
    """The original 26-feature extractor, retained for a fair baseline comparison."""
    window = np.asarray(window, dtype=np.float32)
    ax, ay, az, gx, gy, gz = window.T
    acceleration = np.linalg.norm(window[:, :3], axis=1)
    gyroscope = np.linalg.norm(window[:, 3:], axis=1)
    jerk = np.abs(np.diff(window[:, :3], axis=0)).reshape(-1)
    return np.asarray([ax.mean(), ay.mean(), az.mean(), ax.std(), ay.std(), az.std(),
                       ax.min(), ax.max(), ay.min(), ay.max(), az.min(), az.max(),
                       acceleration.mean(), acceleration.std(), acceleration.max(),
                       gx.mean(), gy.mean(), gz.mean(), gx.std(), gy.std(), gz.std(),
                       gyroscope.mean(), gyroscope.std(), gyroscope.max(), jerk.mean(), jerk.std()], dtype=np.float32)
