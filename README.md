# STITHI

STITHI is a wearable AI-assisted fall-prevention research prototype for SIH1580. The M5StickC Plus2 is a lightweight 6-axis IMU sensor and Wi-Fi telemetry node. The laptop runs the processing and dashboard.

## Current architecture

`M5 IMU -> Wi-Fi -> FastAPI -> Model 1 + personal baseline + stability indicators -> local dashboard`

- **Model 1** answers: what activity is being performed? It is the frozen Random Forest classifier using 26 IMU features, 20-sample windows, and six activity classes.
- **Personal baseline** learns approximately 30 seconds of normal live movement, then reports whether current IMU-derived features are within that person's baseline. It is an engineering comparison, not a medical score.
- **Stability engine** computes lightweight acceleration, gyro, jerk, variability, and orientation indicators on the laptop using experimental, unvalidated thresholds.
- **Model 2** is a separate research/evidence layer for gait characteristics associated with historical fall status in older adults. Raw M5 telemetry is not converted into Model 2 inputs, and Model 2 is not used for live future-fall prediction.

The API is local and intentionally small. Use `POST /imu`, `GET /latest`, `GET /history`, `POST /baseline/start`, and the existing recording endpoints. The dashboard is served at `/dashboard`.

## Calibration and demo flow

1. Start the FastAPI server with `python backend/server.py`.
2. Open `http://localhost:8000/dashboard`.
3. With the M5 connected and the user moving normally, select **Start Baseline** and continue normal movement for about 30 seconds.
4. Demonstrate Model 1 activity recognition, live IMU values, orientation, stability indicators, and baseline comparison.
5. Use the recording controls for local CSV data collection when needed.

## Truthful scope

The current prototype does not calculate actual Line of Gravity/Base of Support, clinical diagnosis, clinical fall-risk percentages, future-fall prediction, or the official Otago Exercise Program. Future work requires multi-segment sensing, validated LOG/BOS estimation, prospective clinical validation, and a properly designed personalized exercise/intervention pathway.
