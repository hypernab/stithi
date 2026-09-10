# STITHI

STITHI is a wearable AI-assisted fall-prevention research prototype for SIH1580. The M5StickC Plus2 is a lightweight 6-axis IMU sensor and Wi-Fi telemetry node. The laptop runs the processing and dashboard.

## Current architecture

`M5 IMU -> Wi-Fi -> FastAPI -> Model 1 + personal baseline + stability indicators -> local dashboard`

- **Model 1** answers: what activity is being performed? It is the frozen Random Forest classifier using 26 IMU features, 20-sample windows, and six activity classes.
- **Personal baseline** learns approximately 30 seconds of normal live movement, then reports whether current IMU-derived features are within that person's baseline. It is an engineering comparison, not a medical score.
- **Stability engine** computes lightweight acceleration, gyro, jerk, variability, and orientation indicators on the laptop using experimental, unvalidated thresholds.
- **Model 2** is a separate research/evidence layer for gait characteristics associated with historical fall status in older adults. Raw M5 telemetry is not converted into Model 2 inputs, and Model 2 is not used for live future-fall prediction.

The API is local and intentionally small. Use `POST /imu`, `GET /latest`, `GET /history`, `POST /baseline/start`, and the existing recording endpoints. The prevention-first dashboard is served at `/dashboard`; the previous technical page remains available at `/dashboard-legacy` for troubleshooting.

The dashboard presents a suggested educational prevention routine covering mobility, strength, balance, and everyday movement. Exercise links use a centralized `EXERCISE_VIDEOS` configuration in `backend/dashboard.html` and fall back to official-source YouTube search pages when a verified embed ID is not configured. STITHI does not prescribe exercises or connect a live activity prediction to a medical recommendation.

## Calibration and demo flow

1. Start the FastAPI server with `python backend/server.py`.
2. Open `http://localhost:8000/dashboard`.
3. With the M5 connected and the user moving normally, select **Start Baseline** and continue normal movement for about 30 seconds.
4. Demonstrate Model 1 activity recognition, live IMU values, orientation, stability indicators, and baseline comparison.
5. Use the recording controls for local CSV data collection when needed.

For a complete local launch from the repository root:

```bash
python backend/server.py
```

Then open `http://localhost:8000/dashboard`. For local M5 testing, set `STITHI_SERVER_BASE_URL` in `src/main.cpp` to the laptop's LAN URL and allow port `8000` through the firewall.

## Running STITHI locally

Install the Python dependencies and start the service from the repository root:

```bash
pip install -r ml/requirements.txt
python backend/server.py
```

Open `http://127.0.0.1:8000/` for the pairing page or `http://127.0.0.1:8000/dashboard` for the unpaired developer dashboard. Local telemetry without a pairing code remains supported.

## Deploying STITHI

The repository includes `render.yaml` for a single-process Render web service:

- Build: `pip install -r ml/requirements.txt`
- Start: `uvicorn backend.server:app --host 0.0.0.0 --port $PORT`
- Required environment variable: `STITHI_PUBLIC_URL=https://<your-render-domain>`
- Optional environment variable: `STITHI_PAIRING_TTL_SECONDS=1800`

Create a Render web service from this repository and set `STITHI_PUBLIC_URL` to the exact HTTPS service URL. Pairing sessions are in memory, so a restart or redeploy invalidates active codes. No public deployment is claimed until the Render service and domain are actually created.

## Pairing a STITHI device

1. Power on the M5StickC Plus2 and connect it to Wi-Fi.
2. The device registers at `/register` and displays its four-digit pair code.
3. Open the public STITHI URL on a phone or laptop.
4. Enter the four-digit code and select **CONNECT**.
5. View the paired live dashboard at `/dashboard/<pair-code>`.

The M5 sends the existing `ax`, `ay`, `az`, `gx`, `gy`, and `gz` fields at 10 Hz, plus the short-lived `pair_code`. The code is a demo/session pairing mechanism, not production-grade authentication. For HTTPS firmware, set `STITHI_SERVER_BASE_URL` once in `src/main.cpp` and provide the deployment CA certificate in `STITHI_ROOT_CA`; certificate verification is not disabled by default.

## Truthful scope

The current prototype does not calculate actual Line of Gravity/Base of Support, clinical diagnosis, clinical fall-risk percentages, future-fall prediction, or a live clinical exercise prescription. The dashboard links to educational prevention material but does not claim to deliver the official Otago Exercise Programme. Future work requires multi-segment sensing, validated LOG/BOS estimation, prospective clinical validation, verified exercise-content integration, and a properly designed personalized exercise/intervention pathway.
