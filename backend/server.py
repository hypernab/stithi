from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from datetime import datetime
from collections import deque
from pathlib import Path
import sys
import math
import csv
import os
import time

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from ml.inference import ActivityRecognitionEngine
from backend.stability import PersonalBaseline, StabilityEngine
from backend.pairing import PairingManager

app = FastAPI()

# ============================================================
# CONFIG
# ============================================================
DATA_FOLDER = BASE_DIR / "data"
DATA_FOLDER.mkdir(parents=True, exist_ok=True)

HISTORY_LENGTH = 500
PAIRING_TTL_SECONDS = int(os.getenv("STITHI_PAIRING_TTL_SECONDS", "1800"))
PUBLIC_URL = os.getenv("STITHI_PUBLIC_URL", "http://127.0.0.1:8000")

# ============================================================
# LIVE DATA
# ============================================================
history = deque(maxlen=HISTORY_LENGTH)
activity_engine = ActivityRecognitionEngine()
stability_engine = StabilityEngine()
personal_baseline = PersonalBaseline()
latest_prediction = None
latest_stability = None
latest_baseline = personal_baseline.status()

otago_completed = False
otago_completed_at = None

latest_data = {
    "timestamp": 0, "ax": 0, "ay": 0, "az": 0,
    "gx": 0, "gy": 0, "gz": 0, "acc_mag": 0, "gyro_mag": 0,
    "stability": None, "received_at": ""
}

sample_count = 0
last_received_monotonic = None

# ============================================================
# RECORDING & PAIRING
# ============================================================
recording = False
current_activity = None
csv_file = None
csv_writer = None
pairing = PairingManager(ttl_seconds=PAIRING_TTL_SECONDS)
active_pair_code = None

HOME_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>STITHI | Pair device</title>
<style>
body { margin: 0; min-height: 100vh; display: grid; place-items: center; background: #eef5f0; color: #10231f; font-family: Georgia, "Times New Roman", serif; }
.card { width: min(92vw, 520px); padding: 42px 32px; background: #fffdf8; border: 1px solid #d9e1d8; border-radius: 16px; box-shadow: 0 18px 45px rgba(16,35,31,.12); text-align: center; }
.brand { margin: 0; font-size: 3.6rem; letter-spacing: .08em; }
.tagline { color: #61736e; font-size: 1.25rem; margin: 10px 0 28px; }
label { display: block; color: #61736e; font: 700 12px Arial, sans-serif; letter-spacing: .12em; text-transform: uppercase; margin-bottom: 10px; }
input { width: 100%; padding: 14px; border: 1px solid #b9cfc5; border-radius: 8px; box-sizing: border-box; font: 700 1.8rem Arial, sans-serif; letter-spacing: .35em; text-align: center; }
button { width: 100%; margin-top: 14px; padding: 14px; border: 0; border-radius: 8px; background: #0b7568; color: white; font: 700 1rem Arial, sans-serif; cursor: pointer; }
.error { min-height: 1.5em; margin-top: 16px; color: #a33e36; font: 600 14px Arial, sans-serif; }
.note { margin-top: 28px; color: #61736e; font: 13px/1.5 Arial, sans-serif; }
</style>
</head>
<body><main class="card"><h1 class="brand">STITHI</h1><p class="tagline">Sense. Understand. Support.</p><form id="pairForm"><label for="pairCode">Enter Pair Code</label><input id="pairCode" inputmode="numeric" autocomplete="one-time-code" maxlength="4" pattern="[0-9]{4}" required><button type="submit">CONNECT</button><div id="error" class="error"></div></form><p class="note">STITHI is a research/prototype system and is not a clinically validated fall predictor.</p></main>
<script>document.getElementById("pairForm").addEventListener("submit", async event => { event.preventDefault(); const code = document.getElementById("pairCode").value.trim(); const error = document.getElementById("error"); if (!/^[0-9]{4}$/.test(code)) { error.textContent = "Enter the four-digit code shown on STITHI."; return; } const response = await fetch(`/pair/${code}`); if (response.ok) { window.location.href = `/dashboard/${code}`; } else { const body = await response.json().catch(() => ({})); error.textContent = body.detail || "Pair code not found."; } });</script>
</body></html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(HOME_HTML)

@app.post("/register")
def register_device(data: dict):
    device_id = str(data.get("device_id", "stithi-device"))[:64]
    session = pairing.register(device_id)
    global active_pair_code
    active_pair_code = session.pair_code
    return {
        "pair_code": session.pair_code,
        "dashboard_url": f"{PUBLIC_URL.rstrip('/')}/dashboard/{session.pair_code}",
        "telemetry_url": f"{PUBLIC_URL.rstrip('/')}/imu",
        "expires_in_seconds": PAIRING_TTL_SECONDS,
    }

@app.get("/pair/{pair_code}")
def check_pair(pair_code: str):
    if not pair_code.isdigit() or len(pair_code) != 4:
        raise HTTPException(status_code=404, detail="Pair code not found.")
    session = pairing.get(pair_code)
    if session is None:
        raise HTTPException(status_code=404, detail="Pair code not found or session has expired.")
    return {"status": "paired", "pair_code": session.pair_code}

@app.post("/compliance")
def receive_compliance(data: dict):
    global otago_completed, otago_completed_at, active_pair_code
    
    pair_code = str(data.get("pair_code", "")).strip()
    if pair_code:
        session = pairing.touch(pair_code)
        if session:
            active_pair_code = pair_code
            
    if data.get("event") == "otago_start":
        otago_completed = True
        otago_completed_at = datetime.now().strftime("%I:%M %p")
        return {"status": "compliance_logged", "time": otago_completed_at}
    
    return {"status": "ignored"}

@app.post("/imu")
def receive_imu(data: dict):
    global latest_data, sample_count, csv_writer, latest_prediction
    global latest_stability, latest_baseline, last_received_monotonic
    global active_pair_code

    pair_code = str(data.get("pair_code", "")).strip()
    if pair_code:
        if pairing.touch(pair_code) is None:
            return {"status": "invalid", "detail": "Pair code expired."}
        active_pair_code = pair_code

    try:
        ax, ay, az = float(data.get("ax", 0)), float(data.get("ay", 0)), float(data.get("az", 0))
        gx, gy, gz = float(data.get("gx", 0)), float(data.get("gy", 0)), float(data.get("gz", 0))
    except (TypeError, ValueError):
        return {"status": "invalid", "detail": "IMU values must be numeric"}

    if not all(math.isfinite(v) for v in (ax, ay, az, gx, gy, gz)):
        return {"status": "invalid", "detail": "IMU values must be finite"}

    acc_mag = math.sqrt(ax**2 + ay**2 + az**2)
    gyro_mag = math.sqrt(gx**2 + gy**2 + gz**2)

    latest_stability = stability_engine.update(ax, ay, az, gx, gy, gz)
    latest_baseline = personal_baseline.update(latest_stability)
    
    latest_data = {
        "timestamp": data.get("timestamp", 0),
        "ax": ax, "ay": ay, "az": az,
        "gx": gx, "gy": gy, "gz": gz,
        "acc_mag": acc_mag, "gyro_mag": gyro_mag,
        "stability": latest_stability,
        "received_at": datetime.now().strftime("%H:%M:%S")
    }

    history.append(latest_data.copy())
    sample_count += 1
    last_received_monotonic = time.monotonic()
    latest_prediction = activity_engine.push_sample(ax, ay, az, gx, gy, gz)

    if recording and csv_writer:
        csv_writer.writerow([data.get("timestamp", 0), ax, ay, az, gx, gy, gz, acc_mag, gyro_mag])
        csv_file.flush()

    return {"status": "received"}

@app.get("/latest")
def get_latest(pair_code: str | None = None):
    if pair_code is not None and pairing.get(pair_code) is None:
        raise HTTPException(status_code=404, detail="This STITHI session has expired.")
    
    last_seen_seconds = None if last_received_monotonic is None else round(time.monotonic() - last_received_monotonic, 3)
    
    # MAGIC FIX: Send the last 80 points to the client to eliminate graph lag
    recent_history = list(history)[-80:]
    
    return {
        "data": latest_data,
        "recent_history": recent_history,
        "samples": sample_count,
        "prediction": latest_prediction,
        "stability": latest_stability,
        "baseline": latest_baseline,
        "connected": last_seen_seconds is not None and last_seen_seconds < 3,
        "last_seen_seconds": last_seen_seconds,
        "pair_code": active_pair_code,
        "otago_completed": otago_completed,
        "otago_completed_at": otago_completed_at
    }

@app.get("/history")
def get_history():
    return list(history)

def _validate_optional_pair(pair_code: str | None) -> None:
    if pair_code is not None and pairing.get(pair_code) is None:
        raise HTTPException(status_code=404, detail="STITHI session expired.")

@app.post("/baseline/start")
def start_baseline(pair_code: str | None = None):
    _validate_optional_pair(pair_code)
    global latest_baseline
    latest_baseline = personal_baseline.start()
    return latest_baseline

@app.get("/baseline/status")
def baseline_status():
    return personal_baseline.status()

@app.post("/record/start/{activity}")
def start_recording(activity: str, pair_code: str | None = None):
    _validate_optional_pair(pair_code)
    global recording, current_activity, csv_file, csv_writer

    if recording: return {"status": "already_recording", "activity": current_activity}

    filename = os.path.join(DATA_FOLDER, f"{activity}.csv")
    csv_file = open(filename, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp", "ax", "ay", "az", "gx", "gy", "gz", "acc_mag", "gyro_mag"])
    
    recording, current_activity = True, activity
    return {"status": "recording", "activity": activity, "file": filename}

@app.post("/record/stop")
def stop_recording(pair_code: str | None = None):
    _validate_optional_pair(pair_code)
    global recording, current_activity, csv_file, csv_writer

    if not recording: return {"status": "not_recording"}
    
    activity = current_activity
    recording, current_activity = False, None
    if csv_file: csv_file.close()
    csv_file, csv_writer = None, None
    
    return {"status": "stopped", "activity": activity}

@app.get("/record/status")
def recording_status():
    return {"recording": recording, "activity": current_activity}

def _dashboard_html(pair_code: str | None = None) -> str:
    dashboard_path = Path(__file__).resolve().parent / "dashboard.html"
    html = dashboard_path.read_text(encoding="utf-8")
    if pair_code is None: return html
    if pairing.get(pair_code) is None:
        raise HTTPException(status_code=404, detail="STITHI session expired.")
    html = html.replace('fetch("/latest"', f'fetch("/latest?pair_code={pair_code}"')
    html = html.replace('fetch("/baseline/start"', f'fetch("/baseline/start?pair_code={pair_code}"')
    html = html.replace('`/record/start/${activity}`', f'`/record/start/${{activity}}?pair_code={pair_code}`')
    html = html.replace('fetch("/record/start/" + activity', f'fetch("/record/start/" + activity + "?pair_code={pair_code}"')
    html = html.replace('fetch("/record/stop"', f'fetch("/record/stop?pair_code={pair_code}"')
    return html

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(_dashboard_html())

@app.get("/dashboard/{pair_code}", response_class=HTMLResponse)
def paired_dashboard(pair_code: str):
    return HTMLResponse(_dashboard_html(pair_code))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
