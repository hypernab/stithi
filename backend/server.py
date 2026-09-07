from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from datetime import datetime
from collections import deque
from pathlib import Path
import sys
import math
import csv
import os

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from ml.inference import ActivityRecognitionEngine
from backend.stability import PersonalBaseline, StabilityEngine

app = FastAPI()

# ============================================================
# CONFIG
# ============================================================

DATA_FOLDER = "../data"
os.makedirs(DATA_FOLDER, exist_ok=True)

HISTORY_LENGTH = 500

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

latest_data = {
    "timestamp": 0,
    "ax": 0,
    "ay": 0,
    "az": 0,
    "gx": 0,
    "gy": 0,
    "gz": 0,
    "acc_mag": 0,
    "gyro_mag": 0,
    "stability": None,
    "received_at": ""
}

sample_count = 0

# ============================================================
# RECORDING
# ============================================================

recording = False
current_activity = None
csv_file = None
csv_writer = None


# ============================================================
# RECEIVE IMU
# ============================================================

@app.post("/imu")
def receive_imu(data: dict):

    global latest_data
    global sample_count
    global csv_writer
    global latest_prediction
    global latest_stability
    global latest_baseline

    try:
        ax = float(data.get("ax", 0))
        ay = float(data.get("ay", 0))
        az = float(data.get("az", 0))
        gx = float(data.get("gx", 0))
        gy = float(data.get("gy", 0))
        gz = float(data.get("gz", 0))
    except (TypeError, ValueError):
        return {"status": "invalid", "detail": "IMU values must be numeric"}

    acc_mag = math.sqrt(
        ax**2 +
        ay**2 +
        az**2
    )

    gyro_mag = math.sqrt(
        gx**2 +
        gy**2 +
        gz**2
    )

    latest_stability = stability_engine.update(ax, ay, az, gx, gy, gz)
    latest_baseline = personal_baseline.update(latest_stability)
    latest_data = {
        "timestamp": data.get("timestamp", 0),
        "ax": ax,
        "ay": ay,
        "az": az,
        "gx": gx,
        "gy": gy,
        "gz": gz,
        "acc_mag": acc_mag,
        "gyro_mag": gyro_mag,
        "stability": latest_stability,
        "received_at": datetime.now().strftime("%H:%M:%S")
    }

    history.append(latest_data.copy())
    sample_count += 1
    latest_prediction = activity_engine.push_sample(ax, ay, az, gx, gy, gz)

    # Save if recording
    if recording and csv_writer:
        csv_writer.writerow([
            data.get("timestamp", 0),
            ax,
            ay,
            az,
            gx,
            gy,
            gz,
            acc_mag,
            gyro_mag
        ])
        csv_file.flush()

    return {"status": "received"}


# ============================================================
# API
# ============================================================

@app.get("/latest")
def get_latest():
    return {
        "data": latest_data,
        "samples": sample_count,
        "prediction": latest_prediction,
        "stability": latest_stability,
        "baseline": latest_baseline
    }


@app.get("/history")
def get_history():
    return list(history)


@app.post("/baseline/start")
def start_baseline():
    global latest_baseline
    latest_baseline = personal_baseline.start()
    return latest_baseline


@app.get("/baseline/status")
def baseline_status():
    return personal_baseline.status()


# ============================================================
# RECORDING
# ============================================================

@app.post("/record/start/{activity}")
def start_recording(activity: str):

    global recording
    global current_activity
    global csv_file
    global csv_writer

    if recording:
        return {
            "status": "already_recording",
            "activity": current_activity
        }

    filename = os.path.join(
        DATA_FOLDER,
        f"{activity}.csv"
    )

    csv_file = open(
        filename,
        "w",
        newline=""
    )

    csv_writer = csv.writer(csv_file)

    csv_writer.writerow([
        "timestamp",
        "ax",
        "ay",
        "az",
        "gx",
        "gy",
        "gz",
        "acc_mag",
        "gyro_mag"
    ])

    recording = True
    current_activity = activity

    print(f"RECORDING STARTED: {activity}")

    return {
        "status": "recording",
        "activity": activity,
        "file": filename
    }


@app.post("/record/stop")
def stop_recording():

    global recording
    global current_activity
    global csv_file
    global csv_writer

    if not recording:
        return {"status": "not_recording"}

    activity = current_activity

    recording = False
    current_activity = None

    if csv_file:
        csv_file.close()

    csv_file = None
    csv_writer = None

    print(f"RECORDING STOPPED: {activity}")

    return {
        "status": "stopped",
        "activity": activity
    }


@app.get("/record/status")
def recording_status():
    return {
        "recording": recording,
        "activity": current_activity
    }


# ============================================================
# DASHBOARD
# ============================================================

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>STHIRAI | Wearable Stability Monitor</title>

<style>
/* =========================================================
   GLOBAL
   ========================================================= */
* {
    box-sizing: border-box;
}

body {
    margin: 0;
    background: #0d1117;
    color: #c9d1d9;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
}

.container {
    max-width: 1400px;
    margin: auto;
    padding: 28px;
}

/* =========================================================
   HEADER
   ========================================================= */
.header {
    margin-bottom: 30px;
}

.logo {
    font-size: 32px;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: 1px;
}

.subtitle {
    color: #8b949e;
    margin-top: 5px;
    font-size: 16px;
    margin-bottom: 20px;
}

.status {
    padding: 6px 14px;
    border-radius: 20px;
    background: #151b23;
    border: 1px solid #30363d;
    font-size: 14px;
    font-weight: 600;
    display: inline-block;
    color: #8b949e;
}

.status.connected {
    color: #3fb950;
    border-color: rgba(63, 185, 80, 0.4);
    background: rgba(63, 185, 80, 0.1);
}

/* =========================================================
   GRID
   ========================================================= */
.grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 16px;
    margin-bottom: 25px;
}

.card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.card h2 {
    margin-top: 0;
    font-size: 18px;
    color: #ffffff;
    margin-bottom: 15px;
}

/* =========================================================
   SENSOR VALUES
   ========================================================= */
.sensor-label {
    color: #8b949e;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 8px;
}

.sensor-value {
    font-size: 32px;
    font-weight: 700;
    color: #ffffff;
}

.indicator {
    display: inline-block;
    padding: 6px 10px;
    border-radius: 6px;
    background: #21262d;
    color: #ffffff;
    font-weight: 700;
}

.indicator.unusual {
    background: #9b2c2c;
}

.recent-list {
    margin: 0;
    padding-left: 20px;
    color: #c9d1d9;
}

/* =========================================================
   CHARTS
   ========================================================= */
.chart-card {
    margin-bottom: 25px;
}

canvas {
    width: 100%;
    height: 320px;
    background: #0d1117;
    border-radius: 8px;
    border: 1px solid #30363d;
}

/* =========================================================
   TOGGLES
   ========================================================= */
.toggles {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 15px;
}

.toggle {
    padding: 6px 14px;
    border-radius: 20px;
    background: #21262d;
    border: 1px solid #30363d;
    cursor: pointer;
    font-size: 13px;
    font-weight: 600;
    color: #c9d1d9;
}

.toggle.active {
    background: #30363d;
}

.toggle.x { border-left: 4px solid #ff5c5c; }
.toggle.y { border-left: 4px solid #58a6ff; }
.toggle.z { border-left: 4px solid #3fb950; }

/* =========================================================
   HEATMAP
   ========================================================= */
.heatmap {
    display: grid;
    grid-template-columns: repeat(20, 1fr);
    gap: 4px;
    background: #0d1117;
    padding: 10px;
    border-radius: 8px;
    border: 1px solid #30363d;
}

.heat-cell {
    aspect-ratio: 1;
    border-radius: 4px;
    background: #161b22;
    transition: background 0.15s;
}

/* =========================================================
   RECORDING PANEL
   ========================================================= */
.recording-controls {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
    margin-top: 15px;
}

button {
    border: 1px solid rgba(255,255,255,0.1);
    padding: 10px 18px;
    border-radius: 8px;
    background: #238636;
    color: white;
    cursor: pointer;
    font-weight: 600;
    font-size: 14px;
    transition: all 0.2s;
}

button:hover {
    background: #2ea043;
}

.stop {
    background: #da3633;
}
.stop:hover {
    background: #f85149;
}

#recordStatus {
    color: #8b949e;
    font-size: 14px;
    margin-left: 10px;
}

.footer {
    margin-top: 40px;
    color: #8b949e;
    font-size: 13px;
    text-align: center;
}
</style>
</head>

<body>

<div class="container">

    <!-- =====================================================
         HEADER
         ===================================================== -->
    <div class="header">
        <div class="logo">STHIRAI</div>
        <div class="subtitle">SIH1580 — Wearable Stability Monitor</div>
        <div id="connection" class="status">● WAITING FOR M5STICK...</div>
    </div>

    <!-- =====================================================
         LIVE BIOMETRICS GRID
         ===================================================== -->
    <div class="grid">
        <div class="card">
            <div class="sensor-label">Acceleration X</div>
            <div id="ax" class="sensor-value">--</div>
        </div>
        <div class="card">
            <div class="sensor-label">Acceleration Y</div>
            <div id="ay" class="sensor-value">--</div>
        </div>
        <div class="card">
            <div class="sensor-label">Acceleration Z</div>
            <div id="az" class="sensor-value">--</div>
        </div>
        <div class="card">
            <div class="sensor-label">Acceleration Magnitude</div>
            <div id="acc" class="sensor-value" style="color: #58a6ff;">--</div>
        </div>
        
        <div class="card">
            <div class="sensor-label">Gyroscope X</div>
            <div id="gx" class="sensor-value">--</div>
        </div>
        <div class="card">
            <div class="sensor-label">Gyroscope Y</div>
            <div id="gy" class="sensor-value">--</div>
        </div>
        <div class="card">
            <div class="sensor-label">Gyroscope Z</div>
            <div id="gz" class="sensor-value">--</div>
        </div>
        <div class="card">
            <div class="sensor-label">Gyro Magnitude</div>
            <div id="gyro" class="sensor-value" style="color: #58a6ff;">--</div>
        </div>

        <div class="card">
            <div class="sensor-label">Samples Received</div>
            <div id="samples" class="sensor-value" style="color: #3fb950;">0</div>
        </div>
        <div class="card">
            <div class="sensor-label">Last Update (Uptime)</div>
            <div id="updated" class="sensor-value" style="font-size: 24px;">--</div>
        </div>
        <div class="card">
            <div class="sensor-label">Predicted Activity</div>
            <div id="predictedActivity" class="sensor-value" style="font-size: 24px;">Waiting</div>
            <div class="sensor-label">Confidence: <span id="activityConfidence">--</span></div>
        </div>
        <div class="card">
            <div class="sensor-label">Movement Status</div>
            <div id="movementStatus" class="indicator">Waiting</div>
        </div>
        <div class="card">
            <div class="sensor-label">Stability Indicator</div>
            <div id="stabilityIndicator" class="indicator">Waiting</div>
            <div id="movementFeedback" class="sensor-label">No unusual movement detected</div>
        </div>
        <div class="card">
            <div class="sensor-label">Personal Baseline</div>
            <div id="baselineStatus" class="indicator">Not calibrated</div>
            <div class="sensor-label">Deviation: <span id="baselineDeviation">--</span></div>
        </div>
        <div class="card">
            <div class="sensor-label">Orientation</div>
            <div class="sensor-value" style="font-size: 24px;">Roll <span id="roll">--</span></div>
            <div class="sensor-value" style="font-size: 24px;">Pitch <span id="pitch">--</span></div>
        </div>
        <div class="card">
            <div class="sensor-label">Recent Activity</div>
            <ol id="recentActivity" class="recent-list"><li>Waiting for a full Model 1 window</li></ol>
        </div>
    </div>

    <!-- =====================================================
         DATA COLLECTION PANEL (MOVED UP)
         ===================================================== -->
    <div class="card chart-card">
        <h2>Data Collection</h2>
        <div style="color: #c9d1d9; font-size: 14px;">
            Activity: <strong id="activity" style="color: #ffffff;">None</strong>
        </div>
        
        <div class="recording-controls">
            <button onclick="startBaseline()">Start Baseline</button>
            <button onclick="startRecording('standing')">Standing</button>
            <button onclick="startRecording('walking')">Walking</button>
            <button onclick="startRecording('sitting')">Sitting</button>
            <button onclick="startRecording('turning')">Turning</button>
            <button onclick="startRecording('slow_walking')">Slow Walking</button>
            <button class="stop" onclick="stopRecording()">STOP</button>
            <span id="recordStatus">Not recording</span>
        </div>
    </div>

    <!-- =====================================================
         ACCELERATION CHART
         ===================================================== -->
    <div class="card chart-card">
        <h2>Acceleration</h2>
        <div class="toggles">
            <div class="toggle x active" onclick="toggleAxis('ax', this)">AX</div>
            <div class="toggle y active" onclick="toggleAxis('ay', this)">AY</div>
            <div class="toggle z active" onclick="toggleAxis('az', this)">AZ</div>
        </div>
        <canvas id="accChart" width="1200" height="320"></canvas>
    </div>

    <!-- =====================================================
         HEATMAP (MOVED DOWN)
         ===================================================== -->
    <div class="card chart-card">
        <h2>Movement Intensity</h2>
        <p class="sensor-label">Recent movement intensity. Brighter cells represent stronger motion.</p>
        <div id="heatmap" class="heatmap"></div>
    </div>

    <!-- =====================================================
         GYROSCOPE CHART
         ===================================================== -->
    <div class="card chart-card">
        <h2>Gyroscope</h2>
        <div class="toggles">
            <div class="toggle x active" onclick="toggleAxis('gx', this)">GX</div>
            <div class="toggle y active" onclick="toggleAxis('gy', this)">GY</div>
            <div class="toggle z active" onclick="toggleAxis('gz', this)">GZ</div>
        </div>
        <canvas id="gyroChart" width="1200" height="320"></canvas>
    </div>

    <div class="footer">
        STHIRAI Prototype • SIH1580 <br>
        MVP: laptop-side activity recognition and IMU-derived stability indicators.<br>
        Future research: validated LOG/BOS estimation, personalized exercise and clinical validation.
    </div>

</div>

<script>
/* =========================================================
   STATE
   ========================================================= */
const axes = {
    ax: true, ay: true, az: true,
    gx: true, gy: true, gz: true
};

let accHistory = [];
let gyroHistory = [];
let recentActivities = [];
const MAX_POINTS = 120;

/* =========================================================
   AXIS TOGGLE
   ========================================================= */
function toggleAxis(axis, element) {
    axes[axis] = !axes[axis];
    element.classList.toggle("active");
}

/* =========================================================
   DRAW CHART
   ========================================================= */
function drawChart(canvas, datasets) {
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    if (datasets.length === 0 || datasets[0].data.length < 2) return;

    let values = [];
    datasets.forEach(d => {
        values = values.concat(d.data); // Fixed array extraction
    });

    let min = Math.min(...values);
    let max = Math.max(...values);

    if (min === max) {
        min -= 1;
        max += 1;
    }

    const range = max - min;

    // Grid
    ctx.strokeStyle = "#30363d";
    ctx.lineWidth = 1;
    for (let y = 0; y <= height; y += 50) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
    }

    // Lines
    datasets.forEach(dataset => {
        if (dataset.data.length < 2) return;

        ctx.beginPath();
        ctx.strokeStyle = dataset.color;
        ctx.lineWidth = 2.5;

        dataset.data.forEach((value, index) => {
            const x = index / (MAX_POINTS - 1) * width;
            const y = height - ((value - min) / range) * height;

            if (index === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();
    });
}

/* =========================================================
   HEATMAP
   ========================================================= */
function updateHeatmap() {
    const container = document.getElementById("heatmap");
    container.innerHTML = "";

    const values = accHistory.map(d => Math.abs(d.ax) + Math.abs(d.ay) + Math.abs(d.az));
    const recent = values.slice(-80);

    for (let i = 0; i < 80; i++) {
        const cell = document.createElement("div");
        cell.className = "heat-cell";

        const value = recent[i] || 0;
        const intensity = Math.min(1, Math.abs(value - 1) / 2);

        // Smooth Gradient Math:
        // Blends smoothly from Blue (0, 120, 215) at 0.0 
        // to Red/Orange (232, 60, 20) at 1.0
        const r = Math.round(0 + (232 - 0) * intensity);
        const g = Math.round(120 + (60 - 120) * intensity);
        const b = Math.round(215 + (20 - 215) * intensity);

        cell.style.background = `rgb(${r},${g},${b})`;
        container.appendChild(cell);
    }
}
/* =========================================================
   UPDATE DATA
   ========================================================= */
async function update() {
    try {
        const response = await fetch("/latest");
        const result = await response.json();
        const d = result.data;

        if (d.timestamp === 0) return;

        const conn = document.getElementById("connection");
        conn.textContent = "● M5Stick CONNECTED";
        conn.classList.add("connected");

        document.getElementById("ax").textContent = d.ax.toFixed(3);
        document.getElementById("ay").textContent = d.ay.toFixed(3);
        document.getElementById("az").textContent = d.az.toFixed(3);
        document.getElementById("acc").textContent = d.acc_mag.toFixed(3);

        document.getElementById("gx").textContent = d.gx.toFixed(3);
        document.getElementById("gy").textContent = d.gy.toFixed(3);
        document.getElementById("gz").textContent = d.gz.toFixed(3);
        document.getElementById("gyro").textContent = d.gyro_mag.toFixed(3);

        document.getElementById("samples").textContent = result.samples;
        document.getElementById("updated").textContent = d.received_at;
        if (result.prediction) {
            document.getElementById("predictedActivity").textContent = result.prediction.activity;
            document.getElementById("activityConfidence").textContent =
                (result.prediction.confidence * 100).toFixed(1) + "%";
            if (recentActivities[0] !== result.prediction.activity) {
                recentActivities.unshift(result.prediction.activity);
                recentActivities = recentActivities.slice(0, 5);
                document.getElementById("recentActivity").innerHTML =
                    recentActivities.map(activity => `<li>${activity}</li>`).join("");
            }
        }

        const stability = result.stability || d.stability;
        if (stability) {
            document.getElementById("movementStatus").textContent = stability.movement_status;
            const stabilityIndicator = document.getElementById("stabilityIndicator");
            stabilityIndicator.textContent = stability.unusual_movement ? "CHECK" : "NORMAL";
            stabilityIndicator.classList.toggle("unusual", stability.unusual_movement);
            document.getElementById("movementFeedback").textContent = stability.unusual_movement
                ? "Unusual movement detected - check balance."
                : "No unusual movement detected";
            document.getElementById("roll").textContent = stability.roll.toFixed(1) + "°";
            document.getElementById("pitch").textContent = stability.pitch.toFixed(1) + "°";
        }

        const baseline = result.baseline;
        if (baseline) {
            const baselineStatus = document.getElementById("baselineStatus");
            baselineStatus.textContent = baseline.movement_status;
            baselineStatus.classList.toggle("unusual", baseline.unusual_movement);
            document.getElementById("baselineDeviation").textContent =
                baseline.baseline_deviation === null ? "--" : baseline.baseline_deviation.toFixed(2);
        }

        accHistory.push({ ax: d.ax, ay: d.ay, az: d.az });
        gyroHistory.push({ gx: d.gx, gy: d.gy, gz: d.gz });

        if (accHistory.length > MAX_POINTS) accHistory.shift();
        if (gyroHistory.length > MAX_POINTS) gyroHistory.shift();

        // Acc Chart Data
        const accSets = [];
        if (axes.ax) accSets.push({ data: accHistory.map(d => d.ax), color: "#ff5c5c" });
        if (axes.ay) accSets.push({ data: accHistory.map(d => d.ay), color: "#58a6ff" });
        if (axes.az) accSets.push({ data: accHistory.map(d => d.az), color: "#3fb950" });
        drawChart(document.getElementById("accChart"), accSets);

        // Gyro Chart Data
        const gyroSets = [];
        if (axes.gx) gyroSets.push({ data: gyroHistory.map(d => d.gx), color: "#ff5c5c" });
        if (axes.gy) gyroSets.push({ data: gyroHistory.map(d => d.gy), color: "#58a6ff" });
        if (axes.gz) gyroSets.push({ data: gyroHistory.map(d => d.gz), color: "#3fb950" });
        drawChart(document.getElementById("gyroChart"), gyroSets);

        updateHeatmap();

    } catch(error) {
        const status = document.getElementById("connection");
        status.textContent = "● M5STICK OFFLINE";
        status.classList.remove("connected");
    }
}

/* =========================================================
   RECORDING
   ========================================================= */
async function startBaseline() {
    await fetch("/baseline/start", { method: "POST" });
    const status = document.getElementById("baselineStatus");
    status.textContent = "Collecting baseline";
    status.classList.remove("unusual");
    document.getElementById("baselineDeviation").textContent = "--";
}

async function startRecording(activity) {
    const response = await fetch("/record/start/" + activity, { method: "POST" });
    const result = await response.json();
    document.getElementById("activity").textContent = activity;
    
    const statusText = document.getElementById("recordStatus");
    statusText.textContent = "🔴 RECORDING: " + activity;
    statusText.style.color = "#ff7b72";
}

async function stopRecording() {
    const response = await fetch("/record/stop", { method: "POST" });
    const result = await response.json();
    document.getElementById("activity").textContent = "None";
    
    const statusText = document.getElementById("recordStatus");
    statusText.textContent = "Recording stopped";
    statusText.style.color = "#8b949e";
}

/* =========================================================
   LOOP
   ========================================================= */
setInterval(update, 200);
update();

</script>
</body>
</html>
"""

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)