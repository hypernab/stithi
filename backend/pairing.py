\import sys
import time
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from datetime import datetime
from twilio.rest import Client

# Add root project path so we can import from ml.inference
sys.path.append(str(Path(__file__).resolve().parent.parent))
from ml.inference import BiomechanicalInferenceEngine

app = FastAPI(title="STITHI Clinical Dashboard")

# Allow dashboard frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the biohealth ML engine
engine = BiomechanicalInferenceEngine(window_size=10)

# In-memory storage for the live dashboard
latest_telemetry = {
    "stability_score": 100.0,
    "activity": "Calibrating...",
    "hardware_state": 0, # 0=Safe, 1=Warning, 2=Fall, 3=SOS
    "sway_angle": 0.0,
    "last_update": None
}

clinical_event_log = []

# ============================================================
# TWILIO WHATSAPP CONFIGURATION
# ============================================================
TWILIO_ACCOUNT_SID = "AC13e694ef6b2c02f19c15cb8f52743f5e"
TWILIO_AUTH_TOKEN = "5748a4cc279401422e3c5426f95961d3"
TWILIO_FROM = "whatsapp:+17372508034"
TWILIO_TO = "whatsapp:+917018749037"
TWILIO_CONTENT_SID = "HXfe5ab5f00277942d4d4200328b4d403c"

last_sos_time = 0  
SOS_COOLDOWN_SECONDS = 60 

def send_emergency_whatsapp():
    """Dispatches the SOS WhatsApp template to the caretaker."""
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            content_sid=TWILIO_CONTENT_SID,
            from_=TWILIO_FROM,
            to=TWILIO_TO
        )
        print(f"✅ WhatsApp SOS Dispatched! SID: {message.sid}")
        print(f"Status: {message.status} | API Version: {message.api_version}")
    except Exception as e:
        print(f"❌ Failed to send WhatsApp message: {e}")

# ============================================================
# FASTAPI ENDPOINTS
# ============================================================
class IMUPayload(BaseModel):
    timestamp: int
    pair_code: str = ""
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float

@app.post("/imu")
async def receive_imu_data(data: IMUPayload):
    global last_sos_time
    
    # Calculate state / sway based on raw data if not provided by device in latest struct
    # Assuming the device still dictates state, or it is inferred here. 
    # If the firmware sets state to 3, we catch it. 
    # For this architecture, if you passed state in the JSON, extract it. 
    # (If state is no longer in the payload from your Render update, you will need to infer SOS from the hardware button logic or add it back to the JSON payload).
    
    # 1. Update hardware-level telemetry
    latest_telemetry["last_update"] = datetime.now().isoformat()

    # 2. Feed the 6-axis data into the Python ML Inference Engine
    inference_result = engine.push_sample(
        data.ax, data.ay, data.az, data.gx, data.gy, data.gz
    )
    
    # 3. If the sliding window is full and evaluated, update the clinical scores
    if inference_result:
        latest_telemetry["stability_score"] = inference_result["stability_score"]
        latest_telemetry["activity"] = inference_result["activity_name"]
        
        # AGENT ACTION: Automatically trigger WhatsApp SOS on critical instability (score < 30)
        # OR if your hardware payload includes a manual state == 3.
        if inference_result["stability_score"] < 30.0:
            latest_telemetry["hardware_state"] = 3
            current_time = time.time()
            if current_time - last_sos_time > SOS_COOLDOWN_SECONDS:
                print("🚨 CRITICAL INSTABILITY DETECTED! Triggering WhatsApp SOS...")
                send_emergency_whatsapp()
                last_sos_time = current_time
        elif inference_result["is_unstable"]:
            latest_telemetry["hardware_state"] = 1
        else:
            latest_telemetry["hardware_state"] = 0

        # Log critical events
        if inference_result["is_unstable"]:
            clinical_event_log.append({
                "time": latest_telemetry["last_update"],
                "activity": latest_telemetry["activity"],
                "stability_score": latest_telemetry["stability_score"]
            })
            if len(clinical_event_log) > 50:
                clinical_event_log.pop(0)
    
    return {"status": "success"}

@app.get("/latest")
def get_latest_status():
    """Endpoint for the HTML dashboard to poll live patient status."""
    return latest_telemetry
    
@app.get("/clinical_log")
def get_clinical_log():
    """Endpoint to download patient micro-stumble history."""
    return {"events": clinical_event_log}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
