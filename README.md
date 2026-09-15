# STITHI
**Wearable Stability & Fall Prevention Monitor | Smart India Hackathon 2026 (SIH1580)**

STITHI is a proactive, low-cost wearable prototype and clinical dashboard designed to monitor the stability of community-dwelling older adults. By combining edge-computed kinematics, cloud-based machine learning, and built-in clinical exercise compliance (Otago Programme), STITHI shifts elderly care from reactive fall detection to proactive fall prevention.

---

## Key Features

*   **Real-Time Kinematic Telemetry:** Streams 6-axis IMU data (Acceleration & Gyroscope) at 4Hz to a cloud backend with zero-lag front-end visualization.
*   **Edge Processing:** Local M5StickC Plus 2 (ESP32) deterministic engine for immediate on-device alerts (<20ms latency) based on movement intensity and thresholds.
*   **Machine Learning Activity Recognition:** Python FastAPI backend running a Scikit-Learn Extra Trees model to classify states (Sitting, Standing, Walking, etc.) based on 26 engineered features.
*   **Clinical Compliance System (Otago):** Built-in NTP-synced Real-Time Clock (RTC) triggers a daily alarm for physiotherapy exercises. Compliance is logged via hardware buttons and synced instantly to the web dashboard.
*   **Secure Device Pairing:** Ephemeral 4-digit pairing codes ensure secure, 1-to-1 connections between the wearable and the web dashboard.

---

## Tech Stack

**Edge Hardware**
*   **Device:** M5StickC Plus 2 (ESP32-PICO-V3-02)
*   **Sensors:** MPU6886 6-Axis IMU (Accelerometer + Gyroscope)
*   **Firmware:** C++ / Arduino Core (M5Unified Library)
*   **Connectivity:** Wi-Fi (HTTP POST with TLS)

**Cloud Backend & ML**
*   **Framework:** FastAPI (Python)
*   **Data Processing:** Pandas, NumPy
*   **Machine Learning:** Scikit-Learn (ExtraTreesClassifier)
*   **Deployment:** Render (Dockerized/Web Service)

**Frontend Clinical Dashboard**
*   **Core:** HTML5, CSS3, Vanilla JavaScript
*   **Visualization:** Custom HTML5 Canvas rendering for high-performance 4Hz telemetry charting.

---

## Hardware Setup (M5StickC Plus 2)

1.  Open `main.cpp` in the Arduino IDE or PlatformIO.
2.  Install the required library: `M5StickCPlus2` (or `M5Unified`).
3.  Update the Wi-Fi credentials and server URL in the configuration section:
    ```cpp
    const char* WIFI_SSID = "YOUR_WIFI_NAME";
    const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
    const char* STITHI_SERVER_BASE_URL = "[https://your-render-url.onrender.com](https://your-render-url.onrender.com)";
    ```
4.  Flash the code to the M5StickC Plus 2.
5.  On boot, the device will connect to Wi-Fi, sync time via NTP to IST (UTC+5:30), and display a 4-digit pairing code.

---

## Backend Setup (Local Development)

1.  Clone this repository:
    ```bash
    git clone [https://github.com/yourusername/stithi.git](https://github.com/yourusername/stithi.git)
    cd stithi
    ```
2.  Install the required Python dependencies:
    ```bash
    pip install fastapi uvicorn pandas
