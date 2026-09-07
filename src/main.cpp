#include <M5StickCPlus2.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <math.h>

// ============================================================
// CONFIGURATION
// ============================================================
const char* WIFI_SSID = "EACCESS";
const char* WIFI_PASSWORD = "hostelnet";
const char* SERVER_URL = "http://172.16.166.179:8000/imu"; // Your laptop IP

const unsigned long SEND_INTERVAL = 100;    // 10 Hz telemetry
const unsigned long SCREEN_INTERVAL = 2000; // 2 sec UI refresh

unsigned long lastSend = 0;
unsigned long lastScreen = 0;

// ============================================================
// BATTERY & UI
// ============================================================
int getBatteryPercent() {
    // Native M5Unified function correctly calculates the curve
    return M5.Power.getBatteryLevel();
}

void updateScreen() {
    M5.Lcd.fillScreen(BLACK);
    M5.Lcd.setTextColor(WHITE);
    M5.Lcd.setTextSize(3);
    M5.Lcd.setCursor(25, 20);
    M5.Lcd.print("STITHI");

    M5.Lcd.setTextSize(2);
    M5.Lcd.setCursor(15, 70);
    M5.Lcd.print("WiFi: ");
    M5.Lcd.setTextColor(WiFi.status() == WL_CONNECTED ? GREEN : RED);
    M5.Lcd.print(WiFi.status() == WL_CONNECTED ? "OK" : "OFF");

    M5.Lcd.setTextColor(WHITE);
    M5.Lcd.setCursor(15, 105);
    M5.Lcd.printf("BAT: %d%%", getBatteryPercent());
}

// ============================================================
// SETUP
// ============================================================
void setup() {
    auto cfg = M5.config();
    M5.begin(cfg);
    M5.Imu.init();

    M5.Lcd.setRotation(1);
    M5.Lcd.fillScreen(BLACK);
    M5.Lcd.setTextColor(WHITE);
    M5.Lcd.setTextSize(3);
    M5.Lcd.setCursor(25, 30);
    M5.Lcd.print("STITHI");
    
    M5.Lcd.setTextSize(2);
    M5.Lcd.setCursor(25, 75);
    M5.Lcd.print("Syncing...");

    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    unsigned long startTime = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - startTime < 10000) {
        delay(100);
    }

    M5.Lcd.fillScreen(BLACK);
    updateScreen(); 
}

// ============================================================
// MAIN LOOP
// ============================================================
void loop() {
    M5.update();
    unsigned long now = millis();

    float ax, ay, az, gx, gy, gz;
    M5.Imu.getAccelData(&ax, &ay, &az);
    M5.Imu.getGyroData(&gx, &gy, &gz);

    // --- TRANSMIT TELEMETRY (PURE DATA COLLECTION) ---
    if (now - lastSend >= SEND_INTERVAL) {
        lastSend = now;
        
        if (WiFi.status() == WL_CONNECTED) {
            HTTPClient http;
            http.begin(SERVER_URL);
            http.addHeader("Content-Type", "application/json");
            http.setTimeout(150); 
            
            String json = "{\"timestamp\":" + String(now) + 
                          ",\"ax\":" + String(ax, 3) + ",\"ay\":" + String(ay, 3) + ",\"az\":" + String(az, 3) + 
                          ",\"gx\":" + String(gx, 3) + ",\"gy\":" + String(gy, 3) + ",\"gz\":" + String(gz, 3) + "}";
            
            http.POST(json);
            http.end();
        }
    }

    // --- UI REFRESH ---
    if (now - lastScreen >= SCREEN_INTERVAL) {
        lastScreen = now;
        updateScreen();
    }
}