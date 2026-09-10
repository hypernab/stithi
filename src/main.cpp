#include <M5StickCPlus2.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include <math.h>

// ============================================================
// CONFIGURATION
// ============================================================
// Set these locally before flashing; do not commit real Wi-Fi credentials.
const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
// Change this one value when moving between local and public deployment.
const char* STITHI_SERVER_BASE_URL = "http://10.73.149.130:8000";
// Paste the public endpoint's CA certificate here before enabling HTTPS.
const char* STITHI_ROOT_CA = "";
const bool STITHI_ALLOW_INSECURE_TLS_DEMO = false;

const unsigned long SEND_INTERVAL = 100;    // 10 Hz telemetry
const unsigned long SCREEN_INTERVAL = 2000; // 2 sec UI refresh

unsigned long lastSend = 0;
unsigned long lastScreen = 0;
unsigned long lastPairAttempt = 0;
String pairCode = "";

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

    M5.Lcd.setCursor(15, 135);
    M5.Lcd.print("PAIR: ");
    M5.Lcd.print(pairCode.length() == 4 ? pairCode : "----");
}

bool beginHttp(HTTPClient& http, WiFiClient& plainClient, WiFiClientSecure& secureClient, const String& url) {
    if (url.startsWith("https://")) {
        if (STITHI_ALLOW_INSECURE_TLS_DEMO) {
            secureClient.setInsecure();
        } else if (strlen(STITHI_ROOT_CA) == 0) {
            return false;
        } else {
            secureClient.setCACert(STITHI_ROOT_CA);
        }
        return http.begin(secureClient, url);
    }
    return http.begin(plainClient, url);
}

String extractJsonString(const String& body, const String& key) {
    String marker = "\"" + key + "\":\"";
    int start = body.indexOf(marker);
    if (start < 0) return "";
    start += marker.length();
    int end = body.indexOf('"', start);
    return end < 0 ? "" : body.substring(start, end);
}

bool registerDevice() {
    HTTPClient http;
    WiFiClient plainClient;
    WiFiClientSecure secureClient;
    String url = String(STITHI_SERVER_BASE_URL) + "/register";
    String deviceId = "m5-" + WiFi.macAddress();
    if (!beginHttp(http, plainClient, secureClient, url)) return false;

    http.addHeader("Content-Type", "application/json");
    http.setTimeout(1500);
    int responseCode = http.POST("{\"device_id\":\"" + deviceId + "\"}");
    if (responseCode <= 0) {
        http.end();
        return false;
    }
    String response = http.getString();
    http.end();
    String receivedCode = extractJsonString(response, "pair_code");
    if (receivedCode.length() != 4) return false;
    pairCode = receivedCode;
    return true;
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
    if (WiFi.status() == WL_CONNECTED) {
        M5.Lcd.setCursor(15, 160);
        M5.Lcd.print("Pairing...");
        registerDevice();
        updateScreen();
    }
}

// ============================================================
// MAIN LOOP
// ============================================================
void loop() {
    M5.update();
    unsigned long now = millis();

    if (WiFi.status() == WL_CONNECTED && pairCode.length() != 4 && now - lastPairAttempt >= 5000) {
        lastPairAttempt = now;
        registerDevice();
    }

    float ax, ay, az, gx, gy, gz;
    M5.Imu.getAccelData(&ax, &ay, &az);
    M5.Imu.getGyroData(&gx, &gy, &gz);

    // --- TRANSMIT TELEMETRY (PURE DATA COLLECTION) ---
    if (now - lastSend >= SEND_INTERVAL) {
        lastSend = now;
        
        if (WiFi.status() == WL_CONNECTED) {
            HTTPClient http;
            WiFiClient plainClient;
            WiFiClientSecure secureClient;
            String url = String(STITHI_SERVER_BASE_URL) + "/imu";
            if (!beginHttp(http, plainClient, secureClient, url)) {
                http.end();
                return;
            }
            http.addHeader("Content-Type", "application/json");
            http.setTimeout(150); 
            
            String json = "{\"timestamp\":" + String(now) + 
                          ",\"pair_code\":\"" + pairCode + "\"" +
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