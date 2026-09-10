#include <M5StickCPlus2.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include <math.h>

// ============================================================
// CONFIGURATION
// ============================================================
const char* WIFI_SSID = "prabh";
const char* WIFI_PASSWORD = "prabhnoor123";
const char* STITHI_SERVER_BASE_URL = "https://stithi.onrender.com";

// Verified for stithi.onrender.com: GlobalSign Root CA.
const char* STITHI_ROOT_CA = R"EOF(
MIIDdTCCAl2gAwIBAgILBAAAAAABFUtaw5QwDQYJKoZIhvcNAQEFBQAwVzELMAkG
A1UEBhMCQkUxGTAXBgNVBAoTEEdsb2JhbFNpZ24gbnYt c2ExEDAOBgNVBAsTB1Jv
b3QgQ0ExGzAZBgNVBAMTEkdsb2JhbFNpZ24gUm9vdCBDQTAeFw05ODA5MDExMjAw
MDBaFw0yODAxMjgxMjAwMDBaMFcxCzAJBgNVBAYTAkJFMRkwFwYDVQQKExBHbG9i
YWxTaWduIG52LXNhMRAwDgYDVQQLEwdSb290IENBMRswGQYDVQQDExJHbG9iYWxT
aWduIFJvb3QgQ0EwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQDaDuaZ
jc6j40+Kfvvxi4Mla+pIH/EqsLmVEQS98GPR4mdmzxzdzxtIK+6NiY6arymAZavp
xy0Sy6scTHAHoT0KMM0VjU/43dSMUBUc71DuxC73/OlS8pF94G3VNTCOXkNz8kHp
1Wrjsok6Vjk4bwY8iGlbKk3Fp1S4bInMm/k8yuX9ifUSPJJ4ltbcdG6TRGHRjcdG
snUOhugZitVtbNV4FpWi6cgKOOvyJBNPc1STE4U6G7weNLWLBYy5d4ux2x8gkasJ
U26Qzns3dLlwR5EiUWMWea6xrkEmCMgZK9FGqkjWZCrXgzT/LCrBbBlDSgeF59N8
9iFo7+ryUp9/k5DPAgMBAAGjQjBAMA4GA1UdDwEB/wQEAwIBBjAPBgNVHRMBAf8E
BTADAQH/MB0GA1UdDgQWBBRge2YaRQ2XyolQL30EzTSo//z9SzANBgkqhkiG9w0B
AQUFAAOCAQEA1nPnfE920I2/7LqivjTFKDK1fPxsnCwrvQmeU79rXqoRSLblCKOz
yj1hTdNGCbM+w6DjY1Ub8rrvrTnhQ7k4o+YviiY776BQVvnGCv04zcQLcFGUl5gE
38NflNUVyRRBnMRddWQVDf9VMOyGj/8N7yy5Y0b2qvzfvGn9LhJIZJrglfCm7ymP
AbEVtQwdpf5pLGkkeB6zpxxxYu7KyJesF12KwvhHhm4qxFYxldBniYUr+WymXUad
DKqC5JlR3XC321Y9YeRq4VzW9v493kHMB65jUr9TU/Qr6cf9tveCX4XSQRjbgbME
HMUfpIBvFSDJ3gyICh3WZlXi/EjJKSZp4A==
)EOF";

const bool STITHI_ALLOW_INSECURE_TLS_DEMO = true; 

const unsigned long SEND_INTERVAL = 100;    // 10 Hz telemetry
const unsigned long SCREEN_INTERVAL = 2000; // 2 sec UI refresh

unsigned long lastSend = 0;
unsigned long lastScreen = 0;
unsigned long lastPairAttempt = 0;
String pairCode = "";
String pairStatus = "WAIT";

// Hardware state tracker
int currentState = 0; // 0=Safe, 3=SOS

// ============================================================
// BATTERY & UI
// ============================================================
int getBatteryPercent() {
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
    M5.Lcd.setCursor(15, 92);
    M5.Lcd.printf("BAT: %d%%", getBatteryPercent());

    M5.Lcd.setCursor(15, 112);
    M5.Lcd.print("PAIR: ");
    M5.Lcd.print(pairCode.length() == 4 ? pairCode : pairStatus);
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
    pairStatus = "PAIRING";
    HTTPClient http;
    WiFiClient plainClient;
    WiFiClientSecure secureClient;
    String url = String(STITHI_SERVER_BASE_URL) + "/register";
    String deviceId = "m5-" + WiFi.macAddress();
    
    if (!beginHttp(http, plainClient, secureClient, url)) {
        pairStatus = "TLS FAIL";
        return false;
    }

    http.addHeader("Content-Type", "application/json");
    http.setTimeout(60000); 
    
    int responseCode = http.POST("{\"device_id\":\"" + deviceId + "\"}");
    if (responseCode <= 0) {
        http.end();
        pairStatus = "HTTP FAIL";
        return false;
    }
    
    String response = http.getString();
    http.end();
    String receivedCode = extractJsonString(response, "pair_code");
    if (receivedCode.length() != 4) {
        pairStatus = "CODE FAIL";
        return false;
    }
    pairCode = receivedCode;
    pairStatus = "READY";
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
        M5.Lcd.setCursor(15, 112);
        M5.Lcd.print("PAIRING...");
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

    // --- BUTTON CONTROLS ---
    if (M5.BtnA.wasPressed()) { currentState = 0; } 
    if (M5.BtnA.pressedFor(1500)) { 
        currentState = 3; 
        M5.Speaker.tone(3000, 500); // Beep to confirm SOS sent
    }

    // --- TRANSMIT TELEMETRY ---
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
            http.setTimeout(2000); 
            
            String json = "{\"timestamp\":" + String(now) + 
                          ",\"pair_code\":\"" + pairCode + "\"" +
                          ",\"ax\":" + String(ax, 3) + ",\"ay\":" + String(ay, 3) + ",\"az\":" + String(az, 3) + 
                          ",\"gx\":" + String(gx, 3) + ",\"gy\":" + String(gy, 3) + ",\"gz\":" + String(gz, 3) + 
                          ",\"state\":" + String(currentState) + "}";
            
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
