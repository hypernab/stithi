#include <M5StickCPlus2.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include <time.h>
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

const unsigned long SEND_INTERVAL = 250;    // 4 Hz telemetry unblocks the buttons
const unsigned long SCREEN_INTERVAL = 2000; 

unsigned long lastSend = 0;
unsigned long lastScreen = 0;
unsigned long lastPairAttempt = 0;
String pairCode = "";
String pairStatus = "WAIT";

// ============================================================
// UI STATE MACHINE & OTAGO ALARM VARS
// ============================================================
enum UIState {
    UI_NORMAL,
    UI_MENU,
    UI_SET_HOUR,
    UI_SET_MINUTE,
    UI_RINGING
};

UIState currentState = UI_NORMAL;
int menuIndex = 0; 
int tempHour = 10;
int tempMinute = 0;
bool holdTriggered = false; 

int targetHour = 10;   
int targetMinute = 0;
bool complianceLogged = false;

// ============================================================
// HELPERS
// ============================================================
int getBatteryPercent() {
    return M5.Power.getBatteryLevel();
}

void updateDashboard() {
    M5.Lcd.fillScreen(BLACK);
    
    M5.Lcd.setTextColor(WHITE);
    M5.Lcd.setTextSize(3);
    M5.Lcd.setCursor(25, 10);
    M5.Lcd.print("STITHI");

    M5.Lcd.setTextSize(2);
    M5.Lcd.setCursor(10, 50);
    M5.Lcd.setTextColor(WHITE);
    M5.Lcd.print("WIFI:");
    M5.Lcd.setTextColor(WiFi.status() == WL_CONNECTED ? GREEN : RED);
    M5.Lcd.print(WiFi.status() == WL_CONNECTED ? "OK " : "NO ");
    
    M5.Lcd.setTextColor(WHITE);
    M5.Lcd.printf("BAT:%d%%", getBatteryPercent());

    M5.Lcd.setCursor(10, 80);
    M5.Lcd.print("PAIR: ");
    M5.Lcd.setTextColor(CYAN);
    M5.Lcd.print(pairCode.length() == 4 ? pairCode : pairStatus);

    M5.Lcd.setCursor(10, 110);
    M5.Lcd.setTextColor(WHITE);
    M5.Lcd.print("ALARM: ");
    M5.Lcd.setTextColor(ORANGE);
    M5.Lcd.printf("%02d:%02d", targetHour, targetMinute);
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
    
    M5.Speaker.setVolume(255); 

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
    
    if (WiFi.status() == WL_CONNECTED) {
        M5.Lcd.setCursor(10, 40);
        M5.Lcd.print("SYNCING TIME...");
        
        configTime(19800, 0, "pool.ntp.org", "time.nist.gov");
        
        struct tm timeinfo;
        // Wait up to 10 seconds for a robust time sync
        if (getLocalTime(&timeinfo, 10000)) { 
            m5::rtc_time_t rtcTime;
            rtcTime.hours   = timeinfo.tm_hour;
            rtcTime.minutes = timeinfo.tm_min;
            rtcTime.seconds = timeinfo.tm_sec;
            M5.Rtc.setTime(&rtcTime);
            
            m5::rtc_date_t rtcDate;
            rtcDate.year  = timeinfo.tm_year + 1900;
            rtcDate.month = timeinfo.tm_mon + 1;
            rtcDate.date  = timeinfo.tm_mday;
            M5.Rtc.setDate(&rtcDate);

            // --- THE HACKATHON FIX: AUTO-SET ALARM 2 MINUTES FROM NOW ---
            targetHour = timeinfo.tm_hour;
            targetMinute = (timeinfo.tm_min + 2) % 60;
            if (targetMinute < 2) { // If it rolled over the hour
                targetHour = (targetHour + 1) % 24;
            }
        } else {
            M5.Lcd.setCursor(10, 60);
            M5.Lcd.print("SYNC FAILED.");
            delay(2000);
        }
        
        M5.Lcd.fillScreen(BLACK);
        updateDashboard(); 
        
        M5.Lcd.setCursor(10, 80);
        M5.Lcd.print("PAIR: FETCHING...");
        registerDevice();
        updateDashboard();
    }
}

// ============================================================
// MAIN LOOP
// ============================================================
void loop() {
    M5.update(); 
    unsigned long now = millis();

    // ==========================================
    // 1. NORMAL DASHBOARD & TELEMETRY STATE
    // ==========================================
    if (currentState == UI_NORMAL) {
        
        m5::rtc_time_t rtcTime;
        M5.Rtc.getTime(&rtcTime);

        if (rtcTime.hours != targetHour) complianceLogged = false;

        if (rtcTime.hours == targetHour && rtcTime.minutes == targetMinute && !complianceLogged) {
            currentState = UI_RINGING;
            lastScreen = 0; 
            return;
        }

        if (M5.BtnB.isPressed()) {
            currentState = UI_MENU;
            menuIndex = 0;
            lastScreen = 0; 
            delay(200); 
            return;
        }

        if (WiFi.status() == WL_CONNECTED && pairCode.length() != 4 && now - lastPairAttempt >= 5000) {
            lastPairAttempt = now;
            registerDevice();
        }

        float ax, ay, az, gx, gy, gz;
        M5.Imu.getAccelData(&ax, &ay, &az);
        M5.Imu.getGyroData(&gx, &gy, &gz);

        if (now - lastSend >= SEND_INTERVAL) {
            lastSend = now;
            if (WiFi.status() == WL_CONNECTED) {
                HTTPClient http;
                WiFiClient plainClient;
                WiFiClientSecure secureClient;
                String url = String(STITHI_SERVER_BASE_URL) + "/imu";
                
                if (beginHttp(http, plainClient, secureClient, url)) {
                    http.addHeader("Content-Type", "application/json");
                    http.setTimeout(1500); 
                    String json = "{\"timestamp\":" + String(now) + ",\"pair_code\":\"" + pairCode + "\"" +
                                  ",\"ax\":" + String(ax, 3) + ",\"ay\":" + String(ay, 3) + ",\"az\":" + String(az, 3) + 
                                  ",\"gx\":" + String(gx, 3) + ",\"gy\":" + String(gy, 3) + ",\"gz\":" + String(gz, 3) + "}";
                    http.POST(json);
                    http.end();
                }
            }
        }

        if (now - lastScreen >= SCREEN_INTERVAL) {
            lastScreen = now;
            updateDashboard();
        }
    }

    // ==========================================
    // 2. ALARM RINGING STATE (LOOPING BEEP)
    // ==========================================
    else if (currentState == UI_RINGING) {
        
        if (now - lastScreen > 1000) {
            M5.Lcd.fillScreen(RED);
            M5.Lcd.setTextColor(WHITE);
            M5.Lcd.setTextSize(3);
            M5.Lcd.setCursor(30, 30);
            M5.Lcd.print("OTAGO!");
            M5.Lcd.setTextSize(2);
            M5.Lcd.setCursor(15, 80);
            M5.Lcd.print("PRESS BTN B");
            
            M5.Speaker.tone(1500, 400); 
            lastScreen = now;
        }

        if (M5.BtnB.isPressed()) {
            currentState = UI_NORMAL;
            complianceLogged = true;
            
            M5.Lcd.fillScreen(GREEN);
            M5.Lcd.setTextColor(BLACK);
            M5.Lcd.setCursor(20, 50);
            M5.Lcd.print("LOGGED!");

            if (WiFi.status() == WL_CONNECTED) {
                HTTPClient http;
                WiFiClient plainClient;
                WiFiClientSecure secureClient;
                String url = String(STITHI_SERVER_BASE_URL) + "/compliance";
                if (beginHttp(http, plainClient, secureClient, url)) {
                    http.addHeader("Content-Type", "application/json");
                    http.setTimeout(3000);
                    String payload = "{\"device_id\":\"m5-" + WiFi.macAddress() + "\", \"event\":\"otago_start\", \"pair_code\":\"" + pairCode + "\"}";
                    http.POST(payload);
                    http.end();
                }
            }
            delay(2000); 
            lastScreen = 0; 
        }
    }

    // ==========================================
    // 3. MENU STATE
    // ==========================================
    else if (currentState == UI_MENU) {
        if (now - lastScreen > 100) {
            M5.Lcd.fillScreen(BLACK);
            M5.Lcd.setTextSize(2);
            M5.Lcd.setTextColor(WHITE);
            M5.Lcd.setCursor(40, 20);
            M5.Lcd.print("- MENU -");
            
            M5.Lcd.setCursor(20, 60);
            M5.Lcd.setTextColor(menuIndex == 0 ? GREEN : WHITE);
            M5.Lcd.print("1. ALARM");
            
            M5.Lcd.setCursor(20, 90);
            M5.Lcd.setTextColor(menuIndex == 1 ? GREEN : WHITE);
            M5.Lcd.print("2. BACK");
            
            lastScreen = now;
        }

        if (M5.BtnB.wasPressed()) {
            menuIndex = (menuIndex + 1) % 2;
            lastScreen = 0;
        }

        if (M5.BtnA.wasPressed()) {
            if (menuIndex == 0) {
                currentState = UI_SET_HOUR;
                m5::rtc_time_t rtcTime;
                M5.Rtc.getTime(&rtcTime);
                tempHour = rtcTime.hours;
                holdTriggered = false;
            } else {
                currentState = UI_NORMAL;
            }
            lastScreen = 0;
        }
    }

    // ==========================================
    // 4. SET HOUR STATE
    // ==========================================
    else if (currentState == UI_SET_HOUR) {
        if (now - lastScreen > 100) {
            M5.Lcd.fillScreen(BLACK);
            M5.Lcd.setTextSize(2);
            M5.Lcd.setTextColor(WHITE);
            M5.Lcd.setCursor(50, 10);
            M5.Lcd.print("SET HOUR");
            
            M5.Lcd.setTextSize(5);
            M5.Lcd.setTextColor(ORANGE);
            M5.Lcd.setCursor(50, 45);
            M5.Lcd.printf("%02d", tempHour);
            
            M5.Lcd.setTextSize(1);
            M5.Lcd.setTextColor(WHITE);
            M5.Lcd.setCursor(15, 110);
            M5.Lcd.print("Press M5: +1 | Hold 5s: Next");
            lastScreen = now;
        }

        if (M5.BtnA.wasReleased()) {
            if (!holdTriggered) {
                tempHour = (tempHour + 1) % 24;
                lastScreen = 0;
            }
            holdTriggered = false; 
        }

        if (M5.BtnA.pressedFor(5000) && !holdTriggered) {
            holdTriggered = true;
            currentState = UI_SET_MINUTE;
            
            m5::rtc_time_t rtcTime;
            M5.Rtc.getTime(&rtcTime);
            tempMinute = rtcTime.minutes;

            M5.Speaker.tone(2000, 150); 
            lastScreen = 0;
        }
    }

    // ==========================================
    // 5. SET MINUTE STATE
    // ==========================================
    else if (currentState == UI_SET_MINUTE) {
        if (now - lastScreen > 100) {
            M5.Lcd.fillScreen(BLACK);
            M5.Lcd.setTextSize(2);
            M5.Lcd.setTextColor(WHITE);
            M5.Lcd.setCursor(40, 10);
            M5.Lcd.print("SET MINUTE");
            
            M5.Lcd.setTextSize(5);
            M5.Lcd.setTextColor(ORANGE);
            M5.Lcd.setCursor(50, 45);
            M5.Lcd.printf("%02d", tempMinute);
            
            M5.Lcd.setTextSize(1);
            M5.Lcd.setTextColor(WHITE);
            M5.Lcd.setCursor(15, 110);
            M5.Lcd.print("Press M5: +1 | Hold 5s: Save");
            lastScreen = now;
        }

        if (M5.BtnA.wasReleased()) {
            if (!holdTriggered) {
                tempMinute = (tempMinute + 1) % 60;
                lastScreen = 0;
            }
            holdTriggered = false;
        }

        if (M5.BtnA.pressedFor(5000) && !holdTriggered) {
            holdTriggered = true;
            targetHour = tempHour;
            targetMinute = tempMinute;
            complianceLogged = false; 
            
            M5.Speaker.tone(2000, 200); 
            M5.Lcd.fillScreen(GREEN);
            M5.Lcd.setTextColor(BLACK);
            M5.Lcd.setTextSize(3);
            M5.Lcd.setCursor(30, 50);
            M5.Lcd.print("SAVED!");
            delay(1500);
            
            currentState = UI_NORMAL;
            lastScreen = 0;
        }
    }
}
