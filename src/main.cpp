#include <Arduino.h>
#include <time.h>
#include <MyLD2410.h>
#include <Firebase_ESP_Client.h>
#include <FreeRTOS.h>
#include <task.h>
#include <WiFi.h>
#include <Preferences.h>
#include <WiFiManager.h>
#include <Adafruit_NeoPixel.h>
#include "INA226.h"
#include "addons/TokenHelper.h"
#include "addons/RTDBHelper.h"
#include "queue.h"
#include "RandomForestModel1.h"

// ─────────────────────────────────────────────────────────────
//  CREDENTIALS
// ─────────────────────────────────────────────────────────────
#define USER_EMAIL    "esp32_radar@gmail.com"
#define USER_PASSWORD "55555555"
#define API_KEY       "AIzaSyBqQAhZefoX7XX6NRjH3wSGXzPNM0dpN6c"
#define DATABASE_URL  "https://j-b2103-default-rtdb.asia-southeast1.firebasedatabase.app/"

// ─────────────────────────────────────────────────────────────
//  PIN & HARDWARE
// ─────────────────────────────────────────────────────────────
#define LED_WIFI      48
#define NUM_LED        1
#define LED_DIM        2
#define RADAR_RX_PIN  18
#define RADAR_TX_PIN  17
#define I2C_SDA        8
#define I2C_SCL        9
#define PWM_FREQ    5000
#define PWM_BIT        8
#define PWM_CHANEL     0
#define MAX_GATES      9

// ─────────────────────────────────────────────────────────────
//  AI
// ─────────────────────────────────────────────────────────────
#define AI_N_FEATURES  12

// Hysteresis — giảm độ nhạy
// ON : phải có người liên tiếp ON_CONFIRM frame → bật đèn
// OFF: phải không có người liên tiếp OFF_CONFIRM frame → tắt đèn
// Tăng hai số này lên → đèn càng khó bật/tắt (ít giật hơn)
#define ON_CONFIRM   30   // frame liên tiếp cần để BẬT
#define OFF_CONFIRM   7    // frame liên tiếp cần để TẮT
#define NOISE_CONFIRM 8     // frame liên tiếp để nhận nhãn NOISE là thật

// ─────────────────────────────────────────────────────────────
//  LOGIC
// ─────────────────────────────────────────────────────────────
#define TIMEOUT_NO_PERSON 10000UL  // thời gian tắt đèn
#define FADE_SPEED_MS     20     

// Giá trị mặc định config 
#define DEFAULT_BRIGHT_ACTIVE     100  
#define DEFAULT_BRIGHT_STILL      150  
#define DEFAULT_BRIGHT_NO_PERSON    0  
#define DEFAULT_LUX_THRESHOLD      50  

// ─────────────────────────────────────────────────────────────
//  STRUCTS
// ─────────────────────────────────────────────────────────────
struct RadarData {
    int     movingCm;
    int     staticCm;
    int     lightvalue;
    uint8_t movingE[MAX_GATES];
    uint8_t staticE[MAX_GATES];
    bool    presence;
};

// Dữ liệu chia sẻ giữa các task
struct SystemData {
    bool    presence;
    uint8_t lamp_percent;   // độ sáng đèn hiện tại (%)
    uint8_t env_lux;        // ánh sáng môi trường
    uint8_t ai_label;       // 0-3
    bool    measure_only;   // true = trời sáng, không bật đèn
};

struct PowerData {
    float  voltage;
    float  current;
    float  power;
    double energy_Wh;
};

// Cấu hình người dùng — đọc từ Firebase, lưu Preferences
struct DeviceConfig {
    uint8_t bright_active;      // % khi ACTIVE (label 3)
    uint8_t bright_still;       // % khi STILL  (label 1)
    uint8_t bright_no_person;   // % khi NO_PERSON (0=tắt hẳn)
    uint8_t lux_threshold;      // ngưỡng môi trường bật đèn
    bool measure_mode;
    bool    changed;            // cờ: có thay đổi cần lưu flash không
};

enum systemstate {
    state_off, state_fade_in, state_on, state_fade_out, state_manual
};

// ─────────────────────────────────────────────────────────────
//  GLOBAL OBJECTS
// ─────────────────────────────────────────────────────────────
Adafruit_NeoPixel pixels(NUM_LED, LED_WIFI, NEO_GRB + NEO_KHZ800);
FirebaseData  fb_data_read;
FirebaseData  fb_data_write;
FirebaseAuth  user;
FirebaseConfig config;
MyLD2410  radar(Serial1);
INA226    INA(0x40);
Preferences flash;
Eloquent::ML::Port::RandomForest aiClassifier;

// ─────────────────────────────────────────────────────────────
//  SHARED STATE & RTC MEMORY
// ─────────────────────────────────────────────────────────────
RTC_DATA_ATTR double rtc_energy_Wh = 0;
RTC_DATA_ATTR uint32_t rtc_energy_magic = 0;

SystemData   coreData;
PowerData    systemPowerData;
DeviceConfig devConfig;

bool    ismanual          = false;
uint8_t brightness_manual = 0;

SemaphoreHandle_t coreDataMutex      = NULL;
SemaphoreHandle_t powerMutex         = NULL;
SemaphoreHandle_t manualMutex        = NULL;
SemaphoreHandle_t configMutex        = NULL;

QueueHandle_t      radarQueue;
EventGroupHandle_t wifiEventGroup;
#define wifi_connected_bit (1 << 0)


TaskHandle_t       taskHandle_radar_t = NULL;
TaskHandle_t       taskHandle_WiFi_t  = NULL;
TaskHandle_t       taskHandle_fbdb_t  = NULL;

// ─────────────────────────────────────────────────────────────
//  FORWARD DECLARATIONS
// ─────────────────────────────────────────────────────────────
void Task_read_Radar      (void* pv);
void task_processing_data (void* pv);
void Task_connect_Wifi    (void* pv);
void Task_sent_fbdb       (void* pv);
void task_ina226          (void* pv);
void task_led_wifi        (void* pv);
void Wifi_event           (WiFiEvent_t event);
void load_config_from_flash();
void save_config_to_flash(const DeviceConfig& cfg);


void load_config_from_flash() {
    flash.begin("cfg", true);   
    devConfig.bright_active    = flash.getUChar("b_active",  DEFAULT_BRIGHT_ACTIVE);
    devConfig.bright_still     = flash.getUChar("b_still",   DEFAULT_BRIGHT_STILL);
    devConfig.bright_no_person = flash.getUChar("b_nop",     DEFAULT_BRIGHT_NO_PERSON);
    devConfig.lux_threshold    = flash.getUChar("lux_thr",   DEFAULT_LUX_THRESHOLD);
    devConfig.measure_mode     = flash.getBool("mea_mode", false);
    devConfig.changed          = false;
    flash.end();
    Serial.printf("[CFG] Load: active=%d still=%d nop=%d lux=%d\n",
        devConfig.bright_active, devConfig.bright_still,
        devConfig.bright_no_person, devConfig.lux_threshold);
}

void save_config_to_flash(const DeviceConfig& cfg) {
    flash.begin("cfg", false);  // read-write
    flash.putUChar("b_active",  cfg.bright_active);
    flash.putUChar("b_still",   cfg.bright_still);
    flash.putUChar("b_nop",     cfg.bright_no_person);
    flash.putUChar("lux_thr",   cfg.lux_threshold);
    flash.putBool("mea_mode", cfg.measure_mode);
    flash.end();
    Serial.println("[CFG] Saved to flash");
}
// ═════════════════════════════════════════════════════════════
//  TASK 1 — ĐỌC RADAR LD2410C
// ═════════════════════════════════════════════════════════════
void Task_read_Radar(void* pv) {
    RadarData currentData;
    Serial1.begin(256000, SERIAL_8N1, RADAR_RX_PIN, RADAR_TX_PIN);
    radar.begin();
    radar.enhancedMode(true);  // bậc chế độ đọc sâu

    for (byte i = 0; i <= 8; i++) {
        radar.setMovingThreshold   (i, 20);
        radar.setStationaryThreshold(i, 20);
    }
    radar.requestAuxConfig();

    while (1) {
        if (radar.check() == MyLD2410::DATA) {
            currentData = {};
            currentData.lightvalue = radar.getLightLevel();
            currentData.presence   = radar.presenceDetected();

            if (currentData.presence) {
                if (radar.stationaryTargetDetected())
                    currentData.staticCm = radar.stationaryTargetDistance();
                if (radar.movingTargetDetected())
                    currentData.movingCm = radar.movingTargetDistance();
            }

            int mIdx = 0;
            radar.getMovingSignals().forEach([&](byte val) {
                if (mIdx < MAX_GATES) currentData.movingE[mIdx++] = val;
            });
            int sIdx = 0;
            radar.getStationarySignals().forEach([&](byte val) {
                if (sIdx < MAX_GATES) currentData.staticE[sIdx++] = val;
            });

            xQueueSend(radarQueue, &currentData, pdMS_TO_TICKS(10));
        }
        vTaskDelay(pdMS_TO_TICKS(50));
    }
}

// ═════════════════════════════════════════════════════════════
//  TASK 2 — XỬ LÝ AI + ĐIỀU KHIỂN ĐÈN
// ═════════════════════════════════════════════════════════════
void task_processing_data(void* pv) {
    RadarData recvData;
    systemstate currentState = state_off;

    // Dimming
    uint8_t brightness    = 0;
    uint8_t targetPercent = 0;
    unsigned long lastfadetick = 0;

    // Timeout
    unsigned long lastDetectedTime = 0;

    // Hysteresis counters
    uint8_t onCounter   = 0;   // frame liên tiếp phát hiện người
    uint8_t offCounter  = 0;   // frame liên tiếp không người
    uint8_t noiseCounter= 0;   // frame liên tiếp label NOISE

    // Trạng thái ổn định (sau khi lọc hysteresis)
    bool    confirmedPresence = false;
    uint8_t confirmedLabel    = 0;
    uint8_t rawLabel          = 0;

    while (1) {
        // ── 1. ĐỌC MANUAL ───────────────────────────────────
        bool    safe_manual  = false;
        uint8_t safe_bright  = 0;
        if (manualMutex &&
            xSemaphoreTake(manualMutex, pdMS_TO_TICKS(10)) == pdTRUE) {
            
            // Ép về chế độ Auto (tự động) nếu rớt mạng
            if (!(xEventGroupGetBits(wifiEventGroup) & wifi_connected_bit)) {
                ismanual = false;
            }

            safe_manual = ismanual;
            safe_bright = brightness_manual;
            xSemaphoreGive(manualMutex);
        }

        // ── 2. ĐỌC CONFIG ───────────────────────────────────
        DeviceConfig localCfg;
        if (configMutex &&
            xSemaphoreTake(configMutex, pdMS_TO_TICKS(10)) == pdTRUE) {
            localCfg = devConfig;
            xSemaphoreGive(configMutex);
        }

        // ── 3. CHUYỂN MANUAL ↔ AUTO ─────────────────────────
        if (safe_manual && currentState != state_manual) {
            currentState = state_manual;
        } else if (!safe_manual && currentState == state_manual) {
            currentState  = state_fade_out;
            onCounter     = 0;
            offCounter    = 0;
        }

        // ── 4. ĐỌC RADAR & CHẠY AI ──────────────────────────
        bool gotFrame = (xQueueReceive(radarQueue, &recvData, 0) == pdTRUE);

        if (gotFrame) {

            // 4a. Build features[12]
            float features[AI_N_FEATURES];
            uint32_t totalMoving = 0;
            uint8_t  maxMoving   = 0;
            for (int i = 0; i < MAX_GATES; i++) {
                totalMoving += recvData.movingE[i];
                if (recvData.movingE[i] > maxMoving) maxMoving = recvData.movingE[i];
            }
            uint32_t totalStatic = 0;
            for (int i = 0; i < MAX_GATES; i++) totalStatic += recvData.staticE[i];

            features[0]  = recvData.movingE[2];
            features[1]  = recvData.movingE[3];
            features[2]  = recvData.movingE[4];
            features[3]  = recvData.staticE[3];
            features[4]  = recvData.staticE[4];
            features[5]  = recvData.staticE[5];
            features[6]  = recvData.staticE[6];
            features[7]  = recvData.staticE[7];
            features[8]  = recvData.staticE[8];
            features[9]  = (float)totalMoving;
            features[10] = (float)totalStatic;
            features[11] = (float)maxMoving;

            // 4b. Predict
            rawLabel = (uint8_t)aiClassifier.predict(features);

            // 4c. ───  NHÃN ────────────────────
            //
            //  Label NOISE (2): phải liên tiếp NOISE_CONFIRM frame
            //  mới được chấp nhận, nếu không giữ nhãn confirmedLabel cũ
            if (rawLabel == 2) {
                noiseCounter++;
                if (noiseCounter < NOISE_CONFIRM) {
                    rawLabel = confirmedLabel;
                } else {
                    noiseCounter = NOISE_CONFIRM; 
                }
            } else {
                noiseCounter = 0;
            }

            bool personInFrame = (rawLabel == 1 || rawLabel == 3);

            if (personInFrame) {
                onCounter++;
                if (onCounter > ON_CONFIRM) onCounter = ON_CONFIRM;
                offCounter = 0;
            } else {
                offCounter++;
                if (offCounter > OFF_CONFIRM) offCounter = OFF_CONFIRM;
                onCounter = 0;
            }

            // Chỉ thay đổi trạng thái khi đủ ngưỡng
            if (!confirmedPresence && onCounter >= ON_CONFIRM) {
                confirmedPresence = true;
                confirmedLabel    = rawLabel;
            } else if (confirmedPresence && offCounter >= OFF_CONFIRM) {
                confirmedPresence = false;
                confirmedLabel    = 0;
            } else if (confirmedPresence && personInFrame) {
                // Cập nhật loại nhãn (1 vs 3) khi đang ở trạng thái ON
                confirmedLabel = rawLabel;
            }

            // 4d. Kiểm tra ánh sáng môi trường
            // Sử dụng bộ lọc hysteresis để loại bỏ nhiễu ánh sáng do chính đèn gây ra 
            // (tránh hiện tượng chập chờn bật/tắt liên tục do đèn hắt vào cảm biến)
            static bool isEnvironmentBright = false;
            
            int offset = 0;
            if (currentState == state_on || currentState == state_fade_in) {
                offset = 40; // Bù trừ độ sáng mà đèn hắt ngược vào cảm biến (VD: 40 lux)
            }

            if (!isEnvironmentBright && recvData.lightvalue >= (localCfg.lux_threshold + offset)) {
                isEnvironmentBright = true;
            } else if (isEnvironmentBright && recvData.lightvalue < (localCfg.lux_threshold - 5)) { // Trừ 5 để tạo thêm độ trễ tĩnh
                isEnvironmentBright = false;
            }
            
            // measure_only = true (không bật đèn) khi bị ép chế độ đo trên Firebase HOẶC môi trường tự nhiên đã đủ sáng
            bool measureOnly = localCfg.measure_mode || isEnvironmentBright;

            // 4e. Cập nhật timeout
            if (confirmedPresence) lastDetectedTime = millis();

            // 4f. Chuyển state FSM
            if (!measureOnly) {
                if (confirmedPresence) {
                    if (currentState == state_off || currentState == state_fade_out)
                        currentState = state_fade_in;
                } else {
                    if (millis() - lastDetectedTime > TIMEOUT_NO_PERSON) {
                        if (currentState == state_on || currentState == state_fade_in)
                            currentState = state_fade_out;
                    }
                }
            } else {
                // Đủ sáng → tắt đèn nếu đang bật
                if (currentState == state_on || currentState == state_fade_in)
                    currentState = state_fade_out;
            }

            // 4g. Cập nhật coreData (trừ lamp_percent — sẽ cập nhật sau PWM fade)
            if (coreDataMutex &&
                xSemaphoreTake(coreDataMutex, pdMS_TO_TICKS(10)) == pdTRUE) {
                coreData.presence     = confirmedPresence;
                coreData.env_lux      = (uint8_t)recvData.lightvalue;
                coreData.ai_label     = confirmedLabel;
                coreData.measure_only = measureOnly;
                xSemaphoreGive(coreDataMutex);
            }
        }

        // ── 5. XÁC ĐỊNH TARGET BRIGHTNESS ───────────────────
        if (currentState == state_manual) {
            targetPercent = safe_bright;
        } else if (currentState == state_off || currentState == state_fade_out) {
            targetPercent = localCfg.bright_no_person;
        } else if (currentState == state_on || currentState == state_fade_in) {
            switch (confirmedLabel) {
                case 3:  targetPercent = localCfg.bright_active;    break;  // di chuyển
                case 1:  targetPercent = localCfg.bright_still;     break;  // đứng yên
                default: targetPercent = localCfg.bright_still;     break;
            }
        }

        // ── 6. PWM dimmer ──────────────────────────────────────
        if (brightness != targetPercent) {
            if (millis() - lastfadetick >= FADE_SPEED_MS) {
                lastfadetick = millis();
                if (brightness < targetPercent) brightness++;
                else                            brightness--;
                ledcWrite(PWM_CHANEL, (uint32_t)brightness * 255 / 100);
            }
        } else {
            if (currentState == state_fade_in)  currentState = state_on;
            if (currentState == state_fade_out) currentState = state_off;
        }
        // ── 7. ĐỒNG BỘ lamp_percent SAU KHI dimmer ────────────
        if (coreDataMutex &&
            xSemaphoreTake(coreDataMutex, pdMS_TO_TICKS(10)) == pdTRUE) {
            coreData.lamp_percent = brightness;
            xSemaphoreGive(coreDataMutex);
        }

        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

// ═════════════════════════════════════════════════════════════
//  TASK 3 — INA226
// ═════════════════════════════════════════════════════════════
void task_ina226(void* pv) {
    if (!INA.begin()) {
        Serial.println("[INA226] I2C thất bại");
        vTaskDelete(NULL);
    }
    INA.setMaxCurrentShunt(4, 0.01);

    float f_vol = 0, f_cur = 0, f_pow = 0;
    const float A = 0.2f;
    unsigned long lastTime = millis();
    bool wasConnected = true;

    while (1) {
        static unsigned long lastSave = millis();
        float vol = INA.getBusVoltage();
        float cur = INA.getCurrent();
        float pow = INA.getPower();

        if (f_vol == 0) { f_vol = vol; f_cur = cur; f_pow = pow; }
        else {
            f_vol = A * vol + (1.f - A) * f_vol;
            f_cur = A * cur + (1.f - A) * f_cur;
            f_pow = A * pow + (1.f - A) * f_pow;
        }

        unsigned long now = millis();
        double E = (f_pow / 1000.0) * ((now - lastTime) / 3600000.0);
        lastTime = now;

        double current_E_wh = 0;
        if (powerMutex &&
            xSemaphoreTake(powerMutex, pdMS_TO_TICKS(10)) == pdTRUE) {
            systemPowerData.voltage    = f_vol;
            systemPowerData.current    = f_cur;
            systemPowerData.power      = f_pow;
            systemPowerData.energy_Wh += E;
            current_E_wh = systemPowerData.energy_Wh;
            xSemaphoreGive(powerMutex);
        } else {
            current_E_wh = systemPowerData.energy_Wh;
        }

        // Lưu vào RTC RAM (tốc độ ánh sáng, 0 hao mòn, giữ nguyên qua mọi Crash/Watchdog Reset)
        rtc_energy_Wh = current_E_wh;
        rtc_energy_magic = 0x12345678;

        // Kiểm tra xem có mất kết nối không để lưu Flash kịp thời (dự phòng)
        bool isConnected = (xEventGroupGetBits(wifiEventGroup) & wifi_connected_bit) != 0;
        if (wasConnected && !isConnected) {
            flash.begin("energy", false);
            flash.putDouble("energy", current_E_wh);
            flash.end();
            Serial.println("[INA226] Đã lưu E_Wh do mất mạng!");
            lastSave = millis();
        }
        wasConnected = isConnected;

        // Lưu định kỳ mỗi 1 phút (Để tối ưu tuổi thọ Flash theo yêu cầu)
        if (millis() - lastSave > 60000UL) {
            flash.begin("energy", false);
            flash.putDouble("energy", current_E_wh);
            flash.end();
            lastSave = millis();
        }
        vTaskDelay(pdMS_TO_TICKS(3000));
    }
}

// ═════════════════════════════════════════════════════════════
//  TASK 4 — WIFI
// ═════════════════════════════════════════════════════════════
void Task_connect_Wifi(void* pv) {
    WiFiManager wm;
    wm.setConfigPortalTimeout(180);
    if (!wm.autoConnect("ESP32_RADAR")) {
        vTaskDelay(pdMS_TO_TICKS(3000));
        ESP.restart();
    }
    Serial.print("[WiFi] IP: ");
    Serial.println(WiFi.localIP());
    WiFi.mode(WIFI_STA);

    while (1) {
        vTaskDelay(pdMS_TO_TICKS(5000));
    }
}

void Wifi_event(WiFiEvent_t event) {
    switch (event) {
    case ARDUINO_EVENT_WIFI_STA_GOT_IP:
        xEventGroupSetBits(wifiEventGroup, wifi_connected_bit);
        configTime(7 * 3600, 0, "pool.ntp.org", "time.nist.gov"); // GMT+7
        break;
    case ARDUINO_EVENT_WIFI_STA_DISCONNECTED:
        xEventGroupClearBits(wifiEventGroup, wifi_connected_bit); break;
    default: break;
    }
}

// ═════════════════════════════════════════════════════════════
//  TASK 5 — LED WIFI
// ═════════════════════════════════════════════════════════════
void task_led_wifi(void* pv) {
    while (1) {
        bool ok = xEventGroupGetBits(wifiEventGroup) & wifi_connected_bit;
        pixels.setPixelColor(0, ok ? pixels.Color(0, 0, 255)
                                   : pixels.Color(255, 0, 0));
        pixels.show();
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

// ═════════════════════════════════════════════════════════════
//  TASK 6 — FIREBASE
// ═════════════════════════════════════════════════════════════
void Task_sent_fbdb(void* pv) {
    xEventGroupWaitBits(wifiEventGroup, wifi_connected_bit,
                        pdFALSE, pdTRUE, portMAX_DELAY);

    config.api_key               = API_KEY;
    config.database_url          = DATABASE_URL;
    config.token_status_callback = tokenStatusCallback;
    user.user.email              = USER_EMAIL;
    user.user.password           = USER_PASSWORD;

    Firebase.begin(&config, &user);
    Firebase.reconnectWiFi(true);

    FirebaseJson payload;
    const char* labelStr[] = {"no_person", "still", "noise", "active"};

    while (1) {
        xEventGroupWaitBits(wifiEventGroup, wifi_connected_bit,
                            pdFALSE, pdTRUE, portMAX_DELAY);

        if (!Firebase.ready()) {
            vTaskDelay(pdMS_TO_TICKS(2000));
            continue;
        }
        DeviceConfig localCfg;
        if (configMutex &&
            xSemaphoreTake(configMutex, pdMS_TO_TICKS(10)) == pdTRUE) {
            localCfg = devConfig;
            xSemaphoreGive(configMutex);
        }
        // ── A. ĐỌC ĐIỀU KHIỂN THỦ CÔNG (TỐI ƯU BẰNG JSON) ──
        bool    t_manual = ismanual;
        uint8_t t_bright = brightness_manual;

        if (Firebase.RTDB.getJSON(&fb_data_read, "/Control")) {
            FirebaseJson json;
            json.setJsonData(fb_data_read.jsonString());
            FirebaseJsonData jsonData;

            json.get(jsonData, "dieu_khien");
            if (jsonData.success) t_manual = jsonData.boolValue;

            json.get(jsonData, "do_sang");
            if (jsonData.success) t_bright = (uint8_t)jsonData.intValue;
        }

        if (manualMutex &&
            xSemaphoreTake(manualMutex, pdMS_TO_TICKS(10)) == pdTRUE) {
            ismanual          = t_manual;
            brightness_manual = t_bright;
            xSemaphoreGive(manualMutex);
        }
        // ── B. ĐỌC CONFIG TỪ FIREBASE & LƯU NẾU ĐỔI ────────
        {
            DeviceConfig newCfg = localCfg;
            bool cfgChanged = false;
            
            if (Firebase.RTDB.getJSON(&fb_data_read, "/Config")) {
                FirebaseJson json;
                json.setJsonData(fb_data_read.jsonString());
                FirebaseJsonData jsonData;
                
                json.get(jsonData, "measure_mode");
                if (jsonData.success) {
                    bool v = jsonData.boolValue;
                    if (v != newCfg.measure_mode) {
                        newCfg.measure_mode = v;
                        cfgChanged = true;
                        Serial.printf("[CFG] measure_mode = %s\n", v ? "ON" : "OFF");
                    }
                }

                json.get(jsonData, "bright_active");
                if (jsonData.success) {
                    uint8_t v = (uint8_t)constrain(jsonData.intValue, 0, 100);
                    if (v != newCfg.bright_active) { newCfg.bright_active = v; cfgChanged = true; }
                }

                json.get(jsonData, "bright_still");
                if (jsonData.success) {
                    uint8_t v = (uint8_t)constrain(jsonData.intValue, 0, 100);
                    if (v != newCfg.bright_still) { newCfg.bright_still = v; cfgChanged = true; }
                }

                json.get(jsonData, "bright_no_person");
                if (jsonData.success) {
                    uint8_t v = (uint8_t)constrain(jsonData.intValue, 0, 100);
                    if (v != newCfg.bright_no_person) { newCfg.bright_no_person = v; cfgChanged = true; }
                }

                json.get(jsonData, "lux_threshold");
                if (jsonData.success) {
                    uint8_t v = (uint8_t)constrain(jsonData.intValue, 0, 255);
                    if (v != newCfg.lux_threshold) { newCfg.lux_threshold = v; cfgChanged = true; }
                }
            }

            if (cfgChanged) {
                // Lưu vào flash ngay
                save_config_to_flash(newCfg);

                // Cập nhật global
                if (configMutex &&
                    xSemaphoreTake(configMutex, pdMS_TO_TICKS(10)) == pdTRUE) {
                    devConfig = newCfg;
                    devConfig.changed = false;
                    xSemaphoreGive(configMutex);
                }
            }
        }
        // ── C. LẤY DỮ LIỆU CỤC BỘ ──────────────────────────
        PowerData  lPow;
        SystemData lSys;

        if (powerMutex &&
            xSemaphoreTake(powerMutex, pdMS_TO_TICKS(10)) == pdTRUE) {
            lPow = systemPowerData;
            xSemaphoreGive(powerMutex);
        }
        if (coreDataMutex &&
            xSemaphoreTake(coreDataMutex, pdMS_TO_TICKS(10)) == pdTRUE) {
            lSys = coreData;
            xSemaphoreGive(coreDataMutex);
        }

        // ── D. GỬI LÊN FIREBASE ──────────────────────────────
        payload.clear();

        // Ánh sáng
        payload.set("light/env_lux",      (int)lSys.env_lux);
        payload.set("light/lamp_percent", (int)lSys.lamp_percent);
        payload.set("light/mode",         lSys.measure_only ? "bright" : "dark");
        payload.set("light/measure_mode", lSys.measure_only);
        payload.set("light/lux_threshold",(int)localCfg.lux_threshold);

        // Chế độ điều khiển (manual/auto) + % độ sáng manual
        bool localManual = false;
        uint8_t localManualBright = 0;
        if (manualMutex &&
            xSemaphoreTake(manualMutex, pdMS_TO_TICKS(10)) == pdTRUE) {
            localManual      = ismanual;
            localManualBright = brightness_manual;
            xSemaphoreGive(manualMutex);
        }
        payload.set("control/is_manual",       localManual);
        payload.set("control/manual_brightness",(int)localManualBright);

        // AI
        uint8_t safeLabel = lSys.ai_label < 4 ? lSys.ai_label : 0;
        payload.set("ai/label_name", labelStr[safeLabel]);

        // ── HEARTBEAT — gửi timestamp để web phát hiện mất kết nối ──
        payload.set("heartbeat/timestamp", (unsigned long)millis());
        payload.set("heartbeat/uptime_s",  (unsigned long)(millis() / 1000));

        // 1. Gửi telemetry chính (SET — ghi đè, web đọc realtime)
        if (!Firebase.RTDB.setJSON(&fb_data_write,
                                    "/SmartNode_01/telemetry", &payload)) {
            Serial.println(fb_data_write.errorReason());
        }

        // 2. Lấy thời gian chung cho History
        struct tm timeinfo;
        bool hasTime = getLocalTime(&timeinfo, 10);
        char timeStringBuff[50] = "1970-01-01 00:00:00";
        if (hasTime) {
            strftime(timeStringBuff, sizeof(timeStringBuff), "%Y-%m-%d %H:%M:%S", &timeinfo);
        }

        // 3. Gửi lịch sử Time-series cho AI Label (Cách ~1 giây một lần — PUSH)
        FirebaseJson historyLabel;
        historyLabel.set("label", (int)safeLabel);
        historyLabel.set("time", timeStringBuff);
        
        if (!Firebase.RTDB.pushJSON(&fb_data_write,
                                    "/SmartNode_01/history_label", &historyLabel)) {
            Serial.println(fb_data_write.errorReason());
        }

        // 4. Gửi lịch sử Time-series cho Energy (E_wh) (Cách 1 phút một lần — PUSH)
        static unsigned long lastEnergyPush = 0;
        if (millis() - lastEnergyPush >= 60000UL || lastEnergyPush == 0) {
            FirebaseJson historyEnergy;
            historyEnergy.set("E_wh", lPow.energy_Wh);
            historyEnergy.set("time", timeStringBuff);

            if (!Firebase.RTDB.pushJSON(&fb_data_write,
                                        "/SmartNode_01/history_energy", &historyEnergy)) {
                Serial.println(fb_data_write.errorReason());
            } else {
                lastEnergyPush = millis();
            }
        }

        vTaskDelay(pdMS_TO_TICKS(1000));  // ~1 giây / lần → web watchdog nhanh hơn
    }
}

// ═════════════════════════════════════════════════════════════
//  SETUP
// ═════════════════════════════════════════════════════════════
void setup() {
    Serial.begin(115200);

    load_config_from_flash();

    flash.begin("energy", true);
    double saved_energy = flash.getDouble("energy", 0.0);
    flash.end();

    // Khắc phục WDT Reset: Nếu RTC RAM còn nguyên và lớn hơn bộ nhớ cứng, ưu tiên dùng RTC (vì RTC liên tục cập nhật tới tấp không sợ hao mòn)
    if (rtc_energy_magic == 0x12345678 && rtc_energy_Wh >= saved_energy) {
        systemPowerData.energy_Wh = rtc_energy_Wh;
        Serial.printf("[BOOT] Phục hồi %f Wh từ RTC RAM (Sống sót sau khi sập)\n", rtc_energy_Wh);
    } else {
        systemPowerData.energy_Wh = saved_energy;
        rtc_energy_Wh = saved_energy;
        rtc_energy_magic = 0x12345678;
        Serial.printf("[BOOT] Tải %f Wh từ Flash NVS\n", saved_energy);
    }

    Wire.begin(I2C_SDA, I2C_SCL);

    wifiEventGroup = xEventGroupCreate();
    WiFi.onEvent(Wifi_event);

    pixels.begin();
    pixels.setBrightness(10);

    powerMutex    = xSemaphoreCreateMutex();
    coreDataMutex = xSemaphoreCreateMutex();
    manualMutex   = xSemaphoreCreateMutex();
    configMutex   = xSemaphoreCreateMutex();

    ledcSetup(PWM_CHANEL, PWM_FREQ, PWM_BIT);
    ledcAttachPin(LED_DIM, PWM_CHANEL);

    radarQueue = xQueueCreate(10, sizeof(RadarData));

    // Core 1
    xTaskCreatePinnedToCore(Task_read_Radar,      "Radar",    4096,  NULL, 4, &taskHandle_radar_t, 1);
    xTaskCreatePinnedToCore(task_processing_data, "Process",  8192,  NULL, 3, NULL,                1);
    xTaskCreatePinnedToCore(task_ina226,          "INA226",   4096,  NULL, 2, NULL,                1);
    xTaskCreatePinnedToCore(task_led_wifi,        "LED_WiFi", 2048,  NULL, 1, NULL,                1);
    // Core 0
    xTaskCreatePinnedToCore(Task_connect_Wifi,    "WiFi",     20480, NULL, 2, &taskHandle_WiFi_t,  0);
    xTaskCreatePinnedToCore(Task_sent_fbdb,       "Firebase", 8192,  NULL, 1, &taskHandle_fbdb_t,  0);
}

void loop() {}