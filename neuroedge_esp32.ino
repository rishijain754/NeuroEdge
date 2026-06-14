/**
 * ═══════════════════════════════════════════════════════════════════════
 *  NeuroEdge — ESP32 Neuromorphic Edge Intelligence Firmware
 *  Version: 1.0
 * ═══════════════════════════════════════════════════════════════════════
 *
 *  HARDWARE REQUIRED:
 *  ──────────────────
 *  - ESP32 DevKit V1 (or any 38-pin ESP32)
 *  - DHT22 Temperature & Humidity Sensor
 *  - MQ135 Air Quality Sensor (analog output)
 *  - PIR Motion Sensor (HC-SR501 or similar)
 *  - 16x2 LCD Display (I2C, address 0x27)
 *  - Piezo Buzzer
 *  - Green LED (Status)
 *  - Red LED (Alert)
 *  - 220Ω resistors for LEDs
 *
 *  PIN MAPPING:
 *  ─────────────
 *  DHT22     → GPIO 15
 *  MQ135     → GPIO 34 (ADC1)
 *  PIR       → GPIO 27
 *  LCD SDA   → GPIO 21
 *  LCD SCL   → GPIO 22
 *  Buzzer    → GPIO 25
 *  Green LED → GPIO 26
 *  Red LED   → GPIO 2 (built-in LED)
 *
 *  REQUIRED LIBRARIES (install via Arduino IDE Library Manager):
 *  ─────────────────────────────────────────────────────────────
 *  - DHT sensor library by Adafruit
 *  - Adafruit Unified Sensor
 *  - LiquidCrystal I2C by Frank de Brabander
 *
 * ═══════════════════════════════════════════════════════════════════════
 */

#include <DHT.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// ─── PIN DEFINITIONS ──────────────────────────────────────────────────
#define DHT_PIN       15
#define MQ135_PIN     34
#define PIR_PIN       27
#define BUZZER_PIN    25
#define LED_GREEN     26
#define LED_RED       2

// ─── SENSOR SETUP ────────────────────────────────────────────────────
#define DHT_TYPE DHT22
DHT dht(DHT_PIN, DHT_TYPE);

// LCD with I2C — address 0x27, 16 columns, 2 rows
LiquidCrystal_I2C lcd(0x27, 16, 2);

// ─── LIF NEURON STRUCTURE ─────────────────────────────────────────────
// Leaky Integrate-and-Fire model for each sensor channel.
// Each sensor feeds one neuron. When membrane_potential crosses
// threshold, the neuron "spikes" — triggering a tier escalation.
struct LIFNeuron {
  float membrane_potential; // Current charge accumulated
  float threshold;          // Voltage needed to fire
  float decay;              // How fast charge leaks out (0–1)
  int   refractory_counter; // Ticks remaining before it can fire again
  int   refractory_period;  // How many ticks to stay silent after spike
  bool  spiked;             // True if neuron fired this tick
};

// ─── NEURON INSTANCES ─────────────────────────────────────────────────
LIFNeuron neuron_temp   = {0, 1.0, 0.85, 0, 3, false};
LIFNeuron neuron_humid  = {0, 1.0, 0.90, 0, 3, false};
LIFNeuron neuron_motion = {0, 0.8, 0.70, 0, 2, false};
LIFNeuron neuron_air    = {0, 1.0, 0.88, 0, 3, false};

// ─── SYSTEM STATE ─────────────────────────────────────────────────────
int   currentTier   = 0;        // 0=IDLE, 1=ACTIVE, 2=ALERT
int   spikeCount    = 0;        // Total spikes recorded
float totalEnergy_uJ = 0.0;     // Cumulative microjoule consumption
unsigned long lastTick = 0;     // Timestamp of last tick
unsigned long tierStartTime = 0;// When current tier began

// ─── POWER MODEL (µW per tier) ───────────────────────────────────────
const float POWER_TIER0_UW = 8.5;
const float POWER_TIER1_UW = 150.0;
const float POWER_TIER2_UW = 2200.0;

// ─── TICK INTERVALS ──────────────────────────────────────────────────
const unsigned long TICK_TIER0_MS = 1000;  // 1 Hz in idle
const unsigned long TICK_TIER1_MS = 100;   // 10 Hz in active
const unsigned long TICK_TIER2_MS = 20;    // 50 Hz in alert

// ─── FUNCTION: Step a LIF Neuron ─────────────────────────────────────
// Feeds input_current into the neuron, applies decay, checks for spike.
// Returns true if the neuron fired this step.
bool stepNeuron(LIFNeuron &n, float input_current) {
  n.spiked = false;

  // Neuron is in refractory period — cannot fire, just decay
  if (n.refractory_counter > 0) {
    n.refractory_counter--;
    n.membrane_potential *= n.decay;
    return false;
  }

  // Integrate: accumulate input and apply leak
  n.membrane_potential = n.membrane_potential * n.decay + input_current;

  // Clamp to prevent runaway
  n.membrane_potential = constrain(n.membrane_potential, 0.0, 2.0);

  // Threshold check — does it fire?
  if (n.membrane_potential >= n.threshold) {
    n.spiked = true;
    n.membrane_potential = 0.0; // Reset after spike
    n.refractory_counter = n.refractory_period;
  }

  return n.spiked;
}

// ─── FUNCTION: Normalize a sensor value to 0–1 ───────────────────────
float normalize(float value, float minVal, float maxVal) {
  return constrain((value - minVal) / (maxVal - minVal), 0.0, 1.0);
}

// ─── FUNCTION: Determine processing tier ─────────────────────────────
// Tier escalates based on how many neurons simultaneously spiked.
int determineTier(bool tSpike, bool hSpike, bool mSpike, bool aSpike) {
  int spikeSum = (int)tSpike + (int)hSpike + (int)mSpike + (int)aSpike;
  if (spikeSum == 0) return 0;       // No activity
  if (spikeSum >= 2) return 2;       // Compound event → ALERT
  return 1;                           // Single spike → ACTIVE
}

// ─── FUNCTION: Update LCD ─────────────────────────────────────────────
void updateLCD(float temp, float humid, int tier, const char* event) {
  lcd.clear();
  lcd.setCursor(0, 0);

  if (tier == 0) {
    lcd.print("NEUROEDGE IDLE");
  } else if (tier == 1) {
    lcd.print("TIER-1: ACTIVE");
  } else {
    lcd.print("TIER-2: ALERT!");
  }

  lcd.setCursor(0, 1);
  // Show temp and humidity on second line
  char buf[17];
  snprintf(buf, sizeof(buf), "T:%.1fC H:%.0f%%", temp, humid);
  lcd.print(buf);
}

// ─── FUNCTION: Set output indicators ─────────────────────────────────
void setOutputs(int tier) {
  if (tier == 0) {
    digitalWrite(LED_GREEN, HIGH);
    digitalWrite(LED_RED, LOW);
    digitalWrite(BUZZER_PIN, LOW);
  } else if (tier == 1) {
    digitalWrite(LED_GREEN, HIGH);
    digitalWrite(LED_RED, LOW);
    digitalWrite(BUZZER_PIN, LOW);
  } else {
    // ALERT — red LED and buzzer pulse
    digitalWrite(LED_GREEN, LOW);
    digitalWrite(LED_RED, HIGH);
    // Short buzzer pulse
    digitalWrite(BUZZER_PIN, HIGH);
    delay(50);
    digitalWrite(BUZZER_PIN, LOW);
  }
}

// ─── FUNCTION: Compute & log power ───────────────────────────────────
float computePower(int tier, unsigned long dt_ms) {
  float power_uw = 0;
  if (tier == 0) power_uw = POWER_TIER0_UW;
  else if (tier == 1) power_uw = POWER_TIER1_UW;
  else power_uw = POWER_TIER2_UW;

  // Energy (µJ) = Power (µW) × Time (s)
  totalEnergy_uJ += power_uw * (dt_ms / 1000.0);

  return power_uw;
}

// ─── SETUP ────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  Serial.println("=== NeuroEdge v1.0 Booting ===");

  // Init sensors
  dht.begin();
  pinMode(PIR_PIN, INPUT);

  // Init outputs
  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_RED, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(LED_GREEN, HIGH);
  digitalWrite(LED_RED, LOW);
  digitalWrite(BUZZER_PIN, LOW);

  // Init LCD
  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0);
  lcd.print("NeuroEdge v1.0");
  lcd.setCursor(0, 1);
  lcd.print("Initializing...");
  delay(1500);
  lcd.clear();

  lastTick = millis();
  Serial.println("[BOOT] Sensors ready. SNN online.");
}

// ─── MAIN LOOP ────────────────────────────────────────────────────────
void loop() {
  unsigned long now = millis();

  // Determine tick interval based on current tier
  unsigned long tickInterval = TICK_TIER0_MS;
  if (currentTier == 1) tickInterval = TICK_TIER1_MS;
  else if (currentTier == 2) tickInterval = TICK_TIER2_MS;

  // Only process on tick boundary — no blocking delay()
  if (now - lastTick < tickInterval) return;

  unsigned long dt_ms = now - lastTick;
  lastTick = now;

  // ── READ SENSORS ──────────────────────────────────────────────────
  float temperature = dht.readTemperature();
  float humidity    = dht.readHumidity();
  int   pir_raw     = digitalRead(PIR_PIN);
  int   mq135_raw   = analogRead(MQ135_PIN); // 0–4095

  // Handle DHT read failure gracefully
  if (isnan(temperature)) temperature = 25.0;
  if (isnan(humidity))    humidity    = 50.0;

  // ── NORMALIZE TO 0–1 ──────────────────────────────────────────────
  float temp_norm   = normalize(temperature, 15.0, 40.0);
  float humid_norm  = normalize(humidity, 20.0, 90.0);
  float pir_norm    = (float)pir_raw; // Already 0 or 1
  float air_norm    = normalize((float)mq135_raw, 0.0, 4095.0);

  // ── STEP LIF NEURONS ──────────────────────────────────────────────
  bool t_spike = stepNeuron(neuron_temp,   temp_norm);
  bool h_spike = stepNeuron(neuron_humid,  humid_norm);
  bool m_spike = stepNeuron(neuron_motion, pir_norm);
  bool a_spike = stepNeuron(neuron_air,    air_norm);

  if (t_spike) spikeCount++;
  if (h_spike) spikeCount++;
  if (m_spike) spikeCount++;
  if (a_spike) spikeCount++;

  // ── LOG SPIKES ────────────────────────────────────────────────────
  if (t_spike) Serial.printf("[SPIKE] sensor=temp,  membrane=%.3f, threshold=%.2f\n", 0.0, neuron_temp.threshold);
  if (h_spike) Serial.printf("[SPIKE] sensor=humid, membrane=%.3f, threshold=%.2f\n", 0.0, neuron_humid.threshold);
  if (m_spike) Serial.printf("[SPIKE] sensor=motion,membrane=%.3f, threshold=%.2f\n", 0.0, neuron_motion.threshold);
  if (a_spike) Serial.printf("[SPIKE] sensor=air,   membrane=%.3f, threshold=%.2f\n", 0.0, neuron_air.threshold);

  // ── TIER CLASSIFICATION ───────────────────────────────────────────
  int newTier = determineTier(t_spike, h_spike, m_spike, a_spike);

  // Auto-decay back to tier 0 after 3 seconds of no spikes
  if (newTier == 0 && currentTier > 0) {
    if (now - tierStartTime > 3000) {
      currentTier = 0;
      Serial.println("[TIER] level=0, trigger=timeout_decay");
    }
  } else if (newTier > currentTier) {
    currentTier = newTier;
    tierStartTime = now;
    const char* trigger = (newTier == 2) ? "compound_spike" : "single_spike";
    Serial.printf("[TIER] level=%d, trigger=%s\n", currentTier, trigger);
  }

  // ── POWER TRACKING ────────────────────────────────────────────────
  float power_uw = computePower(currentTier, dt_ms);
  Serial.printf("[POWER] tier=%d, power_uw=%.2f, total_uj=%.2f\n",
                currentTier, power_uw, totalEnergy_uJ);

  // ── OUTPUTS ───────────────────────────────────────────────────────
  setOutputs(currentTier);
  updateLCD(temperature, humidity, currentTier, "");

  // ── HEARTBEAT STATUS ──────────────────────────────────────────────
  Serial.printf("[STATUS] T=%.1fC H=%.1f%% AQ=%d PIR=%d | Tier=%d | Spikes=%d\n",
                temperature, humidity, mq135_raw, pir_raw,
                currentTier, spikeCount);
}
