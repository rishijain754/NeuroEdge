/*
 * ============================================================================
 *  NeuroEdge — Neuromorphic Edge Intelligence (Wokwi Simulation)
 * ============================================================================
 *
 *  This sketch implements a simplified neuromorphic processing pipeline on an
 *  ESP32, inspired by biological neural systems.  The core idea is that each
 *  sensor feeds a **Leaky Integrate-and-Fire (LIF)** neuron model.  The
 *  neuron accumulates charge (membrane potential) from the incoming stimulus.
 *  If the potential exceeds a threshold the neuron "spikes" — a binary event
 *  that triggers further processing.  Between spikes the potential leaks
 *  (decays) exponentially, mimicking ion-channel leakage in real neurons.
 *
 *  Processing is organised into three tiers:
 *
 *    Tier-0  –  Idle monitoring at 1 Hz.  Ultra-low power.  Only LIF neurons
 *               are evaluated.
 *    Tier-1  –  A single neuron spiked.  Sampling rate jumps to 10 Hz and a
 *               lightweight classifier (weighted sum) runs.
 *    Tier-2  –  Multiple neurons spiked simultaneously.  Full burst analysis
 *               at 50 Hz with a detailed event report.
 *
 *  All output is emitted as structured, JSON-like tagged messages on Serial
 *  so that a host or dashboard can parse them easily.
 *
 *  Hardware (Wokwi virtual):
 *    - DHT22       on GPIO 15   → temperature + humidity
 *    - Potentiometer on GPIO 34 → simulates MQ135 air-quality (ADC)
 *    - Push-button on GPIO 27   → simulates PIR motion sensor
 *    - Built-in LED on GPIO 2   → heartbeat indicator
 *
 *  Libraries:
 *    - DHT sensor library (bundled with Wokwi)
 *
 *  Author : NeuroEdge
 *  Date   : 2026-06-13
 * ============================================================================
 */

#include <DHT.h>

// ─────────────────────────────────────────────────────────────────────────────
//  Pin Definitions
// ─────────────────────────────────────────────────────────────────────────────
#define DHT_PIN        15    // DHT22 data line
#define MQ135_PIN      34    // Analog input — potentiometer simulating MQ135
#define PIR_PIN        27    // Digital input — push-button simulating PIR
#define HEARTBEAT_PIN   2    // On-board LED used as heartbeat indicator

// ─────────────────────────────────────────────────────────────────────────────
//  DHT22 Setup
// ─────────────────────────────────────────────────────────────────────────────
#define DHT_TYPE  DHT22
DHT dht(DHT_PIN, DHT_TYPE);

// ─────────────────────────────────────────────────────────────────────────────
//  Neuromorphic Constants
// ─────────────────────────────────────────────────────────────────────────────

/*
 *  The LIF model is one of the simplest spiking neuron models.
 *
 *      dV/dt = -λ·V + I(t)
 *
 *  In discrete time this becomes:
 *
 *      V[n+1] = decay * V[n] + I[n]
 *
 *  When V exceeds `threshold` the neuron fires (spiked = true), V resets to
 *  zero, and a refractory period starts during which the neuron cannot fire.
 *  This prevents runaway spiking and is biologically motivated by the
 *  absolute refractory period of real neurons (~1-2 ms, here we use a few
 *  loop ticks).
 */

// Number of sensor-neurons in our tiny "cortex"
#define NUM_NEURONS  4

// Neuron indices — one per sensor channel
#define NEURON_TEMP     0
#define NEURON_HUMID    1
#define NEURON_AIR      2
#define NEURON_MOTION   3

// ─────────────────────────────────────────────────────────────────────────────
//  LIF Neuron Structure
// ─────────────────────────────────────────────────────────────────────────────
struct LIFNeuron {
    float membrane_potential;   // Current charge (voltage analogue)
    float threshold;            // Spike threshold — fire when V >= threshold
    float decay;                // Leak factor per tick (0 < decay < 1)
    int   refractory_counter;   // Ticks remaining in refractory period
    int   refractory_period;    // Duration of refractory window (ticks)
    bool  spiked;               // True during the tick the neuron fires
};

// ─────────────────────────────────────────────────────────────────────────────
//  Tier / Power Model
// ─────────────────────────────────────────────────────────────────────────────

/*
 *  Simulated power consumption loosely models a real neuromorphic chip:
 *
 *    Tier-0   ~  8.5 µW   (idle, only LIF evaluation)
 *    Tier-1   ~ 45.0 µW   (classifier active, higher sample rate)
 *    Tier-2   ~210.0 µW   (full burst analysis)
 *
 *  These numbers are deliberately in the micro-watt range to illustrate the
 *  extreme efficiency of event-driven neuromorphic processing vs. conventional
 *  always-on AI inference (which would be mW–W range).
 */

#define TIER_0  0
#define TIER_1  1
#define TIER_2  2

static const float POWER_UW[] = { 8.5f, 45.0f, 210.0f };

// Loop interval targets (milliseconds)
static const unsigned long LOOP_INTERVAL_MS[] = {
    1000,   // Tier-0 → 1 Hz
     100,   // Tier-1 → 10 Hz
      20    // Tier-2 → 50 Hz
};

// How many ticks a tier stays elevated before falling back to Tier-0
#define TIER_HOLD_TICKS  30

// ─────────────────────────────────────────────────────────────────────────────
//  Classification Weights (Tier-1 simple weighted-sum classifier)
// ─────────────────────────────────────────────────────────────────────────────

/*
 *  A trivially simple "classifier" inspired by a single-layer perceptron.
 *  Each sensor's normalised value is multiplied by a weight and summed.
 *  If the result exceeds a decision threshold we label the event as an
 *  anomaly; otherwise it is considered nominal.
 *
 *  In a real deployment this would be replaced by a trained SNN (Spiking
 *  Neural Network) running on dedicated neuromorphic silicon (e.g. Intel
 *  Loihi, BrainChip Akida).
 */

static const float CLASS_WEIGHTS[] = { 0.30f, 0.20f, 0.35f, 0.15f };
#define CLASS_THRESHOLD  0.55f

// ─────────────────────────────────────────────────────────────────────────────
//  Global State
// ─────────────────────────────────────────────────────────────────────────────

LIFNeuron neurons[NUM_NEURONS];

// Sensor names for structured log messages
static const char* SENSOR_NAMES[] = { "temperature", "humidity", "air_quality", "motion" };

int  currentTier        = TIER_0;
int  tierHoldCounter    = 0;          // Counts down while tier is elevated
float totalEnergy_uJ    = 0.0f;      // Accumulated energy in micro-joules

unsigned long lastLoopTime      = 0;
unsigned long lastHeartbeatTime = 0;
bool          heartbeatState    = false;

// Normalised sensor readings (0.0 – 1.0) cached for reuse
float normValues[NUM_NEURONS];

// ─────────────────────────────────────────────────────────────────────────────
//  Forward Declarations
// ─────────────────────────────────────────────────────────────────────────────
void  initNeurons();
void  readSensors();
float normaliseDHT_Temp(float raw);
float normaliseDHT_Humid(float raw);
float normaliseADC(int raw);
int   updateNeurons();
void  runClassifier(int spikeCount, int triggerNeuron);
void  updateTier(int spikeCount, int triggerNeuron);
void  printPowerStatus(unsigned long dt_ms);
void  heartbeat();

// ═════════════════════════════════════════════════════════════════════════════
//  SETUP
// ═════════════════════════════════════════════════════════════════════════════
void setup() {
    Serial.begin(115200);
    while (!Serial) { ; }  // Wait for serial monitor

    Serial.println();
    Serial.println("╔══════════════════════════════════════════════════╗");
    Serial.println("║   NeuroEdge — Neuromorphic Edge Intelligence    ║");
    Serial.println("║   Wokwi ESP32 Simulation  ·  Tier-0/1/2 LIF    ║");
    Serial.println("╚══════════════════════════════════════════════════╝");
    Serial.println();

    // Initialise peripherals
    dht.begin();
    pinMode(MQ135_PIN, INPUT);
    pinMode(PIR_PIN,   INPUT_PULLDOWN);   // Button pulls HIGH on press
    pinMode(HEARTBEAT_PIN, OUTPUT);

    // Initialise neuron array
    initNeurons();

    lastLoopTime = millis();
    lastHeartbeatTime = millis();

    Serial.println("[BOOT] neurons=4, tier=0, state=idle");
    Serial.println();
}

// ═════════════════════════════════════════════════════════════════════════════
//  MAIN LOOP  (non-blocking, millis()-based timing)
// ═════════════════════════════════════════════════════════════════════════════
void loop() {
    unsigned long now = millis();
    unsigned long interval = LOOP_INTERVAL_MS[currentTier];

    // ── Gate: only proceed when the current tier's interval has elapsed ──
    if (now - lastLoopTime < interval) {
        // While waiting, still service the heartbeat LED
        heartbeat();
        return;
    }

    unsigned long dt_ms = now - lastLoopTime;
    lastLoopTime = now;

    // ── Step 1: Read & normalise all sensors ────────────────────────────
    readSensors();

    // ── Step 2: Feed normalised values into LIF neurons & check spikes ─
    int spikeCount = updateNeurons();

    // ── Step 3: Determine which neuron triggered (first spike wins) ─────
    int triggerNeuron = -1;
    for (int i = 0; i < NUM_NEURONS; i++) {
        if (neurons[i].spiked) {
            triggerNeuron = i;
            break;
        }
    }

    // ── Step 4: Update processing tier ──────────────────────────────────
    updateTier(spikeCount, triggerNeuron);

    // ── Step 5: If elevated tier, run classifier / burst analysis ───────
    if (currentTier >= TIER_1) {
        runClassifier(spikeCount, triggerNeuron);
    }

    // ── Step 6: Power accounting ────────────────────────────────────────
    printPowerStatus(dt_ms);

    // ── Step 7: Heartbeat LED ───────────────────────────────────────────
    heartbeat();
}

// ═════════════════════════════════════════════════════════════════════════════
//  NEURON INITIALISATION
// ═════════════════════════════════════════════════════════════════════════════

/*
 *  Each neuron is configured with parameters suited to the dynamics of its
 *  sensor.  Temperature and humidity change slowly so they get a higher
 *  threshold and stronger decay (harder to spike).  Motion is binary so it
 *  gets a low threshold (easy to spike on any detection).
 */
void initNeurons() {
    // Temperature neuron — slow dynamics, moderate threshold
    neurons[NEURON_TEMP].membrane_potential = 0.0f;
    neurons[NEURON_TEMP].threshold          = 0.75f;
    neurons[NEURON_TEMP].decay              = 0.85f;
    neurons[NEURON_TEMP].refractory_counter = 0;
    neurons[NEURON_TEMP].refractory_period  = 5;
    neurons[NEURON_TEMP].spiked             = false;

    // Humidity neuron — similar to temperature
    neurons[NEURON_HUMID].membrane_potential = 0.0f;
    neurons[NEURON_HUMID].threshold          = 0.80f;
    neurons[NEURON_HUMID].decay              = 0.88f;
    neurons[NEURON_HUMID].refractory_counter = 0;
    neurons[NEURON_HUMID].refractory_period  = 5;
    neurons[NEURON_HUMID].spiked             = false;

    // Air quality neuron — moderate dynamics
    neurons[NEURON_AIR].membrane_potential = 0.0f;
    neurons[NEURON_AIR].threshold          = 0.70f;
    neurons[NEURON_AIR].decay              = 0.80f;
    neurons[NEURON_AIR].refractory_counter = 0;
    neurons[NEURON_AIR].refractory_period  = 4;
    neurons[NEURON_AIR].spiked             = false;

    // Motion neuron — fast dynamics, low threshold (binary sensor)
    neurons[NEURON_MOTION].membrane_potential = 0.0f;
    neurons[NEURON_MOTION].threshold          = 0.50f;
    neurons[NEURON_MOTION].decay              = 0.60f;
    neurons[NEURON_MOTION].refractory_counter = 0;
    neurons[NEURON_MOTION].refractory_period  = 3;
    neurons[NEURON_MOTION].spiked             = false;
}

// ═════════════════════════════════════════════════════════════════════════════
//  SENSOR READING & NORMALISATION
// ═════════════════════════════════════════════════════════════════════════════

/*
 *  Normalisation maps raw sensor values into the 0.0–1.0 range expected by
 *  the LIF neurons.  This is analogous to biological receptor cells
 *  converting a physical stimulus (photons, pressure, chemicals) into a
 *  graded receptor potential that feeds into the first neuron.
 */

void readSensors() {
    // ── DHT22: temperature (°C) & humidity (%) ──────────────────────────
    float tempC  = dht.readTemperature();
    float humid  = dht.readHumidity();

    // Guard against NaN from failed reads
    if (isnan(tempC)) tempC = 25.0f;
    if (isnan(humid)) humid = 50.0f;

    normValues[NEURON_TEMP]  = normaliseDHT_Temp(tempC);
    normValues[NEURON_HUMID] = normaliseDHT_Humid(humid);

    // ── MQ135 (potentiometer): 12-bit ADC (0–4095) ─────────────────────
    int adcRaw = analogRead(MQ135_PIN);
    normValues[NEURON_AIR] = normaliseADC(adcRaw);

    // ── PIR (push-button): digital HIGH = motion detected ───────────────
    int pirState = digitalRead(PIR_PIN);
    normValues[NEURON_MOTION] = (pirState == HIGH) ? 1.0f : 0.0f;
}

/*
 *  Temperature normalisation: maps 0 °C → 0.0  and  50 °C → 1.0
 *  Clamped so values outside this range don't break the neuron.
 */
float normaliseDHT_Temp(float raw) {
    float v = (raw - 0.0f) / 50.0f;
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    return v;
}

/*
 *  Humidity normalisation: maps 0 % → 0.0  and  100 % → 1.0
 */
float normaliseDHT_Humid(float raw) {
    float v = raw / 100.0f;
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    return v;
}

/*
 *  ADC normalisation: maps 0 → 0.0  and  4095 → 1.0
 *  (ESP32 default 12-bit resolution)
 */
float normaliseADC(int raw) {
    float v = (float)raw / 4095.0f;
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    return v;
}

// ═════════════════════════════════════════════════════════════════════════════
//  LIF NEURON UPDATE
// ═════════════════════════════════════════════════════════════════════════════

/*
 *  This is the heart of the neuromorphic pipeline.  For each neuron:
 *
 *    1. If in refractory period → decrement counter, skip update.
 *    2. Leak:  V *= decay          (exponential decay toward resting state)
 *    3. Integrate:  V += I_norm    (add normalised sensor current)
 *    4. Fire check: if V >= threshold → SPIKE!  Reset V, enter refractory.
 *
 *  Returns the total number of neurons that spiked this tick.
 */
int updateNeurons() {
    int spikeCount = 0;

    for (int i = 0; i < NUM_NEURONS; i++) {
        LIFNeuron &n = neurons[i];
        n.spiked = false;  // Reset spike flag for this tick

        // ── Refractory period guard ─────────────────────────────────────
        if (n.refractory_counter > 0) {
            n.refractory_counter--;
            continue;  // Neuron is "exhausted", cannot fire
        }

        // ── Leak (passive decay) ────────────────────────────────────────
        n.membrane_potential *= n.decay;

        // ── Integrate (add input current) ───────────────────────────────
        n.membrane_potential += normValues[i];

        // ── Fire check ──────────────────────────────────────────────────
        if (n.membrane_potential >= n.threshold) {
            n.spiked = true;
            spikeCount++;

            // Structured log: spike event
            Serial.print("[SPIKE] sensor=");
            Serial.print(SENSOR_NAMES[i]);
            Serial.print(", membrane=");
            Serial.print(n.membrane_potential, 2);
            Serial.print(", threshold=");
            Serial.println(n.threshold, 2);

            // Reset membrane potential (post-spike reset, biological analogue
            // of the hyperpolarisation phase)
            n.membrane_potential = 0.0f;
            n.refractory_counter = n.refractory_period;
        }
    }

    return spikeCount;
}

// ═════════════════════════════════════════════════════════════════════════════
//  TIER MANAGEMENT
// ═════════════════════════════════════════════════════════════════════════════

/*
 *  Tier transitions implement a biologically-inspired "attention" mechanism.
 *  In the brain, a surprising stimulus causes a shift from diffuse, low-power
 *  monitoring to focused, high-energy processing.  The system returns to idle
 *  (Tier-0) once the stimulus subsides, conserving energy.
 */
void updateTier(int spikeCount, int triggerNeuron) {
    int previousTier = currentTier;

    if (spikeCount >= 2) {
        // Multiple simultaneous spikes → high-alert, full burst
        currentTier     = TIER_2;
        tierHoldCounter = TIER_HOLD_TICKS;
    } else if (spikeCount == 1) {
        // Single spike → elevated attention
        if (currentTier < TIER_1) {
            currentTier = TIER_1;
        }
        tierHoldCounter = TIER_HOLD_TICKS;
    } else {
        // No spikes — count down hold timer
        if (tierHoldCounter > 0) {
            tierHoldCounter--;
        } else {
            currentTier = TIER_0;
        }
    }

    // Log tier transitions
    if (currentTier != previousTier) {
        Serial.print("[TIER] level=");
        Serial.print(currentTier);
        Serial.print(", trigger=");
        if (triggerNeuron >= 0) {
            Serial.print(SENSOR_NAMES[triggerNeuron]);
            Serial.println("_spike");
        } else {
            Serial.println("decay_to_idle");
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  CLASSIFIER  (Tier-1 / Tier-2)
// ═════════════════════════════════════════════════════════════════════════════

/*
 *  A single-layer perceptron-style classifier.  Each normalised sensor value
 *  is multiplied by a learned weight and the products are summed.  If the
 *  weighted sum exceeds a decision threshold the event is classified as an
 *  "anomaly"; otherwise it is "nominal".
 *
 *  In Tier-2 (multi-spike) an extended report is printed with per-sensor
 *  breakdown — this models the "full cortical analysis" that occurs when the
 *  brain receives strong, multi-modal stimulation.
 */
void runClassifier(int spikeCount, int triggerNeuron) {
    // Compute weighted sum (dot product of inputs and weights)
    float weightedSum = 0.0f;
    for (int i = 0; i < NUM_NEURONS; i++) {
        weightedSum += normValues[i] * CLASS_WEIGHTS[i];
    }

    // Compute confidence as a normalised score (0–1)
    float confidence = weightedSum;  // Already in roughly 0–1 range
    if (confidence > 1.0f) confidence = 1.0f;

    const char* result = (weightedSum >= CLASS_THRESHOLD) ? "anomaly" : "nominal";

    // ── Tier-1 output: concise classification ───────────────────────────
    Serial.print("[CLASSIFY] result=");
    Serial.print(result);
    Serial.print(", confidence=");
    Serial.print(confidence, 2);
    Serial.print(", weighted_sum=");
    Serial.println(weightedSum, 3);

    // ── Tier-2 output: full burst analysis report ───────────────────────
    if (currentTier == TIER_2) {
        Serial.println("┌──────────── TIER-2 BURST ANALYSIS ────────────┐");

        // Count how many neurons spiked
        int activeSpikers = 0;
        for (int i = 0; i < NUM_NEURONS; i++) {
            if (neurons[i].spiked) activeSpikers++;
        }

        Serial.print("│  Active spikers   : ");
        Serial.println(activeSpikers);
        Serial.print("│  Classification   : ");
        Serial.println(result);
        Serial.print("│  Confidence       : ");
        Serial.print(confidence * 100.0f, 1);
        Serial.println(" %");
        Serial.println("│");
        Serial.println("│  Per-sensor breakdown:");

        for (int i = 0; i < NUM_NEURONS; i++) {
            Serial.print("│    ");
            Serial.print(SENSOR_NAMES[i]);
            Serial.print(": norm=");
            Serial.print(normValues[i], 3);
            Serial.print(", weight=");
            Serial.print(CLASS_WEIGHTS[i], 2);
            Serial.print(", contrib=");
            Serial.print(normValues[i] * CLASS_WEIGHTS[i], 3);
            Serial.print(", membrane=");
            Serial.print(neurons[i].membrane_potential, 3);
            Serial.print(", spiked=");
            Serial.println(neurons[i].spiked ? "YES" : "no");
        }

        Serial.println("│");

        // Determine alert type based on which combination of sensors spiked
        Serial.print("│  Alert type       : ");
        if (neurons[NEURON_TEMP].spiked && neurons[NEURON_AIR].spiked) {
            Serial.println("FIRE_HAZARD (temp + air quality)");
        } else if (neurons[NEURON_MOTION].spiked && neurons[NEURON_TEMP].spiked) {
            Serial.println("INTRUSION_THERMAL (motion + temp)");
        } else if (neurons[NEURON_MOTION].spiked && neurons[NEURON_AIR].spiked) {
            Serial.println("GAS_LEAK_MOTION (motion + air)");
        } else if (neurons[NEURON_HUMID].spiked && neurons[NEURON_TEMP].spiked) {
            Serial.println("CLIMATE_ANOMALY (humidity + temp)");
        } else {
            Serial.println("MULTI_SENSOR_EVENT (general)");
        }

        Serial.println("└────────────────────────────────────────────────┘");
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  POWER ACCOUNTING
// ═════════════════════════════════════════════════════════════════════════════

/*
 *  Simulated power model.  Energy (µJ) is accumulated each tick:
 *
 *      E += P_tier × dt
 *
 *  where P is the instantaneous power for the current tier (in µW) and dt is
 *  the elapsed time in seconds.  This gives a rough picture of how a real
 *  neuromorphic chip conserves energy by spending most of its time in Tier-0.
 */
void printPowerStatus(unsigned long dt_ms) {
    float dt_s   = (float)dt_ms / 1000.0f;
    float power  = POWER_UW[currentTier];
    float energy = power * dt_s;         // µW × s = µJ
    totalEnergy_uJ += energy;

    Serial.print("[POWER] tier=");
    Serial.print(currentTier);
    Serial.print(", power_uw=");
    Serial.print(power, 1);
    Serial.print(", total_uj=");
    Serial.println(totalEnergy_uJ, 1);
}

// ═════════════════════════════════════════════════════════════════════════════
//  HEARTBEAT LED
// ═════════════════════════════════════════════════════════════════════════════

/*
 *  The heartbeat LED provides a quick visual status:
 *    Tier-0 → slow blink  (1000 ms period)
 *    Tier-1 → medium blink ( 250 ms period)
 *    Tier-2 → fast blink   ( 100 ms period)
 *
 *  This is a non-blocking blink using millis().
 */
void heartbeat() {
    unsigned long now = millis();

    unsigned long blinkInterval;
    switch (currentTier) {
        case TIER_2:  blinkInterval = 100;  break;
        case TIER_1:  blinkInterval = 250;  break;
        default:      blinkInterval = 1000; break;
    }

    if (now - lastHeartbeatTime >= blinkInterval) {
        lastHeartbeatTime = now;
        heartbeatState = !heartbeatState;
        digitalWrite(HEARTBEAT_PIN, heartbeatState ? HIGH : LOW);
    }
}
