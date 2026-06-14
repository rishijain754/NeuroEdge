# NeuroEdge

**NeuroEdge** is a proof-of-concept neuromorphic engineering dashboard that demonstrates how event-driven Spiking Neural Networks (SNNs) can process complex, multi-modal sensor data with minimum energy consumption.

The project has two ways to experience it:
- **[Quick Start](https://rishijain754.github.io/NeuroEdge/)** — Open the web dashboard in a browser. No hardware needed.
- **[Physical Hardware](#️-physical-hardware-esp32-setup)** — Flash an ESP32 and build the real circuit.

---

##  Features

- **Tiered Neural Pipeline:** A Tier-0 → Tier-2 architecture that only escalates compute when genuine spike events are detected.
- **Live Membrane Potential Chart:** Real-time bar chart of each neuron's membrane voltage, updated every simulation tick.
- **Spike Raster & Topology Canvas:** Live neural connectivity graph and historical spike raster using the HTML5 Canvas API.
- **Power Efficiency Tracker:** Compares neuromorphic power draw (µW) against an always-on polling baseline.
- **Neural Event Log:** A scrollable serial-style feed of every SNN event — spikes, tier transitions, and classifications.
- **Neuron Heat Strip:** Firing intensity display for all input and hidden layer neurons.
- **Hardware Simulator View:** Drag-and-drop virtual circuit builder with live LCD, LED, and Buzzer output indicators.
- **Collapsible Sidebar:** Navigation sidebar that collapses to icon-only mode.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Structure | HTML5 (Semantic) |
| Styling | Vanilla CSS3 (Design Tokens, 8px Grid) |
| Logic | Vanilla JavaScript (ES6+) |
| Rendering | HTML5 Canvas API |
| Fonts | Inter, IBM Plex Mono (Google Fonts) |
| Embedded | C++ (Arduino / ESP32) |

---

## Project Structure

```text
NeuroEdge/
├── index.html              # Control Center — main telemetry dashboard
├── simulation.html         # Hardware Simulator — virtual circuit canvas
├── styles.css              # Global design tokens, 8px grid, layout
├── dashboard.js            # SNN engine and Canvas rendering logic
├── neuroedge_esp32.ino     # ESP32 Arduino firmware (physical hardware)
├── wokwi_diagram.json      # Wokwi circuit diagram (virtual testing)
└── README.md               # Project documentation
```

---

## Quick Start — UI Dashboard

No installation, no build step required.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/neuroedge.git
   cd neuroedge
   ```

2. **Open the dashboard:**
   Open `index.html` in any modern browser (Chrome, Firefox, Edge, Safari).

3. **Navigate to the Simulator:**
   Click **"02 Simulator"** in the left sidebar or open `simulation.html` directly.

> **Tip:** You can try the full sensor simulation — adjusting temperature, humidity, air quality, and triggering PIR motion events — entirely in the browser via **`simulation.html`**, without any physical hardware.

---

## Physical Hardware — ESP32 Setup

Flash the firmware to a real ESP32 to run NeuroEdge on actual sensors.

### Parts Required

| Component | Quantity | Notes |
|---|---|---|
| ESP32 DevKit V1 | 1 | Any 38-pin ESP32 board |
| DHT22 Sensor | 1 | Temperature & Humidity |
| MQ135 Sensor | 1 | Air Quality (analog output) |
| PIR Sensor (HC-SR501) | 1 | Motion Detection |
| 16×2 LCD Display (I2C) | 1 | I2C address `0x27` |
| Piezo Buzzer | 1 | Active or passive |
| Green LED | 1 | Status indicator |
| Red LED | 1 | Alert indicator |
| 220Ω Resistors | 2 | For LEDs |
| Breadboard + Jumper Wires | — | — |

### Wiring

| Component | ESP32 Pin |
|---|---|
| DHT22 Data | GPIO 15 |
| DHT22 VCC | 3V3 |
| MQ135 Signal | GPIO 34 |
| PIR Output | GPIO 27 |
| LCD SDA | GPIO 21 |
| LCD SCL | GPIO 22 |
| Buzzer + | GPIO 25 |
| Green LED (via 220Ω) | GPIO 26 |
| Red LED (via 220Ω) | GPIO 2 |
| All GND | GND |

### Installation Steps

**Step 1 — Install Arduino IDE**
Download and install from [arduino.cc/en/software](https://www.arduino.cc/en/software).

**Step 2 — Add ESP32 Board Support**
1. Open Arduino IDE → `File > Preferences`
2. In "Additional Boards Manager URLs", paste:
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
3. Go to `Tools > Board > Boards Manager`, search **esp32**, and install the package by Espressif Systems.

**Step 3 — Install Required Libraries**
Go to `Tools > Manage Libraries...` and install:
- `DHT sensor library` by Adafruit
- `Adafruit Unified Sensor` by Adafruit
- `LiquidCrystal I2C` by Frank de Brabander

**Step 4 — Flash the Firmware**
1. Open `neuroedge_esp32.ino` in Arduino IDE.
2. Connect your ESP32 via USB.
3. Select your board: `Tools > Board > ESP32 Arduino > ESP32 Dev Module`
4. Select the correct port: `Tools > Port > (your COM port)`
5. Click **Upload** (→ arrow button).

**Step 5 — Monitor Output**
Open `Tools > Serial Monitor`, set baud rate to **115200**. You will see live output like:
```
[BOOT] Sensors ready. SNN online.
[STATUS] T=27.3C H=55.2% AQ=1840 PIR=0 | Tier=0 | Spikes=0
[SPIKE]  sensor=temp, membrane=0.000, threshold=1.00
[TIER]   level=1, trigger=single_spike
[POWER]  tier=1, power_uw=150.00, total_uj=1.50
```

---

## Virtual Testing with Wokwi

If you don't have the hardware yet, you can simulate the full circuit in your browser using [Wokwi](https://wokwi.com/).

1. Go to [wokwi.com](https://wokwi.com/) and create a new **ESP32** project.
2. Paste the contents of `neuroedge_esp32.ino` into the code editor.
3. Click the `diagram.json` tab and replace its contents with `wokwi_diagram.json`.
4. Click ▶ **Start Simulation**.

>  **You can also use `simulation.html`** in the browser as a high-level visual alternative to Wokwi — it simulates the same sensor pipeline and neural events without needing Wokwi or any ESP32 at all.

---

## Screenshots

> *Add screenshots here after deployment.*

---

##  License

Distributed under the MIT License. See `LICENSE` for more information.
