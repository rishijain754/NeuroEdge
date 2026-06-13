# 🧠 NeuroEdge — Neuromorphic Multi-Sensor Edge Device Simulator

A biologically-inspired edge computing simulator that processes multi-sensor data (temperature, humidity, motion, air quality) using **Spiking Neural Networks (SNN)** with microwatt-level power budgets. The system activates deeper processing tiers only when neural spikes detect relevant events — just like the human brain.

![Architecture](https://img.shields.io/badge/Architecture-3--Tier_Neuromorphic-blueviolet?style=for-the-badge)
![Power](https://img.shields.io/badge/Power-~8.5_µW_Baseline-green?style=for-the-badge)
![Sensors](https://img.shields.io/badge/Sensors-4_Channel-orange?style=for-the-badge)

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────┐
│  TIER 0 — Always-On Sensor Hub (~8.5 µW)        │
│  • ADC reads at 1 Hz                             │
│  • LIF (Leaky Integrate-and-Fire) threshold      │
│  • Spikes emitted on significant change          │
└──────────────┬───────────────────────────────────┘
               │ Spike event
┌──────────────▼───────────────────────────────────┐
│  TIER 1 — Event Processor (~150 µW, 20ms burst)  │
│  • 4→8→3 SNN pattern recognition                 │
│  • Temporal spike encoding                        │
│  • Classify: Normal / Anomaly / Alert            │
└──────────────┬───────────────────────────────────┘
               │ Alert condition
┌──────────────▼───────────────────────────────────┐
│  TIER 2 — Deep Analysis (~2.2 mW, 100ms burst)   │
│  • Full context-aware inference                   │
│  • Alert generation & logging                     │
└──────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Option 1: Web Dashboard (Zero Dependencies)

Simply open `index.html` in any modern browser:

```
# Windows
start index.html

# macOS / Linux
open index.html
```

The entire neuromorphic simulation runs in JavaScript — no server needed!

### Option 2: Python Backend (Standalone)

```bash
pip install numpy
python sensor_engine.py   # → 60s of synthetic sensor data
python neuro_core.py      # → 100-tick SNN simulation
python power_tracker.py   # → Power budget analysis
```

### Option 3: Virtual Arduino (Wokwi)

1. Go to [wokwi.com](https://wokwi.com)
2. Create a new ESP32 project
3. Replace `sketch.ino` with contents of `wokwi_sketch.ino`
4. Replace `diagram.json` with contents of `wokwi_diagram.json`
5. Click ▶ Run — watch Serial monitor for spike events

---

## 🎮 Dashboard Controls

| Control | Action |
|---------|--------|
| ▶ Start / ⏸ Pause | Toggle simulation |
| ↺ Reset | Full system reset |
| Scenario dropdown | Switch between Normal/Fire/Cooking/HVAC/Storm |
| 🔥 Inject Fire | Force fire alarm scenario |
| Speed slider | 0.2x → 5x simulation speed |

---

## 📊 Dashboard Sections

| Panel | What it Shows |
|-------|---------------|
| **Sensor Gauges** | Real-time circular gauges for all 4 sensors with spike indicators |
| **Tier Indicator** | Glowing rings showing which processing tier is active |
| **Spike Raster** | Scrolling dot plot of neural spikes over time (4 channels) |
| **Power Meter** | Current µW consumption with historical chart |
| **SNN Topology** | Animated 4→8→3 neural network with active connections |
| **Event Log** | Timestamped alerts with severity classification |
| **Membrane Bars** | Live membrane potentials for all 15 neurons |
| **Savings Banner** | Neuromorphic vs always-on energy comparison |

---

## 🔬 Neuromorphic Concepts Implemented

### Leaky Integrate-and-Fire (LIF) Neurons
- Membrane potential accumulates input current
- Leaks by decay factor each tick (τ = 0.9)
- Fires spike when threshold reached → resets + refractory period

### Spike Encoding
- **Rate coding**: Higher sensor values → higher spike probability
- **Delta encoding**: Spikes on significant sensor changes
- **Combined**: Both strategies merged for robust event detection

### Winner-Take-All (WTA) Output
- 3 output neurons compete: Normal, Anomaly, Alert
- Highest membrane potential wins classification

### Homeostatic Plasticity
- Firing rate monitored across 50-tick windows
- Over-active neurons: threshold increased by 5%
- Under-active neurons: threshold decreased by 5%
- Prevents alarm fatigue and dead neurons

---

## 📁 Project Structure

```
NeuroEdge/
├── index.html            ← Dashboard (open in browser)
├── styles.css            ← Design system (dark neural theme)
├── dashboard.js          ← Full JS simulation engine
├── neuro_core.py         ← Python SNN implementation
├── sensor_engine.py      ← Synthetic sensor data generator
├── power_tracker.py      ← Power budget simulator
├── wokwi_sketch.ino      ← ESP32 Arduino sketch for Wokwi
├── wokwi_diagram.json    ← Wokwi circuit layout
├── requirements.txt      ← Python dependencies (numpy)
└── README.md             ← This file
```

---

## 🔧 Technology Stack

| Component | Technology |
|-----------|-----------|
| Dashboard | Vanilla HTML/CSS/JS + Canvas API |
| SNN Engine | Pure JavaScript (browser) + Python (standalone) |
| Design | Glassmorphism, CSS animations, dark theme |
| Virtual HW | Wokwi.com (ESP32 simulator) |
| Fonts | Inter + JetBrains Mono (Google Fonts) |

---

## ⚡ Power Budget Model

| State | Power | Duration |
|-------|-------|----------|
| Deep Sleep | 0.5 µW | Between reads |
| Tier-0 (ADC + LIF) | 8.5 µW | Always on |
| Tier-1 (SNN inference) | 150 µW | ~20 ms burst |
| Tier-2 (Deep analysis) | 2,200 µW | ~100 ms burst |
| Sensor ADC read | 3.0 µW | Per sensor |
| Spike propagation | 0.1 µW | Per spike |

**Typical savings: >90% vs always-on Tier-2 processing**

---

## 📝 License

Educational project — built for learning neuromorphic computing concepts.
