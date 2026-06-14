# NeuroEdge

**NeuroEdge** is a proof-of-concept neuromorphic engineering dashboard and simulation environment. It demonstrates how event-driven Spiking Neural Networks (SNNs) can be utilized on ultra-low-power edge devices to process complex, multi-modal sensor data with minimal compute overhead.

The project features a dual-interface architecture: a live "Control Center" dashboard for monitoring telemetry, and a "Simulator" for testing compound anomaly detection in a virtualized hardware environment.

![NeuroEdge UI Preview](https://via.placeholder.com/1000x500.png?text=NeuroEdge+Dashboard+Preview) *(Replace with actual screenshot)*

---

## ⚡ Features

- **Neuromorphic SNN Core:** A custom Leaky Integrate-and-Fire (LIF) network with homeostatic plasticity, built entirely in Python/NumPy.
- **Event-Driven Architecture:** Implements a tiered processing pipeline (Tier-0 to Tier-2) that only consumes power when anomalous sensor spikes occur.
- **Live Telemetry Dashboard:** A Vanilla JS/Canvas-based frontend rendering real-time membrane potentials, spike raster plots, and network topology.
- **Tactical UI/UX:** A rigorously designed, high-contrast "Tactical Instrumentation" interface built on an 8px grid, utilizing deep slate and cyan aesthetics typical of professional engineering software.
- **Hardware Simulation:** Includes virtualized Arduino/ESP32 sketches (`.ino`) and Wokwi definitions for testing physical micro-controller logic.

---

## 🛠️ Tech Stack

**Frontend:**
- HTML5 / Vanilla CSS3 (Strict Design Tokens, Custom 8px Grid System)
- Vanilla JavaScript (ES6+)
- HTML5 `<canvas>` API for high-performance rendering of neural graphs

**Backend & Simulation Engines:**
- Python 3.9+
- NumPy (Matrix operations and LIF neuron simulation)

**Embedded / Hardware Simulation:**
- C++ (Arduino Framework)
- Wokwi Virtual Environment (`wokwi_diagram.json`)

---

## 📂 Project Structure

```text
NeuroEdge/
├── index.html           # Main Control Center Dashboard UI
├── simulation.html      # Hardware Simulator & Component Testing UI
├── styles.css           # Global design tokens and UI styles
├── dashboard.js         # Core frontend logic and Canvas rendering
├── neuro_core.py        # Python SNN simulation logic and LIF Neurons
├── sensor_engine.py     # Python synthetic multi-sensor data generator
├── power_tracker.py     # Python micro-watt power consumption simulator
├── requirements.txt     # Python dependencies
├── wokwi_sketch.ino     # C++ Arduino sketch for hardware simulation
├── wokwi_diagram.json   # Wokwi circuit definition for ESP32
└── README.md            # Project documentation (You are here)
```

---

## 🚀 Setup & Installation

### 1. Frontend (UI Dashboard)
The frontend requires no build steps or bundlers. 
1. Clone the repository.
2. Open `index.html` or `simulation.html` directly in any modern web browser (Chrome, Firefox, Safari, Edge).

### 2. Backend (Python SNN Engine)
To run the SNN simulation logic and sensor generation scripts:
1. Ensure Python 3.9+ is installed.
2. Open a terminal in the project directory.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the individual engine tests:
   ```bash
   python neuro_core.py
   ```

### 3. Hardware Simulation (Wokwi)
To test the embedded C++ logic:
1. Navigate to [Wokwi.com](https://wokwi.com/).
2. Create a new ESP32 project.
3. Copy the contents of `wokwi_sketch.ino` into the code editor.
4. Replace the default `diagram.json` with the contents of `wokwi_diagram.json`.
5. Click "Start Simulation".

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
