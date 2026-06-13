/* ═══════════════════════════════════════════════════════════════
   NeuroEdge — Dashboard Engine
   Full neuromorphic simulation running in-browser (no server needed)
   ═══════════════════════════════════════════════════════════════ */

// ─────────────────────────────────────────
//  1. LIF NEURON — Leaky Integrate-and-Fire
// ─────────────────────────────────────────
class LIFNeuron {
  constructor(threshold = 1.0, decay = 0.9, refractoryPeriod = 3) {
    this.membranePotential = 0;
    this.threshold = threshold;
    this.decay = decay;
    this.refractoryPeriod = refractoryPeriod;
    this.refractoryCounter = 0;
    this.spiked = false;
    this.spikeHistory = [];
    this.maxHistory = 100;
  }

  step(inputCurrent) {
    this.spiked = false;

    if (this.refractoryCounter > 0) {
      this.refractoryCounter--;
      this.membranePotential *= 0.5; // fast decay during refractory
      this.spikeHistory.push(false);
      if (this.spikeHistory.length > this.maxHistory) this.spikeHistory.shift();
      return false;
    }

    // Leaky integration: V(t+1) = decay * V(t) + I(t)
    this.membranePotential = this.decay * this.membranePotential + inputCurrent;

    // Threshold check — fire if above threshold
    if (this.membranePotential >= this.threshold) {
      this.spiked = true;
      this.membranePotential = 0; // reset after spike
      this.refractoryCounter = this.refractoryPeriod;
    }

    this.spikeHistory.push(this.spiked);
    if (this.spikeHistory.length > this.maxHistory) this.spikeHistory.shift();
    return this.spiked;
  }

  getFiringRate(window = 20) {
    const recent = this.spikeHistory.slice(-window);
    if (recent.length === 0) return 0;
    return recent.filter(s => s).length / recent.length;
  }

  reset() {
    this.membranePotential = 0;
    this.refractoryCounter = 0;
    this.spiked = false;
    this.spikeHistory = [];
  }
}

// ─────────────────────────────────────────
//  2. SPIKE ENCODER
// ─────────────────────────────────────────
class SpikeEncoder {
  /** Rate coding: higher value → higher spike probability */
  rateEncode(value, minVal, maxVal, maxRate = 0.8) {
    const normalized = Math.max(0, Math.min(1, (value - minVal) / (maxVal - minVal)));
    return Math.random() < (normalized * maxRate) ? 1 : 0;
  }

  /** Delta encoding: spike only on significant change */
  deltaEncode(current, previous, threshold = 0.5) {
    return Math.abs(current - previous) > threshold ? 1 : 0;
  }

  /** Combined encoding for sensor values */
  encode(value, minVal, maxVal, prevValue, deltaThreshold) {
    const rate = this.rateEncode(value, minVal, maxVal, 0.6);
    const delta = this.deltaEncode(value, prevValue, deltaThreshold);
    // Spike if either rate or delta triggers
    return Math.min(1, rate * 0.4 + delta * 0.8);
  }
}

// ─────────────────────────────────────────
//  3. SPIKING NEURAL NETWORK
// ─────────────────────────────────────────
class SpikingNeuralNetwork {
  constructor() {
    // 4 input → 8 hidden → 3 output (Normal, Anomaly, Alert)
    this.inputNeurons = Array.from({ length: 4 }, () => new LIFNeuron(0.8, 0.85, 2));
    this.hiddenNeurons = Array.from({ length: 8 }, () => new LIFNeuron(1.0, 0.9, 3));
    this.outputNeurons = Array.from({ length: 3 }, () => new LIFNeuron(1.2, 0.88, 4));

    // Hand-tuned weights: input→hidden (4×8)
    // Rows: temp, humidity, motion, airQuality
    // Hidden neurons specialize: 0-1=temp, 2-3=humidity, 4-5=motion, 6-7=airQuality
    this.weightsIH = [
      [0.7, 0.6, 0.1, 0.0, 0.1, 0.0, 0.2, 0.1],  // temp
      [0.1, 0.0, 0.7, 0.6, 0.0, 0.1, 0.1, 0.2],  // humidity
      [0.0, 0.2, 0.0, 0.1, 0.8, 0.7, 0.1, 0.0],  // motion
      [0.2, 0.1, 0.1, 0.2, 0.0, 0.1, 0.7, 0.8],  // airQuality
    ];

    // Hidden→output (8×3): Normal, Anomaly, Alert
    this.weightsHO = [
      [0.5, 0.3, 0.1],  // h0 (temp specialist)
      [0.4, 0.4, 0.2],  // h1
      [0.5, 0.3, 0.1],  // h2 (humidity specialist)
      [0.4, 0.4, 0.2],  // h3
      [0.1, 0.5, 0.6],  // h4 (motion specialist → anomaly/alert)
      [0.2, 0.4, 0.5],  // h5
      [0.1, 0.4, 0.7],  // h6 (air quality → alert)
      [0.2, 0.3, 0.6],  // h7
    ];

    this.allNeurons = [...this.inputNeurons, ...this.hiddenNeurons, ...this.outputNeurons];
  }

  forward(spikeInputs) {
    // Step 1: Feed spikes to input neurons
    const inputSpikes = spikeInputs.map((current, i) =>
      this.inputNeurons[i].step(current) ? 1 : 0
    );

    // Step 2: Input → Hidden
    const hiddenCurrents = new Array(8).fill(0);
    for (let i = 0; i < 4; i++) {
      for (let h = 0; h < 8; h++) {
        hiddenCurrents[h] += inputSpikes[i] * this.weightsIH[i][h];
      }
    }

    const hiddenSpikes = hiddenCurrents.map((current, h) =>
      this.hiddenNeurons[h].step(current) ? 1 : 0
    );

    // Step 3: Hidden → Output
    const outputCurrents = new Array(3).fill(0);
    for (let h = 0; h < 8; h++) {
      for (let o = 0; o < 3; o++) {
        outputCurrents[o] += hiddenSpikes[h] * this.weightsHO[h][o];
      }
    }

    const outputSpikes = outputCurrents.map((current, o) =>
      this.outputNeurons[o].step(current) ? 1 : 0
    );

    // Winner-Take-All: classify based on output membrane potentials
    const potentials = this.outputNeurons.map(n => n.membranePotential);
    const maxIdx = potentials.indexOf(Math.max(...potentials));
    const total = potentials.reduce((a, b) => a + Math.abs(b), 0) || 1;
    const confidences = potentials.map(p => Math.abs(p) / total);

    const classes = ['normal', 'anomaly', 'alert'];
    return {
      classIndex: maxIdx,
      classification: classes[maxIdx],
      confidences: confidences,
      outputSpikes: outputSpikes,
      hiddenSpikes: hiddenSpikes,
      inputSpikes: inputSpikes,
    };
  }

  getAllMembranePotentials() {
    return this.allNeurons.map(n => n.membranePotential);
  }

  getAllFiringRates() {
    return this.allNeurons.map(n => n.getFiringRate());
  }

  reset() {
    this.allNeurons.forEach(n => n.reset());
  }
}

// ─────────────────────────────────────────
//  4. HOMEOSTATIC REGULATOR
// ─────────────────────────────────────────
class HomeostaticRegulator {
  constructor(highRate = 0.8, lowRate = 0.05, adjustFactor = 0.05) {
    this.highRate = highRate;
    this.lowRate = lowRate;
    this.adjustFactor = adjustFactor;
    this.adaptInterval = 50;
    this.tickCounter = 0;
  }

  adapt(neurons) {
    this.tickCounter++;
    if (this.tickCounter % this.adaptInterval !== 0) return;

    neurons.forEach(neuron => {
      const rate = neuron.getFiringRate(30);
      if (rate > this.highRate) {
        neuron.threshold *= (1 + this.adjustFactor); // increase threshold
      } else if (rate < this.lowRate) {
        neuron.threshold *= (1 - this.adjustFactor); // decrease threshold
        neuron.threshold = Math.max(0.3, neuron.threshold); // floor
      }
    });
  }
}

// ─────────────────────────────────────────
//  5. SENSOR ENGINE (Synthetic Data)
// ─────────────────────────────────────────
class SensorEngine {
  constructor() {
    this.time = 0; // elapsed seconds
    this.scenario = 'normal_day';
    this.anomalyInjections = {};

    // Previous values for delta detection
    this.prev = { temperature: 25, humidity: 55, motion: 0, airQuality: 400 };
  }

  tick(dt = 1) {
    this.time += dt;
    const t = this.time;

    let data;
    switch (this.scenario) {
      case 'fire_alarm':
        data = this._fireAlarm(t);
        break;
      case 'cooking':
        data = this._cooking(t);
        break;
      case 'hvac_failure':
        data = this._hvacFailure(t);
        break;
      case 'storm':
        data = this._storm(t);
        break;
      default:
        data = this._normalDay(t);
    }

    // Apply any manual anomaly injections
    for (const [sensor, severity] of Object.entries(this.anomalyInjections)) {
      if (sensor === 'temperature') data.temperature += severity * 20;
      if (sensor === 'humidity') data.humidity += severity * 30;
      if (sensor === 'motion') data.motion = severity > 0.5 ? 1 : data.motion;
      if (sensor === 'airQuality') data.airQuality += severity * 500;
    }

    // Clamp values
    data.temperature = Math.max(-10, Math.min(80, data.temperature));
    data.humidity = Math.max(0, Math.min(100, data.humidity));
    data.motion = data.motion > 0.5 ? 1 : 0;
    data.airQuality = Math.max(0, Math.min(2000, data.airQuality));

    this.prev = { ...data };
    return data;
  }

  _normalDay(t) {
    // Diurnal temperature: sine wave (18°C–32°C) over simulated day (300s = 1 "day")
    const dayPhase = (t % 300) / 300 * Math.PI * 2;
    const temperature = 25 + 7 * Math.sin(dayPhase) + this._noise(0.3);

    // Humidity inversely correlated
    const humidity = 60 - 15 * Math.sin(dayPhase) + this._noise(1.5);

    // Motion: Poisson events (~5% chance per tick)
    const motion = Math.random() < 0.05 ? 1 : 0;

    // Air quality: slow random walk
    const aqDrift = this.prev.airQuality + this._noise(5) - 0.1 * (this.prev.airQuality - 400);
    const airQuality = Math.max(200, Math.min(700, aqDrift));

    return { temperature, humidity, motion, airQuality };
  }

  _fireAlarm(t) {
    const rampUp = Math.min(1, t / 60); // ramps over 60s
    const temperature = 25 + rampUp * 45 + this._noise(2);
    const humidity = 30 - rampUp * 20 + this._noise(2);
    const motion = Math.random() < (0.1 + rampUp * 0.6) ? 1 : 0;
    const airQuality = 400 + rampUp * 1200 + this._noise(30);
    return { temperature, humidity, motion, airQuality };
  }

  _cooking(t) {
    const phase = Math.sin(t * 0.1);
    const temperature = 28 + 8 * Math.max(0, phase) + this._noise(0.5);
    const humidity = 65 + 15 * Math.max(0, phase) + this._noise(2);
    const motion = Math.random() < 0.3 ? 1 : 0;
    const airQuality = 450 + 200 * Math.max(0, phase) + this._noise(15);
    return { temperature, humidity, motion, airQuality };
  }

  _hvacFailure(t) {
    const ramp = Math.min(1, t / 120);
    const temperature = 22 + ramp * 18 + this._noise(0.5);
    const humidity = 45 + ramp * 35 + this._noise(2);
    const motion = Math.random() < 0.02 ? 1 : 0;
    const airQuality = 380 + ramp * 150 + this._noise(10);
    return { temperature, humidity, motion, airQuality };
  }

  _storm(t) {
    const lightning = Math.random() < 0.08;
    const temperature = 18 - 3 * Math.sin(t * 0.05) + this._noise(1);
    const humidity = 85 + 10 * Math.sin(t * 0.1) + this._noise(3);
    const motion = lightning ? 1 : (Math.random() < 0.15 ? 1 : 0);
    const airQuality = 350 + (lightning ? 200 : 0) + this._noise(10);
    return { temperature, humidity, motion, airQuality };
  }

  _noise(sigma) {
    // Box-Muller transform for Gaussian noise
    const u1 = Math.random();
    const u2 = Math.random();
    return sigma * Math.sqrt(-2 * Math.log(u1 || 0.001)) * Math.cos(2 * Math.PI * u2);
  }

  setScenario(name) {
    this.scenario = name;
    this.time = 0;
    this.anomalyInjections = {};
  }

  injectAnomaly(sensor, severity) {
    this.anomalyInjections[sensor] = severity;
  }

  clearAnomalies() {
    this.anomalyInjections = {};
  }
}

// ─────────────────────────────────────────
//  6. POWER TRACKER
// ─────────────────────────────────────────
class PowerTracker {
  constructor() {
    this.TIER0_UW = 8.5;
    this.TIER1_UW = 150;
    this.TIER2_UW = 2200;
    this.SLEEP_UW = 0.5;
    this.ADC_UW = 3.0;
    this.SPIKE_UW = 0.1;

    this.totalEnergy_uJ = 0;
    this.alwaysOnEnergy_uJ = 0; // comparison: always at tier-2
    this.currentPower_uW = this.TIER0_UW;
    this.history = [];
    this.maxHistory = 300;
    this.tickCount = 0;
  }

  recordTick(tier, numSpikes, dtMs = 1000) {
    this.tickCount++;
    const dtS = dtMs / 1000;

    // Calculate power for this tick
    let power = this.TIER0_UW; // baseline always-on
    power += this.ADC_UW * 4;  // 4 sensor reads
    power += this.SPIKE_UW * numSpikes;

    if (tier >= 1) power += this.TIER1_UW;
    if (tier >= 2) power += this.TIER2_UW;

    this.currentPower_uW = power;
    this.totalEnergy_uJ += power * dtS;
    this.alwaysOnEnergy_uJ += (this.TIER2_UW + this.ADC_UW * 4) * dtS;

    this.history.push({ tick: this.tickCount, tier, power });
    if (this.history.length > this.maxHistory) this.history.shift();

    return power;
  }

  getSavingsPercent() {
    if (this.alwaysOnEnergy_uJ === 0) return 0;
    return ((1 - this.totalEnergy_uJ / this.alwaysOnEnergy_uJ) * 100);
  }

  getAveragePower() {
    if (this.tickCount === 0) return 0;
    return this.totalEnergy_uJ / this.tickCount;
  }

  reset() {
    this.totalEnergy_uJ = 0;
    this.alwaysOnEnergy_uJ = 0;
    this.currentPower_uW = this.TIER0_UW;
    this.history = [];
    this.tickCount = 0;
  }
}

// ─────────────────────────────────────────
//  7. NEURO CORE (Orchestrator)
// ─────────────────────────────────────────
class NeuroCore {
  constructor() {
    this.encoder = new SpikeEncoder();
    this.snn = new SpikingNeuralNetwork();
    this.regulator = new HomeostaticRegulator();
    this.power = new PowerTracker();

    this.spikeBuffer = []; // circular buffer for replay
    this.maxBuffer = 200;

    this.prevSensorData = {
      temperature: 25, humidity: 55, motion: 0, airQuality: 400
    };

    // Sensor encoding params: [min, max, deltaThreshold]
    this.sensorParams = {
      temperature: [10, 60, 2.0],
      humidity: [20, 95, 5.0],
      motion: [0, 1, 0.5],
      airQuality: [200, 1500, 50],
    };
  }

  process(sensorData) {
    // Step 1: Encode sensor values into spike currents
    const sensors = ['temperature', 'humidity', 'motion', 'airQuality'];
    const currents = sensors.map(s => {
      const [min, max, dt] = this.sensorParams[s];
      return this.encoder.encode(sensorData[s], min, max, this.prevSensorData[s], dt);
    });

    // Step 2: Run SNN forward pass
    const result = this.snn.forward(currents);

    // Step 3: Determine active tier
    let tier = 0;
    if (result.classification === 'anomaly') tier = 1;
    if (result.classification === 'alert') tier = 2;

    // Count total spikes this tick
    const totalSpikes = [...result.inputSpikes, ...result.hiddenSpikes, ...result.outputSpikes]
      .filter(s => s).length;

    // Step 4: Record power
    const power = this.power.recordTick(tier, totalSpikes);

    // Step 5: Homeostatic adaptation
    this.regulator.adapt(this.snn.allNeurons);

    // Step 6: Store in spike buffer
    const spikeRecord = {
      tick: this.power.tickCount,
      spikes: {
        temperature: result.inputSpikes[0],
        humidity: result.inputSpikes[1],
        motion: result.inputSpikes[2],
        airQuality: result.inputSpikes[3],
      },
      tier,
      classification: result.classification,
    };
    this.spikeBuffer.push(spikeRecord);
    if (this.spikeBuffer.length > this.maxBuffer) this.spikeBuffer.shift();

    this.prevSensorData = { ...sensorData };

    return {
      spikes: spikeRecord.spikes,
      tier,
      classification: result.classification,
      confidences: result.confidences,
      membranePotentials: this.snn.getAllMembranePotentials(),
      firingRates: this.snn.getAllFiringRates(),
      power_uW: power,
      totalEnergy_uJ: this.power.totalEnergy_uJ,
      savings: this.power.getSavingsPercent(),
      avgPower: this.power.getAveragePower(),
      inputSpikes: result.inputSpikes,
      hiddenSpikes: result.hiddenSpikes,
      outputSpikes: result.outputSpikes,
    };
  }

  reset() {
    this.snn.reset();
    this.power.reset();
    this.spikeBuffer = [];
    this.prevSensorData = { temperature: 25, humidity: 55, motion: 0, airQuality: 400 };
  }
}

// ═══════════════════════════════════════════════════════
//  8. DASHBOARD RENDERER
// ═══════════════════════════════════════════════════════

class Dashboard {
  constructor() {
    this.sensorEngine = new SensorEngine();
    this.neuroCore = new NeuroCore();

    this.running = false;
    this.speed = 1; // simulation speed multiplier
    this.tickCount = 0;
    this.alerts = [];
    this.maxAlerts = 50;

    // Spike raster data
    this.rasterData = {
      temperature: [],
      humidity: [],
      motion: [],
      airQuality: [],
    };
    this.maxRasterPoints = 200;

    // Power chart data
    this.powerHistory = [];
    this.maxPowerHistory = 200;

    // Canvas contexts (initialized in init())
    this.rasterCtx = null;
    this.powerCtx = null;
    this.topoCtx = null;

    // Animation frame ID
    this.animFrameId = null;
    this.lastTime = 0;
    this.accumulator = 0;
    this.tickInterval = 500; // ms between ticks (adjustable via speed)
  }

  init() {
    // Get canvas contexts
    this.rasterCtx = document.getElementById('spikeRasterCanvas').getContext('2d');
    this.powerCtx = document.getElementById('powerChartCanvas').getContext('2d');
    this.topoCtx = document.getElementById('topologyCanvas').getContext('2d');

    // Bind controls
    document.getElementById('btnStartStop').addEventListener('click', () => this.toggleRun());
    document.getElementById('btnReset').addEventListener('click', () => this.reset());
    document.getElementById('btnInjectFire').addEventListener('click', () => this.injectScenario('fire_alarm'));
    document.getElementById('btnInjectCooking').addEventListener('click', () => this.injectScenario('cooking'));
    document.getElementById('scenarioSelect').addEventListener('change', (e) => {
      this.sensorEngine.setScenario(e.target.value);
      this.addAlert('info', `Scenario changed: ${e.target.value}`);
    });
    document.getElementById('speedSlider').addEventListener('input', (e) => {
      this.speed = parseFloat(e.target.value);
      document.getElementById('speedValue').textContent = `${this.speed.toFixed(1)}x`;
    });

    // Initial topology draw
    this.drawTopology();

    // Start
    this.toggleRun();
  }

  toggleRun() {
    this.running = !this.running;
    const btn = document.getElementById('btnStartStop');
    btn.textContent = this.running ? '⏸ Pause' : '▶ Resume';
    btn.classList.toggle('active', this.running);

    if (this.running) {
      this.lastTime = performance.now();
      this.loop();
    } else if (this.animFrameId) {
      cancelAnimationFrame(this.animFrameId);
    }
  }

  reset() {
    this.running = false;
    const btn = document.getElementById('btnStartStop');
    btn.textContent = '▶ Start';
    btn.classList.remove('active');

    if (this.animFrameId) cancelAnimationFrame(this.animFrameId);

    this.sensorEngine = new SensorEngine();
    this.neuroCore = new NeuroCore();
    this.tickCount = 0;
    this.alerts = [];
    this.rasterData = { temperature: [], humidity: [], motion: [], airQuality: [] };
    this.powerHistory = [];

    this.updateGauges({ temperature: 25, humidity: 55, motion: 0, airQuality: 400 });
    this.updateTierIndicator(0);
    this.updatePowerDisplay(8.5, 0, 0, 0);
    this.clearRaster();
    this.clearPowerChart();
    this.updateAlertFeed();
    this.updateMembraneBars(new Array(15).fill(0), []);
    this.updateSavingsBanner(0, 0, 0);
    document.getElementById('simClock').textContent = 'T+0s';
  }

  injectScenario(scenario) {
    this.sensorEngine.setScenario(scenario);
    this.addAlert('alert', `🔥 INJECTED: ${scenario.replace('_', ' ').toUpperCase()}`);
  }

  loop() {
    if (!this.running) return;

    const now = performance.now();
    const delta = now - this.lastTime;
    this.lastTime = now;

    this.accumulator += delta * this.speed;
    const interval = this.tickInterval;

    while (this.accumulator >= interval) {
      this.accumulator -= interval;
      this.simulationTick();
    }

    this.animFrameId = requestAnimationFrame(() => this.loop());
  }

  simulationTick() {
    this.tickCount++;

    // 1. Generate sensor data
    const sensorData = this.sensorEngine.tick(1);

    // 2. Process through neuromorphic core
    const result = this.neuroCore.process(sensorData);

    // 3. Update all visualizations
    this.updateGauges(sensorData);
    this.updateSpikeIndicators(result.spikes);
    this.updateTierIndicator(result.tier);
    this.updatePowerDisplay(result.power_uW, result.totalEnergy_uJ, result.savings, result.avgPower);
    this.updateRasterData(result.spikes);
    this.drawRaster();
    this.updatePowerHistory(result.power_uW);
    this.drawPowerChart();
    this.updateMembraneBars(result.membranePotentials, result.firingRates);
    this.updateSavingsBanner(result.savings, result.totalEnergy_uJ, this.neuroCore.power.alwaysOnEnergy_uJ);
    this.drawTopology(result);

    // 4. Generate alerts on events
    if (result.tier >= 1) {
      this.addAlert(result.classification,
        `${result.classification.toUpperCase()}: Tier-${result.tier} activated | ` +
        `Conf: [${result.confidences.map(c => (c * 100).toFixed(0) + '%').join(', ')}] | ` +
        `Power: ${result.power_uW.toFixed(1)} µW`
      );
    } else if (this.tickCount % 20 === 0) {
      // Periodic heartbeat
      this.addAlert('normal',
        `Heartbeat OK — Avg: ${result.avgPower.toFixed(1)} µW | Savings: ${result.savings.toFixed(1)}%`
      );
    }

    // Update clock
    document.getElementById('simClock').textContent = `T+${this.tickCount}s`;
  }

  // ─── Gauge Updates ───
  updateGauges(data) {
    this._setGauge('temp', data.temperature, -10, 80, '°C', 1);
    this._setGauge('humidity', data.humidity, 0, 100, '%RH', 1);
    this._setGauge('motion', data.motion * 100, 0, 100, data.motion ? 'ACTIVE' : 'IDLE', 0);
    this._setGauge('airquality', data.airQuality, 0, 2000, 'PPM', 0);
  }

  _setGauge(id, value, min, max, unit, decimals) {
    const normalized = Math.max(0, Math.min(1, (value - min) / (max - min)));
    const circumference = 2 * Math.PI * 60; // r=60
    const offset = circumference * (1 - normalized);

    const fill = document.getElementById(`gauge-fill-${id}`);
    const numEl = document.getElementById(`gauge-num-${id}`);
    const unitEl = document.getElementById(`gauge-unit-${id}`);

    if (fill) fill.style.strokeDashoffset = offset;
    if (numEl) numEl.textContent = typeof decimals === 'number' ? value.toFixed(decimals) : value;
    if (unitEl) unitEl.textContent = unit;
  }

  updateSpikeIndicators(spikes) {
    const sensorMap = { temperature: 'temp', humidity: 'humidity', motion: 'motion', airQuality: 'airquality' };
    for (const [sensor, spiked] of Object.entries(spikes)) {
      const el = document.getElementById(`spike-${sensorMap[sensor]}`);
      if (el) {
        el.classList.toggle('active', spiked);
      }
    }
  }

  // ─── Tier Indicator ───
  updateTierIndicator(tier) {
    for (let i = 0; i <= 2; i++) {
      const node = document.getElementById(`tier-node-${i}`);
      if (node) node.classList.toggle('active', i <= tier);
    }
    const conn01 = document.getElementById('tier-conn-01');
    const conn12 = document.getElementById('tier-conn-12');
    if (conn01) conn01.classList.toggle('active', tier >= 1);
    if (conn12) {
      conn12.classList.toggle('active', tier >= 2);
      conn12.classList.toggle('active2', tier >= 2);
    }
  }

  // ─── Power Display ───
  updatePowerDisplay(power, totalEnergy, savings, avgPower) {
    const valEl = document.getElementById('powerValue');
    if (valEl) {
      valEl.textContent = power.toFixed(1);
      valEl.className = 'power-value';
      if (power > 200) valEl.classList.add('tier1');
      if (power > 2000) valEl.classList.add('tier2');
    }

    const totalEl = document.getElementById('statTotalEnergy');
    if (totalEl) totalEl.textContent = (totalEnergy / 1000).toFixed(2) + ' mJ';

    const avgEl = document.getElementById('statAvgPower');
    if (avgEl) avgEl.textContent = avgPower.toFixed(1) + ' µW';

    const savEl = document.getElementById('statSavings');
    if (savEl) savEl.textContent = savings.toFixed(1) + '%';
  }

  // ─── Spike Raster Plot ───
  updateRasterData(spikes) {
    const sensors = ['temperature', 'humidity', 'motion', 'airQuality'];
    sensors.forEach(s => {
      this.rasterData[s].push(spikes[s] ? 1 : 0);
      if (this.rasterData[s].length > this.maxRasterPoints) this.rasterData[s].shift();
    });
  }

  clearRaster() {
    if (!this.rasterCtx) return;
    const canvas = this.rasterCtx.canvas;
    this.rasterCtx.clearRect(0, 0, canvas.width, canvas.height);
  }

  drawRaster() {
    const ctx = this.rasterCtx;
    const canvas = ctx.canvas;
    const dpr = window.devicePixelRatio || 1;

    canvas.width = canvas.clientWidth * dpr;
    canvas.height = canvas.clientHeight * dpr;
    ctx.scale(dpr, dpr);

    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    const leftPad = 75;

    ctx.clearRect(0, 0, w, h);

    // Background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
    ctx.fillRect(0, 0, w, h);

    const sensors = ['temperature', 'humidity', 'motion', 'airQuality'];
    const colors = ['#ff6e40', '#40c4ff', '#eeff41', '#b2ff59'];
    const labels = ['TEMP', 'HUMID', 'MOTION', 'AIR Q'];
    const rowH = h / 4;

    sensors.forEach((sensor, row) => {
      const data = this.rasterData[sensor];
      const y = row * rowH + rowH / 2;

      // Row label
      ctx.fillStyle = colors[row];
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'right';
      ctx.fillText(labels[row], leftPad - 8, y + 4);

      // Horizontal guide line
      ctx.strokeStyle = 'rgba(255,255,255,0.03)';
      ctx.beginPath();
      ctx.moveTo(leftPad, y);
      ctx.lineTo(w, y);
      ctx.stroke();

      // Spike dots
      const dataW = w - leftPad;
      data.forEach((spike, i) => {
        if (spike) {
          const x = leftPad + (i / this.maxRasterPoints) * dataW;
          ctx.beginPath();
          ctx.arc(x, y, 2.5, 0, Math.PI * 2);
          ctx.fillStyle = colors[row];
          ctx.fill();

          // Glow
          ctx.beginPath();
          ctx.arc(x, y, 5, 0, Math.PI * 2);
          ctx.fillStyle = colors[row] + '30';
          ctx.fill();
        }
      });
    });
  }

  // ─── Power Chart ───
  updatePowerHistory(power) {
    this.powerHistory.push(power);
    if (this.powerHistory.length > this.maxPowerHistory) this.powerHistory.shift();
  }

  clearPowerChart() {
    if (!this.powerCtx) return;
    this.powerCtx.clearRect(0, 0, this.powerCtx.canvas.width, this.powerCtx.canvas.height);
  }

  drawPowerChart() {
    const ctx = this.powerCtx;
    const canvas = ctx.canvas;
    const dpr = window.devicePixelRatio || 1;

    canvas.width = canvas.clientWidth * dpr;
    canvas.height = canvas.clientHeight * dpr;
    ctx.scale(dpr, dpr);

    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    const pad = 10;

    ctx.clearRect(0, 0, w, h);

    if (this.powerHistory.length < 2) return;

    const maxPower = Math.max(100, ...this.powerHistory);
    const data = this.powerHistory;

    // Area fill
    const gradient = ctx.createLinearGradient(0, 0, 0, h);
    gradient.addColorStop(0, 'rgba(0, 229, 255, 0.3)');
    gradient.addColorStop(1, 'rgba(0, 229, 255, 0)');

    ctx.beginPath();
    ctx.moveTo(pad, h - pad);
    data.forEach((val, i) => {
      const x = pad + (i / (this.maxPowerHistory - 1)) * (w - 2 * pad);
      const y = h - pad - (val / maxPower) * (h - 2 * pad);
      ctx.lineTo(x, y);
    });
    ctx.lineTo(pad + ((data.length - 1) / (this.maxPowerHistory - 1)) * (w - 2 * pad), h - pad);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    // Line
    ctx.beginPath();
    data.forEach((val, i) => {
      const x = pad + (i / (this.maxPowerHistory - 1)) * (w - 2 * pad);
      const y = h - pad - (val / maxPower) * (h - 2 * pad);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = '#00e5ff';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Tier threshold lines
    const thresholds = [
      { val: 8.5, label: 'T0', color: '#69f0ae40' },
      { val: 150, label: 'T1', color: '#ffab4040' },
      { val: 2200, label: 'T2', color: '#ff525240' },
    ];

    thresholds.forEach(({ val, label, color }) => {
      if (val <= maxPower) {
        const y = h - pad - (val / maxPower) * (h - 2 * pad);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(pad, y);
        ctx.lineTo(w - pad, y);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = color.replace('40', 'aa');
        ctx.font = '9px "JetBrains Mono"';
        ctx.fillText(label, w - pad - 15, y - 3);
      }
    });
  }

  // ─── Membrane Potential Bars ───
  updateMembraneBars(potentials, firingRates) {
    const container = document.getElementById('membraneBars');
    if (!container) return;

    const labels = [
      'IN:T', 'IN:H', 'IN:M', 'IN:A',
      'H:0', 'H:1', 'H:2', 'H:3', 'H:4', 'H:5', 'H:6', 'H:7',
      'O:N', 'O:An', 'O:Al'
    ];
    const colors = [
      '#ff6e40', '#40c4ff', '#eeff41', '#b2ff59',
      '#b388ff', '#b388ff', '#b388ff', '#b388ff', '#b388ff', '#b388ff', '#b388ff', '#b388ff',
      '#69f0ae', '#ffab40', '#ff5252'
    ];

    // Build bars if not yet created
    if (container.children.length === 0) {
      labels.forEach((label, i) => {
        const group = document.createElement('div');
        group.className = 'membrane-bar-group';
        group.innerHTML = `
          <div class="membrane-bar" id="mbar-${i}" style="background: ${colors[i]}; height: 2px;"></div>
          <div class="membrane-bar-label">${label}</div>
        `;
        container.appendChild(group);
      });
    }

    // Update bar heights
    potentials.forEach((p, i) => {
      const bar = document.getElementById(`mbar-${i}`);
      if (bar) {
        const height = Math.min(100, Math.max(2, Math.abs(p) * 100));
        bar.style.height = `${height}px`;
        bar.style.background = colors[i];
        bar.style.opacity = Math.max(0.3, Math.min(1, Math.abs(p)));

        // Flash on spike (high potential)
        if (Math.abs(p) > 0.8) {
          bar.classList.add('spiked');
          setTimeout(() => bar.classList.remove('spiked'), 400);
        }
      }
    });
  }

  // ─── SNN Topology Visualizer ───
  drawTopology(result = null) {
    const ctx = this.topoCtx;
    const canvas = ctx.canvas;
    const dpr = window.devicePixelRatio || 1;

    canvas.width = canvas.clientWidth * dpr;
    canvas.height = canvas.clientHeight * dpr;
    ctx.scale(dpr, dpr);

    const w = canvas.clientWidth;
    const h = canvas.clientHeight;

    ctx.clearRect(0, 0, w, h);

    // Layout: 3 columns — Input(4), Hidden(8), Output(3)
    const layers = [4, 8, 3];
    const layerX = [w * 0.15, w * 0.5, w * 0.85];
    const layerColors = [
      ['#ff6e40', '#40c4ff', '#eeff41', '#b2ff59'], // input
      Array(8).fill('#b388ff'), // hidden
      ['#69f0ae', '#ffab40', '#ff5252'], // output
    ];
    const layerLabels = [
      ['Temp', 'Humid', 'Motion', 'AirQ'],
      ['H0', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7'],
      ['Normal', 'Anomaly', 'Alert'],
    ];

    // Calculate node positions
    const nodePositions = layers.map((count, li) => {
      const positions = [];
      const totalH = (count - 1) * 30;
      const startY = (h - totalH) / 2;
      for (let i = 0; i < count; i++) {
        positions.push({ x: layerX[li], y: startY + i * 30 });
      }
      return positions;
    });

    // Gather spike info
    const spiked = result ? [
      result.inputSpikes || [0, 0, 0, 0],
      result.hiddenSpikes || new Array(8).fill(0),
      result.outputSpikes || [0, 0, 0],
    ] : [new Array(4).fill(0), new Array(8).fill(0), new Array(3).fill(0)];

    // Draw connections: Input→Hidden
    for (let i = 0; i < 4; i++) {
      for (let h = 0; h < 8; h++) {
        const weight = this.neuroCore.snn.weightsIH[i][h];
        if (weight < 0.15) continue;
        const from = nodePositions[0][i];
        const to = nodePositions[1][h];
        const active = spiked[0][i] && weight > 0.3;

        ctx.beginPath();
        ctx.moveTo(from.x + 10, from.y);
        ctx.lineTo(to.x - 10, to.y);
        ctx.strokeStyle = active ?
          `rgba(0, 229, 255, ${0.3 + weight * 0.7})` :
          `rgba(255, 255, 255, ${weight * 0.1})`;
        ctx.lineWidth = active ? 2 : 0.5;
        ctx.stroke();
      }
    }

    // Draw connections: Hidden→Output
    for (let h = 0; h < 8; h++) {
      for (let o = 0; o < 3; o++) {
        const weight = this.neuroCore.snn.weightsHO[h][o];
        if (weight < 0.15) continue;
        const from = nodePositions[1][h];
        const to = nodePositions[2][o];
        const active = spiked[1][h] && weight > 0.3;

        ctx.beginPath();
        ctx.moveTo(from.x + 10, from.y);
        ctx.lineTo(to.x - 10, to.y);
        ctx.strokeStyle = active ?
          `rgba(179, 136, 255, ${0.3 + weight * 0.7})` :
          `rgba(255, 255, 255, ${weight * 0.1})`;
        ctx.lineWidth = active ? 2 : 0.5;
        ctx.stroke();
      }
    }

    // Draw nodes
    layers.forEach((count, li) => {
      for (let i = 0; i < count; i++) {
        const pos = nodePositions[li][i];
        const color = layerColors[li][i];
        const active = spiked[li][i];
        const r = active ? 9 : 7;

        // Glow
        if (active) {
          ctx.beginPath();
          ctx.arc(pos.x, pos.y, 16, 0, Math.PI * 2);
          ctx.fillStyle = color + '30';
          ctx.fill();
        }

        // Node circle
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2);
        ctx.fillStyle = active ? color : color + '60';
        ctx.fill();
        ctx.strokeStyle = color;
        ctx.lineWidth = active ? 2 : 1;
        ctx.stroke();

        // Label
        ctx.fillStyle = active ? '#fff' : '#9fa8c780';
        ctx.font = '9px "JetBrains Mono"';
        ctx.textAlign = 'center';
        ctx.fillText(layerLabels[li][i], pos.x, pos.y + (li === 1 ? -14 : 20));
      }
    });

    // Layer headers
    ctx.fillStyle = '#5c6480';
    ctx.font = '11px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('Input (Tier 0)', layerX[0], 18);
    ctx.fillText('Hidden (Tier 1)', layerX[1], 18);
    ctx.fillText('Output (Tier 2)', layerX[2], 18);
  }

  // ─── Alert Feed ───
  addAlert(severity, message) {
    const time = `${this.tickCount}s`;
    this.alerts.unshift({ time, severity, message });
    if (this.alerts.length > this.maxAlerts) this.alerts.pop();
    this.updateAlertFeed();
  }

  updateAlertFeed() {
    const container = document.getElementById('alertFeed');
    if (!container) return;

    container.innerHTML = this.alerts.map(a => `
      <div class="alert-item ${a.severity}">
        <span class="alert-time">${a.time}</span>
        <span class="alert-severity ${a.severity}">${a.severity}</span>
        <span class="alert-message">${a.message}</span>
      </div>
    `).join('');
  }

  // ─── Savings Banner ───
  updateSavingsBanner(savings, neuroEnergy, alwaysOnEnergy) {
    const savEl = document.getElementById('savingsPercent');
    const neuroEl = document.getElementById('neuroEnergy');
    const alwaysEl = document.getElementById('alwaysOnEnergy');

    if (savEl) savEl.textContent = `${savings.toFixed(1)}%`;
    if (neuroEl) neuroEl.textContent = `${(neuroEnergy / 1000).toFixed(2)} mJ`;
    if (alwaysEl) alwaysEl.textContent = `${(alwaysOnEnergy / 1000).toFixed(2)} mJ`;
  }
}

// ═══════════════════════════════════════════════════════
//  9. INITIALIZATION
// ═══════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  const dashboard = new Dashboard();
  dashboard.init();

  // Expose for debugging
  window.neuroEdge = dashboard;
});
