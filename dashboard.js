/* ═══════════════════════════════════════════════════════════════
   NeuroEdge — Dashboard Engine (Redesigned)
   Simulation core unchanged. Only rendering updated for new DOM.
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
      this.membranePotential *= 0.5;
      this.spikeHistory.push(false);
      if (this.spikeHistory.length > this.maxHistory) this.spikeHistory.shift();
      return false;
    }
    this.membranePotential = this.decay * this.membranePotential + inputCurrent;
    if (this.membranePotential >= this.threshold) {
      this.spiked = true;
      this.membranePotential = 0;
      this.refractoryCounter = this.refractoryPeriod;
    }
    this.spikeHistory.push(this.spiked);
    if (this.spikeHistory.length > this.maxHistory) this.spikeHistory.shift();
    return this.spiked;
  }

  getFiringRate(window = 20) {
    const recent = this.spikeHistory.slice(-window);
    return recent.length === 0 ? 0 : recent.filter(s => s).length / recent.length;
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
  rateEncode(value, minVal, maxVal, maxRate = 0.8) {
    const normalized = Math.max(0, Math.min(1, (value - minVal) / (maxVal - minVal)));
    return Math.random() < (normalized * maxRate) ? 1 : 0;
  }

  deltaEncode(current, previous, threshold = 0.5) {
    return Math.abs(current - previous) > threshold ? 1 : 0;
  }

  encode(value, minVal, maxVal, prevValue, deltaThreshold) {
    const rate = this.rateEncode(value, minVal, maxVal, 0.6);
    const delta = this.deltaEncode(value, prevValue, deltaThreshold);
    return Math.min(1, rate * 0.4 + delta * 0.8);
  }
}

// ─────────────────────────────────────────
//  3. SPIKING NEURAL NETWORK
// ─────────────────────────────────────────
class SpikingNeuralNetwork {
  constructor() {
    this.inputNeurons = Array.from({ length: 4 }, () => new LIFNeuron(0.8, 0.85, 2));
    this.hiddenNeurons = Array.from({ length: 8 }, () => new LIFNeuron(1.0, 0.9, 3));
    this.outputNeurons = Array.from({ length: 3 }, () => new LIFNeuron(1.2, 0.88, 4));

    this.weightsIH = [
      [0.7, 0.6, 0.1, 0.0, 0.1, 0.0, 0.2, 0.1],
      [0.1, 0.0, 0.7, 0.6, 0.0, 0.1, 0.1, 0.2],
      [0.0, 0.2, 0.0, 0.1, 0.8, 0.7, 0.1, 0.0],
      [0.2, 0.1, 0.1, 0.2, 0.0, 0.1, 0.7, 0.8],
    ];

    this.weightsHO = [
      [0.5, 0.3, 0.1],
      [0.4, 0.4, 0.2],
      [0.5, 0.3, 0.1],
      [0.4, 0.4, 0.2],
      [0.1, 0.5, 0.6],
      [0.2, 0.4, 0.5],
      [0.1, 0.4, 0.7],
      [0.2, 0.3, 0.6],
    ];

    this.allNeurons = [...this.inputNeurons, ...this.hiddenNeurons, ...this.outputNeurons];
  }

  forward(spikeInputs) {
    const inputSpikes = spikeInputs.map((current, i) =>
      this.inputNeurons[i].step(current) ? 1 : 0
    );

    const hiddenCurrents = new Array(8).fill(0);
    for (let i = 0; i < 4; i++)
      for (let h = 0; h < 8; h++)
        hiddenCurrents[h] += inputSpikes[i] * this.weightsIH[i][h];

    const hiddenSpikes = hiddenCurrents.map((current, h) =>
      this.hiddenNeurons[h].step(current) ? 1 : 0
    );

    const outputCurrents = new Array(3).fill(0);
    for (let h = 0; h < 8; h++)
      for (let o = 0; o < 3; o++)
        outputCurrents[o] += hiddenSpikes[h] * this.weightsHO[h][o];

    const outputSpikes = outputCurrents.map((current, o) =>
      this.outputNeurons[o].step(current) ? 1 : 0
    );

    const potentials = this.outputNeurons.map(n => n.membranePotential);
    const maxIdx = potentials.indexOf(Math.max(...potentials));
    const total = potentials.reduce((a, b) => a + Math.abs(b), 0) || 1;
    const confidences = potentials.map(p => Math.abs(p) / total);

    return {
      classIndex: maxIdx,
      classification: ['normal', 'anomaly', 'alert'][maxIdx],
      confidences, outputSpikes, hiddenSpikes, inputSpikes,
    };
  }

  getAllMembranePotentials() { return this.allNeurons.map(n => n.membranePotential); }
  getAllFiringRates() { return this.allNeurons.map(n => n.getFiringRate()); }
  reset() { this.allNeurons.forEach(n => n.reset()); }
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
      if (rate > this.highRate) neuron.threshold *= (1 + this.adjustFactor);
      else if (rate < this.lowRate) {
        neuron.threshold *= (1 - this.adjustFactor);
        neuron.threshold = Math.max(0.3, neuron.threshold);
      }
    });
  }
}

// ─────────────────────────────────────────
//  5. SENSOR ENGINE
// ─────────────────────────────────────────
class SensorEngine {
  constructor() {
    this.time = 0;
    this.scenario = 'normal_day';
    this.anomalyInjections = {};
    this.prev = { temperature: 25, humidity: 55, motion: 0, airQuality: 400 };
  }

  tick(dt = 1) {
    this.time += dt;
    const t = this.time;
    let data;
    switch (this.scenario) {
      case 'fire_alarm': data = this._fireAlarm(t); break;
      case 'cooking': data = this._cooking(t); break;
      case 'hvac_failure': data = this._hvacFailure(t); break;
      case 'storm': data = this._storm(t); break;
      default: data = this._normalDay(t);
    }
    for (const [sensor, severity] of Object.entries(this.anomalyInjections)) {
      if (sensor === 'temperature') data.temperature += severity * 20;
      if (sensor === 'humidity') data.humidity += severity * 30;
      if (sensor === 'motion') data.motion = severity > 0.5 ? 1 : data.motion;
      if (sensor === 'airQuality') data.airQuality += severity * 500;
    }
    data.temperature = Math.max(-10, Math.min(80, data.temperature));
    data.humidity = Math.max(0, Math.min(100, data.humidity));
    data.motion = data.motion > 0.5 ? 1 : 0;
    data.airQuality = Math.max(0, Math.min(2000, data.airQuality));
    this.prev = { ...data };
    return data;
  }

  _normalDay(t) {
    const dayPhase = (t % 300) / 300 * Math.PI * 2;
    const temperature = 25 + 7 * Math.sin(dayPhase) + this._noise(0.3);
    const humidity = 60 - 15 * Math.sin(dayPhase) + this._noise(1.5);
    const motion = Math.random() < 0.05 ? 1 : 0;
    const aqDrift = this.prev.airQuality + this._noise(5) - 0.1 * (this.prev.airQuality - 400);
    const airQuality = Math.max(200, Math.min(700, aqDrift));
    return { temperature, humidity, motion, airQuality };
  }

  _fireAlarm(t) {
    const r = Math.min(1, t / 60);
    return {
      temperature: 25 + r * 45 + this._noise(2),
      humidity: 30 - r * 20 + this._noise(2),
      motion: Math.random() < (0.1 + r * 0.6) ? 1 : 0,
      airQuality: 400 + r * 1200 + this._noise(30),
    };
  }

  _cooking(t) {
    const p = Math.sin(t * 0.1);
    return {
      temperature: 28 + 8 * Math.max(0, p) + this._noise(0.5),
      humidity: 65 + 15 * Math.max(0, p) + this._noise(2),
      motion: Math.random() < 0.3 ? 1 : 0,
      airQuality: 450 + 200 * Math.max(0, p) + this._noise(15),
    };
  }

  _hvacFailure(t) {
    const r = Math.min(1, t / 120);
    return {
      temperature: 22 + r * 18 + this._noise(0.5),
      humidity: 45 + r * 35 + this._noise(2),
      motion: Math.random() < 0.02 ? 1 : 0,
      airQuality: 380 + r * 150 + this._noise(10),
    };
  }

  _storm(t) {
    const lightning = Math.random() < 0.08;
    return {
      temperature: 18 - 3 * Math.sin(t * 0.05) + this._noise(1),
      humidity: 85 + 10 * Math.sin(t * 0.1) + this._noise(3),
      motion: lightning ? 1 : (Math.random() < 0.15 ? 1 : 0),
      airQuality: 350 + (lightning ? 200 : 0) + this._noise(10),
    };
  }

  _noise(sigma) {
    const u1 = Math.random(), u2 = Math.random();
    return sigma * Math.sqrt(-2 * Math.log(u1 || 0.001)) * Math.cos(2 * Math.PI * u2);
  }

  setScenario(name) { this.scenario = name; this.time = 0; this.anomalyInjections = {}; }
  injectAnomaly(sensor, severity) { this.anomalyInjections[sensor] = severity; }
  clearAnomalies() { this.anomalyInjections = {}; }
}

// ─────────────────────────────────────────
//  6. POWER TRACKER
// ─────────────────────────────────────────
class PowerTracker {
  constructor() {
    this.TIER0_UW = 8.5; this.TIER1_UW = 150; this.TIER2_UW = 2200;
    this.SLEEP_UW = 0.5; this.ADC_UW = 3.0; this.SPIKE_UW = 0.1;
    this.totalEnergy_uJ = 0; this.alwaysOnEnergy_uJ = 0;
    this.currentPower_uW = this.TIER0_UW;
    this.history = []; this.maxHistory = 300; this.tickCount = 0;
  }

  recordTick(tier, numSpikes, dtMs = 1000) {
    this.tickCount++;
    const dtS = dtMs / 1000;
    let power = this.TIER0_UW + this.ADC_UW * 4 + this.SPIKE_UW * numSpikes;
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
    return this.alwaysOnEnergy_uJ === 0 ? 0 : ((1 - this.totalEnergy_uJ / this.alwaysOnEnergy_uJ) * 100);
  }

  getAveragePower() { return this.tickCount === 0 ? 0 : this.totalEnergy_uJ / this.tickCount; }

  reset() {
    this.totalEnergy_uJ = 0; this.alwaysOnEnergy_uJ = 0;
    this.currentPower_uW = this.TIER0_UW;
    this.history = []; this.tickCount = 0;
  }
}

// ─────────────────────────────────────────
//  7. NEURO CORE
// ─────────────────────────────────────────
class NeuroCore {
  constructor() {
    this.encoder = new SpikeEncoder();
    this.snn = new SpikingNeuralNetwork();
    this.regulator = new HomeostaticRegulator();
    this.power = new PowerTracker();
    this.spikeBuffer = []; this.maxBuffer = 200;
    this.prevSensorData = { temperature: 25, humidity: 55, motion: 0, airQuality: 400 };
    this.sensorParams = {
      temperature: [10, 60, 2.0], humidity: [20, 95, 5.0],
      motion: [0, 1, 0.5], airQuality: [200, 1500, 50],
    };
  }

  process(sensorData) {
    const sensors = ['temperature', 'humidity', 'motion', 'airQuality'];
    const currents = sensors.map(s => {
      const [min, max, dt] = this.sensorParams[s];
      return this.encoder.encode(sensorData[s], min, max, this.prevSensorData[s], dt);
    });

    const result = this.snn.forward(currents);

    let tier = 0;
    if (result.classification === 'anomaly') tier = 1;
    if (result.classification === 'alert') tier = 2;

    const totalSpikes = [...result.inputSpikes, ...result.hiddenSpikes, ...result.outputSpikes].filter(s => s).length;
    const power = this.power.recordTick(tier, totalSpikes);
    this.regulator.adapt(this.snn.allNeurons);

    const spikeRecord = {
      tick: this.power.tickCount,
      spikes: { temperature: result.inputSpikes[0], humidity: result.inputSpikes[1], motion: result.inputSpikes[2], airQuality: result.inputSpikes[3] },
      tier, classification: result.classification,
    };
    this.spikeBuffer.push(spikeRecord);
    if (this.spikeBuffer.length > this.maxBuffer) this.spikeBuffer.shift();
    this.prevSensorData = { ...sensorData };

    return {
      spikes: spikeRecord.spikes, tier,
      classification: result.classification, confidences: result.confidences,
      membranePotentials: this.snn.getAllMembranePotentials(),
      firingRates: this.snn.getAllFiringRates(),
      power_uW: power, totalEnergy_uJ: this.power.totalEnergy_uJ,
      savings: this.power.getSavingsPercent(), avgPower: this.power.getAveragePower(),
      inputSpikes: result.inputSpikes, hiddenSpikes: result.hiddenSpikes, outputSpikes: result.outputSpikes,
    };
  }

  reset() {
    this.snn.reset(); this.power.reset(); this.spikeBuffer = [];
    this.prevSensorData = { temperature: 25, humidity: 55, motion: 0, airQuality: 400 };
  }
}

// ═══════════════════════════════════════════════════════
//  8. DASHBOARD RENDERER (Redesigned for new DOM)
// ═══════════════════════════════════════════════════════

class Dashboard {
  constructor() {
    this.sensorEngine = new SensorEngine();
    this.neuroCore = new NeuroCore();
    this.running = false;
    this.speed = 1;
    this.tickCount = 0;
    this.alerts = [];
    this.maxAlerts = 60;
    this.rasterData = { temperature: [], humidity: [], motion: [], airQuality: [] };
    this.maxRasterPoints = 200;
    this.powerHistory = [];
    this.maxPowerHistory = 200;
    this.rasterCtx = null;
    this.powerCtx = null;
    this.topoCtx = null;
    this.animFrameId = null;
    this.lastTime = 0;
    this.accumulator = 0;
    this.tickInterval = 500;
  }

  init() {
    this.rasterCtx = document.getElementById('spikeRasterCanvas').getContext('2d');
    this.powerCtx = document.getElementById('powerChartCanvas').getContext('2d');
    this.topoCtx = document.getElementById('topologyCanvas').getContext('2d');

    document.getElementById('btnStartStop').addEventListener('click', () => this.toggleRun());
    document.getElementById('btnReset').addEventListener('click', () => this.reset());
    document.getElementById('btnInjectFire').addEventListener('click', () => this.injectScenario('fire_alarm'));
    document.getElementById('scenarioSelect').addEventListener('change', (e) => {
      this.sensorEngine.setScenario(e.target.value);
      this.addAlert('info', `Scenario: ${e.target.value.replace(/_/g, ' ')}`);
    });
    document.getElementById('speedSlider').addEventListener('input', (e) => {
      this.speed = parseFloat(e.target.value);
      document.getElementById('speedValue').textContent = `${this.speed.toFixed(1)}×`;
    });

    this.drawTopology();
    this.toggleRun();
  }

  toggleRun() {
    this.running = !this.running;
    const btn = document.getElementById('btnStartStop');
    btn.textContent = this.running ? 'Pause' : 'Resume';
    btn.classList.toggle('active', this.running);
    document.getElementById('statusPip').classList.toggle('paused', !this.running);

    if (this.running) { this.lastTime = performance.now(); this.loop(); }
    else if (this.animFrameId) cancelAnimationFrame(this.animFrameId);
  }

  reset() {
    this.running = false;
    const btn = document.getElementById('btnStartStop');
    btn.textContent = 'Start';
    btn.classList.remove('active');
    document.getElementById('statusPip').classList.add('paused');
    if (this.animFrameId) cancelAnimationFrame(this.animFrameId);

    this.sensorEngine = new SensorEngine();
    this.neuroCore = new NeuroCore();
    this.tickCount = 0;
    this.alerts = [];
    this.rasterData = { temperature: [], humidity: [], motion: [], airQuality: [] };
    this.powerHistory = [];

    this.updateSensors({ temperature: 25, humidity: 55, motion: 0, airQuality: 400 }, {});
    this.updateTier(0);
    this.updatePower(8.5, 0, 0, 0);
    this.clearCanvas(this.rasterCtx);
    this.clearCanvas(this.powerCtx);
    this.updateAlertFeed();
    this.updateMembrane(new Array(15).fill(0));
    this.updateSavings(0, 0, 0);
    document.getElementById('simClock').textContent = 'T+0s';
  }

  injectScenario(scenario) {
    this.sensorEngine.setScenario(scenario);
    this.addAlert('alert', `Injected: ${scenario.replace(/_/g, ' ')}`);
  }

  loop() {
    if (!this.running) return;
    const now = performance.now();
    const delta = now - this.lastTime;
    this.lastTime = now;
    this.accumulator += delta * this.speed;
    while (this.accumulator >= this.tickInterval) {
      this.accumulator -= this.tickInterval;
      this.simulationTick();
    }
    this.animFrameId = requestAnimationFrame(() => this.loop());
  }

  simulationTick() {
    this.tickCount++;
    const sensorData = this.sensorEngine.tick(1);
    const result = this.neuroCore.process(sensorData);

    this.updateSensors(sensorData, result.spikes);
    this.updateTier(result.tier);
    this.updatePower(result.power_uW, result.totalEnergy_uJ, result.savings, result.avgPower);
    this.pushRaster(result.spikes);
    this.drawRaster();
    this.pushPower(result.power_uW);
    this.drawPowerChart();
    this.updateMembrane(result.membranePotentials);
    this.updateSavings(result.savings, result.totalEnergy_uJ, this.neuroCore.power.alwaysOnEnergy_uJ);
    this.drawTopology(result);

    if (result.tier >= 1) {
      this.addAlert(result.classification,
        `Tier-${result.tier} · ${result.confidences.map(c => (c * 100).toFixed(0) + '%').join(' / ')} · ${result.power_uW.toFixed(0)} µW`
      );
    } else if (this.tickCount % 25 === 0) {
      this.addAlert('normal', `Avg ${result.avgPower.toFixed(1)} µW · ${result.savings.toFixed(1)}% saved`);
    }

    document.getElementById('simClock').textContent = `T+${this.tickCount}s`;
  }

  // ─── Sensor Updates ───
  updateSensors(data, spikes) {
    // Temperature
    const tEl = document.getElementById('val-temp');
    if (tEl) tEl.textContent = data.temperature.toFixed(1);
    this.setBar('bar-temp', data.temperature, -10, 80);

    // Humidity
    const hEl = document.getElementById('val-humid');
    if (hEl) hEl.textContent = data.humidity.toFixed(1);
    this.setBar('bar-humid', data.humidity, 0, 100);

    // Motion
    const mEl = document.getElementById('val-motion');
    if (mEl) mEl.textContent = data.motion ? 'Active' : 'Idle';
    this.setBar('bar-motion', data.motion * 100, 0, 100);

    // Air quality
    const aEl = document.getElementById('val-air');
    if (aEl) aEl.textContent = Math.round(data.airQuality);
    this.setBar('bar-air', data.airQuality, 0, 2000);

    // Spike indicators
    if (spikes) {
      this.setSpike('spike-temp', spikes.temperature);
      this.setSpike('spike-humid', spikes.humidity);
      this.setSpike('spike-motion', spikes.motion);
      this.setSpike('spike-air', spikes.airQuality);
    }
  }

  setBar(id, value, min, max) {
    const el = document.getElementById(id);
    if (!el) return;
    const pct = Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100));
    el.style.width = pct + '%';
  }

  setSpike(id, active) {
    const el = document.getElementById(id);
    if (!el) return;
    if (active) {
      el.classList.add('active');
      el.textContent = '⚡';
      setTimeout(() => { el.classList.remove('active'); el.textContent = '—'; }, 400);
    } else {
      el.textContent = '—';
    }
  }

  // ─── Tier (header) ───
  updateTier(tier) {
    const tierNames = ['Tier 0 · Idle', 'Tier 1 · Event', 'Tier 2 · Alert'];
    const tierLabelEl = document.getElementById('tierText');
    if (tierLabelEl) tierLabelEl.textContent = tierNames[tier];

    document.getElementById('td0').className = 'tier-dot t0';
    document.getElementById('td1').className = tier >= 1 ? 'tier-dot t1' : 'tier-dot';
    document.getElementById('td2').className = tier >= 2 ? 'tier-dot t2' : 'tier-dot';

    // Header power color
    const hdrP = document.getElementById('hdrPower');
    if (hdrP) {
      hdrP.className = 'header-power-val';
      if (tier === 1) hdrP.classList.add('warn');
      if (tier === 2) hdrP.classList.add('crit');
    }

    // Actuators (LCD & Buzzer) UI
    const lcd1 = document.getElementById('lcdLine1');
    const lcd2 = document.getElementById('lcdLine2');
    const buz = document.getElementById('buzzerState');
    if (lcd1) lcd1.textContent = `Tier: ${tier}`;
    if (lcd2) {
      if (tier === 2) lcd2.textContent = "ALERT MODE";
      else if (tier === 1) lcd2.textContent = "EVENT MODE";
      else lcd2.textContent = "IDLE MODE";
    }
    if (buz) {
      if (tier === 2) {
        buz.textContent = "ALARM";
        buz.style.color = "var(--danger)";
        buz.style.textShadow = "0 0 8px rgba(232,84,84,0.6)";
      } else if (tier === 1) {
        buz.textContent = "BEEPING";
        buz.style.color = "var(--warning)";
        buz.style.textShadow = "0 0 8px rgba(232,168,53,0.6)";
      } else {
        buz.textContent = "Silent";
        buz.style.color = "#555e72";
        buz.style.textShadow = "none";
      }
    }
  }

  // ─── Power ───
  updatePower(power, totalEnergy, savings, avgPower) {
    const pEl = document.getElementById('powerValue');
    if (pEl) {
      pEl.textContent = power.toFixed(1);
      pEl.className = 'power-num';
      if (power > 200) pEl.classList.add('warn');
      if (power > 2000) pEl.classList.add('crit');
    }

    const hdrP = document.getElementById('hdrPower');
    if (hdrP) hdrP.textContent = power.toFixed(1);

    const tE = document.getElementById('statTotalEnergy');
    if (tE) tE.textContent = (totalEnergy / 1000).toFixed(2) + ' mJ';
    const aP = document.getElementById('statAvgPower');
    if (aP) aP.textContent = avgPower.toFixed(1) + ' µW';
    const sE = document.getElementById('statSavings');
    if (sE) sE.textContent = savings.toFixed(1) + '%';
  }

  // ─── Raster ───
  pushRaster(spikes) {
    for (const s of ['temperature', 'humidity', 'motion', 'airQuality']) {
      this.rasterData[s].push(spikes[s] ? 1 : 0);
      if (this.rasterData[s].length > this.maxRasterPoints) this.rasterData[s].shift();
    }
  }

  clearCanvas(ctx) {
    if (!ctx) return;
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  }

  drawRaster() {
    const ctx = this.rasterCtx;
    const canvas = ctx.canvas;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvas.clientWidth * dpr;
    canvas.height = canvas.clientHeight * dpr;
    ctx.scale(dpr, dpr);
    const w = canvas.clientWidth, h = canvas.clientHeight, pad = 60;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#0a0d14';
    ctx.fillRect(0, 0, w, h);

    const sensors = ['temperature', 'humidity', 'motion', 'airQuality'];
    const colors = ['#d97545', '#4a9ec9', '#b8bf3a', '#58a65c'];
    const labels = ['TEMP', 'HUMID', 'MOTION', 'AIR Q'];
    const rowH = h / 4;

    sensors.forEach((sensor, row) => {
      const y = row * rowH + rowH / 2;
      const data = this.rasterData[sensor];

      // Label
      ctx.fillStyle = colors[row];
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'right';
      ctx.fillText(labels[row], pad - 8, y + 3);

      // Guide line
      ctx.strokeStyle = '#1e2430';
      ctx.beginPath();
      ctx.moveTo(pad, y);
      ctx.lineTo(w - 4, y);
      ctx.stroke();

      // Dots
      const dataW = w - pad - 4;
      data.forEach((spike, i) => {
        if (!spike) return;
        const x = pad + (i / this.maxRasterPoints) * dataW;
        ctx.beginPath();
        ctx.arc(x, y, 2, 0, Math.PI * 2);
        ctx.fillStyle = colors[row];
        ctx.fill();
      });
    });
  }

  // ─── Power Chart ───
  pushPower(power) {
    this.powerHistory.push(power);
    if (this.powerHistory.length > this.maxPowerHistory) this.powerHistory.shift();
  }

  drawPowerChart() {
    const ctx = this.powerCtx;
    const canvas = ctx.canvas;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvas.clientWidth * dpr;
    canvas.height = canvas.clientHeight * dpr;
    ctx.scale(dpr, dpr);
    const w = canvas.clientWidth, h = canvas.clientHeight, pad = 6;

    ctx.clearRect(0, 0, w, h);
    if (this.powerHistory.length < 2) return;

    const maxP = Math.max(100, ...this.powerHistory);
    const data = this.powerHistory;

    // Area
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, 'rgba(91, 141, 239, 0.2)');
    grad.addColorStop(1, 'rgba(91, 141, 239, 0)');

    ctx.beginPath();
    ctx.moveTo(pad, h - pad);
    data.forEach((v, i) => {
      ctx.lineTo(pad + (i / (this.maxPowerHistory - 1)) * (w - 2 * pad), h - pad - (v / maxP) * (h - 2 * pad));
    });
    ctx.lineTo(pad + ((data.length - 1) / (this.maxPowerHistory - 1)) * (w - 2 * pad), h - pad);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // Line
    ctx.beginPath();
    data.forEach((v, i) => {
      const x = pad + (i / (this.maxPowerHistory - 1)) * (w - 2 * pad);
      const y = h - pad - (v / maxP) * (h - 2 * pad);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = '#5b8def';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  // ─── Membrane Bars ───
  updateMembrane(potentials) {
    const container = document.getElementById('membraneBars');
    if (!container) return;

    const labels = [
      'T', 'H', 'M', 'A',
      'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7',
      'N', 'An', 'Al'
    ];
    const colors = [
      '#d97545', '#4a9ec9', '#b8bf3a', '#58a65c',
      '#7a6cb8', '#7a6cb8', '#7a6cb8', '#7a6cb8', '#7a6cb8', '#7a6cb8', '#7a6cb8', '#7a6cb8',
      '#3ddc84', '#e8a835', '#e85454'
    ];

    if (container.children.length === 0) {
      labels.forEach((label, i) => {
        const col = document.createElement('div');
        col.className = 'mbar-col';
        col.innerHTML = `<div class="mbar" id="mb-${i}" style="background:${colors[i]};height:2px"></div><div class="mbar-key">${label}</div>`;
        container.appendChild(col);
      });
    }

    potentials.forEach((p, i) => {
      const bar = document.getElementById(`mb-${i}`);
      if (!bar) return;
      const h = Math.min(90, Math.max(2, Math.abs(p) * 90));
      bar.style.height = h + 'px';
      bar.style.opacity = Math.max(0.25, Math.min(1, Math.abs(p)));
      if (Math.abs(p) > 0.8) {
        bar.classList.add('flash');
        setTimeout(() => bar.classList.remove('flash'), 300);
      }
    });
  }

  // ─── Topology ───
  drawTopology(result = null) {
    const ctx = this.topoCtx;
    const canvas = ctx.canvas;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvas.clientWidth * dpr;
    canvas.height = canvas.clientHeight * dpr;
    ctx.scale(dpr, dpr);
    const w = canvas.clientWidth, h = canvas.clientHeight;

    ctx.clearRect(0, 0, w, h);

    const layers = [4, 8, 3];
    const layerX = [w * 0.14, w * 0.5, w * 0.86];
    const layerColors = [
      ['#d97545', '#4a9ec9', '#b8bf3a', '#58a65c'],
      Array(8).fill('#7a6cb8'),
      ['#3ddc84', '#e8a835', '#e85454'],
    ];
    const layerLabels = [
      ['Temp', 'Humid', 'Motion', 'AirQ'],
      ['h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7'],
      ['Normal', 'Anomaly', 'Alert'],
    ];

    const nodePos = layers.map((count, li) => {
      const arr = [];
      const totalH = (count - 1) * 28;
      const startY = (h - totalH) / 2;
      for (let i = 0; i < count; i++) arr.push({ x: layerX[li], y: startY + i * 28 });
      return arr;
    });

    const spiked = result ? [
      result.inputSpikes || [0, 0, 0, 0],
      result.hiddenSpikes || new Array(8).fill(0),
      result.outputSpikes || [0, 0, 0],
    ] : [new Array(4).fill(0), new Array(8).fill(0), new Array(3).fill(0)];

    // Connections I→H
    for (let i = 0; i < 4; i++) {
      for (let hh = 0; hh < 8; hh++) {
        const wt = this.neuroCore.snn.weightsIH[i][hh];
        if (wt < 0.15) continue;
        const from = nodePos[0][i], to = nodePos[1][hh];
        const active = spiked[0][i] && wt > 0.3;
        ctx.beginPath();
        ctx.moveTo(from.x + 8, from.y);
        ctx.lineTo(to.x - 8, to.y);
        ctx.strokeStyle = active ? `rgba(91, 141, 239, ${0.4 + wt * 0.6})` : `rgba(255,255,255,${wt * 0.07})`;
        ctx.lineWidth = active ? 1.5 : 0.5;
        ctx.stroke();
      }
    }

    // Connections H→O
    for (let hh = 0; hh < 8; hh++) {
      for (let o = 0; o < 3; o++) {
        const wt = this.neuroCore.snn.weightsHO[hh][o];
        if (wt < 0.15) continue;
        const from = nodePos[1][hh], to = nodePos[2][o];
        const active = spiked[1][hh] && wt > 0.3;
        ctx.beginPath();
        ctx.moveTo(from.x + 8, from.y);
        ctx.lineTo(to.x - 8, to.y);
        ctx.strokeStyle = active ? `rgba(122, 108, 184, ${0.4 + wt * 0.6})` : `rgba(255,255,255,${wt * 0.07})`;
        ctx.lineWidth = active ? 1.5 : 0.5;
        ctx.stroke();
      }
    }

    // Nodes
    layers.forEach((count, li) => {
      for (let i = 0; i < count; i++) {
        const pos = nodePos[li][i];
        const color = layerColors[li][i];
        const on = spiked[li][i];
        const r = on ? 7 : 5;

        ctx.beginPath();
        ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2);
        ctx.fillStyle = on ? color : color + '50';
        ctx.fill();

        if (on) {
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }

        ctx.fillStyle = on ? '#d8dce6' : '#555e72';
        ctx.font = '9px "JetBrains Mono"';
        ctx.textAlign = 'center';
        ctx.fillText(layerLabels[li][i], pos.x, pos.y + (li === 1 ? -12 : 18));
      }
    });

    // Layer titles
    ctx.fillStyle = '#555e72';
    ctx.font = '10px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('Input', layerX[0], 14);
    ctx.fillText('Hidden', layerX[1], 14);
    ctx.fillText('Output', layerX[2], 14);
  }

  // ─── Alerts ───
  addAlert(severity, message) {
    this.alerts.unshift({ time: `${this.tickCount}s`, severity, message });
    if (this.alerts.length > this.maxAlerts) this.alerts.pop();
    this.updateAlertFeed();
  }

  updateAlertFeed() {
    const el = document.getElementById('alertFeed');
    if (!el) return;
    el.innerHTML = this.alerts.map(a => `
      <div class="log-entry">
        <span class="log-time">${a.time}</span>
        <span class="log-tag ${a.severity}">${a.severity}</span>
        <span class="log-msg">${a.message}</span>
      </div>
    `).join('');
  }

  // ─── Savings ───
  updateSavings(savings, neuroE, alwaysE) {
    const s = document.getElementById('savingsPercent');
    const n = document.getElementById('neuroEnergy');
    const a = document.getElementById('alwaysOnEnergy');
    if (s) s.textContent = savings.toFixed(1) + '%';
    if (n) n.textContent = (neuroE / 1000).toFixed(2) + ' mJ';
    if (a) a.textContent = (alwaysE / 1000).toFixed(2) + ' mJ';
  }
}

// ═══ Init ═══
document.addEventListener('DOMContentLoaded', () => {
  const dashboard = new Dashboard();
  dashboard.init();
  window.neuroEdge = dashboard;
});
