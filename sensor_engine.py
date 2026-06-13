"""
sensor_engine.py — Synthetic Multi-Sensor Data Generator for NeuroEdge
======================================================================

Simulates four realistic sensor channels for neuromorphic edge processing:
  1. Temperature (DHT22-like): Diurnal sine wave + Gaussian noise + rare heat spikes
  2. Humidity (%RH): Inversely correlated with temperature + dew point events
  3. Motion (PIR): Poisson-distributed binary events with activity bursts
  4. Air Quality (MQ135-like, PPM): Slow random walk + sudden pollution spikes

The SensorEngine class tracks elapsed simulation time internally and produces
one sample per tick.  Anomalies can be injected programmatically, and several
predefined scenarios (fire_alarm, cooking, etc.) configure the engine for
common test cases.

Dependencies: numpy (>= 1.21), Python standard library.
"""

from __future__ import annotations

import math
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants — physical limits & default tunables
# ---------------------------------------------------------------------------

# Temperature (°C)
TEMP_MIN: float = 18.0          # overnight low
TEMP_MAX: float = 32.0          # afternoon peak
TEMP_NOISE_SIGMA: float = 0.3   # sensor noise std-dev
TEMP_SPIKE_PROB: float = 0.002  # probability of a random heat spike per tick
TEMP_SPIKE_AMPLITUDE: float = 8.0  # °C above normal during a heat spike

# Humidity (%RH)
HUM_MIN: float = 40.0
HUM_MAX: float = 85.0
HUM_NOISE_SIGMA: float = 1.5
HUM_DEW_PROB: float = 0.003    # probability of a dew-point condensation event
HUM_DEW_BOOST: float = 15.0    # %RH jump during dew event

# Motion (PIR — binary)
MOTION_LAMBDA_BASE: float = 0.05   # Poisson rate (events/sec) at baseline
MOTION_BURST_LAMBDA: float = 0.6   # rate during activity bursts
MOTION_BURST_PROB: float = 0.005   # probability of entering a burst each tick
MOTION_BURST_DURATION: float = 10.0  # seconds a burst lasts

# Air quality (PPM)
AQ_BASELINE_LOW: float = 300.0
AQ_BASELINE_HIGH: float = 600.0
AQ_WALK_SIGMA: float = 2.0         # random-walk step size
AQ_SPIKE_PROB: float = 0.003       # probability of a pollution spike per tick
AQ_SPIKE_AMPLITUDE: float = 400.0  # PPM above baseline during a spike
AQ_SPIKE_DECAY: float = 0.92       # exponential decay factor for spikes

# Diurnal period
DIURNAL_PERIOD_S: float = 86400.0  # 24 hours in seconds


class SensorEngine:
    """
    Generates synthetic multi-sensor data one tick at a time.

    Parameters
    ----------
    seed : int or None
        Random seed for reproducibility.  None ⇒ non-deterministic.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng: np.random.Generator = np.random.default_rng(seed)

        # Internal clock (elapsed simulation seconds)
        self._elapsed: float = 0.0

        # --- per-sensor state ---
        # Temperature: no extra state beyond clock
        self._temp_spike_remaining: float = 0.0  # seconds left in a heat spike

        # Humidity: tracks dew-point event countdown
        self._hum_dew_remaining: float = 0.0

        # Motion: burst state
        self._motion_burst_remaining: float = 0.0

        # Air quality: random-walk level + spike residual
        self._aq_level: float = self._rng.uniform(AQ_BASELINE_LOW, AQ_BASELINE_HIGH)
        self._aq_spike_residual: float = 0.0  # decaying spike overlay

        # Anomaly injection overrides  {sensor_name: severity}
        self._injected_anomalies: Dict[str, float] = {}

        # Active scenario modifiers (empty dict = no scenario)
        self._scenario_mods: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tick(self, dt: float = 1.0) -> Dict[str, float]:
        """
        Advance the simulation by *dt* seconds and return current readings.

        Parameters
        ----------
        dt : float
            Time step in seconds (default 1.0).

        Returns
        -------
        dict
            Keys: 'temperature', 'humidity', 'motion', 'air_quality'.
        """
        self._elapsed += dt

        temperature = self._sample_temperature(dt)
        humidity = self._sample_humidity(temperature, dt)
        motion = self._sample_motion(dt)
        air_quality = self._sample_air_quality(dt)

        # Apply any injected anomalies
        temperature = self._apply_anomaly("temperature", temperature)
        humidity = self._apply_anomaly("humidity", humidity)
        motion = self._apply_anomaly("motion", motion)
        air_quality = self._apply_anomaly("air_quality", air_quality)

        return {
            "temperature": round(temperature, 2),
            "humidity": round(float(np.clip(humidity, 0.0, 100.0)), 2),
            "motion": float(motion),
            "air_quality": round(max(air_quality, 0.0), 2),
        }

    def inject_anomaly(self, sensor_name: str, severity: float = 1.0) -> None:
        """
        Force an anomalous reading on *sensor_name* starting next tick.

        Parameters
        ----------
        sensor_name : str
            One of 'temperature', 'humidity', 'motion', 'air_quality'.
        severity : float
            Multiplier (1.0 = moderate anomaly, 3.0 = extreme).
        """
        if sensor_name not in ("temperature", "humidity", "motion", "air_quality"):
            raise ValueError(f"Unknown sensor: {sensor_name}")
        self._injected_anomalies[sensor_name] = severity

    def clear_anomalies(self) -> None:
        """Remove all injected anomaly overrides."""
        self._injected_anomalies.clear()

    def get_scenario(self, name: str) -> Dict[str, float]:
        """
        Activate a predefined scenario and return the modifier dict.

        Supported scenarios
        -------------------
        normal_day   — default parameters, no special modifiers
        fire_alarm   — high temp, rising smoke (AQ), motion bursts
        cooking      — moderate AQ spike, slight temp rise
        hvac_failure — temperature rises steadily, humidity climbs
        storm        — rapid humidity swings, temperature drops, motion noise

        Parameters
        ----------
        name : str
            Scenario identifier.

        Returns
        -------
        dict
            The modifier dictionary applied to the engine.
        """
        scenarios: Dict[str, Dict[str, float]] = {
            "normal_day": {},
            "fire_alarm": {
                "temp_offset": 25.0,        # °C above normal
                "hum_offset": -15.0,        # drier air near fire
                "motion_lambda_mult": 5.0,  # panicked movement
                "aq_offset": 600.0,         # heavy smoke
            },
            "cooking": {
                "temp_offset": 4.0,
                "hum_offset": 5.0,
                "motion_lambda_mult": 1.5,
                "aq_offset": 250.0,
            },
            "hvac_failure": {
                "temp_offset": 10.0,
                "hum_offset": 20.0,
                "motion_lambda_mult": 1.0,
                "aq_offset": 50.0,
            },
            "storm": {
                "temp_offset": -6.0,
                "hum_offset": 30.0,
                "motion_lambda_mult": 2.0,
                "aq_offset": 20.0,
            },
        }
        if name not in scenarios:
            raise ValueError(
                f"Unknown scenario '{name}'. "
                f"Choose from: {list(scenarios.keys())}"
            )
        self._scenario_mods = scenarios[name]
        return dict(self._scenario_mods)  # return a copy

    @property
    def elapsed(self) -> float:
        """Total elapsed simulation time in seconds."""
        return self._elapsed

    # ------------------------------------------------------------------
    # Internal sensor models
    # ------------------------------------------------------------------

    def _sample_temperature(self, dt: float) -> float:
        """Diurnal sine wave + noise + optional heat spike."""
        # Base diurnal cycle: peaks at ~14:00 (offset by π/2 so t=0 is midnight)
        phase = 2.0 * math.pi * self._elapsed / DIURNAL_PERIOD_S - math.pi / 2.0
        amplitude = (TEMP_MAX - TEMP_MIN) / 2.0
        midpoint = (TEMP_MAX + TEMP_MIN) / 2.0
        base_temp = midpoint + amplitude * math.sin(phase)

        # Gaussian sensor noise
        noise = float(self._rng.normal(0.0, TEMP_NOISE_SIGMA))

        # Random heat spikes (e.g., appliance turning on nearby)
        if self._temp_spike_remaining > 0:
            spike = TEMP_SPIKE_AMPLITUDE * (self._temp_spike_remaining / 5.0)
            self._temp_spike_remaining = max(0.0, self._temp_spike_remaining - dt)
        elif self._rng.random() < TEMP_SPIKE_PROB:
            self._temp_spike_remaining = 5.0  # spike lasts ~5 seconds
            spike = TEMP_SPIKE_AMPLITUDE
        else:
            spike = 0.0

        # Scenario modifier
        offset = self._scenario_mods.get("temp_offset", 0.0)

        return base_temp + noise + spike + offset

    def _sample_humidity(self, temperature: float, dt: float) -> float:
        """Humidity inversely correlated with temperature + dew events."""
        # Inverse linear mapping from temperature range to humidity range
        temp_norm = (temperature - TEMP_MIN) / (TEMP_MAX - TEMP_MIN + 1e-9)
        temp_norm = float(np.clip(temp_norm, 0.0, 1.0))
        base_hum = HUM_MAX - temp_norm * (HUM_MAX - HUM_MIN)

        # Gaussian noise
        noise = float(self._rng.normal(0.0, HUM_NOISE_SIGMA))

        # Dew-point condensation events (sudden humidity jump)
        if self._hum_dew_remaining > 0:
            dew = HUM_DEW_BOOST * (self._hum_dew_remaining / 8.0)
            self._hum_dew_remaining = max(0.0, self._hum_dew_remaining - dt)
        elif self._rng.random() < HUM_DEW_PROB:
            self._hum_dew_remaining = 8.0  # event lasts ~8 seconds
            dew = HUM_DEW_BOOST
        else:
            dew = 0.0

        # Scenario modifier
        offset = self._scenario_mods.get("hum_offset", 0.0)

        return base_hum + noise + dew + offset

    def _sample_motion(self, dt: float) -> float:
        """Poisson-distributed binary motion events with activity bursts."""
        # Determine current lambda (events/sec)
        lam = MOTION_LAMBDA_BASE

        # Check if we're in an activity burst
        if self._motion_burst_remaining > 0:
            lam = MOTION_BURST_LAMBDA
            self._motion_burst_remaining = max(
                0.0, self._motion_burst_remaining - dt
            )
        elif self._rng.random() < MOTION_BURST_PROB:
            # Start a new burst
            self._motion_burst_remaining = MOTION_BURST_DURATION
            lam = MOTION_BURST_LAMBDA

        # Apply scenario multiplier
        lam *= self._scenario_mods.get("motion_lambda_mult", 1.0)

        # Probability of at least one event in this dt interval (Poisson)
        prob_event = 1.0 - math.exp(-lam * dt)
        detected = 1.0 if self._rng.random() < prob_event else 0.0

        return detected

    def _sample_air_quality(self, dt: float) -> float:
        """Slow random walk + sudden pollution spikes."""
        # Random walk step (Brownian motion, bounded)
        step = float(self._rng.normal(0.0, AQ_WALK_SIGMA))
        self._aq_level += step
        # Soft-clamp to baseline range
        self._aq_level = float(
            np.clip(self._aq_level, AQ_BASELINE_LOW, AQ_BASELINE_HIGH)
        )

        # Pollution spike trigger
        if self._rng.random() < AQ_SPIKE_PROB:
            self._aq_spike_residual += AQ_SPIKE_AMPLITUDE

        # Decay existing spike residual
        self._aq_spike_residual *= AQ_SPIKE_DECAY

        # Scenario modifier
        offset = self._scenario_mods.get("aq_offset", 0.0)

        return self._aq_level + self._aq_spike_residual + offset

    # ------------------------------------------------------------------
    # Anomaly overlay
    # ------------------------------------------------------------------

    def _apply_anomaly(self, sensor_name: str, value: float) -> float:
        """
        If an anomaly has been injected for *sensor_name*, perturb *value*.

        Anomaly semantics per sensor:
          temperature  →  +20 * severity  (°C)
          humidity     →  +25 * severity  (%RH, clamped)
          motion       →  force to 1.0 if severity >= 0.5
          air_quality  →  +500 * severity (PPM)
        """
        if sensor_name not in self._injected_anomalies:
            return value

        severity = self._injected_anomalies[sensor_name]
        if sensor_name == "temperature":
            return value + 20.0 * severity
        elif sensor_name == "humidity":
            return value + 25.0 * severity
        elif sensor_name == "motion":
            return 1.0 if severity >= 0.5 else value
        elif sensor_name == "air_quality":
            return value + 500.0 * severity
        return value  # fallback


# ======================================================================
# Standalone demo
# ======================================================================

def _demo() -> None:
    """Print 60 seconds of simulated sensor readings."""
    engine = SensorEngine(seed=42)

    header = f"{'Tick':>5s}  {'Temp(°C)':>9s}  {'Hum(%RH)':>9s}  {'Motion':>6s}  {'AQ(PPM)':>9s}"
    print("=" * len(header))
    print("NeuroEdge Sensor Engine — 60-second demo")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for tick in range(60):
        data = engine.tick(dt=1.0)
        mot_str = "YES" if data["motion"] > 0.5 else "---"
        print(
            f"{tick:5d}  "
            f"{data['temperature']:9.2f}  "
            f"{data['humidity']:9.2f}  "
            f"{mot_str:>6s}  "
            f"{data['air_quality']:9.2f}"
        )

    print("-" * len(header))
    print(f"Total elapsed time: {engine.elapsed:.1f}s")
    print("Demo complete.")


if __name__ == "__main__":
    _demo()
