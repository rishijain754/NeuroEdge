"""
neuro_core.py — Neuromorphic Spiking Neural Network Core for NeuroEdge
======================================================================

Implements a biologically-inspired spiking neural network (SNN) pipeline:

  1. **LIFNeuron** — Leaky Integrate-and-Fire neuron with refractory period
  2. **SpikeEncoder** — Converts analogue sensor values into spike trains
     (rate coding, time-to-first-spike, delta encoding)
  3. **SpikingNeuralNetwork** — Feedforward 4→8→3 SNN with hand-tuned weights
     and Winner-Take-All (WTA) classification on the output layer
  4. **HomeostaticRegulator** — Keeps neuron firing rates healthy via
     threshold adaptation (intrinsic plasticity)
  5. **NeuroCore** — Top-level façade that wires everything together

The three output classes are:
    0 = Normal   — routine environmental readings
    1 = Anomaly  — unusual but not critical
    2 = Alert    — urgent / safety-critical

Dependencies: numpy (>= 1.21), Python standard library.
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional, Tuple

import numpy as np


# ======================================================================
# 1. Leaky Integrate-and-Fire Neuron
# ======================================================================

class LIFNeuron:
    """
    A single Leaky Integrate-and-Fire (LIF) neuron.

    At each time step the membrane potential decays toward zero by the
    *decay* factor, then the input current is added.  If the potential
    reaches or exceeds *threshold*, the neuron fires a spike, the
    potential resets to zero, and the neuron enters a refractory period
    during which it ignores input.

    Parameters
    ----------
    threshold : float
        Firing threshold (default 1.0).
    decay : float
        Multiplicative leak factor applied each tick (default 0.9).
    refractory_period : int
        Number of ticks the neuron stays silent after a spike (default 3).
    """

    def __init__(
        self,
        threshold: float = 1.0,
        decay: float = 0.9,
        refractory_period: int = 3,
    ) -> None:
        self.threshold: float = threshold
        self.decay: float = decay
        self.refractory_period: int = refractory_period

        # Dynamic state
        self.membrane_potential: float = 0.0
        self.refractory_counter: int = 0
        self.spike_history: List[bool] = []

    def step(self, input_current: float) -> bool:
        """
        Advance the neuron by one tick.

        Parameters
        ----------
        input_current : float
            Weighted sum of incoming spikes / raw current.

        Returns
        -------
        bool
            True if the neuron fires a spike this tick.
        """
        # If still in refractory period, count down and do not integrate
        if self.refractory_counter > 0:
            self.refractory_counter -= 1
            self.spike_history.append(False)
            return False

        # Leak: multiplicative decay toward resting potential (0)
        self.membrane_potential *= self.decay

        # Integrate: add input current
        self.membrane_potential += input_current

        # Fire?
        if self.membrane_potential >= self.threshold:
            self.membrane_potential = 0.0          # reset
            self.refractory_counter = self.refractory_period
            self.spike_history.append(True)
            return True

        self.spike_history.append(False)
        return False

    def reset(self) -> None:
        """Reset the neuron to its initial resting state."""
        self.membrane_potential = 0.0
        self.refractory_counter = 0
        self.spike_history.clear()

    def get_firing_rate(self, window: int = 20) -> float:
        """
        Compute the recent firing rate over the last *window* ticks.

        Returns
        -------
        float
            Fraction of ticks in the window during which the neuron fired
            (range 0.0–1.0).
        """
        if len(self.spike_history) == 0:
            return 0.0
        recent = self.spike_history[-window:]
        return sum(recent) / len(recent)


# ======================================================================
# 2. Spike Encoder
# ======================================================================

class SpikeEncoder:
    """
    Converts continuous sensor values into discrete spike representations.

    Three encoding strategies are provided:

    * **Rate coding** — higher value ⇒ higher probability of spiking
    * **Time-to-First-Spike (TTFS)** — higher value ⇒ shorter delay
    * **Delta coding** — spike only when the value changes significantly
    """

    def rate_encode(
        self,
        value: float,
        min_val: float,
        max_val: float,
        max_rate: float = 1.0,
    ) -> float:
        """
        Rate-code a value into a spike probability.

        Parameters
        ----------
        value : float
            Current sensor reading.
        min_val, max_val : float
            Expected range of the sensor.
        max_rate : float
            Maximum spike probability (default 1.0).

        Returns
        -------
        float
            Probability of generating a spike this tick (0.0–max_rate).
        """
        # Normalise to [0, 1]
        normed = (value - min_val) / (max_val - min_val + 1e-12)
        normed = float(np.clip(normed, 0.0, 1.0))
        return normed * max_rate

    def ttfs_encode(
        self,
        value: float,
        min_val: float,
        max_val: float,
        max_delay: int = 10,
    ) -> int:
        """
        Time-to-First-Spike encoding.

        A high value results in a short delay (spike arrives quickly);
        a low value results in a long delay.

        Parameters
        ----------
        value : float
            Current sensor reading.
        min_val, max_val : float
            Expected range.
        max_delay : int
            Maximum delay in ticks for the lowest-valued input.

        Returns
        -------
        int
            Delay (in ticks) before the first spike fires.
        """
        normed = (value - min_val) / (max_val - min_val + 1e-12)
        normed = float(np.clip(normed, 0.0, 1.0))
        # Invert: high value → small delay
        delay = int(round((1.0 - normed) * max_delay))
        return max(0, delay)

    def delta_encode(
        self,
        current: float,
        previous: float,
        threshold: float = 0.5,
    ) -> bool:
        """
        Delta (change-based) encoding.

        Fires a spike only when the absolute change between the current
        and previous values exceeds *threshold*.

        Parameters
        ----------
        current : float
            Current reading.
        previous : float
            Previous reading.
        threshold : float
            Minimum change magnitude to trigger a spike.

        Returns
        -------
        bool
            True if a spike should be emitted.
        """
        return abs(current - previous) >= threshold


# ======================================================================
# 3. Spiking Neural Network (4 → 8 → 3)
# ======================================================================

class SpikingNeuralNetwork:
    """
    A small feed-forward SNN with architecture **4 → 8 → 3**.

    * **Input layer**  (4 units): one per sensor channel
      (temperature, humidity, motion, air_quality)
    * **Hidden layer** (8 LIF neurons): feature detectors
    * **Output layer** (3 LIF neurons): classes Normal / Anomaly / Alert

    Weights are hand-tuned (not learned) to produce sensible tier
    classifications for the NeuroEdge sensor suite.  A Winner-Take-All
    (WTA) rule picks the output class.
    """

    CLASS_NAMES: List[str] = ["normal", "anomaly", "alert"]

    def __init__(self) -> None:
        # ----- Hidden layer (8 neurons) -----
        # Neurons 0-1: temperature detectors (one low-pass, one high-pass)
        # Neurons 2-3: humidity detectors
        # Neurons 4-5: motion detectors (sustained vs transient)
        # Neurons 6-7: air-quality detectors
        self.hidden: List[LIFNeuron] = [
            LIFNeuron(threshold=0.8,  decay=0.85, refractory_period=2),  # H0: temp low
            LIFNeuron(threshold=0.7,  decay=0.80, refractory_period=2),  # H1: temp high
            LIFNeuron(threshold=0.85, decay=0.88, refractory_period=2),  # H2: hum low
            LIFNeuron(threshold=0.75, decay=0.82, refractory_period=2),  # H3: hum high
            LIFNeuron(threshold=0.6,  decay=0.75, refractory_period=3),  # H4: motion transient
            LIFNeuron(threshold=0.9,  decay=0.92, refractory_period=2),  # H5: motion sustained
            LIFNeuron(threshold=0.8,  decay=0.85, refractory_period=2),  # H6: AQ low
            LIFNeuron(threshold=0.65, decay=0.78, refractory_period=2),  # H7: AQ high
        ]

        # ----- Output layer (3 neurons) -----
        self.output: List[LIFNeuron] = [
            LIFNeuron(threshold=1.0,  decay=0.88, refractory_period=3),  # O0: Normal
            LIFNeuron(threshold=0.85, decay=0.82, refractory_period=3),  # O1: Anomaly
            LIFNeuron(threshold=0.80, decay=0.78, refractory_period=3),  # O2: Alert
        ]

        # ----- Weight matrices (hand-tuned) -----
        # Input→Hidden  shape (4 inputs, 8 hidden)
        # Rows = input channels [temp, hum, motion, aq]
        # Cols = hidden neurons  [H0 .. H7]
        self.w_ih: np.ndarray = np.array([
            # H0    H1    H2    H3    H4    H5    H6    H7
            [0.50,  0.90,  0.05,  0.05,  0.00,  0.00,  0.00,  0.00],  # temperature
            [0.05,  0.05,  0.50,  0.85,  0.00,  0.00,  0.00,  0.00],  # humidity
            [0.00,  0.00,  0.00,  0.00,  0.80,  0.60,  0.00,  0.00],  # motion
            [0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.50,  0.90],  # air quality
        ], dtype=np.float64)

        # Hidden→Output  shape (8 hidden, 3 output)
        # Columns = output classes [Normal, Anomaly, Alert]
        self.w_ho: np.ndarray = np.array([
            # Normal  Anomaly  Alert
            [ 0.60,   0.10,   0.00],   # H0 (temp low) → mostly normal
            [ 0.00,   0.45,   0.55],   # H1 (temp high) → anomaly/alert
            [ 0.55,   0.15,   0.00],   # H2 (hum low)  → normal
            [ 0.05,   0.50,   0.40],   # H3 (hum high) → anomaly
            [ 0.10,   0.55,   0.35],   # H4 (motion transient) → anomaly
            [ 0.00,   0.30,   0.65],   # H5 (motion sustained) → alert
            [ 0.55,   0.15,   0.00],   # H6 (AQ low) → normal
            [ 0.00,   0.40,   0.60],   # H7 (AQ high) → alert
        ], dtype=np.float64)

    def forward(
        self, spike_inputs: List[bool]
    ) -> Tuple[int, List[float]]:
        """
        Run one forward pass through the network.

        Parameters
        ----------
        spike_inputs : list of bool
            Length-4 binary spike vector [temp, hum, motion, aq].

        Returns
        -------
        class_index : int
            Winning output class (0=Normal, 1=Anomaly, 2=Alert).
        confidences : list of float
            Soft confidence per class (based on membrane potentials after
            the step, NOT on binary spikes — gives smoother output).
        """
        if len(spike_inputs) != 4:
            raise ValueError("Expected exactly 4 input spikes")

        # Convert boolean spikes to float currents (1.0 or 0.0)
        inp = np.array([float(s) for s in spike_inputs], dtype=np.float64)

        # ---- Input → Hidden ----
        hidden_currents = inp @ self.w_ih  # shape (8,)
        hidden_spikes = np.zeros(8, dtype=np.float64)
        for i, neuron in enumerate(self.hidden):
            if neuron.step(hidden_currents[i]):
                hidden_spikes[i] = 1.0

        # ---- Hidden → Output ----
        output_currents = hidden_spikes @ self.w_ho  # shape (3,)
        output_spikes = []
        for i, neuron in enumerate(self.output):
            output_spikes.append(neuron.step(output_currents[i]))

        # ---- Winner-Take-All (WTA) ----
        # Use membrane potentials (post-step) as continuous confidence
        potentials = np.array(
            [n.membrane_potential for n in self.output], dtype=np.float64
        )

        # If any output neuron actually spiked, it wins outright
        # (its potential was reset to 0, so we set its confidence to 1.0)
        spiked_indices = [i for i, s in enumerate(output_spikes) if s]
        if spiked_indices:
            # If multiple spiked, pick the one that was most recently silent
            # (i.e., had highest pre-spike potential → strongest drive).
            # Since potentials are now reset, we favour the first in priority
            # order: Alert > Anomaly > Normal (higher index = higher urgency).
            winner = max(spiked_indices)
            confidences = [0.0, 0.0, 0.0]
            confidences[winner] = 1.0
        else:
            # No spike: derive soft confidences from membrane potentials
            pot_sum = potentials.sum() + 1e-12
            confidences = (potentials / pot_sum).tolist()
            winner = int(np.argmax(potentials))

        return winner, confidences

    def get_tier(self) -> int:
        """
        Derive the active processing tier from recent output firing rates.

        Tier 0 — mostly Normal → sleep mode
        Tier 1 — Anomaly detected → partial wake
        Tier 2 — Alert detected → full processing

        Returns
        -------
        int
            0, 1, or 2.
        """
        rates = [n.get_firing_rate(window=10) for n in self.output]
        # Alert dominates if it fires at all recently
        if rates[2] > 0.15:
            return 2
        if rates[1] > 0.15:
            return 1
        return 0

    def get_all_neurons(self) -> List[LIFNeuron]:
        """Return a flat list of all hidden + output neurons."""
        return self.hidden + self.output


# ======================================================================
# 4. Homeostatic Plasticity Regulator
# ======================================================================

class HomeostaticRegulator:
    """
    Adjusts neuron thresholds to maintain healthy firing rates.

    If a neuron fires too frequently (rate > *high_rate*), its threshold
    is increased by *adjust_factor* to calm it down.  If it fires too
    rarely (rate < *low_rate*), the threshold is decreased to make it
    more excitable.

    The adaptation is applied every *period* ticks.

    Parameters
    ----------
    period : int
        Number of ticks between adaptations (default 50).
    high_rate : float
        Upper firing rate bound (default 0.8).
    low_rate : float
        Lower firing rate bound (default 0.05).
    adjust_factor : float
        Fractional threshold change per adaptation (default 0.05 = 5%).
    window : int
        Window size for computing firing rates (default 20).
    """

    def __init__(
        self,
        period: int = 50,
        high_rate: float = 0.8,
        low_rate: float = 0.05,
        adjust_factor: float = 0.05,
        window: int = 20,
    ) -> None:
        self.period: int = period
        self.high_rate: float = high_rate
        self.low_rate: float = low_rate
        self.adjust_factor: float = adjust_factor
        self.window: int = window
        self._tick_counter: int = 0

    def step(self, neurons: List[LIFNeuron]) -> bool:
        """
        Increment internal counter; adapt thresholds when the period elapses.

        Parameters
        ----------
        neurons : list of LIFNeuron
            All neurons whose thresholds should be regulated.

        Returns
        -------
        bool
            True if adaptation was applied this tick.
        """
        self._tick_counter += 1
        if self._tick_counter >= self.period:
            self._tick_counter = 0
            self.adapt(neurons)
            return True
        return False

    def adapt(self, neurons: List[LIFNeuron]) -> None:
        """
        Perform one round of homeostatic threshold adaptation.

        Parameters
        ----------
        neurons : list of LIFNeuron
            Neurons to regulate.
        """
        for neuron in neurons:
            rate = neuron.get_firing_rate(window=self.window)
            if rate > self.high_rate:
                # Neuron is over-active → raise threshold (harder to fire)
                neuron.threshold *= (1.0 + self.adjust_factor)
            elif rate < self.low_rate:
                # Neuron is under-active → lower threshold (easier to fire)
                neuron.threshold *= (1.0 - self.adjust_factor)
                # Safety floor: never let threshold go below 0.1
                neuron.threshold = max(neuron.threshold, 0.1)


# ======================================================================
# 5. NeuroCore — top-level façade
# ======================================================================

# Sensor value ranges used for encoding
_SENSOR_RANGES: Dict[str, Tuple[float, float]] = {
    "temperature": (10.0, 60.0),
    "humidity":    (0.0,  100.0),
    "motion":     (0.0,  1.0),
    "air_quality": (200.0, 1200.0),
}

# Delta-encoding thresholds per sensor
_DELTA_THRESHOLDS: Dict[str, float] = {
    "temperature": 1.0,   # °C change
    "humidity":    3.0,    # %RH change
    "motion":     0.5,    # binary flip
    "air_quality": 30.0,  # PPM change
}

# Ordered list of sensor keys (must match SNN input order)
_SENSOR_ORDER: List[str] = ["temperature", "humidity", "motion", "air_quality"]


class NeuroCore:
    """
    Top-level neuromorphic processing core.

    Wraps the spike encoder, SNN, and homeostatic regulator into a
    single ``process(sensor_data)`` call that returns a rich result dict.

    Maintains a circular spike-history buffer of the last 200 ticks
    for analysis and visualisation.

    Parameters
    ----------
    history_size : int
        Length of the circular spike-history buffer (default 200).
    seed : int or None
        RNG seed for stochastic encoding (default None).
    """

    # Classification label lookup
    _CLASS_LABELS: List[str] = ["normal", "anomaly", "alert"]

    def __init__(
        self, history_size: int = 200, seed: Optional[int] = None
    ) -> None:
        self.encoder: SpikeEncoder = SpikeEncoder()
        self.snn: SpikingNeuralNetwork = SpikingNeuralNetwork()
        self.regulator: HomeostaticRegulator = HomeostaticRegulator(period=50)
        self._rng: np.random.Generator = np.random.default_rng(seed)

        # Previous sensor values (for delta encoding)
        self._prev: Dict[str, float] = {k: 0.0 for k in _SENSOR_ORDER}

        # Circular spike history buffer: list of dicts (most recent at end)
        self._history_size: int = history_size
        self._history: List[Dict] = []

        # Tick counter
        self._tick: int = 0

    def process(self, sensor_data: Dict[str, float]) -> Dict:
        """
        Run one tick of neuromorphic processing.

        Parameters
        ----------
        sensor_data : dict
            Must contain keys 'temperature', 'humidity', 'motion',
            'air_quality' with float values.

        Returns
        -------
        dict
            Keys:
              spikes           — {sensor: bool} per-channel spike flags
              tier             — active processing tier (0/1/2)
              classification   — 'normal' / 'anomaly' / 'alert'
              confidence       — list[float] per-class confidence
              membrane_potentials — list[float] all neuron potentials
              firing_rates     — list[float] recent firing rates
        """
        self._tick += 1

        # ----------------------------------------------------------
        # 1. Encode each sensor channel into a binary spike
        # ----------------------------------------------------------
        spikes: Dict[str, bool] = {}
        spike_vector: List[bool] = []

        for key in _SENSOR_ORDER:
            val = sensor_data[key]
            lo, hi = _SENSOR_RANGES[key]

            # Rate-encode → stochastic spike
            prob = self.encoder.rate_encode(val, lo, hi, max_rate=1.0)
            rate_spike = bool(self._rng.random() < prob)

            # Delta-encode → deterministic spike on change
            delta_spike = self.encoder.delta_encode(
                val, self._prev[key], threshold=_DELTA_THRESHOLDS[key]
            )

            # Combined: spike if either encoding says so
            combined = rate_spike or delta_spike
            spikes[key] = combined
            spike_vector.append(combined)

            # Store previous value
            self._prev[key] = val

        # ----------------------------------------------------------
        # 2. Forward pass through SNN
        # ----------------------------------------------------------
        class_idx, confidences = self.snn.forward(spike_vector)
        tier = self.snn.get_tier()
        classification = self._CLASS_LABELS[class_idx]

        # ----------------------------------------------------------
        # 3. Homeostatic regulation
        # ----------------------------------------------------------
        all_neurons = self.snn.get_all_neurons()
        self.regulator.step(all_neurons)

        # ----------------------------------------------------------
        # 4. Collect diagnostics
        # ----------------------------------------------------------
        membrane_potentials = [n.membrane_potential for n in all_neurons]
        firing_rates = [n.get_firing_rate(window=20) for n in all_neurons]

        # ----------------------------------------------------------
        # 5. Update circular history buffer
        # ----------------------------------------------------------
        entry = {
            "tick": self._tick,
            "spikes": dict(spikes),
            "tier": tier,
            "classification": classification,
            "confidence": list(confidences),
        }
        self._history.append(entry)
        if len(self._history) > self._history_size:
            self._history.pop(0)  # drop oldest

        # ----------------------------------------------------------
        # 6. Build result
        # ----------------------------------------------------------
        return {
            "spikes": spikes,
            "tier": tier,
            "classification": classification,
            "confidence": confidences,
            "membrane_potentials": membrane_potentials,
            "firing_rates": firing_rates,
        }

    @property
    def tick_count(self) -> int:
        """Number of ticks processed so far."""
        return self._tick

    def get_history(self) -> List[Dict]:
        """Return the full spike-history buffer (most recent last)."""
        return list(self._history)


# ======================================================================
# Standalone demo
# ======================================================================

def _demo() -> None:
    """Simulate 100 ticks with random sensor data and print results."""
    rng = np.random.default_rng(123)
    core = NeuroCore(seed=99)

    print("=" * 72)
    print("NeuroEdge NeuroCore — 100-tick demo with random sensor data")
    print("=" * 72)
    header = (
        f"{'Tick':>5s}  "
        f"{'Spikes':>12s}  "
        f"{'Tier':>4s}  "
        f"{'Class':>8s}  "
        f"{'Confidence':>28s}"
    )
    print(header)
    print("-" * len(header))

    for t in range(100):
        # Generate random sensor readings within plausible ranges
        sensor_data = {
            "temperature": float(rng.uniform(15.0, 45.0)),
            "humidity":    float(rng.uniform(20.0, 95.0)),
            "motion":      float(rng.choice([0.0, 1.0], p=[0.8, 0.2])),
            "air_quality": float(rng.uniform(250.0, 900.0)),
        }

        result = core.process(sensor_data)

        # Format spike flags compactly: T=temp H=hum M=mot A=aq
        sp = result["spikes"]
        spike_str = (
            f"{'T' if sp['temperature'] else '.'}"
            f"{'H' if sp['humidity'] else '.'}"
            f"{'M' if sp['motion'] else '.'}"
            f"{'A' if sp['air_quality'] else '.'}"
        )

        conf_str = "[" + ", ".join(f"{c:.2f}" for c in result["confidence"]) + "]"

        print(
            f"{t:5d}  "
            f"{spike_str:>12s}  "
            f"{result['tier']:4d}  "
            f"{result['classification']:>8s}  "
            f"{conf_str:>28s}"
        )

    print("-" * len(header))
    print(f"Total ticks processed: {core.tick_count}")
    print("Demo complete.")


if __name__ == "__main__":
    _demo()
