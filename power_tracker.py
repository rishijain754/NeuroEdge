"""
power_tracker.py — Microwatt-Level Power Budget Simulator for NeuroEdge
=======================================================================

Models the energy consumption of a neuromorphic edge device operating in
three processing tiers:

  * **Tier 0** — Always-on ultra-low-power monitoring (≈ 8.5 µW)
  * **Tier 1** — Event-triggered partial processing   (≈ 150 µW, 20 ms)
  * **Tier 2** — Full deep analysis                    (≈ 2200 µW, 100 ms)

Additional fixed costs are modelled for ADC sensor reads and per-spike
computations.  The tracker compares actual energy usage against a
hypothetical "always Tier-2" baseline to quantify power savings.

Dependencies: numpy (>= 1.21), Python standard library.
"""

from __future__ import annotations

import sys
from typing import Dict, List, Tuple

import numpy as np


class PowerTracker:
    """
    Tracks cumulative energy consumption and instantaneous power draw
    for a NeuroEdge device across multiple processing tiers.

    All power values are in microwatts (µW) and energy in microjoules (µJ).
    Time durations are in milliseconds (ms) unless stated otherwise.

    Usage
    -----
    >>> tracker = PowerTracker()
    >>> for tick in range(1000):
    ...     power = tracker.record_tick(tier=0, num_spikes=2, dt_ms=10.0)
    >>> print(tracker.get_summary())
    """

    # ----------------------------------------------------------------
    # Hardware power constants (µW)
    # ----------------------------------------------------------------
    TIER0_BASELINE_UW: float = 8.5       # always-on standby
    TIER1_ACTIVE_UW:   float = 150.0     # event-processing mode
    TIER1_DURATION_MS:  float = 20.0     # typical Tier-1 burst length
    TIER2_ACTIVE_UW:   float = 2200.0    # deep analysis mode
    TIER2_DURATION_MS:  float = 100.0    # typical Tier-2 burst length
    SLEEP_UW:          float = 0.5       # deep-sleep between reads
    ADC_READ_UW:       float = 3.0       # per sensor ADC conversion
    SPIKE_COMPUTE_UW:  float = 0.1       # per spike propagation

    # Number of sensors (for ADC cost)
    NUM_SENSORS: int = 4

    def __init__(self) -> None:
        # Cumulative energy in µJ  (energy = power × time)
        self.total_energy_uj: float = 0.0

        # Instantaneous power of the most recent tick (µW)
        self.current_power_uw: float = 0.0

        # History: list of (tick_index, tier, power_uw) tuples
        self.tier_history: List[Tuple[int, int, float]] = []

        # "Always-on Tier-2" hypothetical cumulative energy (for comparison)
        self.always_on_energy: float = 0.0

        # Internal tick counter
        self._tick: int = 0

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def record_tick(
        self,
        tier: int,
        num_spikes: int,
        dt_ms: float = 10.0,
    ) -> float:
        """
        Record the power draw for one simulation tick.

        Parameters
        ----------
        tier : int
            Active processing tier this tick (0, 1, or 2).
        num_spikes : int
            Number of spike propagations this tick (affects compute cost).
        dt_ms : float
            Duration of this tick in milliseconds (default 10.0 ms).

        Returns
        -------
        float
            Total instantaneous power for this tick (µW).
        """
        self._tick += 1
        dt_s = dt_ms / 1000.0  # convert to seconds for energy calculation

        # ---- Base power depends on tier ----
        if tier == 0:
            # Tier 0: always-on baseline + brief ADC reads + sleep between
            base_power = self.TIER0_BASELINE_UW
        elif tier == 1:
            # Tier 1: event processing for a short burst
            # We model the average power over the tick as a weighted blend:
            #   active fraction = min(TIER1_DURATION_MS / dt_ms, 1.0)
            active_frac = min(self.TIER1_DURATION_MS / dt_ms, 1.0)
            idle_frac = 1.0 - active_frac
            base_power = (
                active_frac * self.TIER1_ACTIVE_UW
                + idle_frac * self.TIER0_BASELINE_UW
            )
        elif tier == 2:
            # Tier 2: deep analysis — longer burst
            active_frac = min(self.TIER2_DURATION_MS / dt_ms, 1.0)
            idle_frac = 1.0 - active_frac
            base_power = (
                active_frac * self.TIER2_ACTIVE_UW
                + idle_frac * self.TIER0_BASELINE_UW
            )
        else:
            raise ValueError(f"Invalid tier: {tier}. Must be 0, 1, or 2.")

        # ---- ADC sensor reads (4 sensors, each costs ADC_READ_UW) ----
        adc_power = self.ADC_READ_UW * self.NUM_SENSORS

        # ---- Spike computation cost ----
        spike_power = self.SPIKE_COMPUTE_UW * max(num_spikes, 0)

        # ---- Sleep overhead (small constant) ----
        sleep_power = self.SLEEP_UW

        # ---- Total instantaneous power ----
        total_power = base_power + adc_power + spike_power + sleep_power
        self.current_power_uw = total_power

        # ---- Accumulate energy (E = P × t) ----
        energy_this_tick = total_power * dt_s  # µW * s = µJ
        self.total_energy_uj += energy_this_tick

        # ---- Always-on comparison (Tier-2 every tick) ----
        always_on_power = (
            self.TIER2_ACTIVE_UW
            + adc_power
            + spike_power
            + sleep_power
        )
        self.always_on_energy += always_on_power * dt_s

        # ---- Record history ----
        self.tier_history.append((self._tick, tier, total_power))

        return total_power

    def get_savings_percent(self) -> float:
        """
        Compute the percentage of energy saved compared to always-on Tier-2.

        Returns
        -------
        float
            Savings as a percentage (0.0–100.0).  Returns 0.0 if no ticks
            have been recorded yet.
        """
        if self.always_on_energy <= 0.0:
            return 0.0
        saved = self.always_on_energy - self.total_energy_uj
        return max(0.0, (saved / self.always_on_energy) * 100.0)

    def get_average_power(self) -> float:
        """
        Average power draw over all recorded ticks (µW).

        Returns
        -------
        float
            Mean power in µW.  Returns 0.0 if no history.
        """
        if not self.tier_history:
            return 0.0
        powers = [p for _, _, p in self.tier_history]
        return float(np.mean(powers))

    def get_history(self) -> List[Tuple[int, int, float]]:
        """
        Return the full tick-by-tick history.

        Returns
        -------
        list of (tick, tier, power_uw) tuples.
        """
        return list(self.tier_history)

    def get_summary(self) -> Dict:
        """
        Build a comprehensive power report.

        Returns
        -------
        dict
            Keys:
              total_ticks        — number of ticks recorded
              total_energy_uj    — cumulative energy (µJ)
              average_power_uw   — mean power (µW)
              current_power_uw   — last tick's power (µW)
              always_on_energy_uj — hypothetical always-Tier-2 energy (µJ)
              savings_percent    — energy savings vs always-on (%)
              tier_counts        — {0: n, 1: n, 2: n} breakdown
              peak_power_uw      — maximum instantaneous power seen
              min_power_uw       — minimum instantaneous power seen
        """
        # Count how many ticks at each tier
        tier_counts: Dict[int, int] = {0: 0, 1: 0, 2: 0}
        peak_power: float = 0.0
        min_power: float = float("inf") if self.tier_history else 0.0

        for _, tier, power in self.tier_history:
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            if power > peak_power:
                peak_power = power
            if power < min_power:
                min_power = power

        return {
            "total_ticks": self._tick,
            "total_energy_uj": round(self.total_energy_uj, 4),
            "average_power_uw": round(self.get_average_power(), 4),
            "current_power_uw": round(self.current_power_uw, 4),
            "always_on_energy_uj": round(self.always_on_energy, 4),
            "savings_percent": round(self.get_savings_percent(), 2),
            "tier_counts": tier_counts,
            "peak_power_uw": round(peak_power, 4),
            "min_power_uw": round(min_power, 4),
        }


# ======================================================================
# Standalone demo
# ======================================================================

def _demo() -> None:
    """Simulate 1000 ticks with random tier activations and print report."""
    rng = np.random.default_rng(42)
    tracker = PowerTracker()

    print("=" * 60)
    print("NeuroEdge Power Tracker — 1000-tick simulation")
    print("=" * 60)

    # Tier probability distribution:
    #   80% Tier 0 (idle), 15% Tier 1 (event), 5% Tier 2 (deep)
    tier_probs = [0.80, 0.15, 0.05]

    for tick in range(1000):
        # Random tier selection weighted by probability
        tier = int(rng.choice([0, 1, 2], p=tier_probs))

        # Random number of spike propagations (0–12)
        num_spikes = int(rng.integers(0, 13))

        # Fixed tick interval of 10 ms
        dt_ms = 10.0

        power = tracker.record_tick(tier=tier, num_spikes=num_spikes, dt_ms=dt_ms)

        # Print every 100th tick as a progress sample
        if (tick + 1) % 100 == 0:
            print(
                f"  Tick {tick + 1:5d} | Tier {tier} | "
                f"Power {power:8.2f} µW | "
                f"Cumulative {tracker.total_energy_uj:10.2f} µJ"
            )

    # Final report
    summary = tracker.get_summary()
    print()
    print("-" * 60)
    print("POWER REPORT")
    print("-" * 60)
    print(f"  Total ticks:          {summary['total_ticks']}")
    print(f"  Total energy:         {summary['total_energy_uj']:.4f} µJ")
    print(f"  Average power:        {summary['average_power_uw']:.4f} µW")
    print(f"  Peak power:           {summary['peak_power_uw']:.4f} µW")
    print(f"  Min power:            {summary['min_power_uw']:.4f} µW")
    print(f"  Current (last tick):  {summary['current_power_uw']:.4f} µW")
    print()
    print(f"  Always-on energy:     {summary['always_on_energy_uj']:.4f} µJ")
    print(f"  Energy savings:       {summary['savings_percent']:.2f}%")
    print()
    print("  Tier breakdown:")
    for t in (0, 1, 2):
        count = summary['tier_counts'].get(t, 0)
        pct = (count / summary['total_ticks']) * 100.0 if summary['total_ticks'] else 0
        print(f"    Tier {t}: {count:5d} ticks ({pct:5.1f}%)")
    print("-" * 60)
    print("Demo complete.")


if __name__ == "__main__":
    _demo()
