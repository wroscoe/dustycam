"""Solar + 1S LiPo budget for the n6_speedcam (N6 + LD2415H + DFR0535).

Every draw below is an ESTIMATE until measured on the real pipeline. The two
that matter most are marked; measure them first (README: "What to measure").
Run: python3 cameras/n6_speedcam/hardware/power/budget.py
"""
from dataclasses import dataclass

V_CELL = 3.7                  # nominal 1S
USABLE_DEPTH = 0.8            # LiPo: don't run it to the DFR0535's 2.4 V cutoff every night

# --- what draws what (watts at the rail it sits on) -------------------------------------------
RADAR_W = 12.0 * 0.050        # datasheet: <= 50 mA @ 12 V, always on (it is the trigger)
N6_ACTIVE_W = 5.0 * 0.150     # openmv.io: 150 mA @ 5 V running; WiFi bursts push this up
N6_WIFI_BURST_W = 5.0 * 0.350 # MEASURE: upload bursts on the first N6 app were ~2x
N6_SLEEP_W = 3.7 * 0.0016     # openmv.io: 1.6 mA @ 3.7 V deep sleep (via BAT connector)
N6_VIN_IDLE_W = 5.0 * 0.0015  # BQ24075 charger: ~1.5 mA "when powered" from VIN, sleeping or not
CARRIER_W = 5.0 * 0.003       # power LED 1k @ 5 V (snip it for deployment) + pull-ups
DFR_QUIESCENT_W = V_CELL * 0.003   # DFR0535: overall < 3 mA (OUT3 alone < 1.72 mA)

# --- converter efficiencies, DFR0535 datasheet (3.7 V battery in) ------------------------------
EFF_OUT1_5V = 0.86            # 86 % @ 50 % load
EFF_OUT3_12V = 0.87           # 87 % @ 10 % load (we are ~10 %: 50 mA of 500 mA)
EFF_SOLAR_CHARGE = 0.75       # 78 % @ 1 A, 72 % @ 1.8 A, 18 V MPPT

# --- sun -------------------------------------------------------------------------------------
# Peak-sun-hours per day for a fixed south-facing panel tilted ~latitude, Teton County WY
# (NREL PVWatts-class numbers, rounded down; the valley loses more to terrain shading).
PSH = {"Dec (worst)": 2.0, "Mar/Sep": 4.0, "Jun (best)": 6.0}
PANEL_DERATE = 0.80           # dust, snow, angle, temperature, wiring: fraction of nameplate at MPP


@dataclass(frozen=True)
class Scenario:
    name: str
    n6_duty: float            # fraction of time the N6 is awake (captures + uploads)
    wifi_duty: float          # fraction of awake time in WiFi bursts
    sleeps: bool              # radar-triggered deep sleep between vehicles


SCENARIOS = [
    Scenario("A: always on, no sleep (today's firmware)", n6_duty=1.0, wifi_duty=0.10, sleeps=False),
    Scenario("B: radar-triggered wake, busy road (25 % awake)", n6_duty=0.25, wifi_duty=0.30, sleeps=True),
    Scenario("C: radar-triggered wake, quiet road (5 % awake)", n6_duty=0.05, wifi_duty=0.30, sleeps=True),
]


def battery_side_w(s: Scenario) -> float:
    """Average draw at the LiPo terminals."""
    n6_awake = N6_ACTIVE_W * (1 - s.wifi_duty) + N6_WIFI_BURST_W * s.wifi_duty
    n6_asleep = N6_SLEEP_W + N6_VIN_IDLE_W if s.sleeps else n6_awake
    n6_5v = n6_awake * s.n6_duty + n6_asleep * (1 - s.n6_duty) + CARRIER_W
    return n6_5v / EFF_OUT1_5V + RADAR_W / EFF_OUT3_12V + DFR_QUIESCENT_W


def daily_wh(s: Scenario) -> float:
    return battery_side_w(s) * 24


def battery_mah_for_days(s: Scenario, days: float) -> float:
    """1S capacity for `days` of no sun, using only the usable depth."""
    return daily_wh(s) * days / USABLE_DEPTH / V_CELL * 1000


def panel_w_for(s: Scenario, psh: float) -> float:
    """Nameplate panel that refills a day's use in `psh` peak-sun-hours."""
    return daily_wh(s) / (psh * PANEL_DERATE * EFF_SOLAR_CHARGE)


if __name__ == "__main__":
    print("== Loads (battery side) ==")
    print(f"radar, always on:        {RADAR_W / EFF_OUT3_12V:5.2f} W  ({RADAR_W:.2f} W @ 12 V)")
    print(f"N6 awake, no WiFi:       {N6_ACTIVE_W / EFF_OUT1_5V:5.2f} W")
    print(f"N6 awake, WiFi burst:    {N6_WIFI_BURST_W / EFF_OUT1_5V:5.2f} W   <- MEASURE")
    print(f"N6 deep sleep + charger: {(N6_SLEEP_W + N6_VIN_IDLE_W) / EFF_OUT1_5V:5.3f} W   <- MEASURE")
    print(f"DFR0535 + carrier:       {DFR_QUIESCENT_W + CARRIER_W / EFF_OUT1_5V:5.3f} W")
    print()
    for s in SCENARIOS:
        print(f"== {s.name} ==")
        print(f"  average draw:  {battery_side_w(s):5.2f} W  = {daily_wh(s):5.1f} Wh/day")
        for days in (1, 3):
            print(f"  battery for {days} sunless day(s): {battery_mah_for_days(s, days):6.0f} mAh (1S)")
        for label, psh in PSH.items():
            print(f"  panel to sustain, {label:11s}: {panel_w_for(s, psh):5.1f} W")
        print()
    print("DFR0535 limits: panel <= 20 W nameplate / 30 V open-circuit; charge <= 2 A,")
    print("so the pack should be >= 2000 mAh (1C) unless it has its own current limit.")
