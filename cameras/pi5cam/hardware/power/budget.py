"""Power budget, battery/solar sizing, and 5 V delivery math for a DustyCam.

All draws are ESTIMATES until measured on the real pipeline workload
(see README). Run: python3 hardware/power/budget.py
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Load:
    name: str
    peak_w: float    # worst case (all cores + ISP busy)
    active_w: float  # typical draw while doing work (inference/capture)
    idle_w: float    # draw while waiting for motion


# Estimated draws at 5 V. Pi values from Pi Foundation docs + published
# stress-test measurements; no USB peripherals assumed.
PI5 = Load("Raspberry Pi 5", peak_w=13.0, active_w=8.0, idle_w=2.7)
PI_ZERO_2W = Load("Pi Zero 2 W", peak_w=4.0, active_w=3.0, idle_w=0.8)
CAMERA_HQ = Load("Camera Module HQ", peak_w=1.0, active_w=0.7, idle_w=0.1)

DESIGN_MARGIN = 1.25      # regulator/thermal headroom over computed peak

BATTERY_18650_WH = 12.0   # ~3350 mAh * 3.6 V, per cell
USABLE_DEPTH = 0.8        # don't run Li-ion to the floor
SOLAR_DERATE = 0.15       # avg harvest fraction of panel rating (weather, angle, winter)
CONVERSION_EFF = 0.85     # buck + charge losses

# Pi 5 rail limits
RAIL_V = 5.1              # what the regulator should be set to
BROWNOUT_V = 4.75         # Pi 5 undervoltage warning/instability threshold

# Copper resistance, ohms per meter (one conductor)
AWG_OHM_PER_M = {18: 0.0210, 20: 0.0333, 22: 0.0530, 24: 0.0842}


def peak_draw_w(loads: list[Load]) -> float:
    return sum(l.peak_w for l in loads) * DESIGN_MARGIN


def average_draw_w(loads: list[Load], duty_cycle: float) -> float:
    """Mean system draw given the fraction of time spent active (0..1)."""
    return sum(l.active_w * duty_cycle + l.idle_w * (1 - duty_cycle) for l in loads)


def battery_runtime_h(cells: int, avg_draw_w: float) -> float:
    return cells * BATTERY_18650_WH * USABLE_DEPTH / avg_draw_w


def solar_panel_w(avg_draw_w: float) -> float:
    """Panel rating needed to sustain the average draw indefinitely."""
    return avg_draw_w / (SOLAR_DERATE * CONVERSION_EFF)


def wire_drop_v(awg: int, run_m: float, current_a: float) -> float:
    """Round-trip voltage drop over a supply run of run_m meters."""
    return AWG_OHM_PER_M[awg] * 2 * run_m * current_a


def max_run_m(awg: int, current_a: float, budget_v: float,
              connector_mohm: float = 20.0) -> float:
    """Longest 5 V run that stays inside budget_v of drop, after connectors."""
    wire_budget = budget_v - connector_mohm / 1000 * current_a
    return max(0.0, wire_budget / (AWG_OHM_PER_M[awg] * 2 * current_a))


def battery_current_a(power_w: float, v_cell: float = 3.2,
                      boost_eff: float = 0.9) -> float:
    """Cell-side current when boosting to 5 V; v_cell=3.2 is a nearly-flat cell."""
    return power_w / (v_cell * boost_eff)


if __name__ == "__main__":
    print("== Duty-cycle budget (battery/solar sizing) ==")
    for board, duty in [(PI5, 0.25), (PI_ZERO_2W, 0.25)]:
        loads = [board, CAMERA_HQ]
        avg = average_draw_w(loads, duty)
        print(f"{board.name} + HQ camera @ {duty:.0%} inference duty cycle")
        print(f"  average draw:        {avg:5.1f} W")
        print(f"  runtime on 2x18650:  {battery_runtime_h(2, avg):5.1f} h")
        print(f"  solar to sustain:    {solar_panel_w(avg):5.1f} W panel")
        print()

    print("== Peak rail sizing (regulator selection) ==")
    loads = [PI5, CAMERA_HQ]
    pk = peak_draw_w(loads)
    pk_a = pk / RAIL_V
    print(f"Pi 5 + HQ camera, no USB peripherals:")
    print(f"  computed peak x{DESIGN_MARGIN} margin: {pk:.1f} W = "
          f"{pk_a:.1f} A @ {RAIL_V} V")
    print(f"  (full 5 A / 25 W only needed if USB peripherals are added)")
    print(f"  cell-side current at peak (3.2 V/cell): "
          f"{battery_current_a(pk):.1f} A")
    print(f"  cell-side current at 25 W worst case:   "
          f"{battery_current_a(25):.1f} A")
    print()

    print("== 5 V distribution: voltage drop at 5 A ==")
    budget = RAIL_V - BROWNOUT_V - 0.15  # leave 150 mV for regulator sag
    print(f"drop budget: {RAIL_V} V rail - {BROWNOUT_V} V brownout "
          f"- 0.15 V transient reserve = {budget:.2f} V")
    for awg in sorted(AWG_OHM_PER_M):
        print(f"  {awg} AWG: {wire_drop_v(awg, 0.3, 5):.2f} V per 30 cm run"
              f" -> max run {max_run_m(awg, 5, budget):.2f} m"
              f" (incl. 20 mOhm connector)")
