# Soil sensing notes — repositioning artifact & absolute-moisture options

Findings from analyzing `server/plantlog.db` (dragon tree, STEMMA capacitive
sensor) on 2026-08-08, plus research on upgrading to absolute soil moisture
measurement.

## Why the moisture reading rises after repositioning the sensor

Every placement event in the log shows the same signature: the reading starts
low right after insertion, then **creeps up for ~12–24 h before plateauing**.
Soil temperature was stable through each event, so it isn't thermal.

| Event (from `events` table) | Reading behavior |
|---|---|
| Jul 21 21:58 — sensor added to dragon tree | ~353 in air all afternoon → 804 on insertion → crept to 901 over ~19 h (no watering) |
| Jul 22 ~18:41 — enclosure added, sensor disturbed | dropped 901 → 523 (reseated differently), then crept 523 → 553 over the next day |
| Aug 6 08:32 — sensor pulled and reseated | flat at ~606 before → climbed 606 → ~638 over ~20 h, then normal slow dry-down |

**Cause: soil-contact equilibration, not real moisture change.** The STEMMA
sensor is capacitive and only senses the few mm of soil touching the PCB blade.
Water's dielectric constant is ~80, air's is ~1, so the reading is dominated by
contact quality. Pulling/reinserting the probe leaves air gaps and loose,
disturbed soil against the blade (reads dry). Over the following hours the soil
settles back against the probe and water wicks by capillary action into the
disturbed zone, so capacitance — and the reading — rises until the probe
re-equilibrates with the bulk soil.

**Implications**

- Absolute values are **not comparable across repositions** — each insertion
  gets a new baseline (901 vs 523 for the same soil on Jul 22). Only trends
  within one placement are meaningful.
- Ignore the first ~24 h after any reseat; treat `events` rows as
  baseline-reset markers on the dashboard.
- To speed settling after a reseat: press soil firmly against the blade and
  water lightly right at the sensor.

## Getting absolute soil moisture

"Absolute" means one of three different things:

### 1. Volumetric water content (VWC, % water by volume)

Either calibrate the STEMMA (oven-dry soil = 0%, add known water volumes to
known soil volume, fit raw → VWC; only valid for one soil + one placement,
±3–5% at best), or buy a sensor designed for it — see shopping list below.

### 2. Absolute water amount in the pot — easiest true-absolute option

Put the pot on a **load cell (HX711 + 5–10 kg cell, ~$5)**. 1 g = 1 mL of
water, exactly. Weigh at field capacity (right after watering to drainage) and
at the dry limit → a true 0–100 % plant-available-water scale with no contact
artifacts and no drift from repositioning. HX711 is bit-banged GPIO — trivial
to add to the FeatherS3 logger. The capacitive sensor then becomes a
cross-check/trend signal.

### 3. Water potential (matric tension, kPa) — what plants actually respond to

The agronomically correct "does it need water" measure. Houseplants generally
want watering around −30 to −60 kPa.

## Tension (water-potential) sensor families

- **Tensiometer** — water-filled tube, porous ceramic tip, vacuum transducer
  (e.g. MPX5100). Range 0 to −80 kPa (cavitates beyond). Accurate in the wet
  range; needs refilling; very DIY-able.
- **Granular matrix — Watermark 200SS** — the practical middle ground. Two
  electrodes in a gypsum-buffered matrix; resistance tracks tension. 0 to
  −200 kPa, ~$40, lasts years, zero maintenance.
  - Interface: it's a variable resistor (~500 Ω wet → ~30 kΩ dry). Must be
    excited with **alternating polarity** (steady DC electrolyzes it): drive
    between two GPIOs through a ~7.87 kΩ series resistor, alternate direction,
    read ADC. Published curves convert R + soil temp → kPa
    ([EME Systems notes](http://www.emesystems.com/watermark/buy.html)).
  - Conditioning: ships dry — do 2–3 overnight-soak/day-dry cycles, install
    wet in a slurry-packed hole, then don't move it.
  - Buy: [Irrigation-Mart 200SS-5, $39.95](https://www.irrigation-mart.com/itemdetail/TEN-200SS-5),
    [Amazon single ~$36–40](https://www.amazon.com/Irrometer-200SS-5-Watermark-Moisture-Sensor/dp/B005ZDOBE8),
    [Amazon 4-pack ~$194](https://www.amazon.com/Irrometer-200SS-5-Watermark-Moisture-Sensor/dp/B07MW86VSQ),
    [Mega Depot](https://megadepot.com/product/irrometer-200ss-5-watermark-soil-moisture-sensor-with-5-wire).
    Suffix = cable length (200SS-5 = 5 ft).
- **Gypsum blocks** — same principle, dissolve in 1–3 seasons; superseded.
- **Dielectric matric potential (METER Teros 21)** — capacitive sensor on
  ceramic discs of known retention. −9 to −100,000 kPa, SDI-12, ~$300+.
  Research-grade, overkill here.
- **Heat-dissipation & psychrometers** — niche/lab methods.

## Calibrated VWC sensors (bigger sensing volume, factory curves)

These average over far more soil than the STEMMA's few mm, so they're much less
sensitive to insertion artifacts:

| Sensor | Price | Notes |
|---|---|---|
| [Vegetronix VH400](https://vegetronix.com/soil-moisture-sensor) | ~$37 | **Recommended.** Long blade, true 80 MHz dielectric (salinity-insensitive), potted/non-corroding, published [voltage→VWC curves](https://www.vegetronix.com/TechInfo/Soil-Moisture-Sensor-Primer) (~2% acc). 0–3 V analog, stable 400 ms after power-up — fits the FeatherS3 ADC + LDO2 switched-power scheme directly. VH400-2M/-5M/-10M = cable length. |
| [Truebner SMT50](https://www.truebner.de/en/smt50.php) | €69 | Good sensor (VWC + temp, 0–3 V) but EU-only distributors; worse deal than VH400 from the US. |
| [METER Teros 10](https://metergroup.com/products/teros-10/) | ~$150–250 | Research-grade: ~430 mL sensing volume, ±3% VWC in any mineral soil without site calibration, 10-yr body, 0–2.5 V out. Sold by quote; open-box units appear on eBay. |

## Recommended path

1. **VH400 (~$37)** for calibrated VWC with real sensing area — one ADC pin.
2. **Load cell under the pot (~$5)** for a true absolute water measurement and
   to anchor/verify everything else.
3. Optional: **Watermark 200SS (~$40)** for tension = "water me now" signal.
   Classic pairing with a VWC sensor.
