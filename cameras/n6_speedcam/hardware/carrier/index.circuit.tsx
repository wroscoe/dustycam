/**
 * n6_speedcam carrier v0.1 — OpenMV N6 <-> HLK-LD2415H speed radar
 *
 * What it does (and nothing more):
 *   - breaks the radar's 8-pin J4 pigtail out onto a labelled 0.1" header;
 *   - carries the two power rails from the solar manager (DFRobot DFR0535:
 *     OUT3 = 12 V for the radar, OUT1 = 5 V for the N6's VIN) with bulk
 *     capacitance at the radar end — the radar datasheet says supply ripple
 *     costs sensitivity;
 *   - conditions the radar's trigger output into an active-LOW wake line for
 *     the N6's P11 / WKUP3 ("connect to ground to wakeup"), so the N6 can
 *     deep-sleep between vehicles;
 *   - routes radar TX/RX (3.3 V TTL, 9600 8N1) straight to the N6's UART3
 *     (P4 = TX, P5 = RX). No level shifting needed: both sides are 3.3 V.
 *
 * The LD2415H ships in two trigger variants (datasheet §5): PULSE on pin 1
 * (active HIGH, pulled to VCC = 9–24 V through a PNP — NOT 3.3 V safe) or
 * AA on pin 2 (active LOW, open-collector NPN — safe). Both are handled:
 *
 *   AA    ─────────────────────────────┐
 *                                      ├── WAKE ── N6 P11   (R1 10k to 3V3)
 *   PULSE ── R2 47k ── Q1 2N3904 ──────┘                 (R3 100k B-E)
 *
 * Whichever pin the module populates pulls WAKE low on a detection; the other
 * floats harmlessly (R3 holds Q1 off, R1 holds WAKE high).
 *
 * DESIGNED FOR PERFBOARD FIRST, like piezohat: every part is through-hole and
 * every hole sits on one 2.54 mm lattice (all hole coordinates are odd
 * multiples of 1.27 mm), so the PCB layout is a 1:1 placement map for a
 * protoboard build. The README has the hole-coordinate table. The gerbers
 * still fab a real PCB if wanted.
 *
 * Orientation (top view): radar header along the TOP edge, N6 header along
 * the BOTTOM edge, power in on the LEFT edge. In the case the board stands
 * behind the radar with this top edge up, so the radar pigtail drops
 * straight down onto J_RADAR and the N6 leads leave the bottom edge.
 * PCB frame: board centre = (0,0), mm. Grid unit U = 2.54.
 */

const U = 2.54

// ---------- through-hole footprints (perfboard-compatible, on-grid) ----------
// axial part lying flat, lead spacing p (7.62 = 3 holes for 1/4W R)
const axial = (p: number) => (
  <footprint>
    <platedhole portHints={["pin1"]} shape="circle" pcbX={-p / 2} pcbY={0} holeDiameter={0.8} outerDiameter={1.6} />
    <platedhole portHints={["pin2"]} shape="circle" pcbX={p / 2} pcbY={0} holeDiameter={0.8} outerDiameter={1.6} />
  </footprint>
)
// radial 2-lead part on 2.54 spacing (electrolytic cap, LED): pin1 = +/anode
const radial2 = (hole = 0.9) => (
  <footprint>
    <platedhole portHints={["pin1", "anode"]} shape="circle" pcbX={-U / 2} pcbY={0} holeDiameter={hole} outerDiameter={hole + 0.8} />
    <platedhole portHints={["pin2", "cathode"]} shape="circle" pcbX={U / 2} pcbY={0} holeDiameter={hole} outerDiameter={hole + 0.8} />
  </footprint>
)
// 1×n 0.1" header row, horizontal, pin1 left
const thtRow = (n: number) => (
  <footprint>
    {Array.from({ length: n }, (_, i) => (
      <platedhole
        portHints={[`pin${i + 1}`]}
        shape="circle"
        pcbX={(i - (n - 1) / 2) * U}
        pcbY={0}
        holeDiameter={1.0}
        outerDiameter={1.8}
      />
    ))}
  </footprint>
)
// 1×n 0.1" header column, vertical, pin1 at the TOP
const thtCol = (n: number) => (
  <footprint>
    {Array.from({ length: n }, (_, i) => (
      <platedhole
        portHints={[`pin${i + 1}`]}
        shape="circle"
        pcbX={0}
        pcbY={((n - 1) / 2 - i) * U}
        holeDiameter={1.0}
        outerDiameter={1.8}
      />
    ))}
  </footprint>
)
// TO-92 lying flat, leads in a row on 2.54: E B C left→right (2N3904 flat face toward you)
const to92 = () => (
  <footprint>
    <platedhole portHints={["pin1", "E"]} shape="circle" pcbX={-U} pcbY={0} holeDiameter={0.8} outerDiameter={1.6} />
    <platedhole portHints={["pin2", "B"]} shape="circle" pcbX={0} pcbY={0} holeDiameter={0.8} outerDiameter={1.6} />
    <platedhole portHints={["pin3", "C"]} shape="circle" pcbX={U} pcbY={0} holeDiameter={0.8} outerDiameter={1.6} />
  </footprint>
)

const R_10K = { resistance: "10k", footprint: axial(3 * U) }   // 1/4W axial
const R_47K = { resistance: "47k", footprint: axial(3 * U) }
const R_100K = { resistance: "100k", footprint: axial(3 * U) }
const R_1K = { resistance: "1k", footprint: axial(3 * U) }
const C_100N = { capacitance: "100nF", footprint: axial(U) }   // ceramic disc, 2.54

// radar J4 pin order, exactly as the LD2415H datasheet table (pin 1 first)
const RADAR_PINS = ["PULSE", "AA", "RX", "TX", "B", "A", "GND", "VCC"]
// N6 leads: male header here, male↔female Dupont into the N6's female sockets
const N6_PINS = ["VIN", "GND", "V3V3", "TX_P4", "RX_P5", "WAKE_P11"]
// DFR0535 outputs: 12 V (OUT3, switch set to 12V) and 5 V (OUT1)
const PWR_PINS = ["V12", "GND12", "V5", "GND5"]

export default () => (
  <board
    title="n6_speedcam carrier v0.1 — OpenMV N6 + HLK-LD2415H"
    width={`${16 * U}mm`}
    height={`${12 * U}mm`}
    layers={4}
    material="fr4"
    thickness="1.6mm"
  >
    {/* ---------------- connectors ---------------- */}
    <pinheader
      name="JR"
      pinCount={8}
      pitch="2.54mm"
      gender="male"
      footprint={thtRow(8)}
      pinLabels={RADAR_PINS}
      manufacturerPartNumber="PinHeader_1x08_P2.54mm"
      pcbX={0} pcbY={4.5 * U} pcbRotation={0}
      schX={0} schY={5}
    />
    <pinheader
      name="JN"
      pinCount={6}
      pitch="2.54mm"
      gender="male"
      footprint={thtRow(6)}
      pinLabels={N6_PINS}
      manufacturerPartNumber="PinHeader_1x06_P2.54mm"
      pcbX={-2 * U} pcbY={-4.5 * U} pcbRotation={0}
      schX={0} schY={-5}
    />
    <pinheader
      name="JP"
      pinCount={4}
      pitch="2.54mm"
      gender="male"
      footprint={thtCol(4)}
      pinLabels={PWR_PINS}
      manufacturerPartNumber="JST_XH_B4B-XH-A"
      pcbX={-6.5 * U} pcbY={0} pcbRotation={0}
      schX={-8} schY={0}
    />

    {/* ---------------- trigger conditioning ---------------- */}
    <resistor name="R1" {...R_10K} pcbX={-3 * U} pcbY={1.5 * U} schX={3} schY={1.5} schRotation={90} />
    <resistor name="R2" {...R_47K} pcbX={1 * U} pcbY={2.5 * U} schX={-2} schY={2.5} />
    <resistor name="R3" {...R_100K} pcbX={1 * U} pcbY={1.5 * U} schX={-1} schY={0} schRotation={90} />
    <chip
      name="Q1"
      footprint={to92()}
      pinLabels={{ pin1: "E", pin2: "B", pin3: "C" }}
      manufacturerPartNumber="2N3904"
      pcbX={4.5 * U} pcbY={-0.5 * U}
      schX={1.5} schY={-1}
    />

    {/* ---------------- rail decoupling + power LED ---------------- */}
    <capacitor name="C1" capacitance="100uF" footprint={radial2()} pcbX={6 * U} pcbY={3.5 * U} schX={-5} schY={3.5} schRotation={90} />
    <capacitor name="C2" {...C_100N} pcbX={3 * U} pcbY={3.5 * U} schX={-4} schY={3.5} schRotation={90} />
    <capacitor name="C3" capacitance="100uF" footprint={radial2()} pcbX={-3 * U} pcbY={-3.5 * U} schX={-5} schY={-2.5} schRotation={90} />
    <led name="D1" footprint={radial2(0.8)} pcbX={3 * U} pcbY={-3.5 * U} schX={5} schY={-3} schRotation={-90} />
    <resistor name="R4" {...R_1K} pcbX={6 * U} pcbY={-2.5 * U} schX={5} schY={-1.5} schRotation={90} />

    {/* ---------------- mounting (M2.5, right corners) ---------------- */}
    <hole pcbX={7.5 * U} pcbY={4.5 * U} diameter="2.7mm" />
    <hole pcbX={7.5 * U} pcbY={-4.5 * U} diameter="2.7mm" />
    <hole pcbX={-7.5 * U} pcbY={-4.5 * U} diameter="2.7mm" />
    <hole pcbX={-7.5 * U} pcbY={4.5 * U} diameter="2.7mm" />

    {/* ---------------- silkscreen: install aids ----------------
        (custom THT footprints draw nothing, so every body outline, refdes,
        pin name and polarity mark an installer needs is drawn here) */}
    <silkscreentext text="n6_speedcam carrier v0.1" pcbX={0} pcbY={-5.6 * U} fontSize="0.8mm" />
    <silkscreentext text="RADAR J4 (LD2415H)" pcbX={0} pcbY={5.4 * U} fontSize="0.7mm" />
    <silkscreentext text="N6" pcbX={-5.2 * U} pcbY={-4.5 * U} fontSize="0.8mm" />
    <silkscreentext text="DFR0535" pcbX={-6.5 * U} pcbY={2.6 * U} fontSize="0.6mm" />
    <silkscreenrect pcbX={0} pcbY={4.5 * U} width="20.32mm" height="2.5mm" stroke="solid" strokeWidth="0.15mm" />
    <silkscreenrect pcbX={-2 * U} pcbY={-4.5 * U} width="15.24mm" height="2.5mm" stroke="solid" strokeWidth="0.15mm" />
    <silkscreenrect pcbX={-6.5 * U} pcbY={0} width="5.8mm" height="12.5mm" stroke="solid" strokeWidth="0.15mm" />
    {RADAR_PINS.map((lbl, i) => (
      <silkscreentext text={lbl} pcbX={(i - 3.5) * U} pcbY={3.8 * U} fontSize="0.55mm" />
    ))}
    {N6_PINS.map((lbl, i) => (
      <silkscreentext text={lbl.replace("V3V3", "3V3").split("_")[0]} pcbX={(i - 4.5) * U} pcbY={-3.8 * U} fontSize="0.55mm" />
    ))}
    {N6_PINS.map((lbl, i) => (
      <silkscreentext text={lbl.includes("_") ? lbl.split("_")[1] : ""} pcbX={(i - 4.5) * U} pcbY={-3.2 * U} fontSize="0.5mm" />
    ))}
    {PWR_PINS.map((lbl, i) => (
      <silkscreentext text={lbl.replace("GND12", "GND").replace("GND5", "GND").replace("V12", "12V").replace("V5", "5V")} pcbX={-5.4 * U} pcbY={(1.5 - i) * U} fontSize="0.55mm" />
    ))}
    {/* axial bodies + labels */}
    {[
      { t: "R1 10k", x: -3, y: 1.5 }, { t: "R2 47k", x: 1, y: 2.5 }, { t: "R3 100k", x: 1, y: 1.5 }, { t: "R4 1k", x: 6, y: -2.5 },
    ].map(({ t, x, y }) => (
      <>
        <silkscreenrect pcbX={x * U} pcbY={y * U} width="5.2mm" height="1.5mm" stroke="solid" strokeWidth="0.12mm" />
        <silkscreentext text={t} pcbX={x * U} pcbY={y * U} fontSize="0.6mm" />
      </>
    ))}
    <silkscreenrect pcbX={4.5 * U} pcbY={-0.5 * U} width="6.2mm" height="2.6mm" stroke="solid" strokeWidth="0.12mm" />
    <silkscreentext text="Q1 2N3904  E B C" pcbX={4.5 * U} pcbY={0.4 * U} fontSize="0.55mm" />
    <silkscreentext text="C1 100u 25V +" pcbX={6 * U} pcbY={2.8 * U} fontSize="0.55mm" />
    <silkscreentext text="C2 100n" pcbX={3 * U} pcbY={2.9 * U} fontSize="0.55mm" />
    <silkscreentext text="C3 100u 10V +" pcbX={-3 * U} pcbY={-2.7 * U} fontSize="0.55mm" />
    <silkscreentext text="D1 PWR +" pcbX={3 * U} pcbY={-2.7 * U} fontSize="0.55mm" />
    
    {/* ================= nets ================= */}
    {/* 12 V rail: DFR0535 OUT3 -> radar VCC, bulk + ceramic at the radar header */}
    <trace from="JP.V12" to="net.V12" />
    <trace from="JP.GND12" to="net.GND" />
    <trace from="JR.VCC" to="net.V12" />
    <trace from="JR.GND" to="net.GND" />
    <trace from="C1.pin1" to="net.V12" />
    <trace from="C1.pin2" to="net.GND" />
    <trace from="C2.pin1" to="net.V12" />
    <trace from="C2.pin2" to="net.GND" />

    {/* 5 V rail: DFR0535 OUT1 -> N6 VIN, bulk + power LED */}
    <trace from="JP.V5" to="net.V5" />
    <trace from="JP.GND5" to="net.GND" />
    <trace from="JN.VIN" to="net.V5" />
    <trace from="JN.GND" to="net.GND" />
    <trace from="C3.pin1" to="net.V5" />
    <trace from="C3.pin2" to="net.GND" />
    <trace from="R4.pin1" to="net.V5" />
    <trace from="R4.pin2" to="net.LEDA" />
    <trace from="D1.anode" to="net.LEDA" />
    <trace from="D1.cathode" to="net.GND" />

    {/* 3V3 from the N6 only feeds the WAKE pull-up */}
    <trace from="JN.V3V3" to="net.V3V3" />
    <trace from="R1.pin1" to="net.V3V3" />
    <trace from="R1.pin2" to="net.WAKE" />
    <trace from="JN.WAKE_P11" to="net.WAKE" />

    {/* trigger: AA is open-collector active-low -> straight onto WAKE */}
    <trace from="JR.AA" to="net.WAKE" />
    {/* trigger: PULSE is active-high at VCC level -> inverted by Q1 */}
    <trace from="JR.PULSE" to="net.PULSE" />
    <trace from="R2.pin1" to="net.PULSE" />
    <trace from="R2.pin2" to="net.QB" />
    <trace from="R3.pin1" to="net.QB" />
    <trace from="R3.pin2" to="net.GND" />
    <trace from="Q1.B" to="net.QB" />
    <trace from="Q1.E" to="net.GND" />
    <trace from="Q1.C" to="net.WAKE" />

    {/* UART: radar TX -> N6 RX (P5), radar RX <- N6 TX (P4) */}
    <trace from="JR.TX" to="net.RADAR_TX" />
    <trace from="JN.RX_P5" to="net.RADAR_TX" />
    <trace from="JR.RX" to="net.RADAR_RX" />
    <trace from="JN.TX_P4" to="net.RADAR_RX" />

    <copperpour connectsTo="net.GND" layer="bottom" clearance="0.25mm" />
  </board>
)
