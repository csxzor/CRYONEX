// CRYONEX — SIH 2026 idea submission deck (Team CRYOBLUE, PS SIH26153)
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fs = require("fs");
const fa = require("react-icons/fa");

const OUT = process.argv[2] || "CRYONEX_SIH26153_CRYOBLUE.pptx";
const HB = JSON.parse(fs.readFileSync(__dirname + "/heartbleed.json", "utf8"));
const SIH_LOGO = "image/png;base64," + fs.readFileSync(__dirname + "/sih_2026_logo.png").toString("base64");

// ---- palette ------------------------------------------------------------------------------
const NAVY = "0B1F33";   // dominant dark
const NAVY2 = "13304D";  // raised surface on dark
const ICE = "38BDF8";    // accent (brand: CRYOBLUE)
const ICE_T = "E0F2FE";  // light tint
const AMBER = "F59E0B";  // early warning
const RED = "E5484D";    // attack in progress
const INK = "1E293B";    // body text
const MUTED = "64748B";  // captions
const CARD = "F1F5F9";   // light card
const WHITE = "FFFFFF";
const HEAD = "Arial";
const BODY = "Calibri";

async function icon(Comp, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size: String(size) }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.author = "Team CRYOBLUE";
pres.title = "CRYONEX: Forecasting network attacks before they complete (SIH26153)";

const W = 13.333;

// ---- shared pieces -------------------------------------------------------------------------
function header(slide, title, sub, n) {
  slide.background = { color: WHITE };
  slide.addText(title, { x: 0.6, y: 0.35, w: 9.8, h: 0.7, fontFace: HEAD, fontSize: 30, bold: true, color: NAVY, margin: 0, isTextBox: true });
  if (sub) slide.addText(sub, { x: 0.6, y: 1.02, w: 9.8, h: 0.4, fontFace: BODY, fontSize: 15, color: MUTED, margin: 0, isTextBox: true });
  slide.addImage({ data: SIH_LOGO, x: W - 0.6 - 1.95, y: 0.32, w: 1.95, h: 0.92 });
  slide.addText("SIH 2026  ·  SIH26153  ·  Team CRYOBLUE", { x: 0.6, y: 7.0, w: 6, h: 0.3, fontFace: BODY, fontSize: 10, color: MUTED, margin: 0, isTextBox: true });
  slide.addText(String(n), { x: W - 1.1, y: 7.0, w: 0.5, h: 0.3, fontFace: BODY, fontSize: 10, color: MUTED, align: "right", margin: 0, isTextBox: true });
}

function hexIcon(slide, img, x, y, d, fill) {
  slide.addShape(pres.shapes.HEXAGON, { x, y, w: d, h: d * 0.88, fill: { color: fill }, line: { color: fill } });
  const s = d * 0.46;
  slide.addImage({ data: img, x: x + (d - s) / 2, y: y + (d * 0.88 - s) / 2, w: s, h: s });
}

function card(slide, x, y, w, h, fill = CARD) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.12 });
}

function arrow(slide, x, y, w, color = MUTED) {
  slide.addShape(pres.shapes.LINE, { x, y, w, h: 0, line: { color, width: 2, endArrowType: "triangle" } });
}

(async () => {
  const I = {};
  const need = {
    net: fa.FaNetworkWired, layers: fa.FaLayerGroup, brain: fa.FaBrain, forward: fa.FaForward, search: fa.FaSearch,
    bell: fa.FaBell, shield: fa.FaShieldAlt, eye: fa.FaEye, plug: fa.FaPlug, laptop: fa.FaLaptop, db: fa.FaDatabase,
    check: fa.FaCheckCircle, users: fa.FaUsers, industry: fa.FaIndustry, landmark: fa.FaLandmark, flask: fa.FaFlask,
    clock: fa.FaClock, rupee: fa.FaRupeeSign, lock: fa.FaLock, server: fa.FaServer, globe: fa.FaGlobe, route: fa.FaRoute,
    snow: fa.FaSnowflake, chart: fa.FaChartLine, tags: fa.FaTags, github: fa.FaGithub, fire: fa.FaFireAlt, cogs: fa.FaCogs,
    vial: fa.FaVial, bolt: fa.FaBolt, map: fa.FaMapSigns, question: fa.FaQuestionCircle,
  };
  for (const [k, C] of Object.entries(need)) I[k] = await icon(C, WHITE);
  const Iice = { snow: await icon(fa.FaSnowflake, ICE), check: await icon(fa.FaCheckCircle, "16A34A"),
                 github: await icon(fa.FaGithub, NAVY), arrowR: await icon(fa.FaArrowRight, ICE) };

  // ============================================================================================
  // 1. TITLE
  // ============================================================================================
  {
    const s = pres.addSlide();
    s.background = { color: NAVY };
    // logo on a white card (the SIH logo has dark text)
    card(s, W - 0.6 - 2.9, 0.45, 2.9, 1.45, WHITE);
    s.addImage({ data: SIH_LOGO, x: W - 0.6 - 2.9 + 0.2, y: 0.55, w: 2.5, h: 1.18 });
    // team wordmark
    s.addImage({ data: Iice.snow, x: 0.6, y: 0.62, w: 0.42, h: 0.42 });
    s.addText("CRYOBLUE", { x: 1.12, y: 0.58, w: 4, h: 0.5, fontFace: HEAD, fontSize: 20, bold: true, color: ICE, charSpacing: 4, margin: 0, isTextBox: true });

    s.addText("Forecasting network attacks\nbefore they complete", { x: 0.6, y: 1.95, w: 8.6, h: 1.7, fontFace: HEAD, fontSize: 40, bold: true, color: WHITE, margin: 0, isTextBox: true, valign: "top" });
    s.addText("CRYONEX: a Kill-Chain World Model that learns how a network behaves, simulates its next 5 minutes, and warns defenders, in plain MITRE ATT&CK terms.",
      { x: 0.6, y: 3.75, w: 8.4, h: 0.85, fontFace: BODY, fontSize: 17, color: "CBD5E1", margin: 0, isTextBox: true });

    // kill-chain motif: hexagon stages
    const stages = [["Recon", ICE], ["Initial Access", ICE], ["Lateral Move", AMBER], ["C2", AMBER], ["Exfiltration", RED]];
    stages.forEach(([name, col], i) => {
      const x = 0.6 + i * 1.72, y = 5.0;
      s.addShape(pres.shapes.HEXAGON, { x, y, w: 1.35, h: 0.62, fill: { color: NAVY2 }, line: { color: col, width: 1.5 } });
      s.addText(name, { x, y, w: 1.35, h: 0.62, fontFace: BODY, fontSize: 12, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0, isTextBox: true });
      if (i < stages.length - 1) arrow(s, x + 1.38, y + 0.31, 0.3, "475569");
    });
    s.addText("We predict the next step here, not just detect the last one.", { x: 0.6, y: 5.72, w: 8.6, h: 0.35, fontFace: BODY, fontSize: 13, italic: true, color: "94A3B8", margin: 0, isTextBox: true });

    // details panel
    const rows = [["Problem Statement ID", "SIH26153"], ["Problem Statement Title", "AI based Network Attack Forecasting from Network Traffic Data"],
      ["Theme", "Blockchain & Cybersecurity"], ["PS Category", "Software"], ["Team ID", "141585"], ["Team Name", "CRYOBLUE"]];
    card(s, 9.55, 2.25, 3.18, 4.6, NAVY2);
    let y = 2.42;
    rows.forEach(([k, v]) => {
      const h = k.includes("Title") ? 0.95 : 0.55;
      s.addText([{ text: k.toUpperCase(), options: { fontSize: 9, color: "94A3B8", bold: true, charSpacing: 1, breakLine: true } },
                 { text: v, options: { fontSize: 13, color: WHITE, bold: true } }],
        { x: 9.75, y, w: 2.8, h, fontFace: BODY, margin: 0, valign: "top", isTextBox: true });
      y += h + 0.1;
    });
    s.addNotes("Title slide. Team ID 141585 taken from the draft deck: verify before submission.");
  }

  // ============================================================================================
  // 2. PROPOSED SOLUTION
  // ============================================================================================
  {
    const s = pres.addSlide();
    header(s, "Proposed Solution", "Most security tools tell you an attack happened. CRYONEX tells you one is coming, and what it will do next.", 2);

    // problem -> answer
    const pa = [
      ["Tools judge each connection alone, missing how an intrusion unfolds over time", "Learns how the whole network's state changes every 10 seconds (a world model)"],
      ["Alerts fire only once the attack is already happening", "Simulates the next 5 minutes and raises an early warning before compromise"],
      ["A risk score with no reason gives analysts nothing to act on", "Every alert names the MITRE ATT&CK stage, the traffic that caused it, and the fix"],
    ];
    s.addText("THE PROBLEM", { x: 0.6, y: 1.65, w: 3.9, h: 0.3, fontFace: HEAD, fontSize: 12, bold: true, color: RED, charSpacing: 1, margin: 0, isTextBox: true });
    s.addText("OUR ANSWER", { x: 5.0, y: 1.65, w: 3.9, h: 0.3, fontFace: HEAD, fontSize: 12, bold: true, color: "0284C7", charSpacing: 1, margin: 0, isTextBox: true });
    pa.forEach(([p, a], i) => {
      const y = 2.05 + i * 0.95;
      card(s, 0.6, y, 3.95, 0.8, "FEF2F2");
      s.addText(p, { x: 0.78, y, w: 3.65, h: 0.8, fontFace: BODY, fontSize: 13, color: INK, valign: "middle", margin: 0, isTextBox: true });
      s.addImage({ data: Iice.arrowR, x: 4.62, y: y + 0.27, w: 0.28, h: 0.26 });
      card(s, 5.0, y, 4.0, 0.8, ICE_T);
      s.addText(a, { x: 5.18, y, w: 3.7, h: 0.8, fontFace: BODY, fontSize: 13, color: INK, bold: true, valign: "middle", margin: 0, isTextBox: true });
    });

    // uniqueness
    s.addText("WHAT MAKES IT DIFFERENT", { x: 9.55, y: 1.65, w: 3.2, h: 0.3, fontFace: HEAD, fontSize: 12, bold: true, color: NAVY, charSpacing: 1, margin: 0, isTextBox: true });
    const uq = [[I.forward, "Forecasts, not just detects"], [I.map, "Speaks MITRE ATT&CK"], [I.eye, "Explains every alert"], [I.lock, "Runs fully offline"]];
    uq.forEach(([ic, t], i) => {
      const y = 2.05 + i * 0.72;
      hexIcon(s, ic, 9.55, y, 0.62, NAVY);
      s.addText(t, { x: 10.32, y, w: 2.45, h: 0.55, fontFace: BODY, fontSize: 14, bold: true, color: INK, valign: "middle", margin: 0, isTextBox: true });
    });

    // workflow
    s.addText("HOW IT WORKS", { x: 0.6, y: 5.25, w: 4, h: 0.3, fontFace: HEAD, fontSize: 12, bold: true, color: NAVY, charSpacing: 1, margin: 0, isTextBox: true });
    const wf = [[I.net, "Capture", "PCAP or flow CSV"], [I.layers, "Build state", "92 signals / 10 s"], [I.brain, "Simulate", "next 5 minutes"],
      [I.search, "Explain", "ATT&CK stage + why"], [I.bell, "Alert", "warn, confirm, fix"]];
    wf.forEach(([ic, t, d], i) => {
      const x = 0.6 + i * 2.5;
      hexIcon(s, ic, x, 5.7, 0.78, i === 4 ? AMBER : i === 2 ? "0284C7" : NAVY);
      s.addText([{ text: `${i + 1}. ${t}`, options: { bold: true, fontSize: 14, color: INK, breakLine: true } }, { text: d, options: { fontSize: 12, color: MUTED } }],
        { x: x + 0.9, y: 5.7, w: 1.5, h: 0.72, fontFace: BODY, margin: 0, valign: "middle", isTextBox: true });
      if (i < wf.length - 1) arrow(s, x + 2.2, 6.04, 0.22, "94A3B8");
    });
  }

  // ============================================================================================
  // 3. TECHNICAL APPROACH (pipeline)
  // ============================================================================================
  {
    const s = pres.addSlide();
    header(s, "Technical Approach", "From raw traffic to a forecast an analyst can act on: one pipeline, fully offline.", 3);

    const boxes = [
      [I.net, "Traffic in", "PCAP (packets) or flow records (CSV). Any enterprise network tap."],
      [I.layers, "Network state", "Every 10 s: 92 signals. Flow level (flags, ports, bytes, timing) + packet level (TTL, TCP window, scans)."],
      [I.brain, "World model", "Causal Transformer learns how the state evolves, P(Sₜ₊₁ | Sₜ), with ATT&CK stages built in."],
      [I.forward, "Forward simulation", "Rolls the network 30 steps (5 min) ahead: will this trajectory reach an infiltration?"],
    ];
    const bw = 2.72, gap = 0.36, y0 = 1.75, bh = 2.0;
    boxes.forEach(([ic, t, d], i) => {
      const x = 0.6 + i * (bw + gap);
      card(s, x, y0, bw, bh, i === 2 ? ICE_T : CARD);
      hexIcon(s, ic, x + 0.2, y0 + 0.2, 0.62, i === 2 ? "0284C7" : NAVY);
      s.addText(t, { x: x + 0.95, y: y0 + 0.2, w: bw - 1.05, h: 0.55, fontFace: HEAD, fontSize: 15, bold: true, color: NAVY, valign: "middle", margin: 0, isTextBox: true });
      s.addText(d, { x: x + 0.2, y: y0 + 0.9, w: bw - 0.4, h: 1.0, fontFace: BODY, fontSize: 12.5, color: INK, valign: "top", margin: 0, isTextBox: true });
      if (i < boxes.length - 1) arrow(s, x + bw + 0.04, y0 + bh / 2, gap - 0.08, "94A3B8");
    });

    // outputs
    s.addText("WHAT THE DEFENDER GETS", { x: 0.6, y: 4.05, w: 5, h: 0.3, fontFace: HEAD, fontSize: 12, bold: true, color: NAVY, charSpacing: 1, margin: 0, isTextBox: true });
    const outs = [[I.chart, "Infiltration probability", "a timeline of risk for the next 5 minutes", "0284C7"],
      [I.route, "ATT&CK stage: now and next", "Recon → Initial Access → Lateral → C2 → Exfil", NAVY],
      [I.search, "Why: driving features", "the flags, ports and flows behind each alert", NAVY],
      [I.bell, "Two-level alerts + response", "early warning, attack in progress, D3FEND fixes", AMBER]];
    outs.forEach(([ic, t, d, col], i) => {
      const x = 0.6 + i * 3.08;
      hexIcon(s, ic, x, 4.45, 0.62, col);
      s.addText([{ text: t, options: { bold: true, fontSize: 13.5, color: INK, breakLine: true } }, { text: d, options: { fontSize: 11.5, color: MUTED } }],
        { x: x + 0.75, y: 4.4, w: 2.25, h: 0.8, fontFace: BODY, margin: 0, valign: "middle", isTextBox: true });
    });

    // stack + data + link
    card(s, 0.6, 5.55, 12.13, 1.25, CARD);
    s.addText([{ text: "Tech stack   ", options: { bold: true, color: NAVY } },
               { text: "Python · PyTorch · NFStream (packet parsing) · Polars · scikit-learn · Streamlit + Plotly · MITRE ATT&CK & D3FEND (local copies)", options: { color: INK } }],
      { x: 0.85, y: 5.65, w: 11.7, h: 0.35, fontFace: BODY, fontSize: 12.5, margin: 0, isTextBox: true });
    s.addText([{ text: "Trained on   ", options: { bold: true, color: NAVY } },
               { text: "CIC-IDS2017 (with raw PCAPs) · CSE-CIC-IDS2018 · CTU-13 botnets · DAPT2020 APT: about 60 million real flows", options: { color: INK } }],
      { x: 0.85, y: 6.0, w: 11.7, h: 0.35, fontFace: BODY, fontSize: 12.5, margin: 0, isTextBox: true });
    s.addImage({ data: Iice.github, x: 0.85, y: 6.4, w: 0.28, h: 0.28 });
    s.addText([{ text: "Prototype & code   ", options: { bold: true, color: NAVY } },
               { text: "github.com/csxzor/153", options: { color: "0284C7", hyperlink: { url: "https://github.com/csxzor/153" } } }],
      { x: 1.2, y: 6.37, w: 8, h: 0.35, fontFace: BODY, fontSize: 12.5, margin: 0, isTextBox: true });
  }

  // ============================================================================================
  // 4. PROTOTYPE & RESULTS
  // ============================================================================================
  {
    const s = pres.addSlide();
    header(s, "Working Prototype & Results", "Built, trained and tested on real attacks the model never saw.", 4);

    // chart: real held-out Heartbleed timeline
    card(s, 0.6, 1.6, 7.3, 5.2, CARD);
    s.addText("A real attack, live: Heartbleed (held-out test data)", { x: 0.85, y: 1.72, w: 6.9, h: 0.4, fontFace: HEAD, fontSize: 14, bold: true, color: NAVY, margin: 0, isTextBox: true });
    const idx = HB.t.map((t, i) => i).filter(i => HB.t[i] >= "18:00" && HB.t[i] <= "18:20");
    const labels = idx.map(i => HB.t[i]);
    const risk = idx.map(i => HB.risk[i]);
    s.addChart(pres.charts.LINE, [
      { name: "Attack probability (next 5 min)", labels, values: risk },
      { name: "Alert threshold", labels, values: labels.map(() => Number(HB.threshold.toFixed(2))) },
    ], {
      x: 0.75, y: 2.15, w: 7.0, h: 3.65, chartColors: ["0284C7", "94A3B8"], lineSize: 2.5, lineDataSymbol: "none",
      valAxisMinVal: 0, valAxisMaxVal: 0.8, valAxisLabelFormatCode: "0%", valAxisLabelColor: MUTED, catAxisLabelColor: MUTED,
      valAxisLabelFontSize: 10, catAxisLabelFontSize: 10, valGridLine: { color: "E2E8F0", size: 0.5 }, catGridLine: { style: "none" },
      catAxisLabelFrequency: 2, showLegend: true, legendPos: "b", legendFontSize: 10, legendColor: INK,
    });
    // timeline callouts
    const tags = [[AMBER, "18:08", "Early warning", "a compromise is forecast"], [RED, "18:11", "Attack in progress", "alert confirmed"], [NAVY, "18:12", "Real exploit starts", "from the dataset's labels"]];
    tags.forEach(([col, t, a, b], i) => {
      const x = 0.85 + i * 2.33;
      s.addShape(pres.shapes.OVAL, { x, y: 5.95, w: 0.2, h: 0.2, fill: { color: col }, line: { color: col } });
      s.addText([{ text: `${t}  ${a}`, options: { bold: true, fontSize: 12, color: INK, breakLine: true } }, { text: b, options: { fontSize: 10.5, color: MUTED } }],
        { x: x + 0.28, y: 5.85, w: 2.0, h: 0.62, fontFace: BODY, margin: 0, valign: "top", isTextBox: true });
    });

    // stat callouts
    const stats = [["0.73", "vs 0.51", "F1 score vs the required logistic-regression baseline (same features)"],
      ["1 in 3", "real attacks", "warned before they start, about 5 minutes ahead"],
      ["50%", "next stage", "next ATT&CK stage predicted correctly at real stage changes"],
      ["0.78", "vs 0.57", "on attack types never seen in training (AUPRC)"]];
    stats.forEach(([big, small, d], i) => {
      const x = 8.25 + (i % 2) * 2.3, y = 1.6 + Math.floor(i / 2) * 2.5;
      card(s, x, y, 2.18, 2.35, i === 1 ? "FEF3C7" : ICE_T);
      s.addText(big, { x: x + 0.15, y: y + 0.18, w: 1.95, h: 0.8, fontFace: HEAD, fontSize: 36, bold: true, color: i === 1 ? "B45309" : NAVY, margin: 0, isTextBox: true });
      s.addText(small, { x: x + 0.15, y: y + 0.95, w: 1.95, h: 0.35, fontFace: BODY, fontSize: 13, bold: true, color: INK, margin: 0, isTextBox: true });
      s.addText(d, { x: x + 0.15, y: y + 1.32, w: 1.93, h: 1.0, fontFace: BODY, fontSize: 11.5, color: MUTED, valign: "top", margin: 0, isTextBox: true });
    });
    s.addText("Held-out test data, 4 public datasets, 3 seeds. Full tables: docs/BENCHMARKS.md", { x: 8.25, y: 6.55, w: 4.5, h: 0.25, fontFace: BODY, fontSize: 9, italic: true, color: MUTED, margin: 0, isTextBox: true });
  }

  // ============================================================================================
  // 5. FEASIBILITY & VIABILITY
  // ============================================================================================
  {
    const s = pres.addSlide();
    header(s, "Feasibility & Viability", "Not a plan on paper: the prototype already runs, end to end, on a laptop.", 5);

    // proven
    s.addText("ALREADY PROVEN", { x: 0.6, y: 1.65, w: 3.6, h: 0.3, fontFace: HEAD, fontSize: 12, bold: true, color: NAVY, charSpacing: 1, margin: 0, isTextBox: true });
    const proven = ["Working offline prototype: PCAP/CSV in, forecast out", "Trained on ~60 million real flows from 4 public datasets",
      "Runs on a CPU laptop, no GPU or cloud needed", "75 automated tests; every result reproducible from code"];
    proven.forEach((t, i) => {
      const y = 2.05 + i * 0.72;
      s.addImage({ data: Iice.check, x: 0.6, y: y + 0.08, w: 0.32, h: 0.32 });
      s.addText(t, { x: 1.05, y, w: 3.2, h: 0.55, fontFace: BODY, fontSize: 13, color: INK, valign: "middle", margin: 0, isTextBox: true });
    });

    // challenges -> mitigation
    s.addText("CHALLENGES & HOW WE HANDLE THEM", { x: 4.6, y: 1.65, w: 4.6, h: 0.3, fontFace: HEAD, fontSize: 12, bold: true, color: NAVY, charSpacing: 1, margin: 0, isTextBox: true });
    const ch = [["Every network looks different", "15-minute warm-up: self-calibrates to a new network"],
      ["Some attacks start with no warning sign", "Two alert levels: early warning + attack in progress"],
      ["Analysts distrust black boxes", "Explanations tested: removing the top features cuts risk 500× more than random ones"],
      ["Few labelled multi-stage attacks", "Public datasets + synthetic kill-chain campaigns for training"]];
    ch.forEach(([c, m], i) => {
      const y = 2.05 + i * 0.95;
      card(s, 4.6, y, 4.6, 0.82, CARD);
      s.addText([{ text: c, options: { bold: true, fontSize: 12.5, color: RED, breakLine: true } }, { text: m, options: { fontSize: 12, color: INK } }],
        { x: 4.78, y, w: 4.3, h: 0.82, fontFace: BODY, margin: 0, valign: "middle", isTextBox: true });
    });

    // deployment diagram
    s.addText("DEPLOYMENT", { x: 9.65, y: 1.65, w: 3, h: 0.3, fontFace: HEAD, fontSize: 12, bold: true, color: NAVY, charSpacing: 1, margin: 0, isTextBox: true });
    const dep = [[I.globe, "Internet", MUTED], [I.shield, "Router / firewall (copy of traffic via TAP/SPAN)", NAVY],
      [I.server, "CRYONEX, on-premise, air-gap ready", "0284C7"], [I.bell, "SOC dashboard + SIEM alerts", AMBER]];
    dep.forEach(([ic, t, col], i) => {
      const y = 2.05 + i * 1.05;
      hexIcon(s, ic, 9.65, y, 0.66, col);
      s.addText(t, { x: 10.45, y, w: 2.3, h: 0.6, fontFace: BODY, fontSize: 12.5, bold: true, color: INK, valign: "middle", margin: 0, isTextBox: true });
      if (i < dep.length - 1) s.addShape(pres.shapes.LINE, { x: 9.98, y: y + 0.62, w: 0, h: 0.38, line: { color: "94A3B8", width: 2, endArrowType: "triangle" } });
    });

    card(s, 0.6, 6.0, 12.13, 0.8, NAVY);
    s.addText([{ text: "Viability:  ", options: { bold: true, color: ICE } },
               { text: "open-source stack, public data, commodity hardware, no licence or cloud fees. Deployable in enterprise and Critical Information Infrastructure networks where data must never leave the premises.", options: { color: WHITE } }],
      { x: 0.85, y: 6.0, w: 11.7, h: 0.8, fontFace: BODY, fontSize: 13, valign: "middle", margin: 0, isTextBox: true });
  }

  // ============================================================================================
  // 6. IMPACT & BENEFITS
  // ============================================================================================
  {
    const s = pres.addSlide();
    header(s, "Impact & Benefits", "Minutes of warning turn incident response into incident prevention.", 6);

    const ben = [[I.shield, "Security", "Early warning before compromise completes; the attacker's next stage is named in advance."],
      [I.cogs, "Operational", "Explained alerts cut triage time: analysts see the stage, the reason, the hosts and the fix."],
      [I.rupee, "Economic", "Open source on commodity CPUs: no licences, no cloud bills, reusable across networks."],
      [I.lock, "Strategic", "Runs air-gapped, so sensitive national traffic never leaves the network."]];
    ben.forEach(([ic, t, d], i) => {
      const x = 0.6 + (i % 2) * 3.9, y = 1.75 + Math.floor(i / 2) * 2.5;
      card(s, x, y, 3.7, 2.2, CARD);
      hexIcon(s, ic, x + 0.25, y + 0.25, 0.7, i === 0 ? "0284C7" : NAVY);
      s.addText(t, { x: x + 1.1, y: y + 0.25, w: 2.4, h: 0.62, fontFace: HEAD, fontSize: 17, bold: true, color: NAVY, valign: "middle", margin: 0, isTextBox: true });
      s.addText(d, { x: x + 0.25, y: y + 1.1, w: 3.25, h: 0.95, fontFace: BODY, fontSize: 14.5, color: INK, valign: "top", margin: 0, isTextBox: true });
    });

    // who benefits: nested circles (TAM / SAM / SOM)
    const cx = 10.45, cy = 4.0;
    [[2.35, "E0F2FE"], [1.65, "BAE6FD"], [0.95, "0284C7"]].forEach(([r, col]) => {
      s.addShape(pres.shapes.OVAL, { x: cx - r, y: cy - r * 0.92, w: 2 * r, h: 2 * r * 0.92, fill: { color: col }, line: { color: WHITE, width: 1.5 } });
    });
    s.addText([{ text: "TAM", options: { bold: true, fontSize: 12, color: NAVY, breakLine: true } }, { text: "All enterprise & CII networks", options: { fontSize: 11, color: INK } }],
      { x: cx - 1.3, y: cy - 2.1, w: 2.6, h: 0.5, fontFace: BODY, align: "center", margin: 0, isTextBox: true });
    s.addText([{ text: "SAM", options: { bold: true, fontSize: 12, color: NAVY, breakLine: true } }, { text: "Organisations with a SOC / SIEM", options: { fontSize: 11, color: INK } }],
      { x: cx - 1.2, y: cy - 1.42, w: 2.4, h: 0.5, fontFace: BODY, align: "center", margin: 0, isTextBox: true });
    s.addText([{ text: "SOM", options: { bold: true, fontSize: 12, color: WHITE, breakLine: true } }, { text: "Pilot: CII & government networks", options: { fontSize: 10.5, color: WHITE } }],
      { x: cx - 0.85, y: cy - 0.35, w: 1.7, h: 0.8, fontFace: BODY, align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText("Who benefits: SOC analysts · power, telecom, banking & transport operators · national agencies (NTRO, CERT-In) · security researchers",
      { x: 8.3, y: 6.25, w: 4.45, h: 0.6, fontFace: BODY, fontSize: 11, color: MUTED, align: "center", margin: 0, isTextBox: true });
  }

  // ============================================================================================
  // 7. RESEARCH & REFERENCES
  // ============================================================================================
  {
    const s = pres.addSlide();
    s.background = { color: NAVY };
    s.addText("Research & References", { x: 0.6, y: 0.4, w: 8, h: 0.7, fontFace: HEAD, fontSize: 30, bold: true, color: WHITE, margin: 0, isTextBox: true });
    card(s, W - 0.6 - 2.3, 0.35, 2.3, 1.1, WHITE);
    s.addImage({ data: SIH_LOGO, x: W - 0.6 - 2.3 + 0.13, y: 0.42, w: 2.04, h: 0.96 });

    const colA = [["Core ideas", ICE], ["Ha & Schmidhuber (2018). World Models. arXiv:1803.10122", null],
      ["Hafner et al. (2023). Mastering Diverse Domains through World Models (DreamerV3)", null],
      ["Vaswani et al. (2017). Attention Is All You Need. NeurIPS", null],
      ["Sundararajan et al. (2017). Axiomatic Attribution for Deep Networks (Integrated Gradients). ICML", null],
      ["Frameworks", ICE], ["MITRE ATT&CK Enterprise matrix: attack.mitre.org", null], ["MITRE D3FEND countermeasures: d3fend.mitre.org", null]];
    const colB = [["Datasets", ICE], ["Sharafaldin et al. (2018). CIC-IDS2017, ICISSP (+ Engelen et al. 2021, corrected labels)", null],
      ["CSE-CIC-IDS2018, Canadian Institute for Cybersecurity & CSE", null],
      ["García et al. (2014). CTU-13 botnet dataset. Computers & Security", null],
      ["Myneni et al. (2020). DAPT2020: advanced persistent threat dataset. MLN", null],
      ["Our work", ICE], ["Code, benchmarks & demo: github.com/csxzor/153", null]];
    const para = (items) => items.map(([t, h], i) => ({ text: t, options: h
      ? { bold: true, color: h, fontSize: 16, breakLine: true, paraSpaceBefore: i ? 18 : 0, paraSpaceAfter: 4 }
      : { color: "E2E8F0", fontSize: 14, bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 8 } }));
    s.addText(para(colA), { x: 0.6, y: 1.75, w: 5.9, h: 4.3, fontFace: BODY, valign: "top", margin: 0, isTextBox: true });
    s.addText(para(colB), { x: 6.85, y: 1.75, w: 5.9, h: 4.3, fontFace: BODY, valign: "top", margin: 0, isTextBox: true });

    s.addImage({ data: Iice.snow, x: 0.6, y: 6.35, w: 0.36, h: 0.36 });
    s.addText("See Beyond the Frame. Secure Beyond the Border.", { x: 1.05, y: 6.3, w: 8, h: 0.45, fontFace: HEAD, fontSize: 17, bold: true, italic: true, color: ICE, margin: 0, isTextBox: true });
    s.addText("Team CRYOBLUE  ·  SIH26153", { x: W - 4.6, y: 6.35, w: 4.0, h: 0.4, fontFace: BODY, fontSize: 12, color: "94A3B8", align: "right", margin: 0, isTextBox: true });
  }

  await pres.writeFile({ fileName: OUT });
  console.log("wrote", OUT);
})();
