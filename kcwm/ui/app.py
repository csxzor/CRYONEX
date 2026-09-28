"""CRYONEX operator console (Streamlit, fully offline).

Run: ``kcwm serve`` (or ``streamlit run kcwm/ui/app.py``).

Layout: a header (file, moment to inspect, alert status, key numbers) above four tabs:

1. **Live Risk Dashboard**: attack probability and alert status over time; ATT&CK stage lanes;
   alert log.
2. **Attack Forecast**: the kill-chain position, current stage and likely next stage, and the
   5-minute infiltration forecast at the chosen moment.
3. **Investigation Context**: the traffic features behind the forecast, when the evidence
   appeared, contributing hosts and flagged flows.
4. **Response Guidance**: MITRE ATT&CK tactic and techniques for the stage to prepare for, and
   D3FEND countermeasures.

Every number on screen comes from ``inference.engine.analyze`` on the uploaded or selected
capture.
"""

from __future__ import annotations

import html
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from kcwm import config
from kcwm.features.registry import BY_NAME
from kcwm.inference.engine import RFC1918, analyze, default_bundle, load_bundle, next_stage_probs
from kcwm.kb.knowledge import stage_card
from kcwm.stages import STAGE_SHORT, STAGES

# --- palette ---------------------------------------------------------------------------------
# Status colours (alert states) are reserved and always paired with an icon + label. Stage
# colours are six categorical hues chosen to stay clear of the status colours (validated with
# the dataviz palette checker; stages also get their own labelled row, so colour is never the
# only cue).
NAVY = "#0B1F33"
INK = "#1E293B"
MUTED = "#64748B"
GRID = "#EEF1F5"
LINE = "#0B1F33"
BLUE = "#2A78D6"
RED_DIV = "#E34948"
WARN = "#FAB219"
CRIT = "#D03B3B"
GOOD = "#0CA30C"
STAGE_COLORS = ["#9AA5B1", "#2A78D6", "#EB6834", "#1BAF7A", "#E87BA4", "#008300", "#4A3AA7"]
ATTACK_STAGES = list(range(1, len(STAGES)))
PROTO = {1: "ICMP", 6: "TCP", 17: "UDP"}
NICE = {"Benign": "No attack activity", "Reconnaissance": "Reconnaissance", "InitialAccess": "Initial Access",
        "LateralMovement": "Lateral Movement", "CommandAndControl": "Command & Control",
        "Exfiltration": "Exfiltration", "Impact": "Impact"}


def nice(s: int) -> str:
    return NICE[STAGES[s]]


def fmt_val(v: float, unit: str) -> str:
    """Human-readable feature values: 22 µs, 439 KB, 21.6 flows."""
    unit = (unit or "").strip()
    a = abs(v)
    if unit == "s":
        return f"{v * 1e6:.0f} µs" if a < 1e-3 else f"{v * 1e3:.0f} ms" if a < 1 else f"{v:.1f} s"
    if unit in ("bytes", "B"):
        return f"{v / 1e6:.1f} MB" if a >= 1e6 else f"{v / 1e3:.0f} KB" if a >= 1e3 else f"{v:.0f} B"
    num = f"{v / 1e6:.1f}M" if a >= 1e6 else f"{v / 1e3:.1f}k" if a >= 1e4 else f"{v:.3g}"
    return f"{num} {unit}".strip()
PLOT_CFG = {"displayModeBar": False}

st.set_page_config(page_title="CRYONEX | Network Attack Forecasting", page_icon="🛡️", layout="wide")

st.markdown(
    f"""
<style>
  [data-testid="stToolbar"], footer, [data-testid="stDecoration"] {{ display: none !important; }}
  .block-container {{ padding-top: 3.2rem; padding-bottom: 2rem; max-width: 1500px; }}
  h1, h2, h3 {{ color: {NAVY}; }}
  .kc-title {{ font-size: 2rem !important; font-weight: 800; color: {NAVY}; margin: 0; line-height: 1.2; }}
  .kc-sub {{ color: {MUTED}; font-size: 0.95rem; margin: 0.15rem 0 0.9rem 0; }}
  .kc-banner {{ border-radius: 12px; padding: 0.85rem 1.1rem; font-size: 1.05rem; color: {INK};
               display: flex; gap: 0.7rem; align-items: center; margin: 0.4rem 0 0.9rem 0; }}
  .kc-banner b {{ font-size: 1.1rem; }}
  .kc-kpi {{ background: #F6F8FB; border-radius: 12px; padding: 0.8rem 1rem; height: 100%; }}
  .kc-kpi .lab {{ color: {MUTED}; font-size: 0.8rem; text-transform: uppercase; letter-spacing: .04em; font-weight: 600; }}
  .kc-kpi .val {{ color: {NAVY}; font-size: 1.75rem; font-weight: 800; line-height: 1.25; }}
  .kc-kpi .note {{ color: {MUTED}; font-size: 0.82rem; }}
  .kc-sec {{ font-size: 1.25rem; font-weight: 700; color: {NAVY}; margin: 0.4rem 0 0.1rem 0; }}
  .kc-cap {{ color: {MUTED}; font-size: 0.9rem; margin-bottom: 0.4rem; }}
  .kc-card {{ background: #F6F8FB; border-radius: 12px; padding: 0.9rem 1.05rem; margin-bottom: 0.8rem; }}
  .kc-card h4 {{ margin: 0 0 0.35rem 0; color: {NAVY}; font-size: 1.02rem; }}
  .kc-card p {{ margin: 0 0 0.4rem 0; color: {INK}; font-size: 0.9rem; }}
  .kc-chip {{ display: inline-block; padding: 0.15rem 0.55rem; border-radius: 999px; font-size: 0.78rem;
              font-weight: 600; margin: 0 0.3rem 0.3rem 0; background: #E2E8F0; color: {INK}; }}
  .kc-chain {{ display: flex; flex-wrap: wrap; align-items: center; gap: 0.35rem; margin: 0.3rem 0 0.9rem 0; }}
  .kc-stage {{ border-radius: 10px; padding: 0.55rem 0.8rem; font-weight: 700; font-size: 0.92rem;
               border: 2px solid transparent; text-align: center; min-width: 7.2rem; }}
  .kc-stage small {{ display: block; font-weight: 600; font-size: 0.7rem; letter-spacing: .04em; text-transform: uppercase; }}
  .kc-arrow {{ color: #94A3B8; font-weight: 700; }}
  .kc-big {{ font-size: 1.05rem; color: {INK}; margin: 0.2rem 0; }}
  .kc-big b {{ color: {NAVY}; }}
  [data-testid="stTabs"] [role="tab"] p {{ font-size: 1.05rem; font-weight: 600; }}
</style>
""",
    unsafe_allow_html=True,
)


# --- data ------------------------------------------------------------------------------------

@st.cache_resource
def get_bundle(path: str):
    return load_bundle(path)


@st.cache_data(show_spinner=False)
def run_analysis(path: str, bundle_path: str, cidrs: tuple[str, ...], mode: str):
    return analyze(path, get_bundle(bundle_path), internal_cidrs=list(cidrs), mode=mode)


@st.cache_data(show_spinner=False)
def hosts_and_flows(path: str, bundle_path: str, cidrs: tuple[str, ...], mode: str, window: int):
    """Host attribution (re-simulates without each busy host's recent flows) + flagged flows."""
    from kcwm.explain.hosts import flagged_flows, host_attribution

    a = run_analysis(path, bundle_path, cidrs, mode)
    ha = host_attribution(a, get_bundle(bundle_path), window)
    top = ha.filter(pl.col("contribution") > 0.01)["host"].to_list()[:3]
    ff = flagged_flows(a, window, top) if top else pl.DataFrame()
    return ha, ff, top


def sidebar():
    st.sidebar.markdown("### 🛡️ CRYONEX")
    st.sidebar.caption("Kill-Chain World Model: forecasts network attacks before they complete. Offline, CPU only.")
    src = st.sidebar.radio("Input", ["Sample capture", "Upload PCAP / CSV"])
    path = None
    if src == "Sample capture":
        samples = sorted(p for p in config.resolve("samples").glob("*")
                         if p.suffix in {".gz", ".csv", ".pcap", ".pcapng", ".binetflow"})
        if samples:
            demo = next((k for k, p in enumerate(samples) if "heartbleed" in p.name), 0)  # the demo capture first
            path = str(st.sidebar.selectbox("Capture", samples, index=demo, format_func=lambda p: p.name))
        else:
            st.sidebar.info("No samples yet: run scripts/make_samples.py")
    else:
        up = st.sidebar.file_uploader("PCAP, PCAPNG, CICFlowMeter CSV or CTU binetflow")
        if up is not None:
            d = Path(tempfile.gettempdir()) / "kcwm_uploads"
            d.mkdir(exist_ok=True)
            path = str(d / up.name)
            Path(path).write_bytes(up.getvalue())
    with st.sidebar.expander("Settings"):
        cidrs = st.text_input("Internal networks (CIDR, comma separated)", ",".join(RFC1918))
        mode = st.selectbox("Normalisation", ["warmup", "global"],
                            help="warmup: re-centre on this capture's first 15 min (label-free adaptation to a new network)")
        bundle_path = st.text_input("Model bundle", default_bundle() or "")
    return bundle_path, path, tuple(c.strip() for c in cidrs.split(",") if c.strip()), mode


# --- helpers ---------------------------------------------------------------------------------

def base_layout(fig: go.Figure, *, height: int, title: str | None = None, ypct: bool = False) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=20, t=78 if title else 36, b=10),
        title=dict(text=title, font=dict(size=15, color=NAVY), x=0, xanchor="left", y=0.985, yanchor="top",
                   yref="container") if title else None,
        plot_bgcolor="white", paper_bgcolor="white", font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=13, color=INK),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=12)),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    fig.update_xaxes(showgrid=False, linecolor="#CBD5E1", ticks="outside", tickcolor="#CBD5E1")
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="#CBD5E1")
    if ypct:
        fig.update_yaxes(tickformat=".0%")
    return fig


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs as (start, end) index pairs, inclusive."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def fmt_minutes(n_windows: int) -> str:
    m = n_windows * 10 / 60
    return f"{m:.0f} min" if m >= 1 else f"{n_windows * 10} s"


def section(title: str, caption: str | None = None):
    st.markdown(f"<div class='kc-sec'>{title}</div>", unsafe_allow_html=True)
    if caption:
        st.markdown(f"<div class='kc-cap'>{caption}</div>", unsafe_allow_html=True)


def kpi(col, label: str, value: str, note: str = ""):
    col.markdown(f"<div class='kc-kpi'><div class='lab'>{label}</div><div class='val'>{value}</div>"
                 f"<div class='note'>{note}</div></div>", unsafe_allow_html=True)


# --- tab 1: live risk ------------------------------------------------------------------------

def risk_timeline(a, sel_time, show_parts: bool) -> go.Figure:
    t = a.windows.to_pandas()
    fig = go.Figure()
    level = t["level"].to_numpy() if "level" in t else np.where(t["alert"], 2, 0)
    times = t["time"]
    step = np.timedelta64(10, "s")
    for lv, col, name in ((1, WARN, "⚠ Early warning"), (2, CRIT, "⛔ Attack in progress")):
        for s, e in runs(level == lv):
            fig.add_vrect(x0=times.iloc[s], x1=times.iloc[e] + step, fillcolor=col, opacity=0.22, line_width=0, layer="below")
        # legend entry for the band
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name,
                                 marker=dict(symbol="square", size=12, color=col, opacity=0.5)))
    fig.add_trace(go.Scatter(x=times, y=t["risk"], name="Attack probability (next 5 min)",
                             line=dict(color=LINE, width=2.5), hovertemplate="%{x|%H:%M:%S}  %{y:.0%}<extra></extra>"))
    if show_parts and "detector_risk" in t and t["detector_risk"].notna().any():
        fig.add_trace(go.Scatter(x=times, y=t["wm_risk"], name="World-model forecast",
                                 line=dict(color=BLUE, width=1.5, dash="dot"), hovertemplate="%{y:.0%}<extra>world model</extra>"))
        fig.add_trace(go.Scatter(x=times, y=t["detector_risk"], name="Detector (gradient boosting)",
                                 line=dict(color="#94A3B8", width=1.5, dash="dot"), hovertemplate="%{y:.0%}<extra>detector</extra>"))
    fig.add_hline(y=a.threshold, line_dash="dash", line_color="#94A3B8", line_width=1.5,
                  annotation_text="alert threshold", annotation_position="top left",
                  annotation_font=dict(color=MUTED, size=11))
    if sel_time is not None:
        fig.add_vline(x=sel_time, line_color=NAVY, line_width=1.5, line_dash="dot")
    base_layout(fig, height=400, ypct=True)
    fig.update_yaxes(range=[0, 1.02], title="attack probability")
    fig.update_xaxes(type="date", range=[times.iloc[0], times.iloc[-1] + step], tickformat="%H:%M")
    return fig


def stage_lanes(a, sel_time) -> go.Figure:
    """One labelled row per ATT&CK stage: the model's current-stage belief (dots) and, when the
    file carries labels, the real attack (grey bars)."""
    t = a.windows.to_pandas()
    fig = go.Figure()
    names = [STAGE_SHORT[STAGES[s]] for s in ATTACK_STAGES]
    if a.has_labels:
        truth = t["stage_now"].to_numpy()
        first = True
        for s in ATTACK_STAGES:
            for i0, i1 in runs(truth == s):
                x1 = t["time"].iloc[i1] + np.timedelta64(10, "s")
                fig.add_trace(go.Scatter(x=[t["time"].iloc[i0], x1], y=[STAGE_SHORT[STAGES[s]]] * 2,
                                         mode="lines", line=dict(color="rgba(15,23,42,0.22)", width=16),
                                         name="Actual attack (dataset labels)", showlegend=first, hoverinfo="skip"))
                first = False
    pred = t[t["nowcast_stage"] > 0]
    for s in ATTACK_STAGES:
        p = pred[pred["nowcast_stage"] == s]
        if len(p):
            fig.add_trace(go.Scatter(x=p["time"], y=[STAGE_SHORT[STAGES[s]]] * len(p), mode="markers",
                                     marker=dict(color=STAGE_COLORS[s], size=8, line=dict(color="white", width=1)),
                                     name=f"Model: {STAGE_SHORT[STAGES[s]]}", showlegend=False,
                                     hovertemplate="%{x|%H:%M:%S}  model says " + STAGE_SHORT[STAGES[s]] + "<extra></extra>"))
    if sel_time is not None:
        fig.add_vline(x=sel_time, line_color=NAVY, line_width=1.5, line_dash="dot")
    base_layout(fig, height=320, title="ATT&CK stage over time: model (dots) vs actual attack (grey bars)")
    fig.update_yaxes(categoryorder="array", categoryarray=names[::-1], showgrid=True, type="category",
                     range=[-0.6, len(names) - 0.4])
    fig.update_xaxes(type="date", range=[t["time"].iloc[0], t["time"].iloc[-1]], tickformat="%H:%M")
    return fig


def alert_log(a) -> pl.DataFrame:
    t = a.windows
    level = t["level"].to_numpy()
    risk = t["risk"].to_numpy()
    times = t["time"].to_list()
    rows = []
    for lv, name in ((2, "⛔ Attack in progress"), (1, "⚠ Early warning")):
        for s, e in runs(level == lv):
            rows.append({"Alert": name, "From": times[s].strftime("%H:%M:%S"), "To": times[e].strftime("%H:%M:%S"),
                         "Duration": fmt_minutes(e - s + 1), "Peak probability": round(100 * float(np.nanmax(risk[s:e + 1])))})
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).sort("From")


def tab_live(a):
    section("Live Risk Dashboard",
            "Probability of an infiltration within the next 5 minutes, recomputed every 10 seconds. "
            "Shaded bands show when the system raised alerts.")
    show = st.toggle("Show the two model components (world-model forecast and detector)", value=False)
    st.plotly_chart(risk_timeline(a, st.session_state.get("sel_time"), show), width="stretch", config=PLOT_CFG)
    st.plotly_chart(stage_lanes(a, st.session_state.get("sel_time")), width="stretch", config=PLOT_CFG)
    log = alert_log(a)
    section("Alert log")
    if log.height:
        st.dataframe(log, width="stretch", hide_index=True, height=min(38 * (log.height + 1) + 2, 320),
                     column_config={"Peak probability": st.column_config.ProgressColumn(
                         "Peak probability", min_value=0, max_value=100, format="%d%%")})
    else:
        st.success("No alerts raised in this capture.")


# --- tab 2: forecast -------------------------------------------------------------------------

def chain_html(prog: int, cur: int, nxt: int, nxt_p: float) -> str:
    parts = []
    for s in ATTACK_STAGES:
        name = STAGE_SHORT[STAGES[s]]
        if s == cur and cur > 0:
            style, tag = f"background:{NAVY};color:white;border-color:{NAVY};", "now"
        elif s == nxt:
            style, tag = f"background:white;color:{BLUE};border:2px dashed {BLUE};", f"likely next · {nxt_p:.0%}"
        elif prog > 0 and s <= prog:
            style, tag = "background:#CBD5E1;color:#0F172A;", "reached"
        else:
            style, tag = "background:#F1F5F9;color:#94A3B8;", "&nbsp;"
        parts.append(f"<div class='kc-stage' style='{style}'>{name}<small>{tag}</small></div>")
    return "<div class='kc-chain'>" + "<span class='kc-arrow'>→</span>".join(parts) + "</div>"


def tab_forecast(a, b, i: int, row: pl.DataFrame):
    now = a.nowcast[i]
    cur = int(np.argmax(now))
    prog = int(row["progress"][0])
    probs = next_stage_probs(a.stage_marg[i:i + 1], a.nowcast[i:i + 1], np.array([prog]), b)[0]
    nxt = int(np.argmax(probs))
    t_str = row["time"][0].strftime("%H:%M:%S")

    section("Attack Forecast", f"What the world model expects next, as of {t_str}.")
    st.markdown(chain_html(prog, cur, nxt, float(probs[nxt])), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"<div class='kc-card'><p class='kc-big'>Current stage: <b>{nice(cur)}</b>"
            f" &nbsp;<span class='kc-chip'>{now[cur]:.0%} confidence</span></p>"
            f"<p class='kc-big'>Most likely next stage: <b>{nice(nxt)}</b> &nbsp;<span class='kc-chip'>{probs[nxt]:.0%}</span></p>"
            f"<p style='color:{MUTED};font-size:0.82rem;margin:0'>Next stage = the world model's 5-minute simulation combined "
            f"with kill-chain transitions learned from training data (right on 50% of real stage changes in testing).</p></div>",
            unsafe_allow_html=True)
        order = [s for s in np.argsort(-probs) if s > 0 and probs[s] > 0][:6][::-1]
        fig = go.Figure(go.Bar(x=[probs[s] for s in order], y=[nice(s) for s in order], orientation="h",
                               marker=dict(color=BLUE, cornerradius=4), text=[f"{probs[s]:.0%}" for s in order],
                               textposition="outside", cliponaxis=False, hovertemplate="%{y}: %{x:.0%}<extra></extra>"))
        base_layout(fig, height=300, title="Which stage comes next?")
        fig.update_xaxes(tickformat=".0%", range=[0, min(1.0, max(probs) * 1.25 + 0.05)], showgrid=True, gridcolor=GRID)
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
    with c2:
        p = a.p_infil[i]
        mins = np.arange(1, len(p) + 1) * 10 / 60
        fig = go.Figure(go.Scatter(x=mins, y=p, mode="lines", line=dict(color=LINE, width=2.5), fill="tozeroy",
                                   fillcolor="rgba(11,31,51,0.08)", hovertemplate="in %{x:.1f} min: %{y:.0%}<extra></extra>"))
        fig.add_annotation(x=mins[-1], y=p[-1], text=f"<b>{p[-1]:.0%}</b> within 5 min", showarrow=False,
                           xanchor="right", yanchor="bottom", font=dict(color=NAVY, size=13))
        base_layout(fig, height=300, title="World-model simulation: chance of an infiltration by each minute ahead", ypct=True)
        fig.update_yaxes(range=[0, 1])
        fig.update_xaxes(title="minutes ahead")
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
        st.caption("This is the world model's raw 5-minute simulation. The headline attack probability combines it "
                   "with a second detector and is calibrated, so the two numbers differ.")
        with st.expander("How the model reasons about stage order (learned kill-chain transitions)"):
            tm = b.model.transition_matrix()
            lab = [STAGE_SHORT[s] for s in STAGES]
            hm = go.Figure(go.Heatmap(z=tm, x=lab, y=lab, colorscale=[[0, "#F1F5F9"], [1, "#184F95"]],
                                      hovertemplate="from %{y} to %{x}: %{z:.2f}<extra></extra>", showscale=False))
            base_layout(hm, height=300)
            hm.update_yaxes(title="from", autorange="reversed")
            hm.update_xaxes(title="to")
            st.plotly_chart(hm, width="stretch", config=PLOT_CFG)


# --- tab 3: investigation --------------------------------------------------------------------

def tab_investigate(a, b, i: int, w: int, key: tuple):
    from kcwm.data.sequences import context_index, model_input
    from kcwm.explain.attributions import attention_rollout, integrated_gradients
    from kcwm.explain.counterfactual import group_counterfactual

    section("Investigation Context", "Why the model raised this forecast: the traffic behind it, when it appeared, and who caused it.")
    x = model_input(a.arrays, context_index(np.array([i]), b.context))[0]
    names = list(b.scaler.names)
    att = integrated_gradients(b.model, x, horizon=b.horizon, n_features=len(names), names=names)
    native = b.scaler.inverse(a.arrays.x[i][None])[0]
    ranked = att.top_features(len(names))
    raises = [(n, v) for n, v in ranked if v > 0][:6]
    lowers = [(n, v) for n, v in ranked if v < 0][:2]

    def short(n: str) -> str:
        d = BY_NAME[n].desc
        return d if len(d) <= 62 else d[:60].rsplit(" ", 1)[0] + "…"

    if raises:
        order = raises[::-1]  # largest at the top
        labels = [f"{short(n)}  <b>{fmt_val(native[names.index(n)], BY_NAME[n].unit)}</b>" for n, _ in order]
        vals = [v for _, v in order]
        fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker=dict(color=RED_DIV, cornerradius=4),
                               text=[f"+{100 * v:.0f} pts" for v in vals], textposition="outside", cliponaxis=False,
                               textfont=dict(color=INK, size=13),
                               hovertemplate="%{y}<br>raises the forecast by %{x:.1%}<extra></extra>"))
        base_layout(fig, height=360, title="What is pushing the attack risk up")
        fig.update_xaxes(title="push on the world-model forecast (percentage points)", tickformat=".0%",
                         range=[0, max(vals) * 1.18], showgrid=True, gridcolor=GRID)
        fig.update_yaxes(showgrid=False, tickfont=dict(size=13))
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
    if lowers:
        txt = "; ".join(f"{html.escape(short(n))} ({fmt_val(native[names.index(n)], BY_NAME[n].unit)}, "
                        f"{100 * v:.0f} pts)" for n, v in lowers)
        st.markdown(f"<div class='kc-cap'>Evidence pointing the other way: {txt}.</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        attn = attention_rollout(b.model, x)
        fig = go.Figure(go.Bar(x=np.arange(-len(attn) + 1, 1) * 10 / 60, y=attn, marker=dict(color=BLUE, cornerradius=2),
                               hovertemplate="%{x:.1f} min: %{y:.3f}<extra></extra>"))
        base_layout(fig, height=300, title="When the evidence appeared (model attention)")
        fig.update_xaxes(title="minutes before this moment")
        fig.update_yaxes(title="attention", showticklabels=False)
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
    cf = group_counterfactual(b.model, x, horizon=b.horizon, n_features=len(names), threshold=b.threshold, names=names)
    with c2:
        st.write("")
        if cf["steps"]:
            items = "".join(f"<li>without <b>{html.escape(s['group'])}</b> behaviour: <b>{s['risk_after']:.0%}</b></li>"
                            for s in cf["steps"])
            st.markdown(f"<div class='kc-card'><h4>What if? Remove one kind of behaviour</h4>"
                        f"<p>World-model forecast now <b>{cf['risk']:.0%}</b>. Removing behaviour step by step:</p>"
                        f"<ol style='margin:0'>{items}</ol></div>", unsafe_allow_html=True)

    with st.spinner("Re-simulating without each host to find who is responsible..."):
        ha, ff, top_hosts = hosts_and_flows(*key, w)
    c3, c4 = st.columns([1, 1.6])
    with c3:
        section("Contributing hosts", "Drop in attack probability when each host's recent traffic is removed.")
        if ha.height:
            show = ha.head(6).select(
                pl.col("host").alias("Host"),
                (100 * pl.col("contribution").clip(lower_bound=0)).round(0).alias("Contribution"),
                (100 * pl.col("risk_without")).round(0).alias("Risk without host"))
            st.dataframe(show, width="stretch", hide_index=True, column_config={
                "Contribution": st.column_config.ProgressColumn(
                    "Contribution", min_value=0, max_value=max(5.0, float(show["Contribution"].max())), format="%d pts"),
                "Risk without host": st.column_config.NumberColumn(format="%d%%")})
        else:
            st.info("No busy hosts in the last 2 minutes.")
    with c4:
        section("Flagged flows", "Recent connections of the top contributing hosts: " + (", ".join(top_hosts) or "none") + ".")
        if ff.height:
            view = ff.head(60).select(
                pl.col("time").dt.strftime("%H:%M:%S").alias("Time"),
                pl.concat_str([pl.col("src_ip"), pl.lit(":"), pl.col("src_port").cast(pl.Utf8)]).alias("Source"),
                pl.concat_str([pl.col("dst_ip"), pl.lit(":"), pl.col("dst_port").cast(pl.Utf8)]).alias("Destination"),
                pl.col("protocol").replace_strict(PROTO, default=None, return_dtype=pl.Utf8).fill_null(pl.col("protocol").cast(pl.Utf8)).alias("Proto"),
                (pl.col("fwd_packets") + pl.col("bwd_packets")).cast(pl.Int64).alias("Packets"),
                (pl.col("fwd_bytes") + pl.col("bwd_bytes")).cast(pl.Int64).alias("Bytes"),
                pl.concat_str([pl.lit("SYN "), pl.col("syn_count").cast(pl.Int64).cast(pl.Utf8),
                               pl.lit(" · RST "), pl.col("rst_count").cast(pl.Int64).cast(pl.Utf8)]).alias("Flags"),
                *([pl.col("stage").alias("Label")] if "stage" in ff.columns else []))
            st.dataframe(view, width="stretch", hide_index=True, height=300)
        else:
            st.info("No host contributed noticeably to this forecast.")


# --- tab 4: response -------------------------------------------------------------------------

def tab_respond(a, b, i: int, row: pl.DataFrame):
    now = a.nowcast[i]
    cur = int(np.argmax(now))
    prog = int(row["progress"][0])
    probs = next_stage_probs(a.stage_marg[i:i + 1], a.nowcast[i:i + 1], np.array([prog]), b)[0]
    nxt = int(np.argmax(probs))
    target = nxt if nxt > 0 else max(cur, 1)
    card = stage_card(STAGES[target])
    section("Response Guidance",
            f"Prepare for <b>{nice(target)}</b>: the most likely next stage ({probs[target]:.0%})"
            + (f", after the current <b>{nice(cur)}</b> activity." if cur else "."))
    desc = card.get("description") or ""
    st.markdown(f"<div class='kc-card'><h4>MITRE ATT&CK tactic {html.escape(card['tactic_id'])}: {html.escape(card['tactic_name'])}</h4>"
                + (f"<p>{html.escape(desc[:400])}</p>" if desc else "") + "</div>", unsafe_allow_html=True)

    # recommended first actions: countermeasures shared by the most techniques first
    cms = Counter()
    meta = {}
    for tch in card["techniques"]:
        for cm in tch["countermeasures"]:
            cms[cm["id"]] += 1
            meta[cm["id"]] = cm
    if cms:
        items = "".join(f"<li><b>{html.escape(meta[k]['name'])}</b> <span class='kc-chip'>{html.escape(meta[k]['tactic'])}</span>"
                        f"<span style='color:{MUTED};font-size:0.8rem'>{html.escape(k)}</span></li>"
                        for k, _ in cms.most_common(5))
        st.markdown(f"<div class='kc-card'><h4>✅ Recommended first actions (MITRE D3FEND)</h4><ol style='margin:0'>{items}</ol></div>",
                    unsafe_allow_html=True)

    section("Techniques to watch for")
    cols = st.columns(min(3, max(1, len(card["techniques"]))))
    for k, tch in enumerate(card["techniques"]):
        cm = "".join(f"<span class='kc-chip'>{html.escape(c['tactic'])}: {html.escape(c['name'])}"
                     f"{' *' if c['curated'] else ''}</span>" for c in tch["countermeasures"])
        d = html.escape((tch.get("description") or "")[:220].rsplit(" ", 1)[0]) + "…"
        cols[k % len(cols)].markdown(
            f"<div class='kc-card'><h4>{html.escape(tch['id'])} · {html.escape(tch['name'])}</h4><p>{d}</p>"
            f"<p style='margin:0.3rem 0 0.2rem 0;font-weight:600;color:{NAVY};font-size:0.85rem'>D3FEND countermeasures</p>{cm}</div>",
            unsafe_allow_html=True)
    st.caption("Sources: MITRE ATT&CK Enterprise and MITRE D3FEND, local offline copies. * = mapping curated by the team.")


# --- main ------------------------------------------------------------------------------------

def main():
    bundle_path, path, cidrs, mode = sidebar()
    if not bundle_path or not Path(bundle_path).exists():
        st.warning("No model bundle found. Train one (`kcwm evaluate --models kcwm`) or set KCWM_BUNDLE.")
        return
    if not path:
        st.markdown("<div class='kc-title'>🛡️ CRYONEX · Network Attack Forecasting</div>", unsafe_allow_html=True)
        st.info("Choose a sample capture or upload a PCAP/CSV in the sidebar.")
        return
    with st.spinner("Parsing traffic, building network states, simulating futures..."):
        a = run_analysis(path, bundle_path, cidrs, mode)
    b = get_bundle(bundle_path)
    t = a.windows
    scored = t.filter(pl.col("risk").is_not_nan())
    span = f"{t['time'].min():%d %b %Y, %H:%M} – {t['time'].max():%H:%M}" if t.height else ""
    st.markdown("<div class='kc-title'>🛡️ CRYONEX · Network Attack Forecasting</div>"
                f"<div class='kc-sub'>{html.escape(Path(path).name)} &nbsp;·&nbsp; {a.fmt.upper()} &nbsp;·&nbsp; "
                f"{a.flows.height:,} flows &nbsp;·&nbsp; {span}</div>", unsafe_allow_html=True)
    if scored.height == 0:
        st.warning(f"Capture too short: the model needs {b.context} windows ({b.context * 10 // 60} min) of history.")
        return

    # moment to inspect (defaults to the riskiest moment)
    opts = scored["window"].to_list()
    tmap = dict(zip(scored["window"].to_list(), scored["time"].to_list()))
    default = int(scored.sort("risk", descending=True)["window"][0])
    want = st.query_params.get("t")  # ?t=HH:MM:SS opens at that moment (bookmarkable, for screenshots)
    if want:
        hit = [wid for wid, tm in tmap.items() if f"{tm:%H:%M:%S}" >= want]
        if hit:
            default = int(hit[0])
    w = st.select_slider("Moment to inspect", options=opts, value=default, format_func=lambda x: f"{tmap[x]:%H:%M:%S}")
    row = t.filter(pl.col("window") == w)
    i = int(np.flatnonzero(t["window"].to_numpy() == w)[0])
    st.session_state["sel_time"] = row["time"][0]
    lvl = int(row["level"][0]) if "level" in row.columns else (2 if bool(row["alert"][0]) else 0)
    risk = float(row["risk"][0])
    when = f"{row['time'][0]:%H:%M:%S}"
    if lvl == 2:
        bg, icon, head, text = "#FDECEC", "⛔", "ATTACK IN PROGRESS", f"at {when}: attack probability {risk:.0%}, above the alert threshold."
    elif lvl == 1:
        bg, icon, head, text = "#FFF6DD", "⚠️", "EARLY WARNING", f"at {when}: the world model forecasts a compromise within 5 minutes ({risk:.0%})."
    else:
        bg, icon, head, text = "#E9F7E9", "✅", "NO ALERT", f"at {when}: attack probability {risk:.0%}."
    st.markdown(f"<div class='kc-banner' style='background:{bg}'><span style='font-size:1.4rem'>{icon}</span>"
                f"<span><b>{head}</b> {text}</span></div>", unsafe_allow_html=True)

    level = t["level"].to_numpy() if "level" in t.columns else np.where(t["alert"].to_numpy(), 2, 0)
    peak_row = scored.sort("risk", descending=True).row(0, named=True)
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "Attack probability now", f"{risk:.0%}", f"at {when}, next 5 minutes")
    kpi(k2, "Peak attack probability", f"{peak_row['risk']:.0%}", f"at {peak_row['time']:%H:%M:%S}")
    kpi(k3, "⚠ Early warning time", fmt_minutes(int((level == 1).sum())), f"{len(runs(level == 1))} warning periods")
    kpi(k4, "⛔ Attack in progress", fmt_minutes(int((level == 2).sum())), f"{len(runs(level == 2))} alert periods")
    st.write("")

    t1, t2, t3, t4 = st.tabs(["📈 Live Risk Dashboard", "🔮 Attack Forecast", "🔍 Investigation Context", "🛡️ Response Guidance"])
    with t1:
        tab_live(a)
    with t2:
        tab_forecast(a, b, i, row)
    with t3:
        tab_investigate(a, b, i, w, (path, bundle_path, cidrs, mode))
    with t4:
        tab_respond(a, b, i, row)


main()
