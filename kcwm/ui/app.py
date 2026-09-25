"""KC-WM operator console (Streamlit, fully offline).

Run: ``kcwm serve`` (or ``streamlit run kcwm/ui/app.py``).

Panels: input -> risk timeline -> kill chain -> why -> who -> respond. Every number on
screen comes from ``inference.engine.analyze`` on the uploaded or selected capture.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from kcwm import config
from kcwm.features.registry import BY_NAME, FEATURE_NAMES, N_FEATURES
from kcwm.inference.engine import RFC1918, analyze, default_bundle, load_bundle
from kcwm.kb.knowledge import stage_card
from kcwm.stages import STAGE_SHORT, STAGES

STAGE_COLORS = ["#9aa5b1", "#5b8def", "#f2a33a", "#d9534f", "#8e44ad", "#c0392b", "#2c3e50"]

st.set_page_config(page_title="KC-WM | Attack Forecasting", layout="wide")


@st.cache_resource
def get_bundle(path: str):
    return load_bundle(path)


@st.cache_data(show_spinner=False)
def run_analysis(path: str, bundle_path: str, cidrs: tuple[str, ...], mode: str):
    b = get_bundle(bundle_path)
    a = analyze(path, b, internal_cidrs=list(cidrs), mode=mode)
    return a


def sidebar():
    st.sidebar.title("KC-WM")
    st.sidebar.caption("Kill-Chain World Model: forecasts infiltration before it completes. Offline, CPU.")
    bundle_path = st.sidebar.text_input("Model bundle", default_bundle() or "")
    src = st.sidebar.radio("Input", ["Sample capture", "Upload PCAP / CSV"])
    path = None
    if src == "Sample capture":
        samples = sorted(p for p in config.resolve("samples").glob("*") if p.suffix in {".gz", ".csv", ".pcap", ".pcapng", ".binetflow"})
        if samples:
            path = str(st.sidebar.selectbox("Capture", samples, format_func=lambda p: p.name))
        else:
            st.sidebar.info("No samples yet: run scripts/make_samples.py")
    else:
        up = st.sidebar.file_uploader("PCAP, PCAPNG, CICFlowMeter CSV or CTU binetflow")
        if up is not None:
            d = Path(tempfile.gettempdir()) / "kcwm_uploads"
            d.mkdir(exist_ok=True)
            path = str(d / up.name)
            Path(path).write_bytes(up.getvalue())
    cidrs = st.sidebar.text_input("Internal networks (CIDR, comma separated)", ",".join(RFC1918))
    mode = st.sidebar.selectbox("Normalisation", ["warmup", "global"],
                                help="warmup: re-centre on this capture's first 15 min (label-free adaptation to a new network)")
    return bundle_path, path, tuple(c.strip() for c in cidrs.split(",") if c.strip()), mode


def timeline_chart(a, sel_time):
    t = a.windows.to_pandas()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t["time"], y=t["risk"], name=f"alert score: P(infiltration within {a.horizon * 10 // 60} min)",
                             line=dict(color="#d9534f", width=2)))
    if "detector_risk" in t and t["detector_risk"].notna().any():
        fig.add_trace(go.Scatter(x=t["time"], y=t["wm_risk"], name="world-model forecast",
                                 line=dict(color="#5b8def", width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=t["time"], y=t["detector_risk"], name="detector (gradient boosting)",
                                 line=dict(color="#9aa5b1", width=1, dash="dot")))
    fig.add_hline(y=a.threshold, line_dash="dot", line_color="#555", annotation_text="alert threshold")
    al = t[t["alert"]]
    fig.add_trace(go.Scatter(x=al["time"], y=al["risk"], mode="markers", name="alert",
                             marker=dict(color="#d9534f", size=5, symbol="triangle-up")))
    if a.has_labels:
        lab = t[t["stage_now"] > 0]
        fig.add_trace(go.Scatter(x=lab["time"], y=[-0.06] * len(lab), mode="markers", name="ground truth (labels)",
                                 marker=dict(color=[STAGE_COLORS[s] for s in lab["stage_now"]], size=6, symbol="square"),
                                 text=[STAGES[s] for s in lab["stage_now"]], hoverinfo="text+x"))
    pred = t[t["nowcast_stage"] > 0]
    fig.add_trace(go.Scatter(x=pred["time"], y=[-0.12] * len(pred), mode="markers", name="model: current stage",
                             marker=dict(color=[STAGE_COLORS[int(s)] for s in pred["nowcast_stage"]], size=6),
                             text=[STAGES[int(s)] for s in pred["nowcast_stage"]], hoverinfo="text+x"))
    if sel_time is not None:
        fig.add_vline(x=sel_time, line_color="#333", line_width=1)
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), yaxis=dict(range=[-0.18, 1.02], title="risk"),
                      legend=dict(orientation="h", y=1.12))
    return fig


def main():
    bundle_path, path, cidrs, mode = sidebar()
    st.title("Network attack forecasting: Kill-Chain World Model")
    if not bundle_path or not Path(bundle_path).exists():
        st.warning("No model bundle found. Train one (`kcwm evaluate --models kcwm`) or set KCWM_BUNDLE.")
        return
    if not path:
        st.info("Choose a sample capture or upload a PCAP/CSV in the sidebar.")
        return
    with st.spinner("Parsing traffic, building network states, simulating futures..."):
        a = run_analysis(path, bundle_path, cidrs, mode)
    b = get_bundle(bundle_path)
    t = a.windows
    scored = t.filter(pl.col("risk").is_not_nan())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Flows", f"{a.flows.height:,}")
    c2.metric("Windows (10 s)", f"{t.height:,}")
    c3.metric("Peak risk", f"{np.nanmax(t['risk'].to_numpy()):.0%}" if scored.height else "n/a")
    c4.metric("Alert windows", int(t["alert"].sum()))
    if scored.height == 0:
        st.warning(f"Capture too short: the model needs {b.context} windows ({b.context * 10 // 60} min) of history.")
        return

    idx_opts = scored["window"].to_list()
    default = int(scored.sort("risk", descending=True)["window"][0])
    w = st.select_slider("Inspect window", options=idx_opts, value=default,
                         format_func=lambda x: str(t.filter(pl.col("window") == x)["time"][0]))
    row = t.filter(pl.col("window") == w)
    i = int(np.flatnonzero(t["window"].to_numpy() == w)[0])
    st.plotly_chart(timeline_chart(a, row["time"][0]), width="stretch")

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Forecast at this moment")
        p = a.p_infil[i]
        mins = np.arange(1, len(p) + 1) * 10 / 60
        fig = go.Figure(go.Scatter(x=mins, y=p, fill="tozeroy", line=dict(color="#d9534f")))
        fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="minutes ahead",
                          yaxis=dict(range=[0, 1], title="P(infiltration by then)"))
        st.plotly_chart(fig, width="stretch")
        sm = a.stage_marg[i]
        fig2 = go.Figure()
        for s in range(1, len(STAGES)):
            fig2.add_trace(go.Scatter(x=mins, y=sm[:, s], stackgroup="one", name=STAGE_SHORT[STAGES[s]],
                                      line=dict(color=STAGE_COLORS[s])))
        fig2.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="minutes ahead",
                           yaxis_title="stage probability", legend=dict(orientation="h"))
        st.plotly_chart(fig2, width="stretch")
    with right:
        st.subheader("Kill chain")
        now = a.nowcast[i]
        prog = int(row["progress"][0])
        chips = []
        for s in range(1, len(STAGES)):
            state = "reached" if s <= prog and prog > 0 else ""
            chips.append(f"<span style='padding:3px 8px;margin:2px;border-radius:10px;"
                         f"background:{STAGE_COLORS[s] if state else '#e6e8eb'};color:{'white' if state else '#333'}'>"
                         f"{STAGE_SHORT[STAGES[s]]}</span>")
        st.markdown(" → ".join(chips), unsafe_allow_html=True)
        cur = int(np.argmax(now))
        st.write(f"**Now:** {STAGES[cur]} ({now[cur]:.0%}).  **Heading to:** {STAGES[int(row['heading_to'][0])]}.")
        st.caption("Transition matrix (context-free kill-chain prior as learned)")
        tm = b.model.transition_matrix()
        st.plotly_chart(go.Figure(go.Heatmap(z=tm, x=[STAGE_SHORT[s] for s in STAGES], y=[STAGE_SHORT[s] for s in STAGES],
                                             colorscale="Reds")).update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10)),
                        width="stretch")

    st.subheader("Why: driving features")
    from kcwm.data.sequences import context_index, model_input
    from kcwm.explain.attributions import attention_rollout, integrated_gradients
    from kcwm.explain.counterfactual import group_counterfactual

    x = model_input(a.arrays, context_index(np.array([i]), b.context))[0]
    att = integrated_gradients(b.model, x, horizon=b.horizon, n_features=N_FEATURES)
    cur_raw = a.windows  # native units from the scaler inverse
    native = b.scaler.inverse(a.arrays.x[i][None])[0]
    rows = []
    for name, v in att.top_features(8):
        j = FEATURE_NAMES.index(name)
        rows.append({"feature": BY_NAME[name].desc, "group": BY_NAME[name].group,
                     "value now": f"{native[j]:.3g} {BY_NAME[name].unit}", "push on risk": round(v, 3)})
    cA, cB = st.columns([1.3, 1])
    cA.dataframe(pl.DataFrame(rows), width="stretch", hide_index=True)
    attn = attention_rollout(b.model, x)
    cB.plotly_chart(go.Figure(go.Bar(x=np.arange(-len(attn) + 1, 1) * 10 / 60, y=attn, marker_color="#5b8def"))
                    .update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10), title="Which past windows mattered (attention)",
                                   xaxis_title="minutes before now"), width="stretch")
    cf = group_counterfactual(b.model, x, horizon=b.horizon, n_features=N_FEATURES, threshold=b.threshold)
    if cf["steps"]:
        txt = "; ".join(f"without **{s['group']}** behaviour → {s['risk_after']:.0%}" for s in cf["steps"])
        st.write(f"Counterfactual: risk {cf['risk']:.0%}; {txt}.")
    del cur_raw

    st.subheader("Who: hosts and flows")
    if st.button("Attribute to hosts (re-simulates without each host's recent flows)"):
        from kcwm.explain.hosts import flagged_flows, host_attribution

        with st.spinner("Re-simulating..."):
            ha = host_attribution(a, b, w)
        st.dataframe(ha, width="stretch", hide_index=True)
        top = ha.filter(pl.col("contribution") > 0.01)["host"].to_list()[:3]
        if top:
            st.dataframe(flagged_flows(a, w, top), width="stretch", hide_index=True)

    st.subheader("Respond: ATT&CK and D3FEND")
    target = int(row["heading_to"][0]) if int(row["heading_to"][0]) > 0 else max(cur, 1)
    card = stage_card(STAGES[target])
    st.write(f"**{card['tactic_id']} {card['tactic_name']}**: {card['description']}")
    for tch in card["techniques"]:
        with st.expander(f"{tch['id']} {tch['name']}"):
            st.write(tch.get("description", "")[:400])
            for cm in tch["countermeasures"]:
                tag = " (curated)" if cm["curated"] else ""
                st.write(f"- **{cm['tactic']}**: {cm['id']} {cm['name']}{tag}")


main()
