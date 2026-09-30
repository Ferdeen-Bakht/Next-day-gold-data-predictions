# -*- coding: utf-8 -*-
"""
Streamlit dashboard for the XAU/USD next-day forecast.

Run:  streamlit run app.py
(Requires outputs/results.json produced by `python train_predict.py`.)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = BASE_DIR / "outputs" / "results.json"
BACKTEST_PATH = BASE_DIR / "outputs" / "backtest_predictions.csv"
OHLC_PATH = BASE_DIR / "outputs" / "backtest_ohlc.csv"
PRICE_TYPES = ["Open", "High", "Low", "Close"]

st.set_page_config(page_title="XAU/USD Next-Day Forecast", page_icon="🥇", layout="wide")


@st.cache_data
def load_results(mtime):
    with open(RESULTS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_ohlc(mtime):
    return pd.read_csv(OHLC_PATH, parse_dates=["Date", "Target_Date"])


@st.cache_data
def load_backtest(mtime):
    return pd.read_csv(BACKTEST_PATH, parse_dates=["Date", "Target_Date"])


if not RESULTS_PATH.exists():
    st.error("No results found. Run `python train_predict.py` first.")
    st.stop()

R = load_results(RESULTS_PATH.stat().st_mtime)
bt = load_backtest(BACKTEST_PATH.stat().st_mtime) if BACKTEST_PATH.exists() else None

target_date = pd.Timestamp(R["target_date"])
last_bar = R["last_bar"]
last_close = last_bar["Close"]
preds = R["predictions"]
metrics = R["backtest"]["metrics"]
model_names = [m for m in preds if not m.startswith("Ensemble")]
ensemble_names = [m for m in preds if m.startswith("Ensemble")]

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.title(f"XAU/USD forecast for {target_date:%a %d %b %Y}")
trained_through = R.get("models_trained_through", R["last_date"])
st.caption(
    f"Models trained on {R['n_training_rows_full']:,} labelled daily bars ({R['first_date']} → {trained_through})"
    + (f" and applied to data through **{R['last_date']}** without retraining" if trained_through != R["last_date"]
       else "")
    + f" · {R['n_features']} features · last close **${last_close:,.2f}** on {R['last_date']} · "
    f"generated {R['generated_at']}"
)

with st.sidebar:
    st.header("Settings")
    headline_options = ensemble_names + model_names
    default_headline = "Ensemble (stacked)" if "Ensemble (stacked)" in headline_options else headline_options[0]
    headline = st.selectbox("Headline forecast", headline_options, index=headline_options.index(default_headline))
    selected = st.multiselect("Models on chart", model_names + ensemble_names,
                              default=[headline])
    history_days = st.slider("Chart history (trading days)", 20, len(R["recent_ohlc"]["Date"]), 60)
    st.divider()
    if R.get("ensemble_members"):
        sel = R.get("ensemble_selection", {})
        st.markdown(f"**Ensemble members** (top {sel.get('top_k', len(R['ensemble_members']))} by validation RMSE)")
        st.dataframe(pd.Series({m: sel.get("val_avg_rmse", {}).get(m) for m in R["ensemble_members"]},
                               name="Validation avg RMSE $").map("{:,.2f}".format), width="stretch")
        b_ = R["backtest"]
        st.caption(f"Chosen on the validation period ({b_.get('val_start', '?')} → {b_.get('val_end', '?')}), "
                   "not the test period, so the ensembles' test scores stay fair.")
    if R.get("stacking_weights"):
        st.markdown("**Stacked ensemble weights**")
        sw = pd.DataFrame(R["stacking_weights"])
        sw = sw[(sw > 0.005).any(axis=1)]
        st.dataframe(sw.map("{:.0%}".format), width="stretch")
        st.caption("Fitted on the validation period, separately for each price. "
                   "'Naive' = tomorrow equals today's close. Models with ~0% weight are hidden.")
    st.markdown("**RMSE-weighted ensemble weights** (inverse validation RMSE)")
    st.dataframe(pd.Series(R["ensemble_weights"], name="Weight").map("{:.1%}".format),
                 width="stretch")
    st.caption("⚠️ Educational model output — not financial advice.")

# ------------------------------------------------------------
# KPI row
# ------------------------------------------------------------
hp = preds[headline]
st.subheader(f"{headline} — predicted bar")
cols = st.columns(4, border=True)
for col, p in zip(cols, PRICE_TYPES):
    col.metric(f"Predicted {p}", f"${hp[p]:,.2f}", f"{(hp[p] / last_close - 1) * 100:+.2f}% vs last close")
cols = st.columns(4, border=True)
direction = "▲ Up" if hp["Close"] > last_close else "▼ Down"
cols[0].metric("Direction", direction, f"{hp['Close'] - last_close:+,.2f} $")
cols[1].metric("Expected range (H−L)", f"${hp['High'] - hp['Low']:,.2f}",
               f"20d avg ${R['market_stats']['Avg_Daily_Range_20d']:,.2f}", delta_color="off")
cols[2].metric(f"Last bar {R['last_date']}", f"${last_close:,.2f}",
               f"O {last_bar['Open']:,.0f} · H {last_bar['High']:,.0f} · L {last_bar['Low']:,.0f}",
               delta_color="off")
cols[3].metric("RSI (14)", f"{R['market_stats']['RSI_14']:.1f}",
               f"ATR ${R['market_stats']['ATR_14']:,.2f}", delta_color="off")

bullish = sum(preds[m]["Close"] > last_close for m in model_names)
st.info(f"**Model consensus:** {bullish} of {len(model_names)} models expect a higher close "
        f"than \\${last_close:,.2f}. Close forecasts range "
        f"\\${min(preds[m]['Close'] for m in model_names):,.2f} – "
        f"\\${max(preds[m]['Close'] for m in model_names):,.2f}.")

REGIME_COLORS = {"Bearish": "#ef5350", "Sideways": "#9e9e9e", "Bullish": "#26a69a"}
reg, cls = R.get("regimes"), R.get("classification") or {}
hc = cls.get(headline, {})
if reg or hc.get("p_up_next_day") is not None:
    c1, c2 = st.columns(2, border=True)
    if reg:
        cur = reg["current"]
        top = max(cur, key=cur.get)
        c1.metric(f"Market regime on {R['last_date']}", top,
                  " · ".join(f"{k} {v:.0%}" for k, v in cur.items()), delta_color="off")
    if hc.get("p_up_next_day") is not None:
        pu = hc["p_up_next_day"]
        c2.metric(f"{headline}: Up/Down call for {target_date:%d %b}",
                  f"{'▲ UP' if pu > 0.5 else '▼ DOWN'}  ({pu:.1%} up)",
                  f"classifier test accuracy {hc['accuracy']:.1f}% vs always-up "
                  f"{cls.get('Always up (baseline)', {}).get('accuracy', float('nan')):.1f}%", delta_color="off")

tab_fc, tab_pa, tab_lb, tab_ud, tab_bt, tab_mkt, tab_train = st.tabs(
    ["📈 Forecast", "📉 Predicted vs Actual", "🏆 Leaderboard", "🎯 Up/Down", "🧪 Backtest", "📊 Market stats",
     "⚙️ Training"])

# ------------------------------------------------------------
# Predicted vs Actual tab: hold-out predictions against real prices
# ------------------------------------------------------------
with tab_pa:
    live = pd.DataFrame(R.get("live_track") or [])
    if not live.empty:
        st.markdown("**Live forecasts vs what actually happened** (forecasts made after training, "
                    "then checked against the real prices)")
        dates = sorted(live["target_date"].unique(), reverse=True)
        for d in dates:
            day = live[live["target_date"] == d].copy()
            made_on, prev_close = day["made_on"].iloc[0], day["prev_close"].iloc[0]
            a = day.iloc[0]
            moved = a["actual_Close"] - prev_close
            st.markdown(f"Forecast made on **{made_on}** for **{pd.Timestamp(d):%a %d %b %Y}** · actual: "
                        f"O {a['actual_Open']:,.2f} · H {a['actual_High']:,.2f} · L {a['actual_Low']:,.2f} · "
                        f"C {a['actual_Close']:,.2f} ({'▲' if moved > 0 else '▼'} {moved:+,.2f} vs "
                        f"{prev_close:,.2f})")
            day["Up/Down call"] = day.apply(
                lambda r: "—" if pd.isna(r["p_up"]) else
                f"{'▲' if r['p_up'] > 0.5 else '▼'} {r['p_up']:.0%} {'✅' if r['updown_call_right'] else '❌'}", axis=1)
            view = day.set_index("model")[["pred_Open", "pred_High", "pred_Low", "pred_Close", "close_error",
                                           "avg_abs_error", "Up/Down call"]].rename(columns={
                "pred_Open": "Pred Open", "pred_High": "Pred High", "pred_Low": "Pred Low",
                "pred_Close": "Pred Close", "close_error": "Close error $", "avg_abs_error": "Avg O/H/L/C miss $"})
            view = view.sort_values("Avg O/H/L/C miss $")
            st.dataframe(view.style.format({c: "{:,.2f}" for c in view.columns if c != "Up/Down call"})
                         .background_gradient(subset=["Avg O/H/L/C miss $"], cmap="RdYlGn_r"),
                         width="stretch", height=38 * (len(view) + 1) + 3)
        right = live["updown_call_right"].dropna()
        st.caption(f"{len(dates)} forecast day(s) checked so far. Up/Down calls right: {int(right.sum())} of "
                   f"{len(right)} model-days. One day says very little — the hold-out charts below cover "
                   f"{R['backtest']['test_rows']} days.")
        st.divider()

    if not OHLC_PATH.exists():
        st.info("No hold-out predictions found. Run `python train_predict.py --export-backtest` (no training needed).")
    else:
        bo = load_ohlc(OHLC_PATH.stat().st_mtime)
        avail = [c[:-len(" | Close")] for c in bo.columns if c.endswith(" | Close")]
        best_single = min((m for m in avail if m in metrics and not m.startswith("Ensemble")),
                          key=lambda m: metrics[m]["AvgRMSE"], default=avail[0])
        defaults = list(dict.fromkeys([m for m in (headline, best_single) if m in avail]))
        c1, c2, c3 = st.columns([3, 1, 2])
        pa_models = c1.multiselect("Models", avail, default=defaults, key="pa_models")
        price = c2.selectbox("Price", PRICE_TYPES, index=3, key="pa_price")
        days = c3.slider("Days shown (most recent)", 20, len(bo), min(90, len(bo)), key="pa_days")
        sub = bo.tail(days)
        actual = sub[f"Actual {price}"]
        palette = ["#f9a825", "#29b6f6", "#26c6da", "#ff7043", "#8d6e63", "#ec407a", "#5c6bc0", "#d4e157"]
        actual_color = "#ab47bc"  # purple: visible on both light and dark themes

        # ---- 1. predicted vs actual line chart + tomorrow's forecast
        fig = go.Figure(go.Scatter(x=sub["Target_Date"], y=actual, name=f"Actual {price}", mode="lines+markers",
                                   line=dict(color=actual_color, width=3), marker=dict(size=4)))
        for i, m in enumerate(pa_models):
            color = palette[i % len(palette)]
            fig.add_trace(go.Scatter(x=sub["Target_Date"], y=sub[f"{m} | {price}"], name=f"{m} (predicted)",
                                     line=dict(color=color, width=1.8)))
            if m in preds:  # the forecast for the next, still unknown, day
                fig.add_trace(go.Scatter(
                    x=[sub["Target_Date"].iloc[-1], target_date], y=[actual.iloc[-1], preds[m][price]],
                    mode="lines+markers", line=dict(color=color, dash="dot"),
                    marker=dict(symbol=["circle-open", "star"], size=[6, 16], color=color),
                    name=f"{m} forecast {target_date:%d %b}: ${preds[m][price]:,.2f}"))
        if price == "Close" and pa_models and f"{pa_models[0]} | P(up)" in sub:
            went_up = sub["Actual Close"] > sub["Close_t"]
            called_up = sub[f"{pa_models[0]} | P(up)"] > 0.5
            right = went_up == called_up
            fig.add_trace(go.Scatter(
                x=sub["Target_Date"], y=actual, mode="markers", name=f"{pa_models[0]} Up/Down call right / wrong",
                marker=dict(size=9, symbol=["triangle-up" if c else "triangle-down" for c in called_up],
                            color=["#26a69a" if r else "#ef5350" for r in right], line=dict(width=0)),
                hovertext=[f"called {'UP' if c else 'DOWN'} · {'right' if r else 'wrong'}"
                           for c, r in zip(called_up, right)], hoverinfo="text+x"))
        fig.add_vline(x=sub["Target_Date"].iloc[-1], line_dash="dot", line_color="gray")
        fig.update_layout(title=f"Next-day {price}: predicted vs actual (last {days} days) and forecast for "
                                f"{target_date:%d %b %Y}", height=520, yaxis_title="USD / oz",
                          margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.15),
                          hovermode="x unified")
        st.plotly_chart(fig, width="stretch")
        st.caption(
            f"Each point is the prediction made the evening before, by models trained only on data before "
            f"{R['backtest']['test_start']} (they never saw these days). The ★ is the forecast "
            f"for {target_date:%d %b} from the models retrained on 100% of the data. "
            + ("Triangles on the actual line show the first model's Up/Down call (▲/▼), green = right, "
               "red = wrong." if price == "Close" else ""))

        # ---- 2. accuracy over the selected window
        rows = []
        went_up = sub["Actual Close"] > sub["Close_t"]
        naive_err = sub["Close_t"] - actual
        rows.append({"Model": "Naive (today's close)", "RMSE $": np.sqrt(np.mean(naive_err ** 2)),
                     "MAE $": np.mean(np.abs(naive_err)), "Bias $": np.mean(naive_err),
                     "Price direction right %": np.nan, "Up/Down classifier right %": np.nan})
        for m in pa_models:
            err = sub[f"{m} | {price}"] - actual
            pred_up = sub[f"{m} | Close"] > sub["Close_t"]
            rows.append({
                "Model": m, "RMSE $": np.sqrt(np.mean(err ** 2)), "MAE $": np.mean(np.abs(err)),
                "Bias $": np.mean(err),
                "Price direction right %": np.mean(pred_up == went_up) * 100,
                "Up/Down classifier right %": (np.mean((sub[f"{m} | P(up)"] > 0.5) == went_up) * 100
                                               if f"{m} | P(up)" in sub else np.nan),
            })
        st.markdown(f"**{price} accuracy over these {days} days**")
        st.dataframe(pd.DataFrame(rows).set_index("Model").style.format("{:,.2f}", na_rep="—"), width="stretch")
        st.caption("Bias > 0 means the model predicted too high on average. Direction columns always refer to "
                   "the close (up vs down from the previous close); 'always up' would have been right "
                   f"{went_up.mean() * 100:.1f}% of these days.")

        if pa_models:
            m0 = pa_models[0]
            c1, c2 = st.columns(2)
            # ---- 3. predicted range vs actual range (High / Low)
            with c1:
                fig = go.Figure([
                    go.Scatter(x=sub["Target_Date"], y=sub[f"{m0} | High"], line=dict(width=0), showlegend=False,
                               hoverinfo="skip"),
                    go.Scatter(x=sub["Target_Date"], y=sub[f"{m0} | Low"], fill="tonexty",
                               fillcolor="rgba(41,182,246,0.25)", line=dict(width=0), name=f"{m0} predicted High–Low"),
                    go.Scatter(x=sub["Target_Date"], y=sub["Actual High"], name="Actual High",
                               line=dict(color="#26a69a", width=1.5)),
                    go.Scatter(x=sub["Target_Date"], y=sub["Actual Low"], name="Actual Low",
                               line=dict(color="#ef5350", width=1.5)),
                ])
                inside = ((sub["Actual High"] <= sub[f"{m0} | High"]) & (sub["Actual Low"] >= sub[f"{m0} | Low"])).mean()
                fig.update_layout(title=f"{m0}: predicted vs actual daily range "
                                        f"(fully inside on {inside:.0%} of days)", height=380,
                                  margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.2))
                st.plotly_chart(fig, width="stretch")
            # ---- 4. prediction error over time
            with c2:
                err = sub[f"{m0} | {price}"] - actual
                fig = go.Figure([
                    go.Bar(x=sub["Target_Date"], y=err, name="Error (predicted − actual)",
                           marker_color=["#ef5350" if e > 0 else "#29b6f6" for e in err]),
                    go.Scatter(x=sub["Target_Date"], y=np.sqrt((err ** 2).rolling(20, min_periods=5).mean()),
                               name="Rolling 20-day RMSE", line=dict(color=actual_color, width=2.5)),
                ])
                fig.update_layout(title=f"{m0}: {price} error per day ($)", height=380,
                                  margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.2))
                st.plotly_chart(fig, width="stretch")

            # ---- 5. predicted vs actual daily change
            fig = go.Figure()
            lim = 0
            for i, m in enumerate(pa_models):
                x = (actual / sub["Close_t"] - 1) * 100
                y = (sub[f"{m} | {price}"] / sub["Close_t"] - 1) * 100
                lim = max(lim, x.abs().max(), y.abs().max())
                fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name=m, opacity=0.7,
                                         marker=dict(color=palette[i % len(palette)], size=6)))
            fig.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], mode="lines", name="perfect prediction",
                                     line=dict(color="gray", dash="dash")))
            fig.update_layout(title=f"{price}: predicted vs actual change from the previous close (%)", height=420,
                              xaxis_title="Actual change %", yaxis_title="Predicted change %",
                              margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption("Points on the dashed line would be perfect. A flat cloud around 0 on the vertical axis "
                       "means the model mostly predicts 'little change' for this price.")

# ------------------------------------------------------------
# Up/Down tab: binary classification of tomorrow's close
# ------------------------------------------------------------
with tab_ud:
    if not cls:
        st.info("No classification results yet: run `python train_predict.py`.")
    else:
        baseline = cls.get("Always up (baseline)", {})
        ud = pd.DataFrame(cls).T.rename(columns={
            "accuracy": "Test accuracy %", "auc": "AUC", "precision_up": "Precision (up) %",
            "recall_up": "Recall (up) %", "pred_up_share": "Predicted up %", "val_accuracy": "Validation accuracy %",
            "p_up_next_day": "P(up) tomorrow"})
        ud["Tomorrow"] = ud["P(up) tomorrow"].map(lambda v: "—" if pd.isna(v) else ("▲ UP" if v > 0.5 else "▼ DOWN"))
        ud = ud[["Tomorrow", "P(up) tomorrow", "Test accuracy %", "AUC", "Precision (up) %", "Recall (up) %",
                 "Predicted up %", "Validation accuracy %"]].sort_values("Test accuracy %", ascending=False)
        ud.insert(0, "Rank", [str(i + 1) for i in range(len(ud))])

        ups = sum(1 for m in model_names if cls.get(m, {}).get("p_up_next_day", 0.5) > 0.5)
        st.markdown(f"**Tomorrow ({target_date:%a %d %b %Y}): {ups} of {len(model_names)} models say UP**, "
                    f"{len(model_names) - ups} say DOWN.")
        num = ["P(up) tomorrow", "Test accuracy %", "AUC", "Precision (up) %", "Recall (up) %", "Predicted up %",
               "Validation accuracy %"]
        for c in num:
            ud[c] = pd.to_numeric(ud[c])
        st.dataframe(
            ud.style.format({c: ("{:.1%}" if c == "P(up) tomorrow" else "{:.3f}" if c == "AUC" else "{:.1f}")
                             for c in num}, na_rep="—")
            .background_gradient(subset=["Test accuracy %"], cmap="RdYlGn", vmin=45, vmax=60)
            .apply(lambda r: ["font-weight: bold" if r.name == "Always up (baseline)" else "" for _ in r], axis=1),
            width="stretch", height=38 * (len(ud) + 1) + 3,
        )
        st.caption(
            f"Scored on the {R['backtest']['test_rows']:,}-day hold-out ({R['backtest']['test_start']} → "
            f"{R['backtest']['test_end']}). Gold rose on {baseline.get('accuracy', float('nan')):.1f}% of those days, "
            "so 'Always up' is the bar to beat. AUC 0.5 = no skill; precision = how often an UP call was right; "
            "recall = share of actual up days that were called UP. Neural models learn Up/Down jointly with the "
            "prices; LightGBM uses its own classifier; Chronos-Bolt's P(up) comes from its forecast quantiles; "
            "ensembles average their members' probabilities.")

        pu = ud["P(up) tomorrow"].dropna().sort_values()
        fig = go.Figure(go.Bar(x=pu.values, y=pu.index, orientation="h",
                               marker_color=["#26a69a" if v > 0.5 else "#ef5350" for v in pu.values],
                               text=[f"{v:.1%}" for v in pu.values], textposition="auto"))
        fig.add_vline(x=0.5, line_dash="dash", line_color="gray", annotation_text="50%")
        lo, hi = min(pu.min(), 0.45), max(pu.max(), 0.55)
        fig.update_layout(title=f"P(close up on {target_date:%d %b}) by model", height=40 * len(pu) + 80,
                          xaxis=dict(range=[lo - 0.02, hi + 0.02], tickformat=".0%"),
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------
# Leaderboard tab: models sorted by RMSE / MAE
# ------------------------------------------------------------
with tab_lb:
    rows = []
    for m, met in metrics.items():
        row = {"Model": m,
               "Avg RMSE $": np.mean([met[p]["RMSE"] for p in PRICE_TYPES]),
               "Avg MAE $": np.mean([met[p]["MAE"] for p in PRICE_TYPES])}
        for p in PRICE_TYPES:
            row[f"{p} RMSE $"] = met[p]["RMSE"]
            row[f"{p} MAE $"] = met[p]["MAE"]
        rows.append(row)
    lb = pd.DataFrame(rows)

    sort_options = ["Avg RMSE $", "Avg MAE $"] + [f"{p} {k} $" for p in PRICE_TYPES for k in ("RMSE", "MAE")]
    sort_by = st.selectbox("Sort models by", sort_options, index=0)
    lb = lb.sort_values(sort_by).reset_index(drop=True)
    lb.insert(0, "Rank", [("🥇", "🥈", "🥉")[i] if i < 3 else str(i + 1) for i in range(len(lb))])
    lb = lb.set_index("Model")

    winner = lb.index[0]
    st.success(f"**{winner}** has the lowest {sort_by.replace(' $', '')} "
               f"(\\${lb[sort_by].iloc[0]:,.2f}) on the {R['backtest']['test_rows']:,}-day hold-out "
               f"({R['backtest']['test_start']} → {R['backtest']['test_end']}).")

    num_cols = [c for c in lb.columns if c != "Rank"]
    st.dataframe(
        lb.style.format("{:,.2f}", subset=num_cols)
        .background_gradient(subset=num_cols, cmap="RdYlGn_r", axis=0)
        .apply(lambda s: ["font-weight: bold" if c == sort_by else "" for c in s.index], axis=1),
        width="stretch", height=38 * (len(lb) + 1) + 3,
    )
    st.caption("RMSE penalises large misses more heavily; MAE is the typical miss in dollars. "
               "Both are measured on the next-day Open, High, Low and Close; lower is better. "
               "Colours run green (best) → red (worst) within each column.")

    order = lb.index.tolist()
    fig = go.Figure([
        go.Bar(name="RMSE", x=order, y=lb[sort_by.replace("MAE", "RMSE")], marker_color="#f9a825",
               text=[f"{v:,.2f}" for v in lb[sort_by.replace("MAE", "RMSE")]], textposition="outside"),
        go.Bar(name="MAE", x=order, y=lb[sort_by.replace("RMSE", "MAE")], marker_color="#29b6f6",
               text=[f"{v:,.2f}" for v in lb[sort_by.replace("RMSE", "MAE")]], textposition="outside"),
    ])
    fig.update_layout(barmode="group", height=420, margin=dict(l=10, r=10, t=40, b=10),
                      title=f"{sort_by.replace(' RMSE $', '').replace(' MAE $', '')} RMSE and MAE by model "
                            f"(sorted by {sort_by.replace(' $', '')})",
                      yaxis_title="$", legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------
# Forecast tab
# ------------------------------------------------------------
with tab_fc:
    rows = []
    for m in model_names + ensemble_names:
        p = preds[m]
        met = metrics.get(m)
        rows.append({
            "Model": m,
            "Open": p["Open"], "High": p["High"], "Low": p["Low"], "Close": p["Close"],
            "Close Δ %": (p["Close"] / last_close - 1) * 100,
            "Range $": p["High"] - p["Low"],
            "Backtest avg RMSE $": met["AvgRMSE"] if met else np.nan,
            "Backtest direction %": met["DirectionAccuracy"] if met else np.nan,
            "P(up)": cls.get(m, {}).get("p_up_next_day", np.nan),
            "Up/Down": ("—" if cls.get(m, {}).get("p_up_next_day") is None
                        else "▲ UP" if cls[m]["p_up_next_day"] > 0.5 else "▼ DOWN"),
        })
    table = pd.DataFrame(rows).set_index("Model")
    best_model = table["Backtest avg RMSE $"].idxmin()
    st.markdown(f"**All models — predicted {target_date:%d %b %Y} bar** "
                f"(best backtest model: **{best_model}**)")
    # Ensembles have no backtest of their own; show a dash instead of NaN
    disp = table.copy()
    disp["Backtest avg RMSE $"] = disp["Backtest avg RMSE $"].map(lambda v: "—" if pd.isna(v) else f"{v:,.2f}")
    disp["Backtest direction %"] = disp["Backtest direction %"].map(lambda v: "—" if pd.isna(v) else f"{v:.1f}%")
    st.dataframe(
        disp.style.format({
            "Open": "${:,.2f}", "High": "${:,.2f}", "Low": "${:,.2f}", "Close": "${:,.2f}",
            "Close Δ %": "{:+.3f}%", "Range $": "${:,.2f}", "P(up)": "{:.1%}",
        }, na_rep="—").apply(lambda s: ["background-color: #2e7d3240" if m == best_model else "" for m in s.index],
                 subset=["Backtest avg RMSE $"])
        .background_gradient(subset=["Close Δ %"], cmap="RdYlGn", vmin=-0.3, vmax=0.3),
        width="stretch",
    )

    # Candlestick: history + one predicted candle per selected model
    ro = pd.DataFrame(R["recent_ohlc"]).tail(history_days)
    ro["Date"] = pd.to_datetime(ro["Date"])
    fig = go.Figure(go.Candlestick(
        x=ro["Date"], open=ro["Open"], high=ro["High"], low=ro["Low"], close=ro["Close"],
        name="Actual", increasing_line_color="#26a69a", decreasing_line_color="#ef5350"))
    palette = ["#f9a825", "#7e57c2", "#29b6f6", "#ec407a", "#8d6e63", "#66bb6a", "#ffa726", "#5c6bc0"]
    for i, m in enumerate(selected):
        p = preds[m]
        # offset x slightly so multiple predicted candles don't overlap
        x = target_date + pd.Timedelta(hours=(i - (len(selected) - 1) / 2) * 5)
        fig.add_trace(go.Candlestick(
            x=[x], open=[p["Open"]], high=[p["High"]], low=[p["Low"]], close=[p["Close"]],
            name=f"{m} (pred)", increasing_line_color=palette[i % len(palette)],
            decreasing_line_color=palette[i % len(palette)], opacity=0.85))
    fig.add_vline(x=pd.Timestamp(R["last_date"]) + pd.Timedelta(hours=12),
                  line_dash="dot", line_color="gray")
    fig.update_layout(height=520, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.05), yaxis_title="USD / oz")
    st.plotly_chart(fig, width="stretch")

    # Close forecasts vs last close
    close_df = table["Close"].sort_values()
    fig2 = go.Figure(go.Bar(
        x=close_df.values - last_close, y=close_df.index, orientation="h",
        marker_color=["#26a69a" if v > last_close else "#ef5350" for v in close_df.values],
        text=[f"${v:,.2f}" for v in close_df.values], textposition="auto"))
    fig2.update_layout(title="Predicted close minus last close ($)", height=380,
                       margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig2, width="stretch")

# ------------------------------------------------------------
# Backtest tab
# ------------------------------------------------------------
with tab_bt:
    b = R["backtest"]
    st.markdown(
        f"Models trained on the first 90% ({b['train_rows']:,} train + {b['val_rows']:,} validation rows) "
        f"and scored on the last {b['test_rows']:,} days (**{b['test_start']} → {b['test_end']}**) "
        f"which they never saw. The final forecast above comes from models re-trained on 100% of the data.")
    rows = []
    for m, met in metrics.items():
        row = {"Model": m, "Direction acc. %": met["DirectionAccuracy"], "Avg RMSE $": met["AvgRMSE"]}
        for p in PRICE_TYPES:
            row[f"{p} MAE $"] = met[p]["MAE"]
            row[f"{p} RMSE $"] = met[p]["RMSE"]
            row[f"{p} MAPE %"] = met[p]["MAPE"]
            row[f"{p} ±0.5% hit %"] = met[p]["HitRate"]
        rows.append(row)
    mdf = pd.DataFrame(rows).set_index("Model").apply(pd.to_numeric).sort_values("Avg RMSE $")
    st.dataframe(mdf.style.format("{:.2f}", na_rep="—")
                 .highlight_min(subset=[c for c in mdf.columns if "MAE" in c or "RMSE" in c or "MAPE" in c],
                                color="#2e7d3240"),
                 width="stretch")
    st.caption("Naive baseline = \"tomorrow's O/H/L/C equal today's close\". "
               "A model is only useful if it beats it. Direction accuracy is the share of days where the "
               "predicted close moved the same way as the actual close (50% ≈ coin flip).")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(x=mdf.index, y=mdf["Close RMSE $"],
                               marker_color=["#9e9e9e" if m.startswith("Naive") else "#f9a825" for m in mdf.index]))
        lo, hi = mdf["Close RMSE $"].min(), mdf["Close RMSE $"].max()
        fig.update_layout(title="Close RMSE on hold-out ($, lower is better)", height=360,
                          margin=dict(l=10, r=10, t=40, b=10),
                          yaxis=dict(range=[lo - (hi - lo) * 1.5 - 0.1, hi + (hi - lo) * 0.5 + 0.1]))
        st.plotly_chart(fig, width="stretch")
    with c2:
        dmdf = mdf["Direction acc. %"].dropna().sort_values()
        fig = go.Figure(go.Bar(x=dmdf.values, y=dmdf.index, orientation="h", marker_color="#29b6f6",
                               text=[f"{v:.1f}%" for v in dmdf.values], textposition="auto"))
        fig.add_vline(x=50, line_dash="dash", line_color="gray", annotation_text="coin flip")
        fig.update_layout(title="Direction accuracy on hold-out", height=360,
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")

    if reg and reg.get("backtest_avg_rmse"):
        st.markdown("**Average RMSE ($) by market regime on the hold-out**")
        rr = pd.DataFrame(reg["backtest_avg_rmse"]).T[reg["names"]]
        rr = rr.sort_values("Sideways")
        st.dataframe(rr.style.format("{:,.2f}", na_rep="—").background_gradient(cmap="RdYlGn_r", axis=0),
                     width="stretch")
        st.caption("Hold-out days per regime: " + ", ".join(f"{k} {v}" for k, v in reg["test_days"].items())
                   + ". Regimes come from the mixture fitted on training data only.")

    if bt is not None:
        show = st.multiselect("Models to compare with actual close", model_names, default=model_names[:2])
        window = st.slider("Days shown", 20, len(bt), min(120, len(bt)))
        sub = bt.tail(window)
        fig = go.Figure(go.Scatter(x=sub["Target_Date"], y=sub["Actual_Next_Close"], name="Actual close",
                                   line=dict(color="black", width=2)))
        for m in show:
            fig.add_trace(go.Scatter(x=sub["Target_Date"], y=sub[m], name=m, line=dict(dash="dot")))
        fig.update_layout(title="Actual vs predicted next-day close (hold-out)", height=450,
                          margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, width="stretch")

        if show:
            err = pd.DataFrame({m: sub[m] - sub["Actual_Next_Close"] for m in show})
            fig = go.Figure([go.Histogram(x=err[m], name=m, opacity=0.6, nbinsx=40) for m in show])
            fig.update_layout(barmode="overlay", title="Prediction error distribution ($)", height=350,
                              margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------
# Market stats tab
# ------------------------------------------------------------
with tab_mkt:
    if reg:
        st.markdown("**Market regime detection** (Gaussian mixture on " + ", ".join(reg["inputs"]) + ")")
        prof = pd.DataFrame(reg["profile"]).T.rename(columns={
            "share_of_days_pct": "Share of days %", "avg_20d_return_pct": "Avg 20-day return %",
            "avg_daily_vol_pct": "Avg daily volatility %", "next_day_up_pct": "Next day up %",
            "avg_next_day_return_pct": "Avg next-day return %"})
        c1, c2 = st.columns([2, 3])
        with c1:
            fig = go.Figure(go.Bar(x=list(reg["current"].values()), y=list(reg["current"].keys()), orientation="h",
                                   marker_color=[REGIME_COLORS[k] for k in reg["current"]],
                                   text=[f"{v:.0%}" for v in reg["current"].values()], textposition="auto"))
            fig.update_layout(title=f"Regime probabilities on {R['last_date']}", height=240,
                              xaxis=dict(range=[0, 1], tickformat=".0%"), margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, width="stretch")
        with c2:
            st.dataframe(prof.style.format("{:.2f}"), width="stretch")
            st.caption("Profile of each regime over all history. Regimes separate volatility and trend "
                       "clearly, but the next-day up % barely differs between them.")
        tl = pd.DataFrame(reg["timeline"])
        fig = go.Figure(go.Scatter(x=tl["Date"], y=tl["Close"], mode="lines", line=dict(color="#bdbdbd"),
                                   name="Close", showlegend=False))
        for name in reg["names"]:
            sub = tl[tl["Regime"] == name]
            fig.add_trace(go.Scatter(x=sub["Date"], y=sub["Close"], mode="markers", name=name,
                                     marker=dict(color=REGIME_COLORS[name], size=6)))
        fig.update_layout(title="Close price coloured by detected regime (last 250 days)", height=360,
                          margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")
        st.divider()

    ms = R["market_stats"]
    st.markdown(f"**Indicators as of {R['last_date']}**")
    c = st.columns(4)
    rsi = ms["RSI_14"]
    rsi_label = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
    c[0].metric("RSI (14)", f"{rsi:.1f}", rsi_label, delta_color="off")
    c[1].metric("ATR (14)", f"${ms['ATR_14']:,.2f}", f"{ms['ATR_pct']:.2f}% of price", delta_color="off")
    c[2].metric("Volatility 14d (daily σ)", f"{ms['Volatility_14_pct']:.2f}%",
                f"7d: {ms['Volatility_7_pct']:.2f}%", delta_color="off")
    c[3].metric("Annualized vol (30d)", f"{ms['Annualized_Vol_30d_pct']:.1f}%")

    c = st.columns(4)
    c[0].metric("1-day return", f"{ms['Return_1d_pct']:+.2f}%")
    c[1].metric("5-day return", f"{ms['Return_5d_pct']:+.2f}%")
    c[2].metric("20-day return", f"{ms['Return_20d_pct']:+.2f}%")
    c[3].metric("YTD return", f"{ms['Return_YTD_pct']:+.2f}%")

    c = st.columns(4)
    c[0].metric("52-week high", f"${ms['High_52w']:,.2f}", f"{(last_close / ms['High_52w'] - 1) * 100:+.2f}% from high")
    c[1].metric("52-week low", f"${ms['Low_52w']:,.2f}", f"{(last_close / ms['Low_52w'] - 1) * 100:+.2f}% from low")
    c[2].metric("MACD", f"{ms['MACD']:.2f}", f"signal {ms['MACD_Signal']:.2f}",
                delta_color="normal" if ms["MACD"] > ms["MACD_Signal"] else "inverse")
    c[3].metric("Bollinger position", f"{ms['BB_Position']:.2f}",
                f"band {ms['BB_Lower']:,.0f} – {ms['BB_Upper']:,.0f}", delta_color="off")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=rsi, title={"text": "RSI (14)"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#f9a825"},
                   "steps": [{"range": [0, 30], "color": "#c8e6c9"}, {"range": [30, 70], "color": "#eeeeee"},
                             {"range": [70, 100], "color": "#ffcdd2"}]}))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(fig, width="stretch")
    with c2:
        ma = {k: ms[k] for k in ["SMA_7", "SMA_14", "SMA_30", "EMA_7", "EMA_14", "EMA_30", "BB_Middle"]}
        dist = pd.Series({k: (last_close / v - 1) * 100 for k, v in ma.items()})
        fig = go.Figure(go.Bar(x=dist.index, y=dist.values,
                               marker_color=["#26a69a" if v > 0 else "#ef5350" for v in dist.values],
                               text=[f"${ma[k]:,.0f}" for k in dist.index], textposition="outside"))
        fig.update_layout(title="Close vs moving averages (% above / below)", height=300,
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")

    rets = np.array(R["daily_returns_pct"])
    fig = go.Figure(go.Histogram(x=rets, nbinsx=80, marker_color="#f9a825"))
    for m in model_names:
        fig.add_vline(x=(preds[m]["Close"] / last_close - 1) * 100, line_color="#7e57c2", opacity=0.5)
    fig.update_layout(title=f"Distribution of the last {len(rets):,} daily returns (%) — purple lines = "
                            f"model forecasts for {target_date:%d %b}",
                      height=360, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Mean {rets.mean():+.3f}% · σ {rets.std():.3f}% · "
               f"up days {(rets > 0).mean() * 100:.1f}% · worst {rets.min():.2f}% · best {rets.max():+.2f}%")

# ------------------------------------------------------------
# Training tab
# ------------------------------------------------------------
with tab_train:
    ga = R["ga"]
    st.markdown("**Genetic-algorithm tuned SAG-LSTM**")
    c = st.columns(4)
    c[0].metric("Hidden size", ga["best_hidden_size"])
    c[1].metric("Learning rate", f"{ga['best_learning_rate']:.5f}")
    c[2].metric("Epochs", ga["best_epochs"])
    c[3].metric("Val RMSE (return)", f"{ga['best_val_rmse']:.5f}")
    gl = pd.DataFrame(ga["log"])
    fig = go.Figure([go.Scatter(x=gl["generation"], y=gl["best_fitness"], name="best", mode="lines+markers"),
                     go.Scatter(x=gl["generation"], y=gl["mean_fitness"], name="population mean",
                                mode="lines+markers")])
    fig.update_layout(title="GA fitness by generation (validation RMSE of returns)", height=300,
                      margin=dict(l=10, r=10, t=40, b=10), xaxis_title="Generation")
    st.plotly_chart(fig, width="stretch")

    tr = R["training"]
    st.dataframe(pd.DataFrame({m: {"Patience": v.get("patience"), "Best epoch (early stopping)": v.get("best_epoch"),
                                   "Epochs used for 100% refit": v["epochs"], "Learning rate": v["learning_rate"],
                                   "Train time (s)": round(v["train_seconds"], 1)} for m, v in tr.items()}).T,
                 width="stretch")
    st.caption("Early stopping keeps the epoch with the lowest validation loss (patience 8); the 100% refit then "
               "trains for that many epochs. For LightGBM the numbers are trees (Close target). "
               "Chronos-Bolt is pretrained and used without training.")
    m = st.selectbox("Loss curves for", [k for k, v in tr.items() if v["backtest_train_loss"]])
    v = tr[m]
    fig = go.Figure([
        go.Scatter(y=v["backtest_train_loss"], name="backtest train"),
        go.Scatter(y=v["backtest_val_loss"], name="backtest validation"),
        go.Scatter(y=v["full_train_loss"], name="100% refit train", line=dict(dash="dot")),
    ])
    if v.get("best_epoch"):
        fig.add_vline(x=v["best_epoch"] - 1, line_dash="dash", line_color="gray", annotation_text="best epoch")
    fig.update_layout(title=f"{m} — MSE loss (scaled returns)", height=380,
                      xaxis_title="Tree (Close target)" if m == "LightGBM" else "Epoch",
                      margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")
    att = R.get("cnn_bilstm_attention")
    if att and att.get("weights"):
        st.markdown("**CNN-BiLSTM-Attention — which past days it focused on**")
        st.caption(f"The model reads a {att['window_size']}-day window (Conv2D → BiLSTM → Attention → Dense). "
                   f"Bars show the attention weight given to each day when forecasting {target_date:%d %b %Y}; "
                   "an even spread means no single day dominated.")
        fig = go.Figure(go.Bar(x=att["dates"], y=att["weights"], marker_color="#7e57c2",
                               text=[f"{w:.1%}" for w in att["weights"]], textposition="outside"))
        fig.add_hline(y=1 / att["window_size"], line_dash="dash", line_color="gray",
                      annotation_text="uniform")
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), yaxis_tickformat=".0%",
                          xaxis_type="category")
        st.plotly_chart(fig, width="stretch")

    eemd = R.get("eemd")
    if eemd:
        st.markdown("**EEMD-TCN-GRU — decomposition of the price window it used**")
        st.caption(f"Each day's last {eemd['eemd_window']} closes are split by EEMD ({eemd['trials']} noise trials, "
                   "using only past prices). IMF 1 (fast wiggles) feeds the TCN, IMF 2 (the slower trend: "
                   "remaining IMFs + residue) feeds the GRU; their forecasts are added. "
                   f"Values are % distance from the {R['last_date']} close; IMF 1 + IMF 2 = price.")
        fig = go.Figure([
            go.Scatter(x=eemd["dates"], y=np.add(eemd["imf1_pct"], eemd["imf2_pct"]), name="Price (log % vs last)",
                       line=dict(color="gray", width=3)),
            go.Scatter(x=eemd["dates"], y=eemd["imf2_pct"], name="IMF 2 → GRU", line=dict(color="#29b6f6")),
            go.Scatter(x=eemd["dates"], y=eemd["imf1_pct"], name="IMF 1 → TCN", line=dict(color="#f9a825")),
        ])
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="%",
                          xaxis_type="category", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")

    hw = R.get("helformer")
    if hw:
        st.markdown("**Helformer — learned Holt-Winters decomposition of the forecast window**")
        c = st.columns(len(hw["channels"]))
        for col, ch, a, g in zip(c, hw["channels"], hw["alpha"], hw["gamma"]):
            col.metric(f"{ch} smoothing", f"α {a:.2f}", f"γ {g:.2f}", delta_color="off")
        st.caption("α (level) and γ (seasonal) are learned during training. The close window is split into a "
                   "smoothed level and a seasonal factor; the deseasonalized series X / (L·S) is what the "
                   "attention + LSTM layers see. Values are % vs the last close.")
        fig = go.Figure([
            go.Scatter(x=hw["dates"], y=hw["close_pct"], name="Close", line=dict(color="gray", width=3)),
            go.Scatter(x=hw["dates"], y=hw["level_pct"], name="Level L", line=dict(color="#29b6f6")),
            go.Scatter(x=hw["dates"], y=hw["deseasonalized_pct"], name="Deseasonalized X/(L·S)",
                       line=dict(color="#f9a825")),
        ])
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="%",
                          xaxis_type="category", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")

    with st.expander(f"Input features ({R['n_features']})"):
        st.write(", ".join(R["features"]))
