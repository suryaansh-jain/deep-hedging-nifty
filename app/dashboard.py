from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

DATA_DIRECTORY = Path(__file__).resolve().parent.parent / "data" / "processed" / "dashboard"
STRATEGY_LABELS = {
    "unhedged": "Unhedged",
    "bs_delta": "BS delta",
    "delta_band": "Delta + no-trade band",
    "mlp": "Deep hedge (MLP)",
    "ntb": "Deep hedge (band network)",
    "ntb_tail": "Deep hedge (band net, CVaR95+99)",
}
STRATEGY_COLORS = {"unhedged": "#9e9e9e", "bs_delta": "#1f77b4", "delta_band": "#2ca02c", "mlp": "#ff7f0e", "ntb": "#d62728", "ntb_tail": "#9467bd"}
BASELINES = ("unhedged", "bs_delta", "delta_band")
METRIC_COLUMNS = {
    "mean_pnl_x_premium": "Mean P&L",
    "std_pnl_x_premium": "Std P&L",
    "cvar95_x_premium": "CVaR95",
    "cvar99_x_premium": "CVaR99",
    "mean_total_cost_x_premium": "Mean cost",
    "mean_turnover": "Turnover",
    "fraction_loss_above_5pct_premium": "P(loss > 5% premium)",
}


@st.cache_data
def load_tables() -> dict[str, pd.DataFrame]:
    names = ["test_metrics", "bootstrap_cvar95_differences", "pnl_histograms", "real_data_metrics", "real_data_bootstrap_cvar95_differences", "real_data_windows"]
    return {name: pd.read_csv(DATA_DIRECTORY / f"{name}.csv") for name in names}


@st.cache_data
def load_manifest() -> dict:
    with open(DATA_DIRECTORY / "manifest.json") as handle:
        return json.load(handle)


@st.cache_resource
def load_sample_paths() -> dict[str, np.ndarray]:
    with np.load(DATA_DIRECTORY / "sample_paths.npz") as archive:
        return {key: archive[key] for key in archive.files}


def shown_strategies(policy: str, available: set[str]) -> list[str]:
    return [name for name in [*BASELINES, policy] if name in available]


def metrics_table(metrics: pd.DataFrame, strategies: list[str]) -> pd.DataFrame:
    table = metrics.set_index("strategy").loc[strategies, list(METRIC_COLUMNS)]
    table.index = [STRATEGY_LABELS[name] for name in table.index]
    return table.rename(columns=METRIC_COLUMNS)


def histogram_figure(histograms: pd.DataFrame, strategies: list[str], title: str) -> go.Figure:
    figure = go.Figure()
    for name in strategies:
        rows = histograms[histograms["strategy"] == name]
        figure.add_trace(go.Scatter(x=rows["bin_center"], y=rows["probability"], mode="lines", line={"shape": "hvh", "color": STRATEGY_COLORS[name], "width": 2}, name=STRATEGY_LABELS[name]))
    figure.update_layout(title=title, xaxis_title="terminal P&L / premium (tails clipped to [-5, 2])", yaxis_title="share of paths", yaxis_type="log", height=430, legend={"orientation": "h", "y": -0.25})
    return figure


def frontier_figure(metrics: pd.DataFrame, strategies: list[str], selected_cost: float, title: str) -> go.Figure:
    figure = go.Figure()
    for name in strategies:
        if name == "unhedged":
            continue
        rows = metrics[metrics["strategy"] == name].sort_values("cost_bps")
        figure.add_trace(go.Scatter(x=rows["mean_total_cost_x_premium"], y=rows["cvar95_x_premium"], mode="lines+markers", name=STRATEGY_LABELS[name], line={"color": STRATEGY_COLORS[name]}, text=[f"{cost:g} bps" for cost in rows["cost_bps"]], hovertemplate="%{text}<br>cost %{x:.3f}<br>CVaR95 %{y:.3f}<extra></extra>"))
        chosen = rows[rows["cost_bps"] == selected_cost]
        figure.add_trace(go.Scatter(x=chosen["mean_total_cost_x_premium"], y=chosen["cvar95_x_premium"], mode="markers", marker={"size": 16, "color": STRATEGY_COLORS[name], "line": {"color": "black", "width": 2}}, showlegend=False, hoverinfo="skip"))
    figure.update_layout(title=title, xaxis_title="mean total transaction cost / premium", yaxis_title="CVaR95 of loss / premium", height=450, legend={"orientation": "h", "y": -0.25})
    return figure


def path_figure(paths: dict[str, np.ndarray], simulator: str, cost_bps: float, path_index: int, strategies: list[str]) -> go.Figure:
    days = np.arange(paths[f"{simulator}__prices"].shape[1])
    figure = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.4, 0.2, 0.4], vertical_spacing=0.05, subplot_titles=("Index level (S / K)", "Conditional volatility feature (annualized)", "Hedge position h"))
    figure.add_trace(go.Scatter(x=days, y=paths[f"{simulator}__prices"][path_index], line={"color": "black"}, name="S / K", showlegend=False), row=1, col=1)
    figure.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=1)
    figure.add_trace(go.Scatter(x=days, y=paths[f"{simulator}__volatility"][path_index], line={"color": "#7f7f7f"}, showlegend=False), row=2, col=1)
    for name in strategies:
        key = f"{simulator}__positions__{name}__{cost_bps:g}"
        if name == "unhedged" or key not in paths:
            continue
        figure.add_trace(go.Scatter(x=days[:-1], y=paths[key][path_index], line={"shape": "hv", "color": STRATEGY_COLORS[name], "width": 2}, name=STRATEGY_LABELS[name]), row=3, col=1)
    figure.update_yaxes(range=[-0.05, 1.05], row=3, col=1)
    figure.update_xaxes(title_text="trading day", row=3, col=1)
    figure.update_layout(height=700, legend={"orientation": "h", "y": -0.1})
    return figure


def bootstrap_table(bootstrap: pd.DataFrame, policy: str, cost_bps: float, scale: float) -> pd.DataFrame:
    rows = bootstrap[(bootstrap["strategy"] == policy) & (bootstrap["cost_bps"] == cost_bps)]
    return pd.DataFrame(
        {
            "Versus": [STRATEGY_LABELS[name] for name in rows["versus"]],
            "CVaR95 difference": rows["difference"].to_numpy() / scale,
            "95% CI low": rows["ci_low"].to_numpy() / scale,
            "95% CI high": rows["ci_high"].to_numpy() / scale,
        }
    )


def main() -> None:
    st.set_page_config(page_title="Deep hedging Nifty", layout="wide")
    if not (DATA_DIRECTORY / "manifest.json").exists():
        st.error(f"No dashboard data in {DATA_DIRECTORY}. Run scripts run_01 to run_07 first.")
        return
    manifest = load_manifest()
    tables = load_tables()
    paths = load_sample_paths()
    premium = manifest["premium"]

    st.sidebar.header("Controls")
    cost_bps = float(st.sidebar.select_slider("Transaction cost (bps)", options=manifest["cost_grid"], value=manifest["default_cost_bps"]))
    simulator = st.sidebar.radio("Simulator", manifest["simulators"], format_func=str.upper)
    test_metrics = tables["test_metrics"]
    simulator_metrics = test_metrics[test_metrics["simulator"] == simulator]
    available = set(simulator_metrics["strategy"])
    deep_options = [name for name in ("ntb", "ntb_tail", "mlp") if name in available]
    policy = st.sidebar.selectbox("Deep hedging policy", deep_options, format_func=lambda name: STRATEGY_LABELS[name])
    strategies = shown_strategies(policy, available)
    st.sidebar.caption(f"All values are in units of the option premium C0 = {premium:.4f} x S0 (sigma0 = {manifest['sigma0']:.3f}, r = {manifest['rate']}). Costs are illustrative.")

    st.title("Deep hedging a short Nifty ATM call under transaction costs")
    st.caption(f"{manifest['n_test_paths']:,} simulated test paths (seed {manifest['test_seed']}), {manifest['maturity_days']} trading days, daily rebalancing. Same paths for every strategy.")
    overview, paths_tab, frontier_tab, real_tab = st.tabs(["P&L and metrics", "Hedge paths", "Cost vs risk frontier", "Real Nifty data"])
    at_cost = simulator_metrics[simulator_metrics["cost_bps"] == cost_bps]

    with overview:
        deep_row = at_cost[at_cost["strategy"] == policy].iloc[0]
        delta_row = at_cost[at_cost["strategy"] == "bs_delta"].iloc[0]
        columns = st.columns(3)
        columns[0].metric("CVaR95, deep hedge", f"{deep_row['cvar95_x_premium']:.3f}", f"{deep_row['cvar95_x_premium'] - delta_row['cvar95_x_premium']:+.3f} vs BS delta", delta_color="inverse")
        columns[1].metric("Mean cost, deep hedge", f"{deep_row['mean_total_cost_x_premium']:.3f}", f"{deep_row['mean_total_cost_x_premium'] - delta_row['mean_total_cost_x_premium']:+.3f} vs BS delta", delta_color="inverse")
        columns[2].metric("CVaR99, deep hedge", f"{deep_row['cvar99_x_premium']:.3f}", f"{deep_row['cvar99_x_premium'] - delta_row['cvar99_x_premium']:+.3f} vs BS delta", delta_color="inverse")
        histograms = tables["pnl_histograms"]
        selected_histograms = histograms[(histograms["simulator"] == simulator) & (histograms["cost_bps"] == cost_bps)]
        st.plotly_chart(histogram_figure(selected_histograms, strategies, f"P&L distribution, {simulator.upper()}, {cost_bps:g} bps"), width="stretch")
        st.subheader("Metrics (x premium, except turnover and probability)")
        st.dataframe(metrics_table(at_cost, strategies).style.format("{:.3f}"), width="stretch")
        st.subheader("Paired bootstrap, CVaR95 of deep hedge minus baseline (1000 resamples)")
        st.dataframe(bootstrap_table(tables["bootstrap_cvar95_differences"][tables["bootstrap_cvar95_differences"]["simulator"] == simulator], policy, cost_bps, premium).style.format({"CVaR95 difference": "{:.3f}", "95% CI low": "{:.3f}", "95% CI high": "{:.3f}"}), width="stretch", hide_index=True)

    with paths_tab:
        n_samples = paths[f"{simulator}__prices"].shape[0]
        path_index = st.slider("Sample test path", 0, n_samples - 1, 0)
        st.plotly_chart(path_figure(paths, simulator, cost_bps, path_index, strategies), width="stretch")

    with frontier_tab:
        st.plotly_chart(frontier_figure(simulator_metrics, strategies, cost_bps, f"Cost versus tail risk, {simulator.upper()} (selected cost circled)"), width="stretch")
        cvar_figure = go.Figure()
        for name in strategies:
            if name == "unhedged":
                continue
            rows = simulator_metrics[simulator_metrics["strategy"] == name].sort_values("cost_bps")
            cvar_figure.add_trace(go.Scatter(x=rows["cost_bps"], y=rows["cvar95_x_premium"], mode="lines+markers", name=STRATEGY_LABELS[name], line={"color": STRATEGY_COLORS[name]}))
        cvar_figure.add_vline(x=cost_bps, line_dash="dot", line_color="black")
        cvar_figure.update_layout(title="CVaR95 versus transaction cost", xaxis_title="cost (bps)", yaxis_title="CVaR95 / premium", height=400)
        st.plotly_chart(cvar_figure, width="stretch")

    with real_tab:
        windows = manifest["real_data_windows"]
        independent = windows["non_overlapping"]
        st.warning(
            f"Sanity check on a small sample, not proof. {independent['n_windows']} independent non-overlapping 21-day windows "
            f"({independent['first_start']} to {independent['last_start']}). Synthetic ATM option priced with Black-Scholes at India VIX, "
            "spot index as hedge, no dividends or futures basis. Models were trained on simulated GARCH paths, whatever simulator is selected."
        )
        st.metric("Independent windows", independent["n_windows"])
        label = st.radio("Window set", ["non_overlapping", "overlapping"], format_func=lambda value: "Non-overlapping (primary)" if value == "non_overlapping" else f"Overlapping ({windows['overlapping']['n_windows']} dependent windows, secondary)", horizontal=True)
        real_metrics = tables["real_data_metrics"]
        real_at_cost = real_metrics[(real_metrics["windows"] == label) & (real_metrics["cost_bps"] == cost_bps)]
        real_strategies = shown_strategies(policy, set(real_at_cost["strategy"]))
        st.dataframe(metrics_table(real_at_cost, real_strategies).style.format("{:.3f}"), width="stretch")
        real_bootstrap = tables["real_data_bootstrap_cvar95_differences"]
        st.subheader("Paired bootstrap, CVaR95 of deep hedge minus baseline")
        if label == "overlapping":
            st.caption("Overlapping windows are dependent; this naive bootstrap understates the uncertainty.")
        st.dataframe(bootstrap_table(real_bootstrap[real_bootstrap["windows"] == label], policy, cost_bps, 1.0).style.format({"CVaR95 difference": "{:.3f}", "95% CI low": "{:.3f}", "95% CI high": "{:.3f}"}), width="stretch", hide_index=True)
        real_windows = tables["real_data_windows"]
        selected = real_windows[(real_windows["windows"] == label) & (real_windows["cost_bps"] == cost_bps)]
        figure = go.Figure()
        for name in real_strategies:
            values = selected[selected["strategy"] == name]["pnl_x_premium"].clip(-6, 2)
            figure.add_trace(go.Histogram(x=values, name=STRATEGY_LABELS[name], marker_color=STRATEGY_COLORS[name], opacity=0.55, xbins={"start": -6, "end": 2, "size": 0.2}))
        figure.update_layout(barmode="overlay", title=f"Real-data P&L per window, {cost_bps:g} bps", xaxis_title="P&L / window premium", yaxis_title="windows", height=420)
        st.plotly_chart(figure, width="stretch")


main()
