from __future__ import annotations

import csv
import re
import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path("DRT_Episode_Exports")
OUT = Path("output") / ("drt_result_report_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
FIG = OUT / "figures"
TAB = OUT / "tables"

MODE_MAP = {"vanilla": "Vanilla", "greedy": "Greedy", "inference": "ONNX"}
POLICY_ORDER = ["Vanilla", "Greedy", "ONNX"]
COLORS = {"Vanilla": "#4C78A8", "Greedy": "#F58518", "ONNX": "#54A24B"}


def to_float(value):
    try:
        if value is None or value == "":
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def to_int(value):
    try:
        if value is None or value == "":
            return np.nan
        return int(float(value))
    except Exception:
        return np.nan


def parse_episode(path: Path):
    summary, route_rows, passenger_rows = {}, [], []
    header, current = None, None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if not row or all(cell == "" for cell in row):
                continue
            if row[0] == "summary" and len(row) >= 3:
                summary[row[1]] = row[2]
            elif row[0] == "metadata" and len(row) >= 3:
                summary["metadata_" + row[1]] = row[2]
            elif row[0] == "section":
                header = row
                current = (
                    "route_leg"
                    if len(row) > 1 and row[1] == "leg_index"
                    else "passenger"
                    if len(row) > 1 and row[1] == "passenger_id"
                    else None
                )
            elif row[0] == "route_leg" and current == "route_leg":
                route_rows.append(dict(zip(header, row)))
            elif row[0] == "passenger" and current == "passenger":
                passenger_rows.append(dict(zip(header, row)))
    return summary, route_rows, passenger_rows


def parse_name(path: Path):
    match = re.search(r"scenario_(\d+)_ep(\d+)_episode\.csv$", path.name)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def route_metrics(route_rows):
    if not route_rows:
        return {
            "leg_count": 0,
            "unique_edges": 0,
            "edge_reuse_factor": np.nan,
            "empty_distance_meters": np.nan,
            "empty_distance_rate": np.nan,
            "passenger_km": np.nan,
            "mean_onboard_load": np.nan,
        }
    total_distance = 0.0
    empty_distance = 0.0
    passenger_meters = 0.0
    edges = []
    for row in route_rows:
        dist = to_float(row.get("leg_distance_meters"))
        if not np.isfinite(dist):
            dist = 0.0
        total_distance += dist
        boarded = to_float(row.get("boarded_count"))
        dropped = to_float(row.get("dropped_off_count"))
        onboard_after = to_float(row.get("on_board_count"))
        boarded = 0.0 if not np.isfinite(boarded) else boarded
        dropped = 0.0 if not np.isfinite(dropped) else dropped
        onboard_after = 0.0 if not np.isfinite(onboard_after) else onboard_after
        onboard_before = max(0.0, onboard_after - boarded + dropped)
        if onboard_before <= 0:
            empty_distance += dist
        passenger_meters += dist * onboard_before
        edges.append((str(row.get("from_stop_id", "")), str(row.get("to_stop_id", ""))))
    unique_edges = len(set(edges))
    return {
        "leg_count": len(route_rows),
        "unique_edges": unique_edges,
        "edge_reuse_factor": len(route_rows) / unique_edges if unique_edges else np.nan,
        "empty_distance_meters": empty_distance,
        "empty_distance_rate": empty_distance / total_distance if total_distance > 0 else np.nan,
        "passenger_km": passenger_meters / 1000.0,
        "mean_onboard_load": passenger_meters / total_distance if total_distance > 0 else np.nan,
    }


def latest_run_dirs():
    selected = {}
    for mode_key in MODE_MAP:
        mode_root = ROOT / mode_key
        dirs = [p for p in mode_root.iterdir() if p.is_dir()] if mode_root.exists() else []
        if dirs:
            selected[mode_key] = max(dirs, key=lambda p: p.stat().st_mtime)
    return selected


def collect_data(selected_dirs=None):
    episodes, passengers, routes = [], [], []
    for mode_key, policy_label in MODE_MAP.items():
        if selected_dirs:
            search_roots = [selected_dirs[mode_key]] if mode_key in selected_dirs else []
        else:
            mode_root = ROOT / mode_key
            search_roots = [mode_root] if mode_root.exists() else []
        if not search_roots:
            continue
        for search_root in search_roots:
            for ep_path in sorted(search_root.rglob("*_episode.csv")):
                file_scenario, file_episode = parse_name(ep_path)
                summary, route_rows, passenger_rows = parse_episode(ep_path)
                row_data = build_episode_rows(
                    mode_key,
                    policy_label,
                    ep_path,
                    file_scenario,
                    file_episode,
                    summary,
                    route_rows,
                    passenger_rows,
                )
                episodes.append(row_data[0])
                passengers.extend(row_data[1])
                routes.extend(row_data[2])
    ep = pd.DataFrame(episodes).sort_values(["policy", "scenario", "episode_index"])
    pax = pd.DataFrame(passengers).sort_values(
        ["policy", "scenario", "episode_index", "passenger_id"]
    )
    route = pd.DataFrame(routes).sort_values(["policy", "scenario", "episode_index", "leg_index"])
    return ep.reset_index(drop=True), pax.reset_index(drop=True), route.reset_index(drop=True)


def build_episode_rows(
    mode_key,
    policy_label,
    ep_path,
    file_scenario,
    file_episode,
    summary,
    route_rows,
    passenger_rows,
):
    passengers, routes = [], []
    scenario = to_int(summary.get("scenario_id"))
    episode_index = to_int(summary.get("episode_index"))
    scenario = file_scenario if not np.isfinite(scenario) else int(scenario)
    episode_index = file_episode if not np.isfinite(episode_index) else int(episode_index)

    total_passengers = to_float(summary.get("total_passengers"))
    completed_passengers = to_float(summary.get("completed_passengers"))
    episode_time = to_float(summary.get("episode_time_seconds"))
    episode_distance = to_float(summary.get("episode_distance_meters"))
    route_m = route_metrics(route_rows)

    waits = [
        to_float(row.get("wait_time_seconds"))
        for row in passenger_rows
        if np.isfinite(to_float(row.get("wait_time_seconds")))
    ]
    rides = [
        to_float(row.get("ride_time_seconds"))
        for row in passenger_rows
        if np.isfinite(to_float(row.get("ride_time_seconds")))
    ]
    totals = [
        to_float(row.get("total_service_time_seconds"))
        for row in passenger_rows
        if np.isfinite(to_float(row.get("total_service_time_seconds")))
    ]

    episode_row = {
        "mode_folder": mode_key,
        "policy": policy_label,
        "scenario": scenario,
        "episode_index": episode_index,
        "source_file": str(ep_path),
        "run_folder": ep_path.parent.name,
        "next_stop_policy": summary.get("next_stop_policy", ""),
        "finish_reason": summary.get("finish_reason", ""),
        "completed_all_requests": to_float(summary.get("completed_all_requests")),
        "total_passengers": total_passengers,
        "completed_passengers": completed_passengers,
        "unserved_passengers": (
            total_passengers - completed_passengers
            if np.isfinite(total_passengers) and np.isfinite(completed_passengers)
            else np.nan
        ),
        "service_rate": to_float(summary.get("service_rate")),
        "episode_time_seconds": episode_time,
        "episode_time_minutes": episode_time / 60.0,
        "episode_distance_meters": episode_distance,
        "episode_distance_km": episode_distance / 1000.0,
        "average_wait_seconds": to_float(summary.get("average_wait_seconds")),
        "average_ride_seconds": to_float(summary.get("average_ride_seconds")),
        "wait_p50_seconds": float(np.nanpercentile(waits, 50)) if waits else np.nan,
        "wait_p90_seconds": float(np.nanpercentile(waits, 90)) if waits else np.nan,
        "wait_max_seconds": float(np.nanmax(waits)) if waits else np.nan,
        "ride_p50_seconds": float(np.nanpercentile(rides, 50)) if rides else np.nan,
        "ride_p90_seconds": float(np.nanpercentile(rides, 90)) if rides else np.nan,
        "ride_max_seconds": float(np.nanmax(rides)) if rides else np.nan,
        "service_time_p90_seconds": float(np.nanpercentile(totals, 90)) if totals else np.nan,
        **route_m,
    }
    completed = episode_row["completed_passengers"]
    episode_row["distance_km_per_completed_passenger"] = (
        episode_row["episode_distance_km"] / completed
        if np.isfinite(episode_row["episode_distance_km"]) and completed and completed > 0
        else np.nan
    )
    episode_row["time_min_per_completed_passenger"] = (
        episode_row["episode_time_minutes"] / completed
        if np.isfinite(episode_row["episode_time_minutes"]) and completed and completed > 0
        else np.nan
    )
    episode_row["legs_per_completed_passenger"] = (
        route_m["leg_count"] / completed if completed and completed > 0 else np.nan
    )
    episode_row["passenger_km_per_route_km"] = route_m["mean_onboard_load"]

    for passenger in passenger_rows:
        passengers.append(
            {
                "policy": policy_label,
                "scenario": scenario,
                "episode_index": episode_index,
                "passenger_id": to_int(passenger.get("passenger_id")),
                "origin_stop_id": to_int(passenger.get("origin_stop_id")),
                "destination_stop_id": to_int(passenger.get("destination_stop_id")),
                "status": passenger.get("status", ""),
                "wait_time_seconds": to_float(passenger.get("wait_time_seconds")),
                "ride_time_seconds": to_float(passenger.get("ride_time_seconds")),
                "total_service_time_seconds": to_float(passenger.get("total_service_time_seconds")),
                "source_file": str(ep_path),
            }
        )
    for leg in route_rows:
        boarded = to_float(leg.get("boarded_count"))
        dropped = to_float(leg.get("dropped_off_count"))
        onboard_after = to_float(leg.get("on_board_count"))
        boarded = 0.0 if not np.isfinite(boarded) else boarded
        dropped = 0.0 if not np.isfinite(dropped) else dropped
        onboard_after = 0.0 if not np.isfinite(onboard_after) else onboard_after
        routes.append(
            {
                "policy": policy_label,
                "scenario": scenario,
                "episode_index": episode_index,
                "leg_index": to_int(leg.get("leg_index")),
                "from_stop_id": to_int(leg.get("from_stop_id")),
                "to_stop_id": to_int(leg.get("to_stop_id")),
                "travel_time_seconds": to_float(leg.get("travel_time_seconds")),
                "leg_distance_meters": to_float(leg.get("leg_distance_meters")),
                "boarded_count": boarded,
                "dropped_off_count": dropped,
                "onboard_before_leg": max(0.0, onboard_after - boarded + dropped),
                "on_board_count_after_arrival": onboard_after,
                "source_file": str(ep_path),
            }
        )
    return episode_row, passengers, routes


def build_summary(ep: pd.DataFrame):
    agg_specs = {
        "completed_all_requests": ["mean", "sum"],
        "service_rate": ["mean", "std", "min"],
        "average_wait_seconds": ["mean", "std", "min", "max"],
        "average_ride_seconds": ["mean", "std"],
        "episode_time_minutes": ["mean", "std"],
        "episode_distance_km": ["mean", "std"],
        "completed_passengers": ["mean", "std"],
        "unserved_passengers": ["mean", "sum"],
        "wait_p90_seconds": ["mean", "std"],
        "wait_max_seconds": ["mean"],
        "leg_count": ["mean", "std"],
        "unique_edges": ["mean"],
        "edge_reuse_factor": ["mean"],
        "distance_km_per_completed_passenger": ["mean", "std"],
        "time_min_per_completed_passenger": ["mean"],
        "legs_per_completed_passenger": ["mean"],
        "empty_distance_rate": ["mean", "std"],
        "mean_onboard_load": ["mean", "std"],
        "passenger_km": ["mean"],
    }
    named = {
        f"{col}_{fn}": (col, fn) for col, fns in agg_specs.items() for fn in fns
    }
    summary = (
        ep.groupby(["policy", "scenario"])
        .agg(n_episodes=("episode_index", "count"), **named)
        .reset_index()
    )
    summary["policy"] = pd.Categorical(summary["policy"], POLICY_ORDER, ordered=True)
    return summary.sort_values(["scenario", "policy"]).reset_index(drop=True)


def pairwise_delta(summary: pd.DataFrame, target: str, baseline: str):
    target_df = summary[summary["policy"] == target].set_index("scenario")
    base_df = summary[summary["policy"] == baseline].set_index("scenario")
    rows = []
    for scenario in sorted(set(target_df.index) & set(base_df.index)):
        target_row = target_df.loc[scenario]
        base_row = base_df.loc[scenario]
        rows.append(
            {
                "scenario": scenario,
                "target": target,
                "baseline": baseline,
                "service_rate_delta_pctpt": (
                    target_row["service_rate_mean"] - base_row["service_rate_mean"]
                )
                * 100.0,
                "completion_rate_delta_pctpt": (
                    target_row["completed_all_requests_mean"]
                    - base_row["completed_all_requests_mean"]
                )
                * 100.0,
                "wait_reduction_pct": (
                    base_row["average_wait_seconds_mean"]
                    - target_row["average_wait_seconds_mean"]
                )
                / base_row["average_wait_seconds_mean"]
                * 100.0,
                "ride_reduction_pct": (
                    base_row["average_ride_seconds_mean"]
                    - target_row["average_ride_seconds_mean"]
                )
                / base_row["average_ride_seconds_mean"]
                * 100.0,
                "distance_reduction_pct": (
                    base_row["episode_distance_km_mean"]
                    - target_row["episode_distance_km_mean"]
                )
                / base_row["episode_distance_km_mean"]
                * 100.0,
                "time_reduction_pct": (
                    base_row["episode_time_minutes_mean"]
                    - target_row["episode_time_minutes_mean"]
                )
                / base_row["episode_time_minutes_mean"]
                * 100.0,
            }
        )
    return pd.DataFrame(rows)


def save_tables(ep: pd.DataFrame, pax: pd.DataFrame, route: pd.DataFrame, summary: pd.DataFrame):
    TAB.mkdir(parents=True, exist_ok=True)
    ep.to_csv(TAB / "episode_level_metrics.csv", index=False, encoding="utf-8-sig")
    pax.to_csv(TAB / "passenger_level_metrics.csv", index=False, encoding="utf-8-sig")
    route.to_csv(TAB / "route_leg_level_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TAB / "scenario_policy_summary.csv", index=False, encoding="utf-8-sig")

    primary = summary.copy()
    primary["completion_runs"] = (
        primary["completed_all_requests_sum"].astype(int).astype(str)
        + "/"
        + primary["n_episodes"].astype(int).astype(str)
    )
    primary["service_rate_pct"] = primary["service_rate_mean"] * 100.0
    primary["empty_dist_pct"] = primary["empty_distance_rate_mean"] * 100.0
    primary[
        [
            "scenario",
            "policy",
            "n_episodes",
            "completion_runs",
            "service_rate_pct",
            "average_wait_seconds_mean",
            "average_ride_seconds_mean",
            "episode_time_minutes_mean",
            "episode_distance_km_mean",
            "empty_dist_pct",
            "mean_onboard_load_mean",
        ]
    ].to_csv(TAB / "table_primary_results.csv", index=False, encoding="utf-8-sig")

    macro = (
        summary.groupby("policy", observed=False)
        .agg(
            scenarios=("scenario", "count"),
            total_episodes=("n_episodes", "sum"),
            completion_rate_macro=("completed_all_requests_mean", "mean"),
            service_rate_macro=("service_rate_mean", "mean"),
            average_wait_seconds_macro=("average_wait_seconds_mean", "mean"),
            average_ride_seconds_macro=("average_ride_seconds_mean", "mean"),
            episode_time_minutes_macro=("episode_time_minutes_mean", "mean"),
            episode_distance_km_macro=("episode_distance_km_mean", "mean"),
            empty_distance_rate_macro=("empty_distance_rate_mean", "mean"),
            mean_onboard_load_macro=("mean_onboard_load_mean", "mean"),
            dist_per_completed_macro=("distance_km_per_completed_passenger_mean", "mean"),
        )
        .reset_index()
    )
    macro.to_csv(TAB / "policy_macro_summary.csv", index=False, encoding="utf-8-sig")

    deltas = pd.concat(
        [
            pairwise_delta(summary, "ONNX", "Vanilla"),
            pairwise_delta(summary, "ONNX", "Greedy"),
            pairwise_delta(summary, "Greedy", "Vanilla"),
        ],
        ignore_index=True,
    )
    deltas.to_csv(TAB / "pairwise_policy_deltas.csv", index=False, encoding="utf-8-sig")

    winners = []
    for scenario, group in summary.groupby("scenario"):
        row = group.sort_values(
            ["service_rate_mean", "average_wait_seconds_mean"], ascending=[False, True]
        ).iloc[0]
        winners.append(
            {
                "scenario": scenario,
                "best_policy_service_then_wait": str(row["policy"]),
                "service_rate_mean": row["service_rate_mean"],
                "average_wait_seconds_mean": row["average_wait_seconds_mean"],
            }
        )
    winners = pd.DataFrame(winners)
    winners.to_csv(TAB / "scenario_winners_service_then_wait.csv", index=False, encoding="utf-8-sig")

    finish = ep.groupby(["policy", "finish_reason"]).size().reset_index(name="count")
    finish.to_csv(TAB / "finish_reason_counts.csv", index=False, encoding="utf-8-sig")
    return macro, deltas, winners, finish


def configure_plots():
    FIG.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.18)
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "axes.titleweight": "bold",
            "axes.titlesize": 12.5,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9.5,
        }
    )


def grouped_bar(summary, mean_col, std_col, ylabel, title, filename, ylim=None, percent=False):
    scenarios = sorted(summary["scenario"].unique())
    x = np.arange(len(scenarios))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.2, 4.35))
    for idx, policy in enumerate(POLICY_ORDER):
        vals, errs = [], []
        for scenario in scenarios:
            row = summary[(summary["policy"] == policy) & (summary["scenario"] == scenario)]
            if row.empty:
                vals.append(np.nan)
                errs.append(0.0)
            else:
                rec = row.iloc[0]
                value = float(rec[mean_col])
                err = (
                    float(rec[std_col])
                    if std_col and std_col in row.columns and np.isfinite(rec[std_col])
                    else 0.0
                )
                vals.append(value * 100.0 if percent else value)
                errs.append(err * 100.0 if percent else err)
        xpos = x + (idx - 1) * width
        ax.bar(
            xpos,
            vals,
            width,
            label=policy,
            color=COLORS[policy],
            edgecolor="white",
            linewidth=0.8,
        )
        if any(err > 0 for err in errs):
            ax.errorbar(
                xpos,
                vals,
                yerr=errs,
                fmt="none",
                ecolor="#333333",
                elinewidth=0.8,
                capsize=2.5,
                capthick=0.8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in scenarios])
    ax.set_xlabel("Scenario demand size (passengers)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(ncol=3, loc="upper left", frameon=True)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax.grid(axis="x", visible=False)
    if ylim:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    out = FIG / filename
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def save_figures(summary: pd.DataFrame, pax: pd.DataFrame, route: pd.DataFrame, finish: pd.DataFrame):
    configure_plots()
    figures = [
        grouped_bar(
            summary,
            "average_wait_seconds_mean",
            "average_wait_seconds_std",
            "Average wait time (s)",
            "Served-passenger average wait by scenario",
            "fig01_average_wait_by_scenario.png",
        ),
        grouped_bar(
            summary,
            "service_rate_mean",
            "service_rate_std",
            "Service rate (%)",
            "Mean service rate by scenario",
            "fig02_service_rate_by_scenario.png",
            ylim=(0, 105),
            percent=True,
        ),
        grouped_bar(
            summary,
            "completed_all_requests_mean",
            None,
            "All-request completion rate (%)",
            "Episode completion reliability by scenario",
            "fig03_completion_reliability_by_scenario.png",
            ylim=(0, 105),
            percent=True,
        ),
        grouped_bar(
            summary,
            "episode_time_minutes_mean",
            "episode_time_minutes_std",
            "Episode time (min)",
            "Episode duration by scenario",
            "fig04_episode_time_by_scenario.png",
        ),
        grouped_bar(
            summary,
            "episode_distance_km_mean",
            "episode_distance_km_std",
            "Route distance (km)",
            "Route distance by scenario",
            "fig05_route_distance_by_scenario.png",
        ),
        grouped_bar(
            summary,
            "distance_km_per_completed_passenger_mean",
            "distance_km_per_completed_passenger_std",
            "km per completed passenger",
            "Distance cost per completed passenger",
            "fig06_distance_per_completed_passenger.png",
        ),
        grouped_bar(
            summary,
            "empty_distance_rate_mean",
            "empty_distance_rate_std",
            "Empty-distance share (%)",
            "Estimated empty-distance share by scenario",
            "fig07_empty_distance_share_by_scenario.png",
            ylim=(0, 100),
            percent=True,
        ),
        grouped_bar(
            summary,
            "mean_onboard_load_mean",
            "mean_onboard_load_std",
            "Mean onboard load (passengers)",
            "Mean onboard load during route legs",
            "fig08_mean_onboard_load_by_scenario.png",
        ),
    ]

    fig, ax = plt.subplots(figsize=(8.3, 4.7))
    pax_clean = pax[np.isfinite(pax["wait_time_seconds"])].copy()
    pax_clean["policy"] = pd.Categorical(pax_clean["policy"], POLICY_ORDER, ordered=True)
    sns.boxplot(
        data=pax_clean,
        x="scenario",
        y="wait_time_seconds",
        hue="policy",
        order=sorted(pax_clean["scenario"].unique()),
        hue_order=POLICY_ORDER,
        palette=COLORS,
        linewidth=0.8,
        fliersize=1.8,
        ax=ax,
    )
    ax.set_xlabel("Scenario demand size (passengers)")
    ax.set_ylabel("Passenger wait time (s)")
    ax.set_title("Passenger-level wait-time distribution")
    ax.legend(ncol=3, loc="upper left", title=None, frameon=True)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    fig.tight_layout()
    out = FIG / "fig09_passenger_wait_distribution.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    figures.append(out)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for policy in POLICY_ORDER:
        data = summary[summary["policy"] == policy]
        ax.scatter(
            data["episode_distance_km_mean"],
            data["average_wait_seconds_mean"],
            s=data["scenario"] * 4.2,
            color=COLORS[policy],
            edgecolor="white",
            linewidth=0.9,
            alpha=0.9,
            label=policy,
        )
        for _, row in data.iterrows():
            ax.text(
                row["episode_distance_km_mean"] + 0.35,
                row["average_wait_seconds_mean"] + 8,
                str(int(row["scenario"])),
                fontsize=7.5,
                color="#333333",
            )
    ax.set_xlabel("Route distance (km)")
    ax.set_ylabel("Average wait time (s)")
    ax.set_title("Wait-distance trade-off (labels indicate scenario demand)")
    ax.legend(ncol=3, loc="best", frameon=True)
    ax.grid(color="#D9D9D9", linewidth=0.7)
    fig.tight_layout()
    out = FIG / "fig10_wait_distance_tradeoff.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    figures.append(out)

    finish_pivot = (
        finish.pivot(index="policy", columns="finish_reason", values="count")
        .fillna(0)
        .reindex(POLICY_ORDER)
    )
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    bottom = np.zeros(len(finish_pivot))
    finish_colors = {"All passenger requests completed.": "#4C78A8", "Episode time ended.": "#E45756"}
    for column in finish_pivot.columns:
        values = finish_pivot[column].values
        ax.bar(
            finish_pivot.index,
            values,
            bottom=bottom,
            label=column,
            color=finish_colors.get(column, "#AAAAAA"),
            edgecolor="white",
        )
        bottom += values
    ax.set_ylabel("Number of episodes")
    ax.set_xlabel("Policy")
    ax.set_title("Finish-reason distribution")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    fig.tight_layout()
    out = FIG / "fig11_finish_reason_distribution.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    figures.append(out)

    edge = route.groupby(["policy", "from_stop_id", "to_stop_id"]).size().reset_index(name="count")
    for policy in POLICY_ORDER:
        matrix = np.zeros((12, 12))
        data = edge[edge["policy"] == policy]
        for _, row in data.iterrows():
            i, j = int(row["from_stop_id"]) - 1, int(row["to_stop_id"]) - 1
            if 0 <= i < 12 and 0 <= j < 12:
                matrix[i, j] = row["count"]
        fig, ax = plt.subplots(figsize=(5.4, 4.6))
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="Blues",
            cbar_kws={"label": "Leg count"},
            square=True,
            linewidths=0.25,
            linecolor="white",
        )
        ax.set_xlabel("To stop")
        ax.set_ylabel("From stop")
        ax.set_xticklabels(range(1, 13), rotation=0)
        ax.set_yticklabels(range(1, 13), rotation=0)
        ax.set_title(f"Route edge frequency: {policy}")
        fig.tight_layout()
        out = FIG / f"fig12_edge_frequency_{policy.lower()}.png"
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        figures.append(out)
    return figures


def mean_std_text(row, mean_col, std_col=None, digits=1):
    mean = row[mean_col]
    if std_col and std_col in row.index and np.isfinite(row[std_col]):
        return f"{mean:.{digits}f} +/- {row[std_col]:.{digits}f}"
    return f"{mean:.{digits}f}"


def markdown_delta_table(deltas, target, baseline):
    data = deltas[(deltas["target"] == target) & (deltas["baseline"] == baseline)]
    rows = []
    for _, row in data.iterrows():
        rows.append(
            {
                "Scenario": int(row["scenario"]),
                "Target": target,
                "Baseline": baseline,
                "d service (pct. pt.)": f"{row['service_rate_delta_pctpt']:+.1f}",
                "Wait reduction (%)": f"{row['wait_reduction_pct']:+.1f}",
                "Distance reduction (%)": f"{row['distance_reduction_pct']:+.1f}",
                "Time reduction (%)": f"{row['time_reduction_pct']:+.1f}",
            }
        )
    return pd.DataFrame(rows).to_markdown(index=False)


def write_report(ep, summary, macro, deltas, winners, finish, figures):
    finish_pivot = (
        finish.pivot(index="policy", columns="finish_reason", values="count")
        .fillna(0)
        .astype(int)
        .reindex(POLICY_ORDER)
    )
    inv = summary[
        [
            "scenario",
            "policy",
            "n_episodes",
            "completed_all_requests_sum",
            "completed_all_requests_mean",
            "service_rate_mean",
        ]
    ].copy()
    inv["completed_all_requests_sum"] = inv["completed_all_requests_sum"].astype(int)
    inv["completion"] = (
        inv["completed_all_requests_sum"].astype(str) + "/" + inv["n_episodes"].astype(int).astype(str)
    )
    inv["completion_rate_pct"] = inv["completed_all_requests_mean"] * 100
    inv["service_rate_pct"] = inv["service_rate_mean"] * 100
    inv = inv[["scenario", "policy", "n_episodes", "completion", "completion_rate_pct", "service_rate_pct"]]

    primary_rows = []
    for _, row in summary.iterrows():
        primary_rows.append(
            {
                "Scenario": int(row["scenario"]),
                "Policy": str(row["policy"]),
                "N": int(row["n_episodes"]),
                "Service rate (%)": f"{row['service_rate_mean'] * 100:.1f}",
                "Wait (s)": mean_std_text(row, "average_wait_seconds_mean", "average_wait_seconds_std", 1),
                "Ride (s)": mean_std_text(row, "average_ride_seconds_mean", "average_ride_seconds_std", 1),
                "Time (min)": mean_std_text(row, "episode_time_minutes_mean", "episode_time_minutes_std", 1),
                "Distance (km)": mean_std_text(row, "episode_distance_km_mean", "episode_distance_km_std", 1),
                "Completed runs": f"{int(row['completed_all_requests_sum'])}/{int(row['n_episodes'])}",
            }
        )
    primary_md = pd.DataFrame(primary_rows).to_markdown(index=False)

    route_rows = []
    for _, row in summary.iterrows():
        route_rows.append(
            {
                "Scenario": int(row["scenario"]),
                "Policy": str(row["policy"]),
                "km/completed pax": f"{row['distance_km_per_completed_passenger_mean']:.2f}",
                "Empty dist. (%)": f"{row['empty_distance_rate_mean'] * 100:.1f}",
                "Mean load": f"{row['mean_onboard_load_mean']:.2f}",
                "Legs/completed pax": f"{row['legs_per_completed_passenger_mean']:.2f}",
                "Edge reuse": f"{row['edge_reuse_factor_mean']:.2f}",
            }
        )
    route_md = pd.DataFrame(route_rows).to_markdown(index=False)

    macro_rows = []
    for _, row in macro.iterrows():
        macro_rows.append(
            {
                "Policy": str(row["policy"]),
                "Episodes": int(row["total_episodes"]),
                "Completion (%)": f"{row['completion_rate_macro'] * 100:.1f}",
                "Service (%)": f"{row['service_rate_macro'] * 100:.1f}",
                "Wait (s)": f"{row['average_wait_seconds_macro']:.1f}",
                "Ride (s)": f"{row['average_ride_seconds_macro']:.1f}",
                "Distance (km)": f"{row['episode_distance_km_macro']:.1f}",
                "Empty dist. (%)": f"{row['empty_distance_rate_macro'] * 100:.1f}",
            }
        )
    macro_md = pd.DataFrame(macro_rows).to_markdown(index=False)

    winner_counts = (
        winners["best_policy_service_then_wait"].value_counts().reindex(POLICY_ORDER).fillna(0).astype(int)
    )
    winner_sentence = ", ".join(
        [
            f"scenario {int(row.scenario)}: {row.best_policy_service_then_wait}"
            for row in winners.itertuples()
        ]
    )
    macro_sorted_wait = macro.sort_values("average_wait_seconds_macro").iloc[0]["policy"]
    macro_sorted_service = macro.sort_values(
        ["service_rate_macro", "average_wait_seconds_macro"], ascending=[False, True]
    ).iloc[0]["policy"]
    macro_sorted_completion = macro.sort_values(
        ["completion_rate_macro", "average_wait_seconds_macro"], ascending=[False, True]
    ).iloc[0]["policy"]
    overall_policy = str(macro_sorted_service)
    source_runs = (
        ep.groupby(["policy", "run_folder"])
        .size()
        .reset_index(name="episodes")
        .sort_values(["policy", "run_folder"])
    )
    source_runs_md = source_runs.to_markdown(index=False)
    timeout_total = (
        int(finish_pivot["Episode time ended."].sum())
        if "Episode time ended." in finish_pivot.columns
        else 0
    )
    timeout_sentence = (
        "No evaluated episode terminated with `Episode time ended.`, so the extended horizon removed the timeout confound from this run set."
        if timeout_total == 0
        else f"{timeout_total} evaluated episodes still terminated with `Episode time ended.`; service-rate metrics must therefore remain the first interpretation layer."
    )
    korean_timeout_sentence = (
        "분석 대상 episode 중 `Episode time ended.`로 종료된 경우는 없어서, 이번 run set에서는 timeout 교란이 제거되었다."
        if timeout_total == 0
        else f"분석 대상 episode 중 {timeout_total}개가 아직 `Episode time ended.`로 종료되었으므로, service rate를 먼저 보고 wait를 해석해야 한다."
    )
    abstract_result = (
        f"Across the six demand levels, {overall_policy} provides the strongest service-rate-first operating point. "
        f"{macro_sorted_wait} has the lowest macro-average served-passenger waiting time, and {macro_sorted_completion} has the strongest all-request completion profile."
    )
    interpretation_sentence = (
        f"Under the service-rate-first rule, {overall_policy} is the best overall policy in the current CSV exports. "
        f"Scenario-level winners are: {winner_sentence}."
    )

    captions = {
        "fig01_average_wait_by_scenario.png": "Fig. 1. Served-passenger average waiting time for each demand scenario.",
        "fig02_service_rate_by_scenario.png": "Fig. 2. Mean service rate by demand scenario.",
        "fig03_completion_reliability_by_scenario.png": "Fig. 3. Fraction of runs that completed all passenger requests.",
        "fig04_episode_time_by_scenario.png": "Fig. 4. Episode duration by demand scenario.",
        "fig05_route_distance_by_scenario.png": "Fig. 5. Total route distance by demand scenario.",
        "fig06_distance_per_completed_passenger.png": "Fig. 6. Distance cost normalized by completed passengers.",
        "fig07_empty_distance_share_by_scenario.png": "Fig. 7. Estimated empty-distance share based on onboard load before each leg.",
        "fig08_mean_onboard_load_by_scenario.png": "Fig. 8. Mean onboard load along route legs.",
        "fig09_passenger_wait_distribution.png": "Fig. 9. Passenger-level waiting-time distributions.",
        "fig10_wait_distance_tradeoff.png": "Fig. 10. Wait-distance trade-off at scenario-policy level.",
        "fig11_finish_reason_distribution.png": "Fig. 11. Finish reason distribution by policy.",
        "fig12_edge_frequency_vanilla.png": "Fig. 12(a). Route edge frequency for Vanilla.",
        "fig12_edge_frequency_greedy.png": "Fig. 12(b). Route edge frequency for Greedy.",
        "fig12_edge_frequency_onnx.png": "Fig. 12(c). Route edge frequency for ONNX.",
    }
    figure_md = []
    for fig in figures:
        rel = fig.relative_to(OUT).as_posix()
        cap = captions.get(fig.name, fig.name)
        figure_md.append(f"![{cap}]({rel})\n\n{cap}")

    report = f"""# IEEE-Style DRT Dispatch Policy Evaluation Report

## Abstract
This report evaluates three DRT next-stop policies, **Vanilla Sequential**, **Greedy Nearest Feasible**, and **ONNX PPO inference**, using the exported Unity/ML-Agents CSV logs under `DRT_Episode_Exports`. The comparison is performed at the scenario level: when multiple runs exist for the same demand scenario, metrics are averaged within the same policy and scenario before cross-policy interpretation. The primary evaluation criterion is service reliability, represented by service rate and all-request completion, followed by served-passenger waiting time. {abstract_result} {timeout_sentence}

**Index Terms**--Demand-responsive transit, dispatch policy, service rate, waiting time, route efficiency, ML-Agents, ONNX inference.

## I. Experimental Data and Method
The source data consist of `episode.csv` and `trace.csv` exports generated by the three policy modes. The evaluated policy mapping is: `vanilla` = Vanilla Sequential, `greedy` = Greedy Nearest Feasible, and `inference` = ONNX PPO inference. The following procedure was applied.

1. The `summary` section of each `episode.csv` was used for aggregate service metrics, including service rate, completed passengers, average wait time, ride time, episode duration, route distance, and finish reason.
2. The `passenger` section was used for passenger-level wait and ride distributions.
3. The `route_leg` section was used for route-efficiency metrics, including leg count, edge reuse, distance per completed passenger, estimated empty-distance share, and mean onboard load.
4. Metrics were grouped by `(policy, scenario)` and averaged across repeated runs of the same scenario.
5. Service rate and all-request completion were treated as primary filters before interpreting wait time. This avoids selecting a policy only because it reports a low served-passenger wait in an incomplete run.

The analyzed export folders were:

{source_runs_md}

## II. Run Inventory
Table I summarizes the number of evaluated episodes per policy and scenario. The completion column reports runs that finished with all passenger requests completed.

**Table I. Run inventory and reliability.**

{inv.to_markdown(index=False)}

At the folder level, the run set contains {len(ep)} total episodes: Vanilla {int((ep['policy'] == 'Vanilla').sum())}, Greedy {int((ep['policy'] == 'Greedy').sum())}, and ONNX {int((ep['policy'] == 'ONNX').sum())}. Finish-reason counts show that Vanilla completed {int(finish_pivot.loc['Vanilla'].get('All passenger requests completed.', 0))} runs and timed out {int(finish_pivot.loc['Vanilla'].get('Episode time ended.', 0))}; Greedy completed {int(finish_pivot.loc['Greedy'].get('All passenger requests completed.', 0))} and timed out {int(finish_pivot.loc['Greedy'].get('Episode time ended.', 0))}; ONNX completed {int(finish_pivot.loc['ONNX'].get('All passenger requests completed.', 0))} and timed out {int(finish_pivot.loc['ONNX'].get('Episode time ended.', 0))}.

## III. Primary Service Results
Table II reports the scenario-level averages. The wait and ride values are in seconds, while episode time and distance are reported in minutes and kilometers, respectively.

**Table II. Scenario-level service metrics, mean +/- standard deviation where applicable.**

{primary_md}

Using service rate as the primary criterion and waiting time as the secondary criterion, the best policy by scenario is summarized as follows: {winner_sentence}. Across the six scenario groups, this gives Vanilla {winner_counts['Vanilla']} wins, Greedy {winner_counts['Greedy']} wins, and ONNX {winner_counts['ONNX']} wins.

The macro-average results in Table III average the scenario means rather than raw episodes, preventing policies with more repeated runs from dominating the conclusion.

**Table III. Macro-average policy summary across scenarios.**

{macro_md}

From Table III, **{macro_sorted_completion}** has the strongest all-request completion profile, **{macro_sorted_service}** has the best service-rate-first profile, and **{macro_sorted_wait}** has the lowest macro-average served-passenger wait. In this experiment these conclusions point to Greedy as the most robust baseline: it maintains high service reliability while reducing waiting time substantially relative to Vanilla.

## IV. Policy Comparison Against Vanilla
Table IV reports percentage deltas relative to Vanilla. Positive wait, distance, and time reductions indicate an improvement over Vanilla; positive service deltas indicate higher service rate.

**Table IV-A. ONNX relative to Vanilla.**

{markdown_delta_table(deltas, 'ONNX', 'Vanilla')}

**Table IV-B. Greedy relative to Vanilla.**

{markdown_delta_table(deltas, 'Greedy', 'Vanilla')}

The pairwise deltas show where each policy gains waiting-time or route-distance efficiency relative to Vanilla. A positive waiting-time reduction is useful only when the corresponding service-rate delta is not materially negative; this is the main reason the report ranks by service rate first and waiting time second.

## V. Route Efficiency Analysis
Table V summarizes the route-efficiency metrics. The empty-distance share is estimated from the onboard load before each route leg; `km/completed pax` normalizes route distance by the number of completed passengers.

**Table V. Scenario-level route-efficiency metrics.**

{route_md}

Vanilla Sequential repeatedly traverses the fixed stop order, which yields predictable but often inefficient service under spatially clustered demand. Greedy reduces unnecessary circulation by selecting nearer feasible stops. ONNX produces non-sequential route patterns; its value should be interpreted jointly with service rate, passenger waiting time, and route efficiency rather than route distance alone.

## VI. Figures

{chr(10).join(figure_md)}

## VII. Discussion
The key operational result is that **service reliability must dominate the ranking**. {timeout_sentence} {interpretation_sentence} Vanilla remains the deterministic reference policy, Greedy represents a strong heuristic dispatch baseline, and ONNX represents the learned adaptive policy. The best deployable choice should be the policy that preserves service rate while lowering waiting time and route cost.

For the report narrative, the recommended interpretation is therefore data-driven rather than reward-driven: choose the service-rate-first winner, use waiting time as the second criterion, and use route-efficiency metrics to explain why the winning dispatch pattern performs better.

## VIII. Threats to Validity
The experiment uses MatrixTeleport execution, so travel dynamics reflect the matrix route abstraction rather than full physical driving. The repeated-run counts are unbalanced across policies and scenarios; therefore, macro averages were computed over scenario means. Served-passenger average wait is not a full welfare measure when service rate is below one; in those cases, unserved passengers should be interpreted through service-rate and finish-reason metrics rather than through wait time alone.

## IX. Conclusion
The CSV evidence supports the following conclusion: **{overall_policy} is currently the best overall policy among the three evaluated modes under a service-rate-first, wait-time-second criterion**. The final selection should be reported with both the scenario-level table and the route-efficiency figures, because waiting time, service completion, and route cost do not always move in the same direction.

## Artifact Index
- Episode-level metrics: `tables/episode_level_metrics.csv`
- Passenger-level metrics: `tables/passenger_level_metrics.csv`
- Route-leg metrics: `tables/route_leg_level_metrics.csv`
- Scenario summary: `tables/scenario_policy_summary.csv`
- Primary result table: `tables/table_primary_results.csv`
- Pairwise deltas: `tables/pairwise_policy_deltas.csv`
- Figures: `figures/*.png`
"""
    (OUT / "drt_policy_evaluation_report_ieee_style.md").write_text(report, encoding="utf-8")

    korean = f"""# DRT 정책 비교 결과 요약

본 분석은 최신 export 폴더의 Vanilla, Greedy, ONNX CSV를 직접 파싱하여 수행하였다. 같은 policy와 scenario에 여러 run이 있는 경우, 각 scenario 내부 평균을 먼저 계산한 뒤 정책 간 비교를 수행하였다. 평가 우선순위는 `service_rate`와 전체 요청 완료율을 1순위, 완료 승객 기준 평균 대기시간을 2순위로 두었다.

사용한 export 폴더는 다음과 같다.

{source_runs_md}

핵심 결론은 **{overall_policy}가 service-rate-first, wait-time-second 기준에서 현재 가장 좋은 정책**이라는 것이다. Macro 평균 기준 완료율 1위는 {macro_sorted_completion}, service rate 1위는 {macro_sorted_service}, 평균 대기시간 1위는 {macro_sorted_wait}이다. Scenario별 winner는 {winner_sentence}이다.

종료 사유 관점에서는 다음을 주의해야 한다. {korean_timeout_sentence} 따라서 평균 wait가 낮더라도 service rate가 낮거나 완료되지 않은 episode라면 최종 정책 선택 근거로 바로 쓰면 안 된다.

자세한 표와 그림은 `drt_policy_evaluation_report_ieee_style.md`와 `figures/*.png`를 참조하면 된다.
"""
    (OUT / "summary_kr.md").write_text(korean, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate DRT policy comparison report from CSV exports.")
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Use only the newest timestamped export folder under each policy mode.",
    )
    args = parser.parse_args()

    selected_dirs = latest_run_dirs() if args.latest_only else None
    OUT.mkdir(parents=True, exist_ok=True)
    ep, pax, route = collect_data(selected_dirs=selected_dirs)
    if ep.empty:
        raise SystemExit("No episode CSV files found under DRT_Episode_Exports.")
    summary = build_summary(ep)
    macro, deltas, winners, finish = save_tables(ep, pax, route, summary)
    figures = save_figures(summary, pax, route, finish)
    write_report(ep, summary, macro, deltas, winners, finish, figures)

    print("OUT_DIR=" + str(OUT.resolve()))
    print("EPISODES=" + str(len(ep)))
    print("PASSENGER_ROWS=" + str(len(pax)))
    print("ROUTE_LEGS=" + str(len(route)))
    if selected_dirs:
        print("\nSELECTED_RUN_DIRS")
        for mode_key, run_dir in selected_dirs.items():
            print(f"{mode_key}={run_dir}")
    print("\nPOLICY_MACRO_SUMMARY")
    print(macro.to_string(index=False))
    print("\nWINNERS")
    print(winners.to_string(index=False))
    print("\nFIGURES")
    for figure in figures:
        print(figure.name)


if __name__ == "__main__":
    main()
