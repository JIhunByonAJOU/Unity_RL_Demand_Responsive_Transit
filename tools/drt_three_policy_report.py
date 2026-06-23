#!/usr/bin/env python3
"""Generate a three-policy DRT report from Matrix Teleport episode CSV exports."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


POLICY_ORDER = ["ONNX Inference", "Greedy 1", "Vanilla Sequential"]
POLICY_DIRS = {
    "ONNX Inference": "inference",
    "Greedy 1": "greedy",
    "Vanilla Sequential": "vanilla",
}
POLICY_COLORS = {
    "ONNX Inference": "#005AB5",
    "Greedy 1": "#009E73",
    "Vanilla Sequential": "#D55E00",
}
LOWER_IS_BETTER = {
    "episode_time_seconds",
    "episode_distance_meters",
    "mean_wait_seconds",
    "p95_wait_seconds",
    "max_wait_seconds",
    "mean_ride_completed_seconds",
    "route_leg_count",
    "no_interaction_count",
    "no_interaction_rate",
}
HIGHER_IS_BETTER = {
    "service_rate",
    "completed_passengers",
    "completed_all_requests",
}


@dataclass
class EpisodeBundle:
    policy: str
    path: Path
    summary: dict[str, str]
    route_rows: list[dict[str, str]]
    passenger_rows: list[dict[str, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a DRT ONNX vs Greedy vs Vanilla PDF report.")
    parser.add_argument("--exports-root", default="DRT_Episode_Exports")
    parser.add_argument("--output", default="output/pdf/drt_inference_greedy_vanilla_report.pdf")
    parser.add_argument("--figures-dir", default="output/pdf/drt_inference_greedy_vanilla_figures")
    parser.add_argument("--summary-csv", default="output/pdf/drt_inference_greedy_vanilla_summary.csv")
    parser.add_argument("--run-metrics-csv", default="output/pdf/drt_inference_greedy_vanilla_run_metrics.csv")
    parser.add_argument("--passenger-csv", default="output/pdf/drt_inference_greedy_vanilla_passenger_metrics.csv")
    parser.add_argument("--runs-per-policy", type=int, default=0, help="0 means use all runs in the selected latest folder.")
    return parser.parse_args()


def clean_row(row: list[str]) -> list[str]:
    return [cell.strip() for cell in row]


def to_float(value: object, default: float | None = None) -> float | None:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def to_int(value: object, default: int = 0) -> int:
    parsed = to_float(value)
    if parsed is None or math.isnan(parsed):
        return default
    return int(round(parsed))


def display_policy(raw: str | None) -> str | None:
    token = "".join(ch for ch in (raw or "").lower() if ch.isalnum())
    if token in {"onnxinference", "inference"}:
        return "ONNX Inference"
    if token in {"greedy", "greedynearestfeasible", "greedy1", "greedy1nearestfeasible"}:
        return "Greedy 1"
    if token in {"vanillasequential", "vanilla", "sequential"}:
        return "Vanilla Sequential"
    return raw.strip() if raw else None


def latest_policy_dir(root: Path, policy: str) -> Path:
    policy_root = root / POLICY_DIRS[policy]
    if not policy_root.exists():
        raise FileNotFoundError(f"Missing policy export directory: {policy_root}")
    candidates = [path for path in policy_root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under: {policy_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_episode_csv(path: Path, expected_policy: str) -> EpisodeBundle:
    summary: dict[str, str] = {}
    sections: dict[str, list[dict[str, str]]] = defaultdict(list)
    current_header: list[str] | None = None

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for raw_row in reader:
            row = clean_row(raw_row)
            if not any(row):
                current_header = None
                continue
            if row[0] == "section":
                if len(row) >= 3 and row[1] == "key" and row[2] == "value":
                    current_header = None
                else:
                    current_header = row
                continue
            if row[0] == "summary" and len(row) >= 3:
                summary[row[1]] = row[2]
                continue
            if current_header is None:
                continue
            section_name = row[0]
            mapped = {
                header: row[index] if index < len(row) else ""
                for index, header in enumerate(current_header)
            }
            sections[section_name].append(mapped)

    policy = display_policy(summary.get("next_stop_policy") or summary.get("policy")) or expected_policy
    if policy != expected_policy:
        policy = expected_policy
    return EpisodeBundle(
        policy=policy,
        path=path,
        summary=summary,
        route_rows=sections.get("route_leg", []),
        passenger_rows=sections.get("passenger", []),
    )


def load_policy_runs(root: Path, runs_per_policy: int) -> tuple[list[EpisodeBundle], dict[str, Path]]:
    bundles: list[EpisodeBundle] = []
    selected_dirs: dict[str, Path] = {}
    for policy in POLICY_ORDER:
        run_dir = latest_policy_dir(root, policy)
        selected_dirs[policy] = run_dir
        episode_paths = sorted(run_dir.glob("*_episode.csv"))
        if runs_per_policy > 0:
            episode_paths = episode_paths[-runs_per_policy:]
        for path in episode_paths:
            bundles.append(read_episode_csv(path, policy))
    return bundles, selected_dirs


def numeric_frame(rows: list[dict[str, str]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    for column in frame.columns:
        if column == "section":
            continue
        series = frame[column].astype(str)
        nonempty = series.str.strip().ne("")
        converted = pd.to_numeric(frame[column], errors="coerce")
        if int(converted.notna().sum()) == int(nonempty.sum()):
            frame[column] = converted
    return frame


def repeated_abab_count(stop_sequence: Iterable[int]) -> int:
    stops = [stop for stop in stop_sequence if stop > 0]
    count = 0
    for index in range(len(stops) - 3):
        if stops[index] == stops[index + 2] and stops[index + 1] == stops[index + 3] and stops[index] != stops[index + 1]:
            count += 1
    return count


def passenger_id_list(value: object) -> list[int]:
    text = str(value or "").strip()
    if not text:
        return []
    result: list[int] = []
    for part in text.split("|"):
        parsed = to_int(part, -1)
        if parsed > 0:
            result.append(parsed)
    return result


def summarize_episode(bundle: EpisodeBundle) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    summary = bundle.summary
    route = numeric_frame(bundle.route_rows)
    passengers = numeric_frame(bundle.passenger_rows)
    episode_index = to_int(summary.get("episode_index"), 0)

    if not route.empty:
        boarded_count = pd.to_numeric(route.get("boarded_count", pd.Series(dtype=float)), errors="coerce").fillna(0)
        dropped_count = pd.to_numeric(route.get("dropped_off_count", pd.Series(dtype=float)), errors="coerce").fillna(0)
        route_leg_count = len(route)
        no_interaction_count = int(((boarded_count + dropped_count) <= 0).sum())
        no_interaction_rate = no_interaction_count / route_leg_count if route_leg_count else 0.0
        terminal_on_board = to_int(route.iloc[-1].get("on_board_count"), 0)
        max_on_board = int(pd.to_numeric(route.get("on_board_count", pd.Series(dtype=float)), errors="coerce").max())
        stop_sequence = [to_int(value, -1) for value in route.get("arrived_stop_id", [])]
        abab_count = repeated_abab_count(stop_sequence)
        unique_stops_visited = len({stop for stop in stop_sequence if stop > 0})
    else:
        route_leg_count = 0
        no_interaction_count = 0
        no_interaction_rate = 0.0
        terminal_on_board = 0
        max_on_board = 0
        abab_count = 0
        unique_stops_visited = 0

    if not passengers.empty:
        wait = pd.to_numeric(passengers.get("wait_time_seconds", pd.Series(dtype=float)), errors="coerce").dropna()
        completed = passengers[passengers.get("status", "") == "Completed"] if "status" in passengers.columns else passengers
        completed_wait = pd.to_numeric(completed.get("wait_time_seconds", pd.Series(dtype=float)), errors="coerce").dropna()
        completed_ride = pd.to_numeric(completed.get("ride_time_seconds", pd.Series(dtype=float)), errors="coerce").dropna()
        unfinished = passengers[passengers.get("status", "") != "Completed"] if "status" in passengers.columns else passengers.iloc[0:0]
    else:
        wait = pd.Series(dtype=float)
        completed_wait = pd.Series(dtype=float)
        completed_ride = pd.Series(dtype=float)
        unfinished = pd.DataFrame()

    def percentile(series: pd.Series, q: float) -> float:
        return float(np.percentile(series.to_numpy(dtype=float), q)) if len(series) else float("nan")

    total_passengers = to_int(summary.get("total_passengers"), len(passengers))
    completed_passengers = to_int(summary.get("completed_passengers"), len(passengers) - len(unfinished))

    metrics: dict[str, object] = {
        "policy": bundle.policy,
        "episode_index": episode_index,
        "scenario_id": summary.get("scenario_id", ""),
        "scenario_description": summary.get("scenario_description", ""),
        "source_file": str(bundle.path),
        "finish_reason": summary.get("finish_reason", ""),
        "completed_all_requests": to_int(summary.get("completed_all_requests"), 0),
        "total_passengers": total_passengers,
        "completed_passengers": completed_passengers,
        "unfinished_passengers": max(0, total_passengers - completed_passengers),
        "service_rate": to_float(summary.get("service_rate"), 0.0),
        "unity_average_wait_seconds": to_float(summary.get("average_wait_seconds"), float("nan")),
        "unity_average_ride_seconds": to_float(summary.get("average_ride_seconds"), float("nan")),
        "mean_wait_seconds": float(wait.mean()) if len(wait) else float("nan"),
        "median_wait_seconds": float(wait.median()) if len(wait) else float("nan"),
        "p90_wait_seconds": percentile(wait, 90),
        "p95_wait_seconds": percentile(wait, 95),
        "max_wait_seconds": float(wait.max()) if len(wait) else float("nan"),
        "mean_completed_wait_seconds": float(completed_wait.mean()) if len(completed_wait) else float("nan"),
        "mean_ride_completed_seconds": float(completed_ride.mean()) if len(completed_ride) else float("nan"),
        "median_ride_completed_seconds": float(completed_ride.median()) if len(completed_ride) else float("nan"),
        "p95_ride_completed_seconds": percentile(completed_ride, 95),
        "max_ride_completed_seconds": float(completed_ride.max()) if len(completed_ride) else float("nan"),
        "episode_time_seconds": to_float(summary.get("episode_time_seconds"), float("nan")),
        "episode_distance_meters": to_float(summary.get("episode_distance_meters"), float("nan")),
        "route_leg_count": route_leg_count,
        "no_interaction_count": no_interaction_count,
        "no_interaction_rate": no_interaction_rate,
        "terminal_on_board": terminal_on_board,
        "max_on_board": max_on_board,
        "abab_pattern_count": abab_count,
        "unique_stops_visited": unique_stops_visited,
    }

    for frame in (route, passengers):
        if not frame.empty:
            frame.insert(0, "policy", bundle.policy)
            frame.insert(1, "episode_index", episode_index)
            frame.insert(2, "source_file", str(bundle.path))
    return metrics, route, passengers


def build_dataframes(bundles: list[EpisodeBundle]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    route_frames: list[pd.DataFrame] = []
    passenger_frames: list[pd.DataFrame] = []
    for bundle in bundles:
        metrics, route, passengers = summarize_episode(bundle)
        metric_rows.append(metrics)
        if not route.empty:
            route_frames.append(route)
        if not passengers.empty:
            passenger_frames.append(passengers)
    metrics_df = pd.DataFrame(metric_rows)
    route_df = pd.concat(route_frames, ignore_index=True) if route_frames else pd.DataFrame()
    passenger_df = pd.concat(passenger_frames, ignore_index=True) if passenger_frames else pd.DataFrame()
    return metrics_df, route_df, passenger_df


def aggregate_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_names = [
        "service_rate",
        "completed_passengers",
        "completed_all_requests",
        "episode_time_seconds",
        "episode_distance_meters",
        "mean_wait_seconds",
        "p95_wait_seconds",
        "max_wait_seconds",
        "mean_ride_completed_seconds",
        "route_leg_count",
        "no_interaction_count",
        "terminal_on_board",
        "abab_pattern_count",
    ]
    rows = []
    for policy in POLICY_ORDER:
        subset = metrics_df[metrics_df["policy"] == policy]
        for metric in metric_names:
            values = pd.to_numeric(subset[metric], errors="coerce").dropna()
            rows.append(
                {
                    "policy": policy,
                    "metric": metric,
                    "mean": float(values.mean()) if len(values) else float("nan"),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "min": float(values.min()) if len(values) else float("nan"),
                    "max": float(values.max()) if len(values) else float("nan"),
                    "n": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def agg_value(agg_df: pd.DataFrame, policy: str, metric: str, field: str = "mean") -> float:
    row = agg_df[(agg_df["policy"] == policy) & (agg_df["metric"] == metric)]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][field])


def fmt(value: float | int | None, digits: int = 1, suffix: str = "") -> str:
    if value is None:
        return "-"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric):
        return "-"
    if digits == 0:
        return f"{numeric:,.0f}{suffix}"
    return f"{numeric:,.{digits}f}{suffix}"


def fmt_mean_std(agg_df: pd.DataFrame, policy: str, metric: str, digits: int = 1, suffix: str = "") -> str:
    mean = agg_value(agg_df, policy, metric, "mean")
    std = agg_value(agg_df, policy, metric, "std")
    n = int(agg_value(agg_df, policy, metric, "n"))
    if math.isnan(mean):
        return "-"
    if n > 1 and not math.isclose(std, 0.0, abs_tol=1e-9):
        return f"{fmt(mean, digits, suffix)} +/- {fmt(std, digits, suffix)}"
    return fmt(mean, digits, suffix)


def pct_change(new: float, baseline: float) -> float:
    if math.isnan(new) or math.isnan(baseline) or math.isclose(baseline, 0.0):
        return float("nan")
    return (new - baseline) / baseline


def best_policy_for_metric(agg_df: pd.DataFrame, metric: str) -> str:
    values = {policy: agg_value(agg_df, policy, metric, "mean") for policy in POLICY_ORDER}
    values = {policy: value for policy, value in values.items() if not math.isnan(value)}
    if not values:
        return "-"
    if metric in HIGHER_IS_BETTER:
        return max(values, key=values.get)
    return min(values, key=values.get)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.5,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
        }
    )


def label_bars(ax, bars, digits: int = 0, suffix: str = "") -> None:
    for bar in bars:
        height = bar.get_height()
        if math.isnan(height):
            continue
        ax.annotate(
            fmt(height, digits, suffix),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )


def plot_key_metrics(agg_df: pd.DataFrame, figures_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    items = [
        ("service_rate", "Service rate", "%", 1, lambda v: v * 100),
        ("mean_wait_seconds", "Mean passenger wait", "s", 0, lambda v: v),
        ("mean_ride_completed_seconds", "Mean completed ride", "s", 0, lambda v: v),
        ("episode_time_seconds", "Episode completion time", "s", 0, lambda v: v),
    ]
    x = np.arange(len(POLICY_ORDER))
    for ax, (metric, title, suffix, digits, transform) in zip(axes.flatten(), items):
        means = [transform(agg_value(agg_df, policy, metric, "mean")) for policy in POLICY_ORDER]
        stds = [transform(agg_value(agg_df, policy, metric, "std")) for policy in POLICY_ORDER]
        bars = ax.bar(x, means, yerr=stds, capsize=3, color=[POLICY_COLORS[p] for p in POLICY_ORDER], edgecolor="#222222", linewidth=0.6)
        label_bars(ax, bars, digits, suffix if suffix == "%" else "")
        ax.set_xticks(x)
        ax.set_xticklabels(["ONNX", "Greedy", "Vanilla"], rotation=0)
        ax.set_title(title)
        ax.set_ylabel(suffix)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    fig.suptitle("Fig. 1. Key operational metrics by next-stop policy", fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = figures_dir / "01_key_operational_metrics.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_episode_diagnostics(agg_df: pd.DataFrame, figures_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65))
    items = [
        ("route_leg_count", "Route legs", "legs", 0),
        ("no_interaction_count", "No-interaction stops", "count", 1),
        ("terminal_on_board", "Terminal onboard", "passengers", 1),
    ]
    x = np.arange(len(POLICY_ORDER))
    for ax, (metric, title, ylabel, digits) in zip(axes, items):
        means = [agg_value(agg_df, policy, metric, "mean") for policy in POLICY_ORDER]
        bars = ax.bar(x, means, color=[POLICY_COLORS[p] for p in POLICY_ORDER], edgecolor="#222222", linewidth=0.6)
        label_bars(ax, bars, digits)
        ax.set_xticks(x)
        ax.set_xticklabels(["ONNX", "Greedy", "Vanilla"])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    fig.suptitle("Fig. 2. Route quality and completion diagnostics", fontweight="bold", y=1.03)
    fig.tight_layout()
    path = figures_dir / "02_route_quality_diagnostics.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def policy_passenger_values(passenger_df: pd.DataFrame, value_col: str, completed_only: bool = False) -> list[np.ndarray]:
    values = []
    for policy in POLICY_ORDER:
        subset = passenger_df[passenger_df["policy"] == policy]
        if completed_only and "status" in subset.columns:
            subset = subset[subset["status"] == "Completed"]
        series = pd.to_numeric(subset.get(value_col, pd.Series(dtype=float)), errors="coerce").dropna()
        values.append(series.to_numpy(dtype=float))
    return values


def plot_passenger_distributions(passenger_df: pd.DataFrame, figures_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    configs = [
        ("wait_time_seconds", False, "Passenger wait-time distribution", "Wait time (s)"),
        ("ride_time_seconds", True, "Completed ride-time distribution", "Ride time (s)"),
    ]
    for ax, (column, completed_only, title, ylabel) in zip(axes, configs):
        values = policy_passenger_values(passenger_df, column, completed_only)
        box = ax.boxplot(values, patch_artist=True, tick_labels=["ONNX", "Greedy", "Vanilla"], showmeans=True)
        for patch, policy in zip(box["boxes"], POLICY_ORDER):
            patch.set_facecolor(POLICY_COLORS[policy])
            patch.set_alpha(0.65)
        for median in box["medians"]:
            median.set_color("#111111")
            median.set_linewidth(1.1)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    fig.suptitle("Fig. 3. Passenger-level wait and ride distributions", fontweight="bold", y=1.03)
    fig.tight_layout()
    path = figures_dir / "03_passenger_distributions.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_passenger_by_id(passenger_df: pd.DataFrame, figures_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True)
    configs = [
        ("wait_time_seconds", "Mean wait by passenger id", "Wait time (s)", False),
        ("ride_time_seconds", "Mean completed ride by passenger id", "Ride time (s)", True),
    ]
    for ax, (column, title, ylabel, completed_only) in zip(axes, configs):
        for policy in POLICY_ORDER:
            subset = passenger_df[passenger_df["policy"] == policy]
            if completed_only and "status" in subset.columns:
                subset = subset[subset["status"] == "Completed"]
            if subset.empty:
                continue
            grouped = (
                subset.assign(**{column: pd.to_numeric(subset[column], errors="coerce")})
                .groupby("passenger_id")[column]
                .mean()
                .sort_index()
            )
            ax.plot(grouped.index, grouped.values, marker="o", markersize=3, linewidth=1.2, label=policy, color=POLICY_COLORS[policy])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[-1].set_xlabel("Passenger id")
    axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=15))
    axes[0].legend(loc="upper left", ncol=3, frameon=False)
    fig.suptitle("Fig. 4. Passenger-by-passenger service profile", fontweight="bold", y=1.01)
    fig.tight_layout()
    path = figures_dir / "04_passenger_by_id_profile.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def sample_completion_curve(route_df: pd.DataFrame, policy: str, time_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subset = route_df[route_df["policy"] == policy]
    curves = []
    for _, episode_route in subset.groupby("episode_index"):
        episode_route = episode_route.sort_values("arrival_time_seconds")
        times = pd.to_numeric(episode_route["arrival_time_seconds"], errors="coerce").to_numpy(dtype=float)
        completed = pd.to_numeric(episode_route["completed_passenger_count"], errors="coerce").to_numpy(dtype=float)
        values = np.zeros_like(time_grid)
        cursor = 0
        last = 0.0
        for index, t in enumerate(time_grid):
            while cursor < len(times) and times[cursor] <= t:
                last = completed[cursor]
                cursor += 1
            values[index] = last
        curves.append(values)
    if not curves:
        empty = np.zeros_like(time_grid)
        return empty, empty, empty
    matrix = np.vstack(curves)
    return np.median(matrix, axis=0), np.percentile(matrix, 25, axis=0), np.percentile(matrix, 75, axis=0)


def plot_completion_progression(route_df: pd.DataFrame, metrics_df: pd.DataFrame, figures_dir: Path) -> Path:
    max_time = pd.to_numeric(metrics_df["episode_time_seconds"], errors="coerce").max()
    time_grid = np.linspace(0, max_time * 1.02, 240)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for policy in POLICY_ORDER:
        median, low, high = sample_completion_curve(route_df, policy, time_grid)
        ax.plot(time_grid, median, label=policy, color=POLICY_COLORS[policy], linewidth=1.8)
        ax.fill_between(time_grid, low, high, color=POLICY_COLORS[policy], alpha=0.12, linewidth=0)
    ax.set_title("Median completed passengers over episode time")
    ax.set_xlabel("Episode time (s)")
    ax.set_ylabel("Completed passengers")
    ax.set_ylim(0, 31)
    ax.legend(loc="lower right", frameon=True, fontsize=8)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
    fig.suptitle("Fig. 5. Completion progression", fontweight="bold", y=1.03)
    fig.tight_layout()
    path = figures_dir / "05_completion_progression.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_stop_activity(route_df: pd.DataFrame, figures_dir: Path) -> Path:
    records = []
    for policy in POLICY_ORDER:
        subset = route_df[route_df["policy"] == policy].copy()
        if subset.empty:
            continue
        subset["boarded_count"] = pd.to_numeric(subset["boarded_count"], errors="coerce").fillna(0)
        subset["dropped_off_count"] = pd.to_numeric(subset["dropped_off_count"], errors="coerce").fillna(0)
        subset["arrived_stop_id"] = pd.to_numeric(subset["arrived_stop_id"], errors="coerce").fillna(-1).astype(int)
        per_episode = subset.groupby(["episode_index", "arrived_stop_id"])[["boarded_count", "dropped_off_count"]].sum().reset_index()
        per_episode["interactions"] = per_episode["boarded_count"] + per_episode["dropped_off_count"]
        for stop in sorted(per_episode["arrived_stop_id"].unique()):
            if stop < 1:
                continue
            stop_values = per_episode[per_episode["arrived_stop_id"] == stop]["interactions"]
            records.append({"policy": policy, "stop": stop, "interactions": float(stop_values.mean())})
    data = pd.DataFrame(records)
    stops = sorted(data["stop"].unique()) if not data.empty else []
    heat = np.zeros((len(POLICY_ORDER), len(stops)))
    for row_index, policy in enumerate(POLICY_ORDER):
        for col_index, stop in enumerate(stops):
            value = data[(data["policy"] == policy) & (data["stop"] == stop)]["interactions"]
            heat[row_index, col_index] = float(value.iloc[0]) if len(value) else 0.0
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    im = ax.imshow(heat, cmap="Blues", aspect="auto")
    ax.set_yticks(np.arange(len(POLICY_ORDER)))
    ax.set_yticklabels(["ONNX", "Greedy", "Vanilla"])
    ax.set_xticks(np.arange(len(stops)))
    ax.set_xticklabels([str(stop) for stop in stops])
    ax.set_xlabel("Stop id")
    ax.set_title("Mean board + drop interactions per stop and episode")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, fmt(heat[i, j], 1), ha="center", va="center", fontsize=6, color="#111111")
    fig.colorbar(im, ax=ax, fraction=0.032, pad=0.02, label="Interactions")
    fig.suptitle("Fig. 6. Stop activity pattern", fontweight="bold", y=1.05)
    fig.tight_layout()
    path = figures_dir / "06_stop_activity_pattern.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_wait_ride_frontier(metrics_df: pd.DataFrame, figures_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.6, 4.1))
    for policy in POLICY_ORDER:
        subset = metrics_df[metrics_df["policy"] == policy]
        x = pd.to_numeric(subset["mean_wait_seconds"], errors="coerce")
        y = pd.to_numeric(subset["mean_ride_completed_seconds"], errors="coerce")
        sizes = pd.to_numeric(subset["episode_distance_meters"], errors="coerce")
        if len(sizes.dropna()) and not math.isclose(float(sizes.max()), float(sizes.min())):
            marker_sizes = 35 + 80 * (sizes - sizes.min()) / (sizes.max() - sizes.min())
        else:
            marker_sizes = np.full(len(subset), 55.0)
        ax.scatter(x, y, s=marker_sizes, alpha=0.75, label=policy, color=POLICY_COLORS[policy], edgecolor="#222222", linewidth=0.5)
        ax.scatter([x.mean()], [y.mean()], s=150, marker="*", color=POLICY_COLORS[policy], edgecolor="#111111", linewidth=0.8)
    ax.set_title("Wait-ride trade-off frontier")
    ax.set_xlabel("Mean passenger wait (s)")
    ax.set_ylabel("Mean completed ride (s)")
    ax.legend(loc="best", frameon=True)
    fig.suptitle("Fig. 7. Episode-level policy frontier", fontweight="bold", y=1.02)
    fig.tight_layout()
    path = figures_dir / "07_wait_ride_frontier.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def make_figures(metrics_df: pd.DataFrame, route_df: pd.DataFrame, passenger_df: pd.DataFrame, figures_dir: Path) -> list[Path]:
    configure_matplotlib()
    figures_dir.mkdir(parents=True, exist_ok=True)
    for old in figures_dir.glob("*.png"):
        old.unlink()
    return [
        plot_key_metrics(aggregate_metrics(metrics_df), figures_dir),
        plot_episode_diagnostics(aggregate_metrics(metrics_df), figures_dir),
        plot_passenger_distributions(passenger_df, figures_dir),
        plot_passenger_by_id(passenger_df, figures_dir),
        plot_completion_progression(route_df, metrics_df, figures_dir),
        plot_stop_activity(route_df, figures_dir),
        plot_wait_ride_frontier(metrics_df, figures_dir),
    ]


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def table_paragraph(text: object, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(str(text)), style)


def wrap_table_rows(rows: list[list[object]], header_style: ParagraphStyle, cell_style: ParagraphStyle) -> list[list[Paragraph]]:
    wrapped = []
    for row_index, row in enumerate(rows):
        style = header_style if row_index == 0 else cell_style
        wrapped.append([table_paragraph(cell, style) for cell in row])
    return wrapped


def scaled_image(path: Path, width: float) -> Image:
    image = Image(str(path))
    aspect = image.imageHeight / max(1, image.imageWidth)
    image.drawWidth = width
    image.drawHeight = width * aspect
    return image


def table_style(header_fill: colors.Color = colors.HexColor("#E9EEF7")) -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), header_fill),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111111")),
            ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#B8B8B8")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F7F7")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
    )


def metric_delta_text(agg_df: pd.DataFrame, policy: str, baseline_policy: str, metric: str, digits: int = 1, suffix: str = "") -> str:
    value = agg_value(agg_df, policy, metric, "mean")
    baseline = agg_value(agg_df, baseline_policy, metric, "mean")
    delta = value - baseline
    pct = pct_change(value, baseline)
    if math.isnan(delta):
        return "-"
    if math.isnan(pct):
        return f"{fmt(delta, digits, suffix)}"
    return f"{fmt(delta, digits, suffix)} ({pct:+.1%})"


def build_findings(agg_df: pd.DataFrame) -> list[str]:
    onnx_wait = agg_value(agg_df, "ONNX Inference", "mean_wait_seconds")
    greedy_wait = agg_value(agg_df, "Greedy 1", "mean_wait_seconds")
    vanilla_wait = agg_value(agg_df, "Vanilla Sequential", "mean_wait_seconds")
    onnx_sr = agg_value(agg_df, "ONNX Inference", "service_rate")
    greedy_sr = agg_value(agg_df, "Greedy 1", "service_rate")
    vanilla_sr = agg_value(agg_df, "Vanilla Sequential", "service_rate")
    onnx_ride = agg_value(agg_df, "ONNX Inference", "mean_ride_completed_seconds")
    greedy_ride = agg_value(agg_df, "Greedy 1", "mean_ride_completed_seconds")
    vanilla_ride = agg_value(agg_df, "Vanilla Sequential", "mean_ride_completed_seconds")
    findings = [
        f"Wait-time leader: {best_policy_for_metric(agg_df, 'mean_wait_seconds')} with mean all-passenger wait {fmt(min(onnx_wait, greedy_wait, vanilla_wait), 1)} s.",
        f"Full-completion status: ONNX={fmt(onnx_sr * 100, 1, '%')}, Greedy={fmt(greedy_sr * 100, 1, '%')}, Vanilla={fmt(vanilla_sr * 100, 1, '%')}.",
        f"Compared with Vanilla, ONNX changes mean wait by {metric_delta_text(agg_df, 'ONNX Inference', 'Vanilla Sequential', 'mean_wait_seconds', 1, ' s')} and mean completed ride by {metric_delta_text(agg_df, 'ONNX Inference', 'Vanilla Sequential', 'mean_ride_completed_seconds', 1, ' s')}.",
        f"Compared with Greedy 1, ONNX changes mean wait by {metric_delta_text(agg_df, 'ONNX Inference', 'Greedy 1', 'mean_wait_seconds', 1, ' s')} and mean completed ride by {metric_delta_text(agg_df, 'ONNX Inference', 'Greedy 1', 'mean_ride_completed_seconds', 1, ' s')}.",
    ]
    if onnx_sr >= 0.999 and greedy_sr >= 0.999:
        better_wait = "ONNX" if onnx_wait < greedy_wait else "Greedy 1"
        better_ride = "ONNX" if onnx_ride < greedy_ride else "Greedy 1"
        findings.append(f"Among full-completion non-vanilla policies, {better_wait} is better on wait time while {better_ride} is better on completed ride time.")
    return findings


def build_metric_table(agg_df: pd.DataFrame) -> list[list[str]]:
    rows = [["Metric", "ONNX Inference", "Greedy 1", "Vanilla", "Best", "ONNX vs Vanilla", "ONNX vs Greedy"]]
    metric_specs = [
        ("service_rate", "Service rate", 3, ""),
        ("completed_passengers", "Completed passengers", 1, ""),
        ("episode_time_seconds", "Episode time", 1, " s"),
        ("episode_distance_meters", "Distance", 1, " m"),
        ("mean_wait_seconds", "Mean wait, all passengers", 1, " s"),
        ("p95_wait_seconds", "P95 wait", 1, " s"),
        ("max_wait_seconds", "Max wait", 1, " s"),
        ("mean_ride_completed_seconds", "Mean completed ride", 1, " s"),
        ("route_leg_count", "Route legs", 1, ""),
        ("no_interaction_count", "No-interaction stops", 1, ""),
        ("terminal_on_board", "Terminal onboard", 1, ""),
        ("abab_pattern_count", "ABAB pattern count", 1, ""),
    ]
    for metric, label, digits, suffix in metric_specs:
        rows.append(
            [
                label,
                fmt_mean_std(agg_df, "ONNX Inference", metric, digits, suffix),
                fmt_mean_std(agg_df, "Greedy 1", metric, digits, suffix),
                fmt_mean_std(agg_df, "Vanilla Sequential", metric, digits, suffix),
                best_policy_for_metric(agg_df, metric),
                metric_delta_text(agg_df, "ONNX Inference", "Vanilla Sequential", metric, digits, suffix),
                metric_delta_text(agg_df, "ONNX Inference", "Greedy 1", metric, digits, suffix),
            ]
        )
    return rows


def source_table(metrics_df: pd.DataFrame, selected_dirs: dict[str, Path]) -> list[list[str]]:
    rows = [["Policy", "Export folder", "Episodes", "Scenario", "Full-completion episodes"]]
    for policy in POLICY_ORDER:
        subset = metrics_df[metrics_df["policy"] == policy]
        scenario = ", ".join(sorted(str(value) for value in subset["scenario_description"].dropna().unique()))
        full = int(pd.to_numeric(subset["completed_all_requests"], errors="coerce").fillna(0).sum())
        path = selected_dirs[policy]
        short_path = "/".join(path.parts[-2:]) if len(path.parts) >= 2 else str(path)
        rows.append([policy, short_path, str(len(subset)), scenario, f"{full}/{len(subset)}"])
    return rows


def save_csv_outputs(metrics_df: pd.DataFrame, passenger_df: pd.DataFrame, agg_df: pd.DataFrame, args: argparse.Namespace) -> None:
    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)
    agg_df.to_csv(args.summary_csv, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(args.run_metrics_csv, index=False, encoding="utf-8-sig")
    passenger_df.to_csv(args.passenger_csv, index=False, encoding="utf-8-sig")


def build_pdf(
    output_path: Path,
    metrics_df: pd.DataFrame,
    route_df: pd.DataFrame,
    passenger_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    selected_dirs: dict[str, Path],
    figure_paths: list[Path],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontName="Times-Bold",
        fontSize=17,
        leading=20,
        alignment=TA_CENTER,
        spaceAfter=5,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=8.5,
        leading=10,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#444444"),
        spaceAfter=8,
    )
    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontName="Times-Bold",
        fontSize=12,
        leading=14,
        spaceBefore=8,
        spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Times-Roman",
        fontSize=9,
        leading=11,
        alignment=TA_JUSTIFY,
        spaceAfter=5,
    )
    bullet_style = ParagraphStyle(
        "Bullet",
        parent=body_style,
        leftIndent=10,
        firstLineIndent=-6,
        bulletIndent=0,
    )
    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=body_style,
        fontName="Times-Bold",
        fontSize=7.7,
        leading=8.5,
        alignment=0,
    )
    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=body_style,
        fontName="Times-Roman",
        fontSize=7.6,
        leading=8.4,
        alignment=0,
    )
    caption_style = ParagraphStyle(
        "Caption",
        parent=styles["BodyText"],
        fontName="Times-Italic",
        fontSize=8,
        leading=9,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#333333"),
        spaceBefore=3,
        spaceAfter=7,
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=13 * mm,
        leftMargin=13 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
        title="DRT Next-Stop Policy Evaluation",
    )
    story = []
    story.append(paragraph("DRT Next-Stop Policy Evaluation", title_style))
    story.append(paragraph("ONNX Inference vs Greedy 1 Nearest Feasible vs Vanilla Sequential", subtitle_style))
    story.append(paragraph(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} from Matrix Teleport episode CSV exports.", subtitle_style))

    story.append(paragraph("Abstract", heading_style))
    story.append(
        paragraph(
            "This report compares three next-stop selection policies on the same 30-passenger DRT scenario: the trained ONNX policy, "
            "a newly added Greedy 1 nearest-feasible heuristic, and the Vanilla Sequential baseline. The evaluation uses exported "
            "episode summaries, route-leg logs, and passenger records. The primary outcomes are service completion, passenger wait "
            "time, completed ride time, route length, and no-interaction behavior.",
            body_style,
        )
    )

    story.append(paragraph("Data Sources", heading_style))
    source = Table(
        wrap_table_rows(source_table(metrics_df, selected_dirs), table_header_style, table_cell_style),
        colWidths=[26 * mm, 65 * mm, 15 * mm, 43 * mm, 25 * mm],
        repeatRows=1,
    )
    source.setStyle(table_style())
    story.append(source)
    story.append(Spacer(1, 5))
    story.append(
        paragraph(
            "Important interpretation note: the compared runs reuse the same scenario and Matrix Teleport travel-time matrix. "
            "Repeated episodes are therefore deterministic replications unless the selected policy injects randomness. Means and "
            "standard deviations are reported for auditability, not as a substitute for multi-scenario statistical testing.",
            body_style,
        )
    )

    story.append(paragraph("Key Findings", heading_style))
    for finding in build_findings(agg_df):
        story.append(Paragraph(f"- {finding}", bullet_style))

    story.append(paragraph("Aggregate Metric Table", heading_style))
    metric_table = Table(
        wrap_table_rows(build_metric_table(agg_df), table_header_style, table_cell_style),
        colWidths=[34 * mm, 24 * mm, 24 * mm, 24 * mm, 22 * mm, 27 * mm, 27 * mm],
        repeatRows=1,
    )
    metric_table.setStyle(table_style(colors.HexColor("#DCE6F2")))
    story.append(metric_table)

    story.append(PageBreak())
    story.append(paragraph("Operational Results", heading_style))
    intro = (
        "The figures below separate three questions: whether passengers are completed, how long they wait, and whether the route "
        "contains operationally wasteful stops. For incomplete policies, the all-passenger wait statistics are emphasized because "
        "confirmed-wait averages can hide passengers who were never delivered."
    )
    story.append(paragraph(intro, body_style))
    for index, figure_path in enumerate(figure_paths[:4], start=1):
        story.append(scaled_image(figure_path, 175 * mm))
        story.append(paragraph(f"Figure {index}. {figure_path.stem.replace('_', ' ').title()}.", caption_style))

    story.append(PageBreak())
    story.append(paragraph("Route and Stop Diagnostics", heading_style))
    story.append(
        paragraph(
            "The completion curve and stop-activity heatmap expose whether a policy completes the final onboard passengers or "
            "circulates without useful interactions. No-interaction stop counts are a direct diagnostic for the ABAB-style failure "
            "mode discussed during training.",
            body_style,
        )
    )
    for index, figure_path in enumerate(figure_paths[4:], start=5):
        story.append(scaled_image(figure_path, 175 * mm))
        story.append(paragraph(f"Figure {index}. {figure_path.stem.replace('_', ' ').title()}.", caption_style))

    story.append(PageBreak())
    story.append(paragraph("Per-Policy Diagnostic Summary", heading_style))
    diag_rows = [["Policy", "Main strength", "Main weakness", "Use as baseline?"]]
    onnx_wait = agg_value(agg_df, "ONNX Inference", "mean_wait_seconds")
    greedy_wait = agg_value(agg_df, "Greedy 1", "mean_wait_seconds")
    vanilla_sr = agg_value(agg_df, "Vanilla Sequential", "service_rate")
    diag_rows.append(
        [
            "ONNX Inference",
            "Lowest wait; full completion.",
            "Ride/time slightly above Greedy 1.",
            "Primary proposed method.",
        ]
    )
    diag_rows.append(
        [
            "Greedy 1",
            "Strong non-learning baseline; no empty stops.",
            "Myopic; wait higher than ONNX.",
            "Yes, strong baseline.",
        ]
    )
    diag_rows.append(
        [
            "Vanilla Sequential",
            "Simple fixed-route lower baseline.",
            f"Incomplete service ({fmt(vanilla_sr * 100, 1, '%')}); many empty stops.",
            "Use as weak baseline only.",
        ]
    )
    diag_table = Table(
        wrap_table_rows(diag_rows, table_header_style, table_cell_style),
        colWidths=[28 * mm, 48 * mm, 64 * mm, 34 * mm],
        repeatRows=1,
    )
    diag_table.setStyle(table_style(colors.HexColor("#EAF3E8")))
    story.append(diag_table)
    story.append(Spacer(1, 7))
    conclusion = (
        f"Conclusion: Under the current scenario, the best operational interpretation depends on whether the trained policy beats "
        f"the Greedy 1 baseline on the paper's primary objective. ONNX mean wait is {fmt(onnx_wait, 1)} s and Greedy 1 mean wait is "
        f"{fmt(greedy_wait, 1)} s. If ONNX is lower while preserving full completion, the learned policy provides value beyond a "
        f"simple dispatch heuristic. If Greedy 1 is lower, the current model still improves over Vanilla but needs either broader "
        f"training, adjusted reward shaping, or a stronger experimental claim focused on learning-vs-greedy trade-offs."
    )
    story.append(paragraph(conclusion, body_style))

    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)


def page_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Times-Roman", 7)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(13 * mm, 8 * mm, "DRT next-stop policy comparison")
    canvas.drawRightString(A4[0] - 13 * mm, 8 * mm, f"Page {doc.page}")
    canvas.restoreState()


def main() -> int:
    args = parse_args()
    root = Path(args.exports_root)
    bundles, selected_dirs = load_policy_runs(root, args.runs_per_policy)
    metrics_df, route_df, passenger_df = build_dataframes(bundles)
    agg_df = aggregate_metrics(metrics_df)
    figures_dir = Path(args.figures_dir)
    figure_paths = make_figures(metrics_df, route_df, passenger_df, figures_dir)
    save_csv_outputs(metrics_df, passenger_df, agg_df, args)
    build_pdf(Path(args.output), metrics_df, route_df, passenger_df, agg_df, selected_dirs, figure_paths)

    print(f"PDF: {args.output}")
    print(f"Figures: {figures_dir}")
    print(f"Summary CSV: {args.summary_csv}")
    for policy in POLICY_ORDER:
        subset = metrics_df[metrics_df["policy"] == policy]
        print(
            f"{policy}: n={len(subset)}, service_rate={fmt(agg_value(agg_df, policy, 'service_rate'), 3)}, "
            f"mean_wait={fmt(agg_value(agg_df, policy, 'mean_wait_seconds'), 1)}s, "
            f"mean_ride={fmt(agg_value(agg_df, policy, 'mean_ride_completed_seconds'), 1)}s, "
            f"episode_time={fmt(agg_value(agg_df, policy, 'episode_time_seconds'), 1)}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
