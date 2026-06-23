from __future__ import annotations

import csv
import math
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path("DRT_Episode_Exports") / "inference"
OUTPUT_ROOT = Path("output") / "drt_train_progress"
PDF_PATH = Path("output") / "pdf" / "DRT_train_progress.pdf"
PHASES = [
    ("Begin", "DRT_train_begin"),
    ("Mid", "DRT_train_mid"),
    ("Final", "DRT_train_final"),
]
COLORS = {
    "Begin": "#BDBDBD",
    "Mid": "#737373",
    "Final": "#1F4E79",
}

FONT_REGULAR = "MalgunGothic"
FONT_BOLD = "MalgunGothic-Bold"
FONT_REGULAR_PATH = Path(r"C:\Windows\Fonts\malgun.ttf")
FONT_BOLD_PATH = Path(r"C:\Windows\Fonts\malgunbd.ttf")


def register_fonts() -> None:
    if FONT_REGULAR_PATH.exists() and FONT_BOLD_PATH.exists():
        pdfmetrics.registerFont(TTFont(FONT_REGULAR, str(FONT_REGULAR_PATH)))
        pdfmetrics.registerFont(TTFont(FONT_BOLD, str(FONT_BOLD_PATH)))
        plt.rcParams["font.family"] = "Malgun Gothic"
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


def f(value: object) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def i(value: object) -> int:
    try:
        return int(float(str(value)))
    except Exception:
        return 0


def fmt(value: object, digits: int = 1, suffix: str = "") -> str:
    value_f = f(value)
    if math.isnan(value_f):
        return "-"
    return f"{value_f:,.{digits}f}{suffix}"


def fmt_pct(value: object, digits: int = 1) -> str:
    value_f = f(value)
    if math.isnan(value_f):
        return "-"
    return f"{value_f * 100:,.{digits}f}%"


def fmt_signed_seconds(value: object) -> str:
    value_f = f(value)
    if math.isnan(value_f):
        return "-"
    return f"{value_f:+,.1f}s"


def safe_para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(text), style)


def gini(values: list[float]) -> float:
    clean = sorted(v for v in values if not math.isnan(v) and v >= 0.0)
    if not clean:
        return float("nan")
    total = sum(clean)
    if total == 0.0:
        return float("nan")
    n = len(clean)
    weighted_sum = sum((idx + 1) * value for idx, value in enumerate(clean))
    return float((2.0 * weighted_sum / (n * total)) - ((n + 1.0) / n))


def passenger_onboard_count_for_leg(leg: dict[str, str], passengers: list[dict[str, str]]) -> int:
    departure = f(leg.get("departure_time_seconds"))
    arrival = f(leg.get("arrival_time_seconds"))
    if math.isnan(departure) or math.isnan(arrival):
        return 0
    count = 0
    for passenger in passengers:
        pickup = f(passenger.get("pickup_time_seconds"))
        dropoff = f(passenger.get("dropoff_time_seconds"))
        if math.isnan(pickup) or pickup > departure + 1e-6:
            continue
        if not math.isnan(dropoff) and dropoff < arrival - 1e-6:
            continue
        count += 1
    return count


def parse_episode(path: Path) -> tuple[dict[str, str], list[dict[str, str]], list[dict[str, str]]]:
    summary: dict[str, str] = {}
    passengers: list[dict[str, str]] = []
    route_legs: list[dict[str, str]] = []
    section_header: list[str] | None = None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row or all(cell == "" for cell in row):
                continue
            if row[0] == "summary" and len(row) >= 3:
                summary[row[1]] = row[2]
            elif row[0] == "section":
                section_header = row[1:]
            elif row[0] in {"passenger", "route_leg"} and section_header is not None:
                values = {
                    section_header[idx]: row[idx + 1] if idx + 1 < len(row) else ""
                    for idx in range(len(section_header))
                }
                if row[0] == "passenger":
                    passengers.append(values)
                else:
                    route_legs.append(values)
    return summary, passengers, route_legs


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    episode_rows: list[dict[str, object]] = []
    passenger_rows: list[dict[str, object]] = []
    leg_rows: list[dict[str, object]] = []

    for phase_order, (phase, folder_name) in enumerate(PHASES):
        folder = ROOT / folder_name
        episode_files = sorted(folder.glob("*_episode.csv"))
        if not episode_files:
            raise SystemExit(f"No episode CSV files found in {folder}")
        for episode_file in episode_files:
            summary, passengers, route_legs = parse_episode(episode_file)
            episode_index = i(summary.get("episode_index"))
            scenario_id = i(summary.get("scenario_id"))
            leg_onboard_counts = [
                passenger_onboard_count_for_leg(leg, passengers) for leg in route_legs
            ]
            leg_distances = [f(leg.get("leg_distance_meters")) for leg in route_legs]
            total_distance_m = float(sum(d for d in leg_distances if not math.isnan(d)))
            if total_distance_m == 0.0:
                total_distance_m = f(summary.get("episode_distance_meters"))
            loaded_distance_m = float(
                sum(
                    distance
                    for distance, onboard_count in zip(leg_distances, leg_onboard_counts)
                    if not math.isnan(distance) and onboard_count > 0
                )
            )
            visited_stops = set()
            for leg in route_legs:
                from_stop = i(leg.get("from_stop_id"))
                arrived_stop = i(leg.get("arrived_stop_id") or leg.get("to_stop_id"))
                if from_stop:
                    visited_stops.add(from_stop)
                if arrived_stop:
                    visited_stops.add(arrived_stop)
            passenger_waits = [
                f(passenger.get("wait_time_seconds"))
                for passenger in passengers
                if not math.isnan(f(passenger.get("wait_time_seconds")))
            ]
            passenger_rides = [
                f(passenger.get("ride_time_seconds"))
                for passenger in passengers
                if passenger.get("status") == "Completed"
                and not math.isnan(f(passenger.get("ride_time_seconds")))
            ]
            completed_passengers = i(summary.get("completed_passengers"))
            episode_time_seconds = f(summary.get("episode_time_seconds"))
            episode_rows.append(
                {
                    "phase": phase,
                    "phase_order": phase_order,
                    "folder": folder_name,
                    "episode_file": episode_file.name,
                    "episode_index": episode_index,
                    "scenario_id": scenario_id,
                    "scenario_description": summary.get("scenario_description", ""),
                    "travel_mode": summary.get("travel_mode", ""),
                    "travel_execution_mode": summary.get("travel_execution_mode", ""),
                    "next_stop_policy": summary.get("next_stop_policy", ""),
                    "finish_reason": summary.get("finish_reason", ""),
                    "completed_all_requests": i(summary.get("completed_all_requests")),
                    "total_passengers": i(summary.get("total_passengers")),
                    "completed_passengers": completed_passengers,
                    "service_rate": f(summary.get("service_rate")),
                    "average_wait_seconds": f(summary.get("average_wait_seconds")),
                    "median_wait_seconds": float(pd.Series(passenger_waits).median())
                    if passenger_waits
                    else float("nan"),
                    "max_wait_seconds": max(passenger_waits) if passenger_waits else float("nan"),
                    "wait_gini": gini(passenger_waits),
                    "average_ride_seconds": f(summary.get("average_ride_seconds")),
                    "median_ride_seconds": float(pd.Series(passenger_rides).median())
                    if passenger_rides
                    else float("nan"),
                    "max_ride_seconds": max(passenger_rides) if passenger_rides else float("nan"),
                    "ride_gini": gini(passenger_rides),
                    "episode_time_seconds": episode_time_seconds,
                    "episode_distance_meters": total_distance_m,
                    "distance_per_completed_km": total_distance_m / 1000.0 / completed_passengers
                    if completed_passengers
                    else float("nan"),
                    "time_per_completed_seconds": episode_time_seconds / completed_passengers
                    if completed_passengers
                    else float("nan"),
                    "loaded_distance_meters": loaded_distance_m,
                    "loaded_distance_ratio": loaded_distance_m / total_distance_m
                    if total_distance_m
                    else float("nan"),
                    "visited_stop_count": len(visited_stops),
                    "route_leg_count": len(route_legs),
                }
            )
            for passenger in passengers:
                passenger_rows.append(
                    {
                        "phase": phase,
                        "phase_order": phase_order,
                        "folder": folder_name,
                        "episode_file": episode_file.name,
                        "episode_index": episode_index,
                        "passenger_id": i(passenger.get("passenger_id")),
                        "status": passenger.get("status", ""),
                        "wait_time_seconds": f(passenger.get("wait_time_seconds")),
                        "ride_time_seconds": f(passenger.get("ride_time_seconds")),
                        "total_service_time_seconds": f(passenger.get("total_service_time_seconds")),
                    }
                )
            for leg, onboard_count in zip(route_legs, leg_onboard_counts):
                leg_rows.append(
                    {
                        "phase": phase,
                        "phase_order": phase_order,
                        "folder": folder_name,
                        "episode_file": episode_file.name,
                        "episode_index": episode_index,
                        "leg_index": i(leg.get("leg_index")),
                        "from_stop_id": i(leg.get("from_stop_id")),
                        "to_stop_id": i(leg.get("to_stop_id")),
                        "arrived_stop_id": i(leg.get("arrived_stop_id") or leg.get("to_stop_id")),
                        "departure_time_seconds": f(leg.get("departure_time_seconds")),
                        "arrival_time_seconds": f(leg.get("arrival_time_seconds")),
                        "boarded_count": i(leg.get("boarded_count")),
                        "dropped_off_count": i(leg.get("dropped_off_count")),
                        "waiting_count": i(leg.get("waiting_count")),
                        "on_board_count": i(leg.get("on_board_count")),
                        "completed_passenger_count": i(leg.get("completed_passenger_count")),
                        "travel_time_seconds": f(leg.get("travel_time_seconds")),
                        "leg_distance_meters": f(leg.get("leg_distance_meters")),
                        "onboard_during_leg": onboard_count,
                    }
                )

    episodes = pd.DataFrame(episode_rows)
    passengers = pd.DataFrame(passenger_rows)
    legs = pd.DataFrame(leg_rows)

    summary_rows: list[dict[str, object]] = []
    for phase, _ in PHASES:
        ep = episodes[episodes["phase"] == phase]
        ps = passengers[passengers["phase"] == phase]
        completed_ps = ps[ps["status"] == "Completed"]
        waits = ps["wait_time_seconds"].dropna().astype(float).tolist()
        rides = completed_ps["ride_time_seconds"].dropna().astype(float).tolist()
        service_times = ps["total_service_time_seconds"].dropna().astype(float).tolist()
        wait_mean = float(pd.Series(waits).mean()) if waits else float("nan")
        ride_mean = float(pd.Series(rides).mean()) if rides else float("nan")
        service_time_mean = float(pd.Series(service_times).mean()) if service_times else float("nan")
        finish_counts = ep["finish_reason"].value_counts().to_dict()
        status_counts = ps["status"].value_counts().to_dict()
        summary_rows.append(
            {
                "phase": phase,
                "phase_order": int(ep["phase_order"].iloc[0]),
                "folder": str(ep["folder"].iloc[0]),
                "episodes": len(ep),
                "completed_runs": int(ep["completed_all_requests"].sum()),
                "completion_rate": float(ep["completed_all_requests"].mean()),
                "service_rate": float(ep["service_rate"].mean()),
                "completed_passengers_mean": float(ep["completed_passengers"].mean()),
                "total_passengers_mean": float(ep["total_passengers"].mean()),
                "wait_min_seconds": min(waits) if waits else float("nan"),
                "wait_mean_seconds": wait_mean,
                "wait_median_seconds": float(pd.Series(waits).median()) if waits else float("nan"),
                "wait_max_seconds": max(waits) if waits else float("nan"),
                "wait_gini": gini(waits),
                "average_wait_seconds": wait_mean,
                "average_wait_std": float(ep["average_wait_seconds"].std(ddof=0)),
                "ride_min_seconds": min(rides) if rides else float("nan"),
                "ride_mean_seconds": ride_mean,
                "ride_median_seconds": float(pd.Series(rides).median()) if rides else float("nan"),
                "ride_max_seconds": max(rides) if rides else float("nan"),
                "ride_gini": gini(rides),
                "average_ride_seconds": ride_mean,
                "service_time_mean_seconds": service_time_mean,
                "service_time_min_seconds": min(service_times) if service_times else float("nan"),
                "service_time_median_seconds": float(pd.Series(service_times).median())
                if service_times
                else float("nan"),
                "service_time_max_seconds": max(service_times) if service_times else float("nan"),
                "episode_time_seconds": float(ep["episode_time_seconds"].mean()),
                "episode_time_minutes": float(ep["episode_time_seconds"].mean() / 60.0),
                "total_distance_km": float(ep["episode_distance_meters"].mean() / 1000.0),
                "episode_distance_km": float(ep["episode_distance_meters"].mean() / 1000.0),
                "distance_per_completed_km": float(ep["distance_per_completed_km"].mean()),
                "time_per_completed_minutes": float(ep["time_per_completed_seconds"].mean() / 60.0),
                "loaded_distance_ratio": float(ep["loaded_distance_ratio"].mean()),
                "visited_stop_count": float(ep["visited_stop_count"].mean()),
                "finished_by_completion": int(finish_counts.get("All passenger requests completed.", 0)),
                "finished_by_timeout": int(finish_counts.get("Episode time ended.", 0)),
                "completed_passenger_rows": int(status_counts.get("Completed", 0)),
                "waiting_passenger_rows": int(status_counts.get("Waiting", 0)),
                "onboard_passenger_rows": int(status_counts.get("OnBoard", 0)),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("phase_order").reset_index(drop=True)

    final = summary.iloc[-1]
    for metric in [
        "service_rate",
        "average_wait_seconds",
        "average_ride_seconds",
        "episode_time_seconds",
        "episode_distance_km",
    ]:
        summary[f"delta_vs_begin_{metric}"] = summary[metric] - float(summary.iloc[0][metric])
        summary[f"relative_vs_begin_{metric}"] = summary[f"delta_vs_begin_{metric}"] / float(summary.iloc[0][metric])
    return episodes, passengers, legs, summary


def save_tables(episodes: pd.DataFrame, passengers: pd.DataFrame, legs: pd.DataFrame, summary: pd.DataFrame) -> None:
    tables = OUTPUT_ROOT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(tables / "episode_level_metrics.csv", index=False, encoding="utf-8-sig")
    passengers.to_csv(tables / "passenger_level_metrics.csv", index=False, encoding="utf-8-sig")
    legs.to_csv(tables / "route_leg_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(tables / "train_progress_summary.csv", index=False, encoding="utf-8-sig")
    with (tables / "train_progress_kpi_values.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        csv.writer(handle).writerows(build_summary_rows(summary))
    with (tables / "train_progress_comparison_result.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        csv.writer(handle).writerows(build_delta_rows(summary))
    with (tables / "train_progress_service_quality.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        csv.writer(handle).writerows(passenger_quality_rows(summary))
    with (tables / "train_progress_operational_efficiency.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        csv.writer(handle).writerows(efficiency_rows(summary))
    with (tables / "train_progress_checkpoint_delta.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        csv.writer(handle).writerows(checkpoint_delta_rows(summary))


def annotate_bars(ax, values: list[float], digits: int = 1, suffix: str = "") -> None:
    for patch, value in zip(ax.patches, values):
        ax.annotate(
            f"{value:.{digits}f}{suffix}",
            (patch.get_x() + patch.get_width() / 2, value),
            ha="center",
            va="bottom",
            fontsize=7.5,
            xytext=(0, 3),
            textcoords="offset points",
        )


def apply_ieee_axes(ax) -> None:
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.5, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(axis="both", labelsize=8)


def bar_ax(
    ax,
    summary: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    digits: int = 1,
    suffix: str = "",
    scale: float = 1.0,
) -> None:
    phases = summary["phase"].tolist()
    values = (summary[metric].astype(float) * scale).tolist()
    ax.bar(phases, values, color=[COLORS[p] for p in phases], width=0.58, edgecolor="#333333", linewidth=0.35)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=8)
    if values:
        ax.set_ylim(0, max(values) * 1.22 if max(values) > 0 else 1)
    apply_ieee_axes(ax)
    annotate_bars(ax, values, digits=digits, suffix=suffix)


def grouped_metric_chart(
    ax,
    summary: pd.DataFrame,
    metrics: list[str],
    labels: list[str],
    title: str,
    ylabel: str,
    scale: float = 1.0,
) -> None:
    phases = summary["phase"].tolist()
    x_positions = np.arange(len(metrics))
    width = 0.22
    hatch_patterns = ["", "//", "\\\\"]
    for phase_idx, phase in enumerate(phases):
        phase_values = summary[summary["phase"] == phase].iloc[0]
        values = np.array([float(phase_values[metric]) * scale for metric in metrics], dtype=float)
        ax.bar(
            x_positions + (phase_idx - 1) * width,
            values,
            width=width,
            label=phase,
            color=COLORS[phase],
            edgecolor="#333333",
            linewidth=0.35,
            hatch=hatch_patterns[phase_idx % len(hatch_patterns)],
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=8)
    max_value = max(
        float(summary[metric].max()) * scale
        for metric in metrics
        if metric in summary
    )
    ax.set_ylim(0, max_value * 1.18 if max_value else 1)
    apply_ieee_axes(ax)


def passenger_boxplot(
    ax,
    passengers: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    completed_only: bool = False,
) -> None:
    phases = [phase for phase, _ in PHASES]
    data: list[list[float]] = []
    for phase in phases:
        phase_passengers = passengers[passengers["phase"] == phase]
        if completed_only:
            phase_passengers = phase_passengers[phase_passengers["status"] == "Completed"]
        values = phase_passengers[metric].dropna().astype(float).tolist()
        data.append(values if values else [float("nan")])

    box = ax.boxplot(
        data,
        tick_labels=phases,
        patch_artist=True,
        showmeans=True,
        whis=[0, 100],
        widths=0.55,
        meanprops={
            "marker": "D",
            "markerfacecolor": "#FFFFFF",
            "markeredgecolor": "#222222",
            "markersize": 4.2,
        },
        medianprops={"color": "#111111", "linewidth": 1.1},
        whiskerprops={"color": "#333333", "linewidth": 0.8},
        capprops={"color": "#333333", "linewidth": 0.8},
        boxprops={"edgecolor": "#333333", "linewidth": 0.8},
        flierprops={"markersize": 0},
    )
    for patch, phase in zip(box["boxes"], phases):
        patch.set_facecolor(COLORS[phase])
        patch.set_alpha(0.75)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=8)
    apply_ieee_axes(ax)


def frontier_ax(ax, summary: pd.DataFrame) -> None:
    for _, row in summary.iterrows():
        phase = str(row["phase"])
        ax.scatter(
            row["total_distance_km"],
            row["service_time_mean_seconds"],
            s=72,
            color=COLORS[phase],
            edgecolor="#222222",
            linewidth=0.55,
            zorder=3,
        )
        ax.annotate(
            phase,
            (row["total_distance_km"], row["service_time_mean_seconds"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8.5,
        )
    ax.set_title("Time-distance frontier", fontsize=10)
    ax.set_xlabel("Total driving distance (km/episode)", fontsize=8)
    ax.set_ylabel("Mean passenger service time (s)", fontsize=8)
    apply_ieee_axes(ax)


def build_figures(episodes: pd.DataFrame, passengers: pd.DataFrame, summary: pd.DataFrame) -> list[Path]:
    figs = OUTPUT_ROOT / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    plt.rcParams.update(
        {
            "axes.titleweight": "bold",
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.45), dpi=180)
    bar_ax(axes[0], summary, "service_rate", "Service rate", "%", digits=1, suffix="%", scale=100.0)
    bar_ax(axes[1], summary, "completion_rate", "Full-completion run rate", "%", digits=1, suffix="%", scale=100.0)
    bar_ax(axes[2], summary, "loaded_distance_ratio", "Loaded distance ratio", "%", digits=1, suffix="%", scale=100.0)
    for ax in axes:
        ax.set_ylim(0, 110)
    fig.tight_layout()
    path = figs / "fig01_service_loaded_ratio.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), dpi=180)
    passenger_boxplot(
        axes[0],
        passengers,
        "wait_time_seconds",
        "Passenger wait time",
        "seconds",
    )
    passenger_boxplot(
        axes[1],
        passengers,
        "ride_time_seconds",
        "Passenger ride time",
        "seconds",
        completed_only=True,
    )
    fig.tight_layout()
    path = figs / "fig02_wait_ride_time.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), dpi=180)
    frontier_ax(axes[0], summary)
    passenger_boxplot(
        axes[1],
        passengers,
        "total_service_time_seconds",
        "Passenger service time",
        "seconds",
    )
    fig.tight_layout()
    path = figs / "fig03_service_time_frontier.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.7), dpi=180)
    bar_ax(axes[0, 0], summary, "total_distance_km", "Total driving distance", "km", digits=1)
    bar_ax(axes[0, 1], summary, "episode_time_minutes", "Episode time", "minutes", digits=1)
    bar_ax(axes[1, 0], summary, "distance_per_completed_km", "Distance per completed passenger", "km/pass.", digits=2)
    bar_ax(axes[1, 1], summary, "time_per_completed_minutes", "Time per completed passenger", "min/pass.", digits=2)
    fig.tight_layout()
    path = figs / "fig04_operational_efficiency.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.45), dpi=180)
    bar_ax(axes[0], summary, "wait_gini", "Wait inequality", "Gini", digits=3)
    bar_ax(axes[1], summary, "ride_gini", "Ride inequality", "Gini", digits=3)
    bar_ax(axes[2], summary, "visited_stop_count", "Visited stop count", "stops", digits=0)
    fig.tight_layout()
    path = figs / "fig05_inequality_stops.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    return paths


def table_style(font_size: float = 7.5, header_bg: str = "#E6E6E6") -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD if FONT_BOLD_PATH.exists() else "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), FONT_REGULAR if FONT_REGULAR_PATH.exists() else "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("LEADING", (0, 0), (-1, -1), font_size + 1.4),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#A8A8A8")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F7F7")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
    )


def styled_table(rows: list[list[object]], col_widths: list[float] | None = None, font_size: float = 7.5) -> LongTable:
    table = LongTable(rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(table_style(font_size=font_size))
    return table


def scaled_image(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as img:
        width_px, height_px = img.size
    ratio = min(max_width / width_px, max_height / height_px)
    return Image(str(path), width=width_px * ratio, height=height_px * ratio)


def short_finish(reason: object) -> str:
    text = str(reason)
    if text == "All passenger requests completed.":
        return "All completed"
    if text == "Episode time ended.":
        return "Time ended"
    return text


def build_source_rows(episodes: pd.DataFrame) -> list[list[str]]:
    rows = [["Phase", "Export folder", "Episodes", "Mode", "Scenario", "Finish reasons"]]
    for phase, folder in PHASES:
        ep = episodes[episodes["phase"] == phase]
        reasons = ep["finish_reason"].value_counts()
        reason_text = ", ".join(f"{short_finish(reason)}: {count}" for reason, count in reasons.items())
        scenarios = ", ".join(str(int(v)) for v in sorted(ep["scenario_id"].unique()))
        modes = ", ".join(
            sorted(
                {
                    f"{row.travel_mode}/{row.travel_execution_mode}"
                    for row in ep.itertuples()
                    if row.travel_mode or row.travel_execution_mode
                }
            )
        )
        rows.append([phase, folder, str(len(ep)), modes, scenarios, reason_text])
    return rows


def value_text(value: object, kind: str) -> str:
    if kind == "pct":
        return fmt_pct(value)
    if kind == "seconds":
        return f"{fmt(value)}"
    if kind == "minutes":
        return f"{fmt(value)}"
    if kind == "km":
        return f"{fmt(value, 2)}"
    if kind == "ratio":
        return fmt_pct(value)
    if kind == "gini":
        return fmt(value, 3)
    if kind == "count":
        return fmt(value, 1)
    return fmt(value)


def delta_text(begin_value: object, final_value: object, kind: str) -> str:
    begin_f = f(begin_value)
    final_f = f(final_value)
    if math.isnan(begin_f) or math.isnan(final_f):
        return "-"
    delta = final_f - begin_f
    relative = delta / begin_f * 100.0 if begin_f else float("nan")
    if kind in {"pct", "ratio"}:
        return f"{delta * 100.0:+.1f} pp"
    if kind == "gini":
        return f"{delta:+.3f}"
    if kind == "count":
        return f"{delta:+.1f}"
    suffix = "s" if kind == "seconds" else "m" if kind == "minutes" else "km" if kind == "km" else ""
    if math.isnan(relative):
        return f"{delta:+,.1f}{suffix}"
    return f"{delta:+,.1f}{suffix} ({relative:+.1f}%)"


def kpi_specs() -> list[tuple[str, str, str, str]]:
    return [
        ("Service rate", "service_rate", "%", "pct"),
        ("Wait time - min", "wait_min_seconds", "s", "seconds"),
        ("Wait time - mean", "wait_mean_seconds", "s", "seconds"),
        ("Wait time - median", "wait_median_seconds", "s", "seconds"),
        ("Wait time - max", "wait_max_seconds", "s", "seconds"),
        ("Ride time - min", "ride_min_seconds", "s", "seconds"),
        ("Ride time - mean", "ride_mean_seconds", "s", "seconds"),
        ("Ride time - median", "ride_median_seconds", "s", "seconds"),
        ("Ride time - max", "ride_max_seconds", "s", "seconds"),
        ("Passenger service time - min", "service_time_min_seconds", "s", "seconds"),
        ("Passenger service time - mean", "service_time_mean_seconds", "s", "seconds"),
        ("Passenger service time - median", "service_time_median_seconds", "s", "seconds"),
        ("Passenger service time - max", "service_time_max_seconds", "s", "seconds"),
        ("Total driving distance", "total_distance_km", "km/episode", "km"),
        ("Episode time", "episode_time_minutes", "min/episode", "minutes"),
        ("Distance per completed passenger", "distance_per_completed_km", "km/pax", "km"),
        ("Operation time per completed passenger", "time_per_completed_minutes", "min/pax", "minutes"),
        ("Loaded distance ratio", "loaded_distance_ratio", "%", "ratio"),
        ("Wait-time inequality", "wait_gini", "Gini", "gini"),
        ("Ride-time inequality", "ride_gini", "Gini", "gini"),
        ("Visited stop count", "visited_stop_count", "stops", "count"),
    ]


def build_summary_rows(summary: pd.DataFrame) -> list[list[str]]:
    rows = [["Metric", "Unit", "Begin", "Mid", "Final", "Final vs Begin"]]
    phase_rows = {str(row["phase"]): row for _, row in summary.iterrows()}
    begin = phase_rows["Begin"]
    final = phase_rows["Final"]
    for label, metric, unit, kind in kpi_specs():
        rows.append(
            [
                label,
                unit,
                value_text(begin[metric], kind),
                value_text(phase_rows["Mid"][metric], kind),
                value_text(final[metric], kind),
                delta_text(begin[metric], final[metric], kind),
            ]
        )
    return rows


def passenger_quality_rows(summary: pd.DataFrame) -> list[list[str]]:
    rows = [
        [
            "Checkpoint",
            "Runs",
            "Complete",
            "Service",
            "Wait mean",
            "Wait median",
            "Wait max",
            "Wait Gini",
            "Ride mean",
            "Ride median",
            "Ride max",
            "Ride Gini",
        ]
    ]
    for _, row in summary.iterrows():
        rows.append(
            [
                str(row["phase"]),
                str(int(row["episodes"])),
                f"{int(row['completed_runs'])}/{int(row['episodes'])}",
                fmt_pct(row["service_rate"]),
                f"{fmt(row['wait_mean_seconds'])} s",
                f"{fmt(row['wait_median_seconds'])} s",
                f"{fmt(row['wait_max_seconds'])} s",
                fmt(row["wait_gini"], 3),
                f"{fmt(row['ride_mean_seconds'])} s",
                f"{fmt(row['ride_median_seconds'])} s",
                f"{fmt(row['ride_max_seconds'])} s",
                fmt(row["ride_gini"], 3),
            ]
        )
    return rows


def efficiency_rows(summary: pd.DataFrame) -> list[list[str]]:
    rows = [
        [
            "Checkpoint",
            "Total km",
            "Episode min",
            "km/pass.",
            "min/pass.",
            "Loaded ratio",
            "Svc time mean",
            "Svc time median",
            "Stops",
        ]
    ]
    for _, row in summary.iterrows():
        rows.append(
            [
                str(row["phase"]),
                f"{fmt(row['total_distance_km'])} km",
                f"{fmt(row['episode_time_minutes'])} min",
                f"{fmt(row['distance_per_completed_km'], 2)}",
                f"{fmt(row['time_per_completed_minutes'], 2)}",
                fmt_pct(row["loaded_distance_ratio"]),
                f"{fmt(row['service_time_mean_seconds'])} s",
                f"{fmt(row['service_time_median_seconds'])} s",
                fmt(row["visited_stop_count"], 0),
            ]
        )
    return rows


def checkpoint_delta_rows(summary: pd.DataFrame) -> list[list[str]]:
    phase_rows = {str(row["phase"]): row for _, row in summary.iterrows()}
    begin = phase_rows["Begin"]
    mid = phase_rows["Mid"]
    final = phase_rows["Final"]
    rows = [["Metric", "Better", "Mid vs Begin", "Final vs Begin"]]
    metrics = [
        ("Service rate", "service_rate", "pct", "higher"),
        ("Wait mean", "wait_mean_seconds", "seconds", "lower"),
        ("Wait median", "wait_median_seconds", "seconds", "lower"),
        ("Ride mean", "ride_mean_seconds", "seconds", "lower"),
        ("Service time mean", "service_time_mean_seconds", "seconds", "lower"),
        ("Total distance", "total_distance_km", "km", "lower"),
        ("Episode time", "episode_time_minutes", "minutes", "lower"),
        ("Distance per passenger", "distance_per_completed_km", "km", "lower"),
        ("Loaded distance ratio", "loaded_distance_ratio", "ratio", "higher"),
        ("Wait Gini", "wait_gini", "gini", "lower"),
        ("Ride Gini", "ride_gini", "gini", "lower"),
    ]
    for label, metric, kind, direction in metrics:
        rows.append(
            [
                label,
                direction,
                delta_text(begin[metric], mid[metric], kind),
                delta_text(begin[metric], final[metric], kind),
            ]
        )
    return rows


def build_delta_rows(summary: pd.DataFrame) -> list[list[str]]:
    phase_rows = {str(row["phase"]): row for _, row in summary.iterrows()}
    rows = [["Metric", "Better", "Final vs Begin", "Final vs Mid"]]
    metrics = [
        ("Service rate", "service_rate", "pct", "higher"),
        ("Wait mean", "wait_mean_seconds", "seconds", "lower"),
        ("Wait median", "wait_median_seconds", "seconds", "lower"),
        ("Ride mean", "ride_mean_seconds", "seconds", "lower"),
        ("Total distance", "total_distance_km", "km", "lower"),
        ("Episode time", "episode_time_minutes", "minutes", "lower"),
        ("Distance per passenger", "distance_per_completed_km", "km", "lower"),
        ("Loaded distance ratio", "loaded_distance_ratio", "ratio", "higher"),
        ("Wait Gini", "wait_gini", "gini", "lower"),
    ]
    final = phase_rows["Final"]
    for label, metric, kind, direction in metrics:
        rows.append(
            [
                label,
                direction,
                delta_text(phase_rows["Begin"][metric], final[metric], kind),
                delta_text(phase_rows["Mid"][metric], final[metric], kind),
            ]
        )
    return rows


def build_episode_rows(episodes: pd.DataFrame) -> list[list[str]]:
    rows = [
        [
            "Phase",
            "Ep",
            "Finish",
            "Completed",
            "Service",
            "Wait mean",
            "Ride mean",
            "Dist.",
            "Time",
            "Loaded",
            "Stops",
        ]
    ]
    for _, row in episodes.sort_values(["phase_order", "episode_index"]).iterrows():
        rows.append(
            [
                str(row["phase"]),
                str(int(row["episode_index"])),
                short_finish(row["finish_reason"]),
                f"{int(row['completed_passengers'])}/{int(row['total_passengers'])}",
                fmt_pct(row["service_rate"]),
                f"{fmt(row['average_wait_seconds'])}s",
                f"{fmt(row['average_ride_seconds'])}s",
                f"{fmt(row['episode_distance_meters'] / 1000.0, 2)}km",
                f"{fmt(row['episode_time_seconds'] / 60.0)}m",
                fmt_pct(row["loaded_distance_ratio"]),
                fmt(row["visited_stop_count"], 0),
            ]
        )
    return rows


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont(FONT_REGULAR if FONT_REGULAR_PATH.exists() else "Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(1.4 * cm, 1.0 * cm, "DRT_train_progress")
    canvas.drawRightString(A4[0] - 1.4 * cm, 1.0 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf(episodes: pd.DataFrame, passengers: pd.DataFrame, summary: pd.DataFrame, figures: list[Path]) -> None:
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=1.15 * cm,
        leftMargin=1.15 * cm,
        topMargin=1.15 * cm,
        bottomMargin=1.45 * cm,
    )
    font = FONT_REGULAR if FONT_REGULAR_PATH.exists() else "Helvetica"
    font_bold = FONT_BOLD if FONT_BOLD_PATH.exists() else "Helvetica-Bold"
    styles = {
        "title": ParagraphStyle(
            "title",
            fontName=font_bold,
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#111111"),
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            fontName=font,
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#444444"),
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "h1",
            fontName=font_bold,
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#111111"),
            spaceBefore=7,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body",
            fontName=font,
            fontSize=8.8,
            leading=12.2,
            textColor=colors.HexColor("#222222"),
            alignment=TA_LEFT,
            spaceAfter=5,
        ),
    }

    final = summary[summary["phase"] == "Final"].iloc[0]
    begin = summary[summary["phase"] == "Begin"].iloc[0]
    mid = summary[summary["phase"] == "Mid"].iloc[0]
    wait_gain = begin["wait_mean_seconds"] - final["wait_mean_seconds"]
    ride_gain = begin["ride_mean_seconds"] - final["ride_mean_seconds"]
    service_time_gain = begin["service_time_mean_seconds"] - final["service_time_mean_seconds"]
    distance_gain = begin["total_distance_km"] - final["total_distance_km"]
    time_gain = begin["episode_time_minutes"] - final["episode_time_minutes"]

    story: list = []
    story.append(safe_para("DRT Train-Progress ONNX Evaluation", styles["title"]))
    story.append(
        safe_para(
            "Scenario 30 new demand. Checkpoint order is fixed as Begin, Mid, and Final; tables provide report-ready values and figures provide analysis evidence.",
            styles["subtitle"],
        )
    )
    story.append(safe_para("핵심 해석", styles["h1"]))
    story.append(
        safe_para(
            "Begin checkpoint는 30명 중 26명만 완료하고 시간 종료로 끝난다. 반면 Mid와 Final은 모두 100.0% service rate와 full-completion run을 달성한다. "
            "따라서 가장 중요한 결과는 ONNX 학습 진행 후 실패하던 demand set이 완전 서비스 가능한 정책으로 바뀌었다는 점이다.",
            styles["body"],
        )
    )
    story.append(
        safe_para(
            f"Final은 Begin 대비 평균 대기시간을 {wait_gain:.1f}s, 평균 탑승시간을 {ride_gain:.1f}s, 승객 체감시간을 {service_time_gain:.1f}s 줄인다. "
            f"또한 총 주행거리는 {distance_gain:.2f}km, episode time은 {time_gain:.1f}min 감소한다. "
            "이번 MatrixTeleport export에서는 Final이 Mid보다 평균 대기시간, 평균 탑승시간, episode time, 총 주행거리 모두 낮아 세 checkpoint 중 가장 좋은 운영 지표를 보인다.",
            styles["body"],
        )
    )
    story.append(safe_para("Source Coverage", styles["h1"]))
    story.append(
        styled_table(
            build_source_rows(episodes),
            col_widths=[1.45 * cm, 3.2 * cm, 1.25 * cm, 3.0 * cm, 1.25 * cm, 7.1 * cm],
            font_size=6.9,
        )
    )
    story.append(Spacer(1, 0.12 * cm))
    story.append(safe_para("표 1. 승객 서비스 품질 KPI", styles["h1"]))
    story.append(
        styled_table(
            passenger_quality_rows(summary),
            col_widths=[
                1.55 * cm,
                0.9 * cm,
                1.25 * cm,
                1.25 * cm,
                1.45 * cm,
                1.55 * cm,
                1.45 * cm,
                1.25 * cm,
                1.45 * cm,
                1.55 * cm,
                1.45 * cm,
                1.25 * cm,
            ],
            font_size=5.75,
        )
    )
    story.append(Spacer(1, 0.12 * cm))
    story.append(safe_para("표 2. 운영 효율 KPI", styles["h1"]))
    story.append(
        styled_table(
            efficiency_rows(summary),
            col_widths=[1.55 * cm, 1.65 * cm, 1.65 * cm, 1.6 * cm, 1.6 * cm, 1.75 * cm, 2.25 * cm, 2.25 * cm, 1.25 * cm],
            font_size=6.2,
        )
    )

    story.append(PageBreak())
    story.append(safe_para("표 3. Begin 기준 checkpoint 개선량", styles["h1"]))
    story.append(
        styled_table(
            checkpoint_delta_rows(summary),
            col_widths=[4.0 * cm, 1.8 * cm, 5.2 * cm, 5.2 * cm],
            font_size=6.35,
        )
    )
    story.append(Spacer(1, 0.18 * cm))
    story.append(safe_para("완주성과 loaded 운행", styles["h1"]))
    story.append(
        safe_para(
            "Service rate와 full-completion rate가 동시에 올라간다는 점이 핵심이다. Loaded distance ratio는 Final에서 약간 낮지만, 완주성과 전체 운행 비용 개선 폭을 함께 보면 Begin보다 명확히 안정적인 서비스 상태다.",
            styles["body"],
        )
    )
    story.append(scaled_image(figures[0], max_width=18.0 * cm, max_height=6.4 * cm))
    story.append(Spacer(1, 0.18 * cm))
    story.append(safe_para("승객 시간 분석", styles["h1"]))
    story.append(
        safe_para(
            "대기와 탑승을 따로 보면 Final은 Begin 대비 평균, 중앙값, 최대값을 모두 크게 낮춘다. Mid도 full service를 달성하지만, 이번 export에서는 Final이 승객 시간 지표와 운행 비용을 함께 낮춘 더 안정적인 checkpoint다.",
            styles["body"],
        )
    )
    story.append(scaled_image(figures[1], max_width=18.0 * cm, max_height=6.8 * cm))

    story.append(PageBreak())
    story.append(safe_para("시간-거리 frontier와 승객 체감시간", styles["h1"]))
    story.append(
        safe_para(
            "Frontier 관점에서는 좌하단이 좋은 checkpoint다. Begin은 긴 운행거리와 큰 승객 체감시간을 동시에 보이며 지배당한다. Mid는 full service로 실패를 해소했지만 운행 시간이 길고, Final은 더 낮은 거리와 체감시간으로 이동한 최종 효율점이다.",
            styles["body"],
        )
    )
    story.append(scaled_image(figures[2], max_width=18.0 * cm, max_height=7.1 * cm))
    story.append(Spacer(1, 0.2 * cm))
    story.append(safe_para("운영 효율 분석", styles["h1"]))
    story.append(scaled_image(figures[3], max_width=18.0 * cm, max_height=10.5 * cm))

    story.append(PageBreak())
    story.append(safe_para("불평등과 stop coverage", styles["h1"]))
    story.append(
        safe_para(
            "Wait Gini는 Final이 Begin보다 높아진다. 즉 평균 성능은 크게 좋아졌지만, 승객 간 대기시간 균등성은 아직 개선 여지가 있다. 반대로 Ride Gini는 낮아져 탑승시간 분포는 더 안정적이다. 방문 stop 수는 세 checkpoint 모두 12개로 동일해 coverage 차이가 아니라 순서와 타이밍 차이가 성능을 만든다.",
            styles["body"],
        )
    )
    story.append(scaled_image(figures[4], max_width=18.0 * cm, max_height=6.8 * cm))
    story.append(Spacer(1, 0.16 * cm))
    story.append(safe_para("표 4. Episode-level detail", styles["h1"]))
    story.append(
        styled_table(
            build_episode_rows(episodes),
            col_widths=[1.2 * cm, 0.8 * cm, 2.7 * cm, 1.35 * cm, 1.25 * cm, 1.55 * cm, 1.55 * cm, 1.45 * cm, 1.35 * cm, 1.35 * cm, 0.95 * cm],
            font_size=6.2,
        )
    )
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        KeepTogether(
            [
                safe_para("Caveat", styles["h1"]),
                safe_para(
                    "Begin has one episode, Mid has three episodes, and Final has two episodes. All three exports use matrix/MatrixTeleport execution in the current input folders. "
                    "Therefore, the report should be cited as latest-run export evidence rather than a large-sample statistical claim. Loaded distance ratio is the share of vehicle distance traveled with at least one passenger onboard. Wait and ride inequality are Gini coefficients computed from passenger-level times.",
                    styles["body"],
                ),
            ]
        )
    )

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    register_fonts()
    episodes, passengers, legs, summary = load_data()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    save_tables(episodes, passengers, legs, summary)
    figures = build_figures(episodes, passengers, summary)
    build_pdf(episodes, passengers, summary, figures)
    print(PDF_PATH)
    print(OUTPUT_ROOT / "tables" / "train_progress_summary.csv")


if __name__ == "__main__":
    main()
