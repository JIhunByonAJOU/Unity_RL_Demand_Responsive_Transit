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
    TableStyle,
)


ROOT = Path("DRT_Episode_Exports")
OUTPUT_ROOT = Path("output") / "drt_compare_30Pas_new"
PDF_PATH = Path("output") / "pdf" / "DRT_compare_30Pas_new.pdf"

POLICY_SPECS = [
    ("Vanilla", ROOT / "vanilla", "vanilla"),
    ("FIFO", ROOT / "fifo", "fifo"),
    ("ONNX", ROOT / "inference", "inference"),
]
POLICY_ORDER = [policy for policy, _, _ in POLICY_SPECS]
POLICY_COLORS = {
    "Vanilla": "#BDBDBD",
    "FIFO": "#737373",
    "ONNX": "#1F4E79",
}

FONT_REGULAR = "MalgunGothic"
FONT_BOLD = "MalgunGothic-Bold"
FONT_REGULAR_PATH = Path(r"C:\Windows\Fonts\malgun.ttf")
FONT_BOLD_PATH = Path(r"C:\Windows\Fonts\malgunbd.ttf")


def register_fonts() -> tuple[str, str]:
    if FONT_REGULAR_PATH.exists() and FONT_BOLD_PATH.exists():
        pdfmetrics.registerFont(TTFont(FONT_REGULAR, str(FONT_REGULAR_PATH)))
        pdfmetrics.registerFont(TTFont(FONT_BOLD, str(FONT_BOLD_PATH)))
        plt.rcParams["font.family"] = "Malgun Gothic"
        font = FONT_REGULAR
        font_bold = FONT_BOLD
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
        font = "Helvetica"
        font_bold = "Helvetica-Bold"
    plt.rcParams["axes.unicode_minus"] = False
    return font, font_bold


def to_float(value: object) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def to_int(value: object) -> int:
    try:
        return int(float(str(value)))
    except Exception:
        return 0


def fmt(value: object, digits: int = 1, suffix: str = "") -> str:
    value_f = to_float(value)
    if math.isnan(value_f):
        return "-"
    return f"{value_f:,.{digits}f}{suffix}"


def fmt_pct(value: object, digits: int = 1) -> str:
    value_f = to_float(value)
    if math.isnan(value_f):
        return "-"
    return f"{value_f * 100.0:,.{digits}f}%"


def fmt_delta_abs(value: float, suffix: str = "", digits: int = 1) -> str:
    if math.isnan(value):
        return "-"
    return f"{value:+,.{digits}f}{suffix}"


def fmt_delta_pct(value: float, digits: int = 1) -> str:
    if math.isnan(value):
        return "-"
    return f"{value:+,.{digits}f}%"


def gini(values: list[float]) -> float:
    clean = np.array([v for v in values if not math.isnan(v) and v >= 0.0], dtype=float)
    if len(clean) == 0 or float(clean.sum()) == 0.0:
        return float("nan")
    clean.sort()
    n = len(clean)
    index = np.arange(1, n + 1)
    return float((2.0 * np.dot(index, clean) / (n * clean.sum())) - ((n + 1.0) / n))


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(text), style)


def parse_episode(path: Path) -> tuple[dict[str, str], list[dict[str, str]], list[dict[str, str]]]:
    summary: dict[str, str] = {}
    route_legs: list[dict[str, str]] = []
    passengers: list[dict[str, str]] = []
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
            elif row[0] == "route_leg" and section_header is not None:
                route_legs.append(
                    {
                        section_header[idx]: row[idx + 1] if idx + 1 < len(row) else ""
                        for idx in range(len(section_header))
                    }
                )
            elif row[0] == "passenger" and section_header is not None:
                passengers.append(
                    {
                        section_header[idx]: row[idx + 1] if idx + 1 < len(row) else ""
                        for idx in range(len(section_header))
                    }
                )
    return summary, route_legs, passengers


def select_export_folder(mode_root: Path, token: str) -> tuple[Path, list[Path]]:
    pattern = f"drt_matrix_{token}_scenario_30_*_episode.csv"
    candidates: list[tuple[Path, list[Path]]] = []
    for folder in sorted([p for p in mode_root.iterdir() if p.is_dir()]) if mode_root.exists() else []:
        files = sorted(folder.glob(pattern))
        if files:
            candidates.append((folder, files))
    if not candidates:
        raise SystemExit(f"No compare episode CSV files found under {mode_root} with pattern {pattern}")

    def score(item: tuple[Path, list[Path]]) -> tuple[int, float]:
        folder, files = item
        newest_file_time = max(file.stat().st_mtime for file in files)
        return (len(files), newest_file_time)

    return max(candidates, key=score)


def passenger_onboard_count_for_leg(leg: dict[str, str], passengers: list[dict[str, str]]) -> int:
    departure = to_float(leg.get("departure_time_seconds"))
    arrival = to_float(leg.get("arrival_time_seconds"))
    count = 0
    for passenger in passengers:
        pickup = to_float(passenger.get("pickup_time_seconds"))
        dropoff = to_float(passenger.get("dropoff_time_seconds"))
        if math.isnan(pickup) or pickup > departure + 1e-6:
            continue
        if not math.isnan(dropoff) and dropoff < arrival - 1e-6:
            continue
        count += 1
    return count


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    episode_rows: list[dict[str, object]] = []
    passenger_rows: list[dict[str, object]] = []
    leg_rows: list[dict[str, object]] = []

    for policy_order, (policy, mode_root, token) in enumerate(POLICY_SPECS):
        folder, episode_files = select_export_folder(mode_root, token)
        for episode_file in episode_files:
            summary, route_legs, passengers = parse_episode(episode_file)
            episode_index = to_int(summary.get("episode_index"))
            leg_onboard_counts = [
                passenger_onboard_count_for_leg(leg, passengers) for leg in route_legs
            ]
            leg_distances = [to_float(leg.get("leg_distance_meters")) for leg in route_legs]
            total_distance_m = float(sum(d for d in leg_distances if not math.isnan(d)))
            loaded_distance_m = float(
                sum(
                    distance
                    for distance, onboard_count in zip(leg_distances, leg_onboard_counts)
                    if not math.isnan(distance) and onboard_count > 0
                )
            )
            passenger_distance_m = float(
                sum(
                    distance * onboard_count
                    for distance, onboard_count in zip(leg_distances, leg_onboard_counts)
                    if not math.isnan(distance)
                )
            )
            visited_stops = set()
            for leg in route_legs:
                from_stop = to_int(leg.get("from_stop_id"))
                arrived_stop = to_int(leg.get("arrived_stop_id") or leg.get("to_stop_id"))
                if from_stop:
                    visited_stops.add(from_stop)
                if arrived_stop:
                    visited_stops.add(arrived_stop)

            completed_passengers = to_int(summary.get("completed_passengers"))
            episode_time_seconds = to_float(summary.get("episode_time_seconds"))
            passenger_waits = [
                to_float(passenger.get("wait_time_seconds"))
                for passenger in passengers
                if not math.isnan(to_float(passenger.get("wait_time_seconds")))
            ]
            passenger_rides = [
                to_float(passenger.get("ride_time_seconds"))
                for passenger in passengers
                if passenger.get("status") == "Completed"
                and not math.isnan(to_float(passenger.get("ride_time_seconds")))
            ]
            episode_rows.append(
                {
                    "policy": policy,
                    "policy_order": policy_order,
                    "run_folder": str(folder),
                    "episode_file": episode_file.name,
                    "episode_index": episode_index,
                    "scenario_id": to_int(summary.get("scenario_id")),
                    "scenario_description": summary.get("scenario_description", ""),
                    "finish_reason": summary.get("finish_reason", ""),
                    "completed_all_requests": to_int(summary.get("completed_all_requests")),
                    "total_passengers": to_int(summary.get("total_passengers")),
                    "completed_passengers": completed_passengers,
                    "service_rate": to_float(summary.get("service_rate")),
                    "average_wait_seconds": to_float(summary.get("average_wait_seconds")),
                    "median_wait_seconds": float(np.median(passenger_waits)) if passenger_waits else float("nan"),
                    "max_wait_seconds": max(passenger_waits) if passenger_waits else float("nan"),
                    "wait_gini": gini(passenger_waits),
                    "average_ride_seconds": to_float(summary.get("average_ride_seconds")),
                    "median_ride_seconds": float(np.median(passenger_rides)) if passenger_rides else float("nan"),
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
                    "loaded_distance_ratio": loaded_distance_m / total_distance_m if total_distance_m else float("nan"),
                    "empty_distance_ratio": 1.0 - loaded_distance_m / total_distance_m
                    if total_distance_m
                    else float("nan"),
                    "passenger_km": passenger_distance_m / 1000.0,
                    "avg_onboard_load_distance_weighted": passenger_distance_m / total_distance_m
                    if total_distance_m
                    else float("nan"),
                    "visited_stop_count": len(visited_stops),
                    "route_leg_count": len(route_legs),
                }
            )
            for passenger in passengers:
                passenger_rows.append(
                    {
                        "policy": policy,
                        "policy_order": policy_order,
                        "episode_index": episode_index,
                        "episode_file": episode_file.name,
                        "passenger_id": to_int(passenger.get("passenger_id")),
                        "status": passenger.get("status", ""),
                        "wait_time_seconds": to_float(passenger.get("wait_time_seconds")),
                        "ride_time_seconds": to_float(passenger.get("ride_time_seconds")),
                        "total_service_time_seconds": to_float(passenger.get("total_service_time_seconds")),
                    }
                )
            for leg, onboard_count in zip(route_legs, leg_onboard_counts):
                leg_rows.append(
                    {
                        "policy": policy,
                        "policy_order": policy_order,
                        "episode_index": episode_index,
                        "episode_file": episode_file.name,
                        "leg_index": to_int(leg.get("leg_index")),
                        "from_stop_id": to_int(leg.get("from_stop_id")),
                        "to_stop_id": to_int(leg.get("to_stop_id")),
                        "arrived_stop_id": to_int(leg.get("arrived_stop_id") or leg.get("to_stop_id")),
                        "departure_time_seconds": to_float(leg.get("departure_time_seconds")),
                        "arrival_time_seconds": to_float(leg.get("arrival_time_seconds")),
                        "travel_time_seconds": to_float(leg.get("travel_time_seconds")),
                        "leg_distance_meters": to_float(leg.get("leg_distance_meters")),
                        "boarded_count": to_int(leg.get("boarded_count")),
                        "dropped_off_count": to_int(leg.get("dropped_off_count")),
                        "onboard_during_leg": onboard_count,
                    }
                )

    episodes = pd.DataFrame(episode_rows)
    passengers = pd.DataFrame(passenger_rows)
    legs = pd.DataFrame(leg_rows)
    summary_rows: list[dict[str, object]] = []
    for policy in POLICY_ORDER:
        ep = episodes[episodes["policy"] == policy]
        ps = passengers[passengers["policy"] == policy]
        completed_ps = ps[ps["status"] == "Completed"]
        waits = ps["wait_time_seconds"].dropna().astype(float).tolist()
        rides = completed_ps["ride_time_seconds"].dropna().astype(float).tolist()
        finish_counts = ep["finish_reason"].value_counts().to_dict()
        summary_rows.append(
            {
                "policy": policy,
                "policy_order": int(ep["policy_order"].iloc[0]),
                "run_folder": str(ep["run_folder"].iloc[0]),
                "episodes": len(ep),
                "completed_runs": int(ep["completed_all_requests"].sum()),
                "scenario_id": int(ep["scenario_id"].iloc[0]),
                "service_rate": float(ep["service_rate"].mean()),
                "completed_passengers_mean": float(ep["completed_passengers"].mean()),
                "total_passengers_mean": float(ep["total_passengers"].mean()),
                "wait_mean_seconds": float(np.mean(waits)),
                "wait_min_seconds": float(np.min(waits)),
                "wait_median_seconds": float(np.median(waits)),
                "wait_max_seconds": float(np.max(waits)),
                "wait_gini": gini(waits),
                "ride_mean_seconds": float(np.mean(rides)),
                "ride_min_seconds": float(np.min(rides)),
                "ride_median_seconds": float(np.median(rides)),
                "ride_max_seconds": float(np.max(rides)),
                "ride_gini": gini(rides),
                "total_distance_km": float(ep["episode_distance_meters"].mean() / 1000.0),
                "episode_time_minutes": float(ep["episode_time_seconds"].mean() / 60.0),
                "distance_per_completed_km": float(ep["distance_per_completed_km"].mean()),
                "time_per_completed_minutes": float(ep["time_per_completed_seconds"].mean() / 60.0),
                "loaded_distance_ratio": float(ep["loaded_distance_ratio"].mean()),
                "visited_stop_count": float(ep["visited_stop_count"].mean()),
                "finished_by_completion": int(finish_counts.get("All passenger requests completed.", 0)),
                "finished_by_timeout": int(finish_counts.get("Episode time ended.", 0)),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("policy_order").reset_index(drop=True)
    deltas = build_delta_table(summary)
    return episodes, passengers, legs, summary, deltas


def build_delta_table(summary: pd.DataFrame) -> pd.DataFrame:
    onnx = summary[summary["policy"] == "ONNX"].iloc[0]
    rows: list[dict[str, object]] = []
    metrics = [
        ("service_rate", "Service rate", "higher"),
        ("wait_mean_seconds", "Wait mean", "lower"),
        ("wait_median_seconds", "Wait median", "lower"),
        ("wait_max_seconds", "Wait max", "lower"),
        ("wait_gini", "Wait Gini", "lower"),
        ("ride_mean_seconds", "Ride mean", "lower"),
        ("ride_median_seconds", "Ride median", "lower"),
        ("ride_max_seconds", "Ride max", "lower"),
        ("ride_gini", "Ride Gini", "lower"),
        ("total_distance_km", "Total distance", "lower"),
        ("episode_time_minutes", "Episode time", "lower"),
        ("distance_per_completed_km", "Distance per passenger", "lower"),
        ("time_per_completed_minutes", "Time per passenger", "lower"),
        ("loaded_distance_ratio", "Loaded distance ratio", "higher"),
        ("visited_stop_count", "Visited stops", "lower"),
    ]
    for baseline_policy in ["Vanilla", "FIFO"]:
        baseline = summary[summary["policy"] == baseline_policy].iloc[0]
        for metric_key, metric_label, direction in metrics:
            target = float(onnx[metric_key])
            base = float(baseline[metric_key])
            absolute_delta = target - base
            relative_delta = absolute_delta / base * 100.0 if base else float("nan")
            if direction == "lower":
                improvement = (base - target) / base * 100.0 if base else float("nan")
            else:
                improvement = (target - base) / base * 100.0 if base else float("nan")
            rows.append(
                {
                    "baseline": baseline_policy,
                    "target": "ONNX",
                    "metric": metric_label,
                    "direction": direction,
                    "baseline_value": base,
                    "onnx_value": target,
                    "absolute_delta": absolute_delta,
                    "relative_delta_pct": relative_delta,
                    "improvement_pct": improvement,
                }
            )
    return pd.DataFrame(rows)


def save_tables(
    episodes: pd.DataFrame,
    passengers: pd.DataFrame,
    legs: pd.DataFrame,
    summary: pd.DataFrame,
    deltas: pd.DataFrame,
) -> None:
    tables = OUTPUT_ROOT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(tables / "episode_kpi_metrics.csv", index=False, encoding="utf-8-sig")
    passengers.to_csv(tables / "passenger_kpi_metrics.csv", index=False, encoding="utf-8-sig")
    legs.to_csv(tables / "route_leg_kpi_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(tables / "policy_kpi_summary.csv", index=False, encoding="utf-8-sig")
    deltas.to_csv(tables / "onnx_kpi_delta_vs_baselines.csv", index=False, encoding="utf-8-sig")


def annotate_bars(ax, values: list[float], digits: int = 1, suffix: str = "") -> None:
    for patch, value in zip(ax.patches, values):
        if math.isnan(value):
            continue
        ax.annotate(
            f"{value:.{digits}f}{suffix}",
            (patch.get_x() + patch.get_width() / 2.0, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
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


def bar_ax(ax, summary: pd.DataFrame, metric: str, title: str, ylabel: str, digits: int = 1, suffix: str = "") -> None:
    values = summary[metric].astype(float).tolist()
    policies = summary["policy"].tolist()
    ax.bar(policies, values, color=[POLICY_COLORS[p] for p in policies], width=0.58)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=8)
    apply_ieee_axes(ax)
    annotate_bars(ax, values, digits=digits, suffix=suffix)


def grouped_bar(ax, summary: pd.DataFrame, metrics: list[tuple[str, str]], title: str, ylabel: str) -> None:
    policies = summary["policy"].tolist()
    x = np.arange(len(policies))
    width = 0.22
    hatch_patterns = ["", "//", "\\\\"]
    for idx, (metric, label) in enumerate(metrics):
        values = summary[metric].astype(float).to_numpy()
        ax.bar(
            x + (idx - 1) * width,
            values,
            width=width,
            label=label,
            color=[POLICY_COLORS[p] for p in policies],
            edgecolor="#333333",
            linewidth=0.35,
            hatch=hatch_patterns[idx % len(hatch_patterns)],
        )
    ax.set_xticks(x, policies)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.legend(frameon=False, fontsize=8, ncols=len(metrics), loc="upper center", bbox_to_anchor=(0.5, -0.13))
    apply_ieee_axes(ax)


def build_figures(summary: pd.DataFrame) -> list[Path]:
    figures = OUTPUT_ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), dpi=180)
    bar_ax(axes[0], summary, "service_rate", "Service rate", "ratio", digits=2)
    bar_ax(axes[1], summary, "loaded_distance_ratio", "Loaded distance ratio", "ratio", digits=2)
    fig.tight_layout()
    path = figures / "fig01_service_loaded_ratio.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), dpi=180)
    grouped_bar(
        axes[0],
        summary,
        [
            ("wait_mean_seconds", "Mean"),
            ("wait_median_seconds", "Median"),
            ("wait_max_seconds", "Max"),
        ],
        "Passenger wait time",
        "seconds",
    )
    grouped_bar(
        axes[1],
        summary,
        [
            ("ride_mean_seconds", "Mean"),
            ("ride_median_seconds", "Median"),
            ("ride_max_seconds", "Max"),
        ],
        "Passenger ride time",
        "seconds",
    )
    fig.tight_layout()
    path = figures / "fig02_wait_ride_times.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.7), dpi=180)
    bar_ax(axes[0, 0], summary, "total_distance_km", "Total driving distance", "km", digits=1)
    bar_ax(axes[0, 1], summary, "episode_time_minutes", "Episode time", "minutes", digits=1)
    bar_ax(axes[1, 0], summary, "distance_per_completed_km", "Distance per completed passenger", "km/pass.", digits=2)
    bar_ax(axes[1, 1], summary, "time_per_completed_minutes", "Time per completed passenger", "min/pass.", digits=2)
    fig.tight_layout()
    path = figures / "fig03_operational_efficiency.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), dpi=180)
    bar_ax(axes[0], summary, "wait_gini", "Wait inequality", "Gini", digits=3)
    bar_ax(axes[1], summary, "ride_gini", "Ride inequality", "Gini", digits=3)
    bar_ax(axes[2], summary, "visited_stop_count", "Visited stop count", "stops", digits=0)
    fig.tight_layout()
    path = figures / "fig04_fairness_stops.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def table_style(font: str, font_bold: str, font_size: float = 7.0) -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6E6E6")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("FONTNAME", (0, 0), (-1, 0), font_bold),
            ("FONTNAME", (0, 1), (-1, -1), font),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("LEADING", (0, 0), (-1, -1), font_size + 1.3),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#A8A8A8")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F7F7")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
    )


def styled_table(rows: list[list[object]], font: str, font_bold: str, col_widths: list[float] | None = None, font_size: float = 7.0) -> LongTable:
    table = LongTable(rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(table_style(font, font_bold, font_size=font_size))
    return table


def scaled_image(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as img:
        width_px, height_px = img.size
    ratio = min(max_width / width_px, max_height / height_px)
    return Image(str(path), width=width_px * ratio, height=height_px * ratio)


def source_rows(summary: pd.DataFrame) -> list[list[str]]:
    rows = [["Policy", "Export folder", "Episodes", "Scenario", "Finish"]]
    for _, row in summary.iterrows():
        finish = f"Completed: {int(row['finished_by_completion'])}, Timeout: {int(row['finished_by_timeout'])}"
        rows.append(
            [
                str(row["policy"]),
                str(Path(str(row["run_folder"])).as_posix()),
                str(int(row["episodes"])),
                str(int(row["scenario_id"])),
                finish,
            ]
        )
    return rows


def qos_rows(summary: pd.DataFrame) -> list[list[str]]:
    rows = [
        [
            "Policy",
            "Runs",
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
                str(row["policy"]),
                str(int(row["episodes"])),
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
            "Policy",
            "Total km",
            "Episode min",
            "km/pass.",
            "min/pass.",
            "Loaded ratio",
            "Visited stops",
        ]
    ]
    for _, row in summary.iterrows():
        rows.append(
            [
                str(row["policy"]),
                f"{fmt(row['total_distance_km'])} km",
                f"{fmt(row['episode_time_minutes'])} min",
                f"{fmt(row['distance_per_completed_km'], 2)}",
                f"{fmt(row['time_per_completed_minutes'], 2)}",
                fmt_pct(row["loaded_distance_ratio"]),
                fmt(row["visited_stop_count"], 0),
            ]
        )
    return rows


def onnx_comparison_rows(deltas: pd.DataFrame) -> list[list[str]]:
    selected_metrics = [
        "Wait mean",
        "Wait median",
        "Wait max",
        "Ride mean",
        "Ride median",
        "Ride max",
        "Total distance",
        "Episode time",
        "Distance per passenger",
        "Time per passenger",
        "Loaded distance ratio",
        "Wait Gini",
        "Ride Gini",
    ]
    rows = [["Metric", "ONNX vs Vanilla", "ONNX vs FIFO", "Direction"]]
    for metric in selected_metrics:
        row = [metric]
        direction_text = ""
        for baseline in ["Vanilla", "FIFO"]:
            selected = deltas[(deltas["baseline"] == baseline) & (deltas["metric"] == metric)].iloc[0]
            direction = str(selected["direction"])
            direction_text = "Higher is better" if direction == "higher" else "Lower is better"
            abs_delta = float(selected["absolute_delta"])
            rel_delta = float(selected["relative_delta_pct"])
            if metric in {"Loaded distance ratio", "Wait Gini", "Ride Gini"}:
                if metric == "Loaded distance ratio":
                    value = f"{abs_delta * 100.0:+.1f} pp ({rel_delta:+.1f}%)"
                else:
                    value = f"{abs_delta:+.3f} ({rel_delta:+.1f}%)"
            elif metric == "Total distance":
                value = f"{abs_delta:+.2f} km ({rel_delta:+.1f}%)"
            elif metric == "Distance per passenger":
                value = f"{abs_delta:+.3f} km/pass. ({rel_delta:+.1f}%)"
            elif metric == "Episode time":
                value = f"{abs_delta:+.2f} min ({rel_delta:+.1f}%)"
            elif metric == "Time per passenger":
                value = f"{abs_delta:+.3f} min/pass. ({rel_delta:+.1f}%)"
            else:
                value = f"{abs_delta:+.1f} s ({rel_delta:+.1f}%)"
            row.append(value)
        row.append(direction_text)
        rows.append(row)
    return rows


def footer(font: str):
    def draw(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont(font, 8)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(1.35 * cm, 1.0 * cm, "DRT_compare_30Pas_new")
        canvas.drawRightString(A4[0] - 1.35 * cm, 1.0 * cm, f"Page {doc.page}")
        canvas.restoreState()

    return draw


def build_pdf(summary: pd.DataFrame, deltas: pd.DataFrame, figures: list[Path], font: str, font_bold: str) -> None:
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=1.15 * cm,
        leftMargin=1.15 * cm,
        topMargin=1.15 * cm,
        bottomMargin=1.4 * cm,
    )
    styles = {
        "title": ParagraphStyle(
            "title",
            fontName=font_bold,
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            fontName=font,
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#444444"),
            spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "h1",
            fontName=font_bold,
            fontSize=12,
            leading=15,
            alignment=TA_LEFT,
            spaceBefore=7,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body",
            fontName=font,
            fontSize=8.8,
            leading=12.2,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#222222"),
            spaceAfter=5,
        ),
    }

    vanilla = summary[summary["policy"] == "Vanilla"].iloc[0]
    fifo = summary[summary["policy"] == "FIFO"].iloc[0]
    onnx = summary[summary["policy"] == "ONNX"].iloc[0]
    wait_gain_vanilla = vanilla["wait_mean_seconds"] - onnx["wait_mean_seconds"]
    wait_gain_fifo = fifo["wait_mean_seconds"] - onnx["wait_mean_seconds"]
    ride_gain_fifo = fifo["ride_mean_seconds"] - onnx["ride_mean_seconds"]
    distance_gain_vanilla = vanilla["total_distance_km"] - onnx["total_distance_km"]

    story: list = []
    story.append(para("DRT_compare_30Pas_new", styles["title"]))
    story.append(
        para(
            "Scenario 30 new demand. Policy order is fixed as Vanilla, FIFO, and ONNX in all tables and figures.",
            styles["subtitle"],
        )
    )
    story.append(para("Source Coverage", styles["h1"]))
    story.append(
        styled_table(
            source_rows(summary),
            font,
            font_bold,
            col_widths=[2.0 * cm, 6.4 * cm, 1.6 * cm, 1.5 * cm, 4.9 * cm],
            font_size=6.7,
        )
    )
    story.append(Spacer(1, 0.12 * cm))
    story.append(para("KPI Summary: Passenger Service Quality", styles["h1"]))
    story.append(
        styled_table(
            qos_rows(summary),
            font,
            font_bold,
            col_widths=[
                1.55 * cm,
                1.05 * cm,
                1.25 * cm,
                1.55 * cm,
                1.65 * cm,
                1.55 * cm,
                1.35 * cm,
                1.55 * cm,
                1.65 * cm,
                1.55 * cm,
                1.35 * cm,
            ],
            font_size=6.1,
        )
    )
    story.append(Spacer(1, 0.12 * cm))
    story.append(para("KPI Summary: Operational Efficiency", styles["h1"]))
    story.append(
        styled_table(
            efficiency_rows(summary),
            font,
            font_bold,
            col_widths=[1.8 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 2.2 * cm, 2.0 * cm],
            font_size=6.7,
        )
    )
    story.append(Spacer(1, 0.12 * cm))
    story.append(para("Result", styles["h1"]))
    story.append(
        para(
            "All policies complete all passengers in this 30-passenger scenario, so the main separation comes from "
            f"passenger time and operational efficiency. ONNX has the lowest mean wait ({onnx['wait_mean_seconds']:.1f} s), "
            f"lowest median wait ({onnx['wait_median_seconds']:.1f} s), lowest mean ride ({onnx['ride_mean_seconds']:.1f} s), "
            f"and lowest median ride ({onnx['ride_median_seconds']:.1f} s). Relative to FIFO, ONNX reduces mean wait by "
            f"{wait_gain_fifo:.1f} s and mean ride by {ride_gain_fifo:.1f} s. Relative to Vanilla, ONNX reduces mean wait "
            f"by {wait_gain_vanilla:.1f} s and total driving distance by {distance_gain_vanilla:.1f} km.",
            styles["body"],
        )
    )
    story.append(
        para(
            "The main caveat is tail and fairness behavior: FIFO has lower maximum wait and lower wait/ride Gini than ONNX. "
            "Therefore, the strongest claim is that ONNX is best on average passenger time and vehicle efficiency, while "
            "FIFO remains competitive on worst-case and inequality metrics.",
            styles["body"],
        )
    )

    story.append(PageBreak())
    story.append(para("ONNX Delta Against Baselines", styles["h1"]))
    story.append(
        styled_table(
            onnx_comparison_rows(deltas),
            font,
            font_bold,
            col_widths=[3.4 * cm, 4.8 * cm, 4.8 * cm, 3.1 * cm],
            font_size=6.6,
        )
    )
    story.append(Spacer(1, 0.18 * cm))
    story.append(para("Service and Loaded Operation", styles["h1"]))
    story.append(scaled_image(figures[0], max_width=17.8 * cm, max_height=7.0 * cm))

    story.append(PageBreak())
    story.append(para("Passenger Time KPIs", styles["h1"]))
    story.append(scaled_image(figures[1], max_width=18.1 * cm, max_height=7.5 * cm))
    story.append(Spacer(1, 0.25 * cm))
    story.append(para("Operational Efficiency KPIs", styles["h1"]))
    story.append(scaled_image(figures[2], max_width=18.1 * cm, max_height=11.5 * cm))

    story.append(PageBreak())
    story.append(para("Inequality and Stop Coverage", styles["h1"]))
    story.append(scaled_image(figures[3], max_width=18.1 * cm, max_height=7.3 * cm))
    story.append(Spacer(1, 0.15 * cm))
    story.append(para("Metric Definitions", styles["h1"]))
    story.append(
        para(
            "Wait and ride inequality are reported as Gini coefficients over passenger-level wait and completed-passenger "
            "ride times. Loaded distance ratio is the share of vehicle distance traveled with at least one passenger onboard. "
            "Distance and time per completed passenger divide episode-level vehicle distance or episode duration by the number "
            "of completed passengers.",
            styles["body"],
        )
    )
    story.append(
        KeepTogether(
            [
                para("Interpretive Summary", styles["h1"]),
                para(
                    "For the 30new comparison, ONNX is the preferred policy when the objective prioritizes mean/median "
                    "passenger waiting time, mean/median ride time, and vehicle efficiency. It preserves full service rate "
                    "while using essentially the same distance as FIFO and substantially less distance than Vanilla. The "
                    "remaining weakness is fairness/tail risk, where FIFO produces smaller Gini values and a lower maximum wait.",
                    styles["body"],
                ),
            ]
        )
    )

    doc.build(story, onFirstPage=footer(font), onLaterPages=footer(font))


def main() -> None:
    font, font_bold = register_fonts()
    episodes, passengers, legs, summary, deltas = load_data()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    save_tables(episodes, passengers, legs, summary, deltas)
    figures = build_figures(summary)
    build_pdf(summary, deltas, figures, font, font_bold)
    print(PDF_PATH)
    print(OUTPUT_ROOT / "tables" / "policy_kpi_summary.csv")
    print(OUTPUT_ROOT / "tables" / "onnx_kpi_delta_vs_baselines.csv")


if __name__ == "__main__":
    main()
