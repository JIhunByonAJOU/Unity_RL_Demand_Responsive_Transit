#!/usr/bin/env python3
"""Generate a compact 14/18/22/30 DRT policy comparison report.

This report intentionally selects the latest run folder per policy and scenario
so newly exported ONNX inference runs are not mixed with older inference CSVs.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
from reportlab.lib import colors as rl_colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drt_three_policy_report import (  # noqa: E402
    POLICY_COLORS,
    POLICY_ORDER,
    display_policy,
    fmt,
    read_episode_csv,
    summarize_episode,
)


SCENARIOS = ["14", "18", "22", "30"]
SCENARIO_VALUES = [14, 18, 22, 30]
POLICY_DIR_TO_DISPLAY = {
    "inference": "ONNX Inference",
    "greedy": "Greedy 1",
    "vanilla": "Vanilla Sequential",
}
METRICS = [
    "service_rate",
    "completed_all_requests",
    "completed_passengers",
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
LOWER_IS_BETTER = {
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
}
HIGHER_IS_BETTER = {"service_rate", "completed_passengers", "completed_all_requests"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build latest 14/18/22/30 DRT policy comparison report.")
    parser.add_argument("--exports-root", default="DRT_Episode_Exports")
    parser.add_argument("--output", default="output/pdf/drt_latest_14_30_policy_report.pdf")
    parser.add_argument("--figures-dir", default="output/pdf/drt_latest_14_30_policy_figures")
    parser.add_argument("--summary-csv", default="output/pdf/drt_latest_14_30_policy_summary.csv")
    parser.add_argument("--run-metrics-csv", default="output/pdf/drt_latest_14_30_policy_run_metrics.csv")
    parser.add_argument("--passenger-csv", default="output/pdf/drt_latest_14_30_policy_passenger_metrics.csv")
    return parser.parse_args()


def to_float(value: object, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result


def to_int(value: object, default: int = 0) -> int:
    number = to_float(value)
    if math.isnan(number):
        return default
    return int(round(number))


def run_folder(path: Path, exports_root: Path) -> Path:
    parts = path.parts
    idx = parts.index(exports_root.name)
    return Path(*parts[: idx + 3])


def scan_episode_files(exports_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(exports_root.rglob("*_episode.csv")):
        try:
            policy_dir = path.parts[path.parts.index(exports_root.name) + 1]
        except (ValueError, IndexError):
            continue
        expected_policy = POLICY_DIR_TO_DISPLAY.get(policy_dir)
        if expected_policy is None:
            continue
        try:
            bundle = read_episode_csv(path, expected_policy)
        except Exception as exc:  # noqa: BLE001
            print(f"Skipping unreadable CSV: {path} ({exc})", file=sys.stderr)
            continue
        scenario = str(bundle.summary.get("scenario_id") or "")
        if scenario not in SCENARIOS:
            continue
        policy = display_policy(bundle.summary.get("next_stop_policy") or bundle.summary.get("policy")) or expected_policy
        if policy not in POLICY_ORDER:
            continue
        rows.append(
            {
                "path": path,
                "policy_dir": policy_dir,
                "policy": expected_policy,
                "reported_policy": policy,
                "scenario_id": scenario,
                "scenario_description": bundle.summary.get("scenario_description", ""),
                "run_dir": run_folder(path, exports_root),
                "mtime": path.stat().st_mtime,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No matching 14/18/22/30 inference/greedy/vanilla episode CSVs were found.")
    return frame


def select_latest_run_files(index_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_rows: list[pd.DataFrame] = []
    source_rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        for policy in POLICY_ORDER:
            subset = index_df[(index_df["scenario_id"] == scenario) & (index_df["policy"] == policy)]
            if subset.empty:
                continue
            run_stats = (
                subset.groupby("run_dir", as_index=False)
                .agg(mtime=("mtime", "max"), episodes=("path", "count"))
                .sort_values(["mtime", "episodes"], ascending=[False, False])
            )
            selected_run = run_stats.iloc[0]["run_dir"]
            run_files = subset[subset["run_dir"] == selected_run].copy()
            selected_rows.append(run_files)
            source_rows.append(
                {
                    "scenario_id": scenario,
                    "policy": policy,
                    "run_dir": selected_run,
                    "episodes": len(run_files),
                    "latest_mtime": float(run_files["mtime"].max()),
                    "scenario_description": ", ".join(sorted(set(run_files["scenario_description"].astype(str)))),
                }
            )
    if not selected_rows:
        raise RuntimeError("No latest run selections could be made.")
    return pd.concat(selected_rows, ignore_index=True), pd.DataFrame(source_rows)


def load_records(selected_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    passenger_frames: list[pd.DataFrame] = []
    route_frames: list[pd.DataFrame] = []
    for _, row in selected_df.iterrows():
        bundle = read_episode_csv(Path(row["path"]), str(row["policy"]))
        metrics, route, passengers = summarize_episode(bundle)
        metrics["scenario_id"] = str(row["scenario_id"])
        metrics["scenario_passengers"] = to_int(row["scenario_id"])
        metrics["selected_run_dir"] = str(row["run_dir"])
        metric_rows.append(metrics)
        if not route.empty:
            route["scenario_id"] = str(row["scenario_id"])
            route["scenario_passengers"] = to_int(row["scenario_id"])
            route_frames.append(route)
        if not passengers.empty:
            passengers["scenario_id"] = str(row["scenario_id"])
            passengers["scenario_passengers"] = to_int(row["scenario_id"])
            passenger_frames.append(passengers)
    metrics_df = pd.DataFrame(metric_rows)
    route_df = pd.concat(route_frames, ignore_index=True) if route_frames else pd.DataFrame()
    passenger_df = pd.concat(passenger_frames, ignore_index=True) if passenger_frames else pd.DataFrame()
    return metrics_df, route_df, passenger_df


def aggregate(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        for policy in POLICY_ORDER:
            subset = metrics_df[(metrics_df["scenario_id"] == scenario) & (metrics_df["policy"] == policy)]
            for metric in METRICS:
                values = pd.to_numeric(subset.get(metric, pd.Series(dtype=float)), errors="coerce").dropna()
                rows.append(
                    {
                        "scenario_id": scenario,
                        "scenario_passengers": to_int(scenario),
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


def agg_value(agg_df: pd.DataFrame, scenario: str, policy: str, metric: str, field: str = "mean") -> float:
    row = agg_df[(agg_df["scenario_id"] == scenario) & (agg_df["policy"] == policy) & (agg_df["metric"] == metric)]
    if row.empty:
        return float("nan")
    return to_float(row.iloc[0][field])


def fmt_mean_std(agg_df: pd.DataFrame, scenario: str, policy: str, metric: str, digits: int = 1, suffix: str = "") -> str:
    mean = agg_value(agg_df, scenario, policy, metric, "mean")
    std = agg_value(agg_df, scenario, policy, metric, "std")
    n = to_int(agg_value(agg_df, scenario, policy, metric, "n"))
    if math.isnan(mean):
        return "-"
    if n > 1 and not math.isclose(std, 0.0, abs_tol=1e-9):
        return f"{fmt(mean, digits, suffix)}+/-{fmt(std, digits, suffix)}"
    return fmt(mean, digits, suffix)


def best_policy(agg_df: pd.DataFrame, scenario: str, metric: str) -> str:
    values = {
        policy: agg_value(agg_df, scenario, policy, metric)
        for policy in POLICY_ORDER
        if not math.isnan(agg_value(agg_df, scenario, policy, metric))
    }
    if not values:
        return "-"
    if metric in HIGHER_IS_BETTER:
        return max(values, key=values.get)
    return min(values, key=values.get)


def pct_delta(value: float, baseline: float) -> float:
    if math.isnan(value) or math.isnan(baseline) or math.isclose(baseline, 0.0):
        return float("nan")
    return 100.0 * (value - baseline) / baseline


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


def series(agg_df: pd.DataFrame, policy: str, metric: str) -> list[float]:
    return [agg_value(agg_df, scenario, policy, metric) for scenario in SCENARIOS]


def plot_service_metrics(agg_df: pd.DataFrame, figures_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.1))
    items = [
        ("service_rate", "Service completion rate", "%", lambda x: x * 100),
        ("mean_wait_seconds", "Mean passenger wait", "s", lambda x: x),
        ("p95_wait_seconds", "P95 passenger wait", "s", lambda x: x),
        ("mean_ride_completed_seconds", "Mean completed ride", "s", lambda x: x),
    ]
    for ax, (metric, title, ylabel, transform) in zip(axes.flatten(), items):
        for policy in POLICY_ORDER:
            values = [transform(v) for v in series(agg_df, policy, metric)]
            ax.plot(SCENARIO_VALUES, values, marker="o", linewidth=1.8, color=POLICY_COLORS[policy], label=policy)
        ax.set_title(title)
        ax.set_xlabel("Passenger scenario")
        ax.set_ylabel(ylabel)
        ax.set_xticks(SCENARIO_VALUES)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[0, 0].legend(loc="best", frameon=True)
    fig.suptitle("Fig. 1. Service quality across passenger loads", fontweight="bold", y=1.01)
    fig.tight_layout()
    path = figures_dir / "01_service_quality_by_scenario.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_onnx_deltas(agg_df: pd.DataFrame, figures_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    x = np.arange(len(SCENARIOS))
    for ax, baseline in zip(axes, ["Greedy 1", "Vanilla Sequential"]):
        wait_delta = [
            pct_delta(
                agg_value(agg_df, scenario, "ONNX Inference", "mean_wait_seconds"),
                agg_value(agg_df, scenario, baseline, "mean_wait_seconds"),
            )
            for scenario in SCENARIOS
        ]
        ride_delta = [
            pct_delta(
                agg_value(agg_df, scenario, "ONNX Inference", "mean_ride_completed_seconds"),
                agg_value(agg_df, scenario, baseline, "mean_ride_completed_seconds"),
            )
            for scenario in SCENARIOS
        ]
        ax.axhline(0, color="#222222", linewidth=0.8)
        ax.bar(x - 0.17, wait_delta, width=0.34, color="#005AB5", edgecolor="#222222", linewidth=0.4, label="Mean wait")
        ax.bar(x + 0.17, ride_delta, width=0.34, color="#56B4E9", edgecolor="#222222", linewidth=0.4, label="Mean ride")
        ax.set_title(f"ONNX vs {baseline}")
        ax.set_xticks(x)
        ax.set_xticklabels(SCENARIOS)
        ax.set_xlabel("Passenger scenario")
        ax.set_ylabel("% change; lower is better")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    axes[0].legend(loc="best", frameon=True)
    fig.suptitle("Fig. 2. Learned policy deltas against baselines", fontweight="bold", y=1.04)
    fig.tight_layout()
    path = figures_dir / "02_onnx_delta_vs_baselines.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_route_diagnostics(agg_df: pd.DataFrame, figures_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9))
    items = [
        ("no_interaction_count", "No-interaction stops", "count"),
        ("terminal_on_board", "Terminal onboard", "passengers"),
        ("abab_pattern_count", "ABAB loop motifs", "count"),
    ]
    x = np.arange(len(SCENARIOS))
    width = 0.25
    for ax, (metric, title, ylabel) in zip(axes, items):
        for offset, policy in zip([-width, 0, width], POLICY_ORDER):
            y = series(agg_df, policy, metric)
            ax.bar(x + offset, y, width=width, label=policy, color=POLICY_COLORS[policy], edgecolor="#222222", linewidth=0.4)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(SCENARIOS)
        ax.set_xlabel("Passenger scenario")
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[0].legend(loc="best", fontsize=7, frameon=True)
    fig.suptitle("Fig. 3. Route quality diagnostics", fontweight="bold", y=1.05)
    fig.tight_layout()
    path = figures_dir / "03_route_quality_diagnostics.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_best_policy_map(agg_df: pd.DataFrame, figures_dir: Path) -> Path:
    metrics = [
        ("service_rate", "Service rate"),
        ("mean_wait_seconds", "Mean wait"),
        ("p95_wait_seconds", "P95 wait"),
        ("mean_ride_completed_seconds", "Mean ride"),
        ("no_interaction_count", "No-int stops"),
    ]
    code = {"ONNX Inference": 0, "Greedy 1": 1, "Vanilla Sequential": 2, "-": np.nan}
    matrix = np.zeros((len(metrics), len(SCENARIOS)))
    for i, (metric, _) in enumerate(metrics):
        for j, scenario in enumerate(SCENARIOS):
            matrix[i, j] = code[best_policy(agg_df, scenario, metric)]
    cmap = ListedColormap([POLICY_COLORS[p] for p in POLICY_ORDER])
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(np.arange(len(SCENARIOS)))
    ax.set_xticklabels(SCENARIOS)
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels([label for _, label in metrics])
    ax.set_xlabel("Passenger scenario")
    short = {"ONNX Inference": "ONNX", "Greedy 1": "Greedy", "Vanilla Sequential": "Vanilla", "-": "-"}
    for i, (metric, _) in enumerate(metrics):
        for j, scenario in enumerate(SCENARIOS):
            ax.text(j, i, short[best_policy(agg_df, scenario, metric)], ha="center", va="center", fontsize=8, color="#111111")
    fig.suptitle("Fig. 4. Best policy by metric and load", fontweight="bold", y=1.05)
    fig.tight_layout()
    path = figures_dir / "04_best_policy_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def make_figures(agg_df: pd.DataFrame, figures_dir: Path) -> list[Path]:
    configure_matplotlib()
    figures_dir.mkdir(parents=True, exist_ok=True)
    for old in figures_dir.glob("*.png"):
        old.unlink()
    return [
        plot_service_metrics(agg_df, figures_dir),
        plot_onnx_deltas(agg_df, figures_dir),
        plot_route_diagnostics(agg_df, figures_dir),
        plot_best_policy_map(agg_df, figures_dir),
    ]


def table_style(header_fill: rl_colors.Color = rl_colors.HexColor("#E9EEF7")) -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), header_fill),
            ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.2),
            ("GRID", (0, 0), (-1, -1), 0.25, rl_colors.HexColor("#B8B8B8")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor("#F7F7F7")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]
    )


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def cell(text: object, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(str(text)), style)


def wrap(rows: list[list[object]], header_style: ParagraphStyle, cell_style: ParagraphStyle) -> list[list[Paragraph]]:
    return [[cell(value, header_style if row_index == 0 else cell_style) for value in row] for row_index, row in enumerate(rows)]


def scaled_image(path: Path, width: float) -> Image:
    image = Image(str(path))
    aspect = image.imageHeight / max(1, image.imageWidth)
    image.drawWidth = width
    image.drawHeight = width * aspect
    return image


def source_rows(source_df: pd.DataFrame, metrics_df: pd.DataFrame) -> list[list[object]]:
    rows = [["Scenario", "Policy", "Selected folder", "Runs", "Full", "Source"]]
    for scenario in SCENARIOS:
        for policy in POLICY_ORDER:
            source = source_df[(source_df["scenario_id"] == scenario) & (source_df["policy"] == policy)]
            subset = metrics_df[(metrics_df["scenario_id"] == scenario) & (metrics_df["policy"] == policy)]
            if source.empty:
                rows.append([scenario, policy, "-", 0, "0/0", "-"])
                continue
            folder = "/".join(Path(str(source.iloc[0]["run_dir"])).parts[-2:])
            desc = str(source.iloc[0]["scenario_description"])
            rows.append(
                [
                    scenario,
                    policy,
                    folder,
                    len(subset),
                    f"{int(pd.to_numeric(subset['completed_all_requests'], errors='coerce').sum())}/{len(subset)}",
                    desc,
                ]
            )
    return rows


def metric_rows(agg_df: pd.DataFrame) -> list[list[object]]:
    rows = [["Scenario", "Policy", "SR", "Mean wait", "P95 wait", "Mean ride", "Episode time", "No-int", "End OB"]]
    for scenario in SCENARIOS:
        for policy in POLICY_ORDER:
            rows.append(
                [
                    scenario,
                    policy,
                    fmt_mean_std(agg_df, scenario, policy, "service_rate", 3),
                    fmt_mean_std(agg_df, scenario, policy, "mean_wait_seconds", 1, " s"),
                    fmt_mean_std(agg_df, scenario, policy, "p95_wait_seconds", 1, " s"),
                    fmt_mean_std(agg_df, scenario, policy, "mean_ride_completed_seconds", 1, " s"),
                    fmt_mean_std(agg_df, scenario, policy, "episode_time_seconds", 1, " s"),
                    fmt_mean_std(agg_df, scenario, policy, "no_interaction_count", 1),
                    fmt_mean_std(agg_df, scenario, policy, "terminal_on_board", 1),
                ]
            )
    return rows


def build_findings(agg_df: pd.DataFrame) -> list[str]:
    findings: list[str] = []
    onnx_wait_wins = [scenario for scenario in SCENARIOS if best_policy(agg_df, scenario, "mean_wait_seconds") == "ONNX Inference"]
    full_completion = {
        scenario: agg_value(agg_df, scenario, "ONNX Inference", "service_rate")
        for scenario in SCENARIOS
    }
    findings.append(
        "Latest ONNX is best on mean wait in "
        f"{', '.join(onnx_wait_wins) if onnx_wait_wins else 'no scenario'}; Greedy remains the strongest low-load benchmark when it wins wait."
    )
    findings.append(
        "Across the selected ONNX runs, service rates are "
        + ", ".join(f"{s}: {fmt(full_completion[s] * 100, 1, '%')}" for s in SCENARIOS)
        + ". Interpret wait-time gains only together with completion and terminal-onboard diagnostics."
    )
    for scenario in SCENARIOS:
        onnx_wait = agg_value(agg_df, scenario, "ONNX Inference", "mean_wait_seconds")
        greedy_wait = agg_value(agg_df, scenario, "Greedy 1", "mean_wait_seconds")
        vanilla_wait = agg_value(agg_df, scenario, "Vanilla Sequential", "mean_wait_seconds")
        onnx_ride = agg_value(agg_df, scenario, "ONNX Inference", "mean_ride_completed_seconds")
        findings.append(
            f"Scenario {scenario}: ONNX wait={fmt(onnx_wait, 1)} s, Greedy={fmt(greedy_wait, 1)} s, "
            f"Vanilla={fmt(vanilla_wait, 1)} s; ONNX completed-ride time={fmt(onnx_ride, 1)} s."
        )
    return findings


def build_conclusion(agg_df: pd.DataFrame) -> str:
    wait_winners = {scenario: best_policy(agg_df, scenario, "mean_wait_seconds") for scenario in SCENARIOS}
    onnx_wins = sum(1 for winner in wait_winners.values() if winner == "ONNX Inference")
    greedy_wins = sum(1 for winner in wait_winners.values() if winner == "Greedy 1")
    if onnx_wins >= greedy_wins:
        return (
            "The latest ONNX policy is competitive as a demand-aware routing policy over the evaluated 14-30 passenger range. "
            "The paper claim should emphasize load-dependent gains, while using Greedy 1 as the primary non-learning baseline."
        )
    return (
        "The latest ONNX policy does not dominate Greedy 1 over the full 14-30 passenger range. "
        "The paper claim should therefore focus on where learning improves wait under denser loads, and treat Greedy 1 as a strong heuristic baseline."
    )


def page_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Times-Roman", 7)
    canvas.setFillColor(rl_colors.HexColor("#666666"))
    canvas.drawString(13 * mm, 8 * mm, "DRT 14/18/22/30 policy comparison")
    canvas.drawRightString(A4[0] - 13 * mm, 8 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf(output: Path, source_df: pd.DataFrame, metrics_df: pd.DataFrame, agg_df: pd.DataFrame, figures: list[Path]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("Title", parent=styles["Title"], fontName="Times-Bold", fontSize=17, leading=20, alignment=TA_CENTER)
    subtitle = ParagraphStyle("Subtitle", parent=styles["Normal"], fontName="Times-Roman", fontSize=8.4, leading=10, alignment=TA_CENTER, textColor=rl_colors.HexColor("#444444"), spaceAfter=7)
    heading = ParagraphStyle("Heading", parent=styles["Heading2"], fontName="Times-Bold", fontSize=12, leading=14, spaceBefore=7, spaceAfter=5)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontName="Times-Roman", fontSize=9, leading=11, alignment=TA_JUSTIFY, spaceAfter=5)
    bullet = ParagraphStyle("Bullet", parent=body, leftIndent=10, firstLineIndent=-6, bulletIndent=0, alignment=TA_LEFT)
    caption = ParagraphStyle("Caption", parent=body, fontName="Times-Italic", fontSize=8, leading=9, alignment=TA_CENTER, textColor=rl_colors.HexColor("#333333"))
    table_header = ParagraphStyle("TableHeader", parent=body, fontName="Times-Bold", fontSize=7.0, leading=8.0)
    table_cell = ParagraphStyle("TableCell", parent=body, fontName="Times-Roman", fontSize=6.9, leading=7.8)

    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=13 * mm,
        leftMargin=13 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
        title="DRT Latest 14-30 Policy Evaluation",
    )
    story: list[object] = []
    story.append(paragraph("DRT Latest 14-30 Policy Evaluation", title))
    story.append(paragraph("ONNX Inference vs Greedy 1 vs Vanilla Sequential", subtitle))
    story.append(paragraph(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} from Matrix Teleport episode CSV exports.", subtitle))
    story.append(paragraph("Abstract", heading))
    story.append(
        paragraph(
            "This report compares the newly exported ONNX inference runs against Greedy 1 and Vanilla Sequential baselines for 14, 18, 22, and 30 passenger scenarios. "
            "Only the latest run folder per policy and scenario is selected to avoid mixing older ONNX checkpoints with the current experiment.",
            body,
        )
    )
    story.append(paragraph("Key Findings", heading))
    for finding in build_findings(agg_df):
        story.append(Paragraph(f"- {finding}", bullet))
    story.append(paragraph("Conclusion", heading))
    story.append(paragraph(build_conclusion(agg_df), body))

    story.append(paragraph("Selected CSV Sources", heading))
    source_table = Table(
        wrap(source_rows(source_df, metrics_df), table_header, table_cell),
        colWidths=[14 * mm, 28 * mm, 48 * mm, 14 * mm, 16 * mm, 52 * mm],
        repeatRows=1,
    )
    source_table.setStyle(table_style())
    story.append(source_table)

    story.append(PageBreak())
    story.append(paragraph("Scenario-Level Summary", heading))
    table = Table(
        wrap(metric_rows(agg_df), table_header, table_cell),
        colWidths=[14 * mm, 28 * mm, 16 * mm, 24 * mm, 24 * mm, 24 * mm, 25 * mm, 16 * mm, 17 * mm],
        repeatRows=1,
    )
    table.setStyle(table_style(rl_colors.HexColor("#DCE6F2")))
    story.append(table)

    story.append(PageBreak())
    story.append(paragraph("Figures", heading))
    captions = [
        "Service completion, wait time, and ride time across passenger loads.",
        "ONNX percentage change against Greedy and Vanilla baselines.",
        "Route diagnostics for empty stops, terminal onboard passengers, and ABAB motifs.",
        "Best policy by metric and load.",
    ]
    for index, fig in enumerate(figures, start=1):
        story.append(scaled_image(fig, 175 * mm))
        story.append(paragraph(f"Figure {index}. {captions[index - 1]}", caption))
        if index == 2:
            story.append(PageBreak())

    story.append(Spacer(1, 5))
    story.append(paragraph("Recommended Next Comparison", heading))
    story.append(
        paragraph(
            "For the next paper-strength comparison, keep cap=4 and evaluate the same three policies over multiple random seeds per scenario. "
            "Then add a capacity sensitivity table only after the demand-load comparison is stable; otherwise capacity changes can obscure whether the routing policy or the easier load condition caused the improvement.",
            body,
        )
    )
    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)


def save_outputs(metrics_df: pd.DataFrame, passenger_df: pd.DataFrame, agg_df: pd.DataFrame, args: argparse.Namespace) -> None:
    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)
    agg_df.to_csv(args.summary_csv, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(args.run_metrics_csv, index=False, encoding="utf-8-sig")
    passenger_df.to_csv(args.passenger_csv, index=False, encoding="utf-8-sig")


def main() -> int:
    args = parse_args()
    exports_root = Path(args.exports_root)
    index_df = scan_episode_files(exports_root)
    selected_df, source_df = select_latest_run_files(index_df)
    metrics_df, _route_df, passenger_df = load_records(selected_df)
    agg_df = aggregate(metrics_df)
    figures = make_figures(agg_df, Path(args.figures_dir))
    save_outputs(metrics_df, passenger_df, agg_df, args)
    build_pdf(Path(args.output), source_df, metrics_df, agg_df, figures)

    print(f"PDF: {args.output}")
    print(f"Figures: {args.figures_dir}")
    print(f"Summary CSV: {args.summary_csv}")
    print(f"Run metrics CSV: {args.run_metrics_csv}")
    for scenario in SCENARIOS:
        print(f"Scenario {scenario}")
        for policy in POLICY_ORDER:
            print(
                f"  {policy}: n={to_int(agg_value(agg_df, scenario, policy, 'service_rate', 'n'))}, "
                f"SR={fmt(agg_value(agg_df, scenario, policy, 'service_rate'), 3)}, "
                f"wait={fmt(agg_value(agg_df, scenario, policy, 'mean_wait_seconds'), 1)}s, "
                f"p95={fmt(agg_value(agg_df, scenario, policy, 'p95_wait_seconds'), 1)}s, "
                f"ride={fmt(agg_value(agg_df, scenario, policy, 'mean_ride_completed_seconds'), 1)}s, "
                f"noInt={fmt(agg_value(agg_df, scenario, policy, 'no_interaction_count'), 1)}, "
                f"endOB={fmt(agg_value(agg_df, scenario, policy, 'terminal_on_board'), 1)}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
