#!/usr/bin/env python3
"""Generate a multi-scenario DRT policy sweep report."""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from reportlab.lib import colors
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


SCENARIOS = ["14", "18", "22", "30", "40", "50"]
SCENARIO_VALUES = [14, 18, 22, 30, 40, 50]
POLICY_DIR_TO_DISPLAY = {
    "inference": "ONNX Inference",
    "greedy": "Greedy 1",
    "vanilla": "Vanilla Sequential",
}
LOWER_IS_BETTER = {
    "mean_wait_seconds",
    "p95_wait_seconds",
    "max_wait_seconds",
    "mean_ride_completed_seconds",
    "episode_time_seconds",
    "episode_distance_meters",
    "route_leg_count",
    "no_interaction_count",
}
HIGHER_IS_BETTER = {
    "service_rate",
    "completed_passengers",
    "completed_all_requests",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build scenario-sweep DRT policy comparison report.")
    parser.add_argument("--exports-root", default="DRT_Episode_Exports")
    parser.add_argument("--output", default="output/pdf/drt_scenario_sweep_policy_report.pdf")
    parser.add_argument("--figures-dir", default="output/pdf/drt_scenario_sweep_policy_figures")
    parser.add_argument("--summary-csv", default="output/pdf/drt_scenario_sweep_policy_summary.csv")
    parser.add_argument("--run-metrics-csv", default="output/pdf/drt_scenario_sweep_policy_run_metrics.csv")
    parser.add_argument("--passenger-csv", default="output/pdf/drt_scenario_sweep_policy_passenger_metrics.csv")
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


def load_records(exports_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    route_frames: list[pd.DataFrame] = []
    passenger_frames: list[pd.DataFrame] = []

    for path in sorted(exports_root.rglob("*_episode.csv")):
        try:
            policy_dir = path.parts[path.parts.index(exports_root.name) + 1]
        except (ValueError, IndexError):
            continue
        expected_policy = POLICY_DIR_TO_DISPLAY.get(policy_dir)
        if expected_policy is None:
            continue

        bundle = read_episode_csv(path, expected_policy)
        scenario_id = str(bundle.summary.get("scenario_id") or "")
        if scenario_id not in SCENARIOS:
            continue

        reported_policy = display_policy(bundle.summary.get("next_stop_policy") or bundle.summary.get("policy"))
        if reported_policy not in POLICY_ORDER:
            continue

        metrics, route, passengers = summarize_episode(bundle)
        metrics["scenario_id"] = scenario_id
        metrics["scenario_passengers"] = to_int(scenario_id)
        metric_rows.append(metrics)
        source_rows.append(
            {
                "policy": metrics["policy"],
                "scenario_id": scenario_id,
                "run_dir": Path(path).parent,
                "scenario_description": metrics.get("scenario_description", ""),
            }
        )
        if not route.empty:
            route["scenario_id"] = scenario_id
            route["scenario_passengers"] = to_int(scenario_id)
            route_frames.append(route)
        if not passengers.empty:
            passengers["scenario_id"] = scenario_id
            passengers["scenario_passengers"] = to_int(scenario_id)
            passenger_frames.append(passengers)

    metrics_df = pd.DataFrame(metric_rows)
    if metrics_df.empty:
        raise RuntimeError("No matching 14/18/22/30/40/50 episode CSV records found.")
    metrics_df["scenario_id"] = metrics_df["scenario_id"].astype(str)
    route_df = pd.concat(route_frames, ignore_index=True) if route_frames else pd.DataFrame()
    passenger_df = pd.concat(passenger_frames, ignore_index=True) if passenger_frames else pd.DataFrame()
    source_df = pd.DataFrame(source_rows).drop_duplicates()
    return metrics_df, route_df, passenger_df, source_df


def aggregate(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
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
    rows = []
    for scenario in SCENARIOS:
        for policy in POLICY_ORDER:
            subset = metrics_df[(metrics_df["scenario_id"] == scenario) & (metrics_df["policy"] == policy)]
            for metric in metrics:
                values = pd.to_numeric(subset[metric], errors="coerce").dropna()
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
    return float(row.iloc[0][field])


def mean_series(agg_df: pd.DataFrame, policy: str, metric: str) -> list[float]:
    return [agg_value(agg_df, scenario, policy, metric) for scenario in SCENARIOS]


def best_policy(agg_df: pd.DataFrame, scenario: str, metric: str) -> str:
    values = {policy: agg_value(agg_df, scenario, policy, metric) for policy in POLICY_ORDER}
    values = {policy: value for policy, value in values.items() if not math.isnan(value)}
    if not values:
        return "-"
    if metric in HIGHER_IS_BETTER:
        return max(values, key=values.get)
    return min(values, key=values.get)


def fmt_mean_std(agg_df: pd.DataFrame, scenario: str, policy: str, metric: str, digits: int = 1, suffix: str = "") -> str:
    mean = agg_value(agg_df, scenario, policy, metric, "mean")
    std = agg_value(agg_df, scenario, policy, metric, "std")
    n = to_int(agg_value(agg_df, scenario, policy, metric, "n"))
    if math.isnan(mean):
        return "-"
    if n > 1 and not math.isclose(std, 0.0, abs_tol=1e-8):
        return f"{fmt(mean, digits, suffix)} +/- {fmt(std, digits, suffix)}"
    return fmt(mean, digits, suffix)


def pct_delta(value: float, baseline: float) -> float:
    if math.isnan(value) or math.isnan(baseline) or math.isclose(baseline, 0.0):
        return float("nan")
    return (value - baseline) / baseline


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


def plot_metric_lines(agg_df: pd.DataFrame, figures_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    items = [
        ("mean_wait_seconds", "Mean passenger wait", "s"),
        ("p95_wait_seconds", "P95 passenger wait", "s"),
        ("mean_ride_completed_seconds", "Mean completed ride", "s"),
        ("episode_time_seconds", "Episode completion time", "s"),
    ]
    for ax, (metric, title, ylabel) in zip(axes.flatten(), items):
        for policy in POLICY_ORDER:
            ax.plot(
                SCENARIO_VALUES,
                mean_series(agg_df, policy, metric),
                marker="o",
                linewidth=1.8,
                label=policy,
                color=POLICY_COLORS[policy],
            )
        ax.set_title(title)
        ax.set_xlabel("Scenario passenger count")
        ax.set_ylabel(ylabel)
        ax.set_xticks(SCENARIO_VALUES)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[0, 0].legend(loc="best", frameon=True)
    fig.suptitle("Fig. 1. Scenario-load sweep: passenger service metrics", fontweight="bold", y=1.01)
    fig.tight_layout()
    path = figures_dir / "01_scenario_load_service_metrics.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_completion_and_interactions(agg_df: pd.DataFrame, figures_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    items = [
        ("service_rate", "Service rate", "%", lambda v: v * 100),
        ("no_interaction_count", "No-interaction stops", "count", lambda v: v),
        ("route_leg_count", "Route legs", "legs", lambda v: v),
    ]
    x = np.arange(len(SCENARIOS))
    width = 0.25
    for ax, (metric, title, ylabel, transform) in zip(axes, items):
        for offset, policy in zip([-width, 0, width], POLICY_ORDER):
            y = [transform(agg_value(agg_df, scenario, policy, metric)) for scenario in SCENARIOS]
            ax.bar(x + offset, y, width=width, label=policy, color=POLICY_COLORS[policy], edgecolor="#222222", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(SCENARIOS)
        ax.set_title(title)
        ax.set_xlabel("Passengers")
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[0].legend(loc="lower left", fontsize=7, frameon=True)
    fig.suptitle("Fig. 2. Completion and route-efficiency diagnostics", fontweight="bold", y=1.05)
    fig.tight_layout()
    path = figures_dir / "02_completion_route_diagnostics.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_onnx_delta(agg_df: pd.DataFrame, figures_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True)
    baselines = ["Greedy 1", "Vanilla Sequential"]
    x = np.arange(len(SCENARIOS))
    for ax, baseline in zip(axes, baselines):
        wait_deltas = []
        ride_deltas = []
        for scenario in SCENARIOS:
            onnx_wait = agg_value(agg_df, scenario, "ONNX Inference", "mean_wait_seconds")
            base_wait = agg_value(agg_df, scenario, baseline, "mean_wait_seconds")
            onnx_ride = agg_value(agg_df, scenario, "ONNX Inference", "mean_ride_completed_seconds")
            base_ride = agg_value(agg_df, scenario, baseline, "mean_ride_completed_seconds")
            wait_deltas.append(100 * pct_delta(onnx_wait, base_wait))
            ride_deltas.append(100 * pct_delta(onnx_ride, base_ride))
        ax.axhline(0, color="#222222", linewidth=0.8)
        ax.bar(x - 0.16, wait_deltas, width=0.32, label="Wait", color="#005AB5", edgecolor="#222222", linewidth=0.4)
        ax.bar(x + 0.16, ride_deltas, width=0.32, label="Ride", color="#56B4E9", edgecolor="#222222", linewidth=0.4)
        ax.set_title(f"ONNX vs {baseline}")
        ax.set_xticks(x)
        ax.set_xticklabels(SCENARIOS)
        ax.set_xlabel("Passengers")
        ax.set_ylabel("% change (lower is better)")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    axes[0].legend(loc="upper left", frameon=True)
    fig.suptitle("Fig. 3. ONNX percentage change against baselines", fontweight="bold", y=1.04)
    fig.tight_layout()
    path = figures_dir / "03_onnx_delta_vs_baselines.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_policy_frontier(metrics_df: pd.DataFrame, figures_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.2))
    for ax, scenario in zip(axes, ["14", "30", "50"]):
        subset = metrics_df[metrics_df["scenario_id"] == scenario]
        for policy in POLICY_ORDER:
            s = subset[subset["policy"] == policy]
            ax.scatter(
                s["mean_wait_seconds"],
                s["mean_ride_completed_seconds"],
                s=45,
                alpha=0.75,
                label=policy,
                color=POLICY_COLORS[policy],
                edgecolor="#222222",
                linewidth=0.4,
            )
            ax.scatter(
                [s["mean_wait_seconds"].mean()],
                [s["mean_ride_completed_seconds"].mean()],
                s=150,
                marker="*",
                color=POLICY_COLORS[policy],
                edgecolor="#111111",
                linewidth=0.8,
            )
        ax.set_title(f"Scenario {scenario}")
        ax.set_xlabel("Mean wait (s)")
        ax.set_ylabel("Mean completed ride (s)")
    axes[0].legend(loc="best", fontsize=7, frameon=True)
    fig.suptitle("Fig. 4. Wait-ride frontier under sparse, medium, and high demand", fontweight="bold", y=1.04)
    fig.tight_layout()
    path = figures_dir / "04_wait_ride_frontier_sparse_medium_high.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_best_policy_heatmap(agg_df: pd.DataFrame, figures_dir: Path) -> Path:
    metrics = ["mean_wait_seconds", "p95_wait_seconds", "mean_ride_completed_seconds", "episode_time_seconds", "no_interaction_count"]
    labels = ["Mean wait", "P95 wait", "Mean ride", "Episode time", "No-interaction"]
    policy_codes = {"ONNX Inference": 0, "Greedy 1": 1, "Vanilla Sequential": 2, "-": np.nan}
    matrix = np.zeros((len(metrics), len(SCENARIOS)))
    for i, metric in enumerate(metrics):
        for j, scenario in enumerate(SCENARIOS):
            matrix[i, j] = policy_codes[best_policy(agg_df, scenario, metric)]
    cmap = colors_to_cmap([POLICY_COLORS[p] for p in POLICY_ORDER])
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(np.arange(len(SCENARIOS)))
    ax.set_xticklabels(SCENARIOS)
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Scenario passenger count")
    for i in range(len(metrics)):
        for j, scenario in enumerate(SCENARIOS):
            winner = best_policy(agg_df, scenario, metrics[i])
            short = {"ONNX Inference": "ONNX", "Greedy 1": "Greedy", "Vanilla Sequential": "Vanilla"}.get(winner, "-")
            ax.text(j, i, short, ha="center", va="center", color="#111111", fontsize=8)
    fig.suptitle("Fig. 5. Best policy by scenario and metric", fontweight="bold", y=1.04)
    fig.tight_layout()
    path = figures_dir / "05_best_policy_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def colors_to_cmap(hex_colors: list[str]):
    from matplotlib.colors import ListedColormap

    return ListedColormap(hex_colors)


def make_figures(metrics_df: pd.DataFrame, agg_df: pd.DataFrame, figures_dir: Path) -> list[Path]:
    configure_matplotlib()
    figures_dir.mkdir(parents=True, exist_ok=True)
    for old in figures_dir.glob("*.png"):
        old.unlink()
    return [
        plot_metric_lines(agg_df, figures_dir),
        plot_completion_and_interactions(agg_df, figures_dir),
        plot_onnx_delta(agg_df, figures_dir),
        plot_policy_frontier(metrics_df, figures_dir),
        plot_best_policy_heatmap(agg_df, figures_dir),
    ]


def table_style(header_fill: colors.Color = colors.HexColor("#E9EEF7")) -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), header_fill),
            ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.6),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#B8B8B8")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F7F7")]),
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
    return [[cell(v, header_style if i == 0 else cell_style) for v in row] for i, row in enumerate(rows)]


def scaled_image(path: Path, width: float) -> Image:
    image = Image(str(path))
    aspect = image.imageHeight / max(1, image.imageWidth)
    image.drawWidth = width
    image.drawHeight = width * aspect
    return image


def source_rows(source_df: pd.DataFrame, metrics_df: pd.DataFrame) -> list[list[object]]:
    rows = [["Scenario", "Policy", "Run folder", "Episodes", "Full", "Description"]]
    for scenario in SCENARIOS:
        for policy in POLICY_ORDER:
            subset = metrics_df[(metrics_df["scenario_id"] == scenario) & (metrics_df["policy"] == policy)]
            s = source_df[(source_df["scenario_id"] == scenario) & (source_df["policy"] == policy)]
            folders = sorted({"/".join(Path(path).parts[-2:]) for path in s["run_dir"].astype(str)})
            desc = ", ".join(sorted(s["scenario_description"].astype(str).unique()))
            rows.append(
                [
                    scenario,
                    policy,
                    ", ".join(folders),
                    len(subset),
                    f"{int(subset['completed_all_requests'].sum())}/{len(subset)}",
                    desc,
                ]
            )
    return rows


def scenario_metric_rows(agg_df: pd.DataFrame) -> list[list[object]]:
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
    findings = []
    for scenario in SCENARIOS:
        wait_winner = best_policy(agg_df, scenario, "mean_wait_seconds")
        sr = {policy: agg_value(agg_df, scenario, policy, "service_rate") for policy in POLICY_ORDER}
        onnx_wait = agg_value(agg_df, scenario, "ONNX Inference", "mean_wait_seconds")
        greedy_wait = agg_value(agg_df, scenario, "Greedy 1", "mean_wait_seconds")
        findings.append(
            f"Scenario {scenario}: wait-time winner is {wait_winner}; ONNX wait={fmt(onnx_wait, 1)} s, "
            f"Greedy wait={fmt(greedy_wait, 1)} s, service rates ONNX/Greedy/Vanilla="
            f"{fmt(sr['ONNX Inference'] * 100, 1, '%')}/{fmt(sr['Greedy 1'] * 100, 1, '%')}/{fmt(sr['Vanilla Sequential'] * 100, 1, '%')}."
        )
    onnx_wait_wins = [scenario for scenario in SCENARIOS if best_policy(agg_df, scenario, "mean_wait_seconds") == "ONNX Inference"]
    greedy_wait_wins = [scenario for scenario in SCENARIOS if best_policy(agg_df, scenario, "mean_wait_seconds") == "Greedy 1"]
    findings.append(
        f"Mean-wait winners by load are ONNX={', '.join(onnx_wait_wins) if onnx_wait_wins else 'none'} and "
        f"Greedy={', '.join(greedy_wait_wins) if greedy_wait_wins else 'none'}. This identifies whether the learned policy is a high-load specialist or a general policy."
    )
    findings.append(
        "The low-load cases remain important controls because they expose over-specialization: a learned policy can reduce wait under dense demand while still losing to Greedy on sparse demand."
    )
    findings.append(
        "The 40- and 50-passenger runs extend the original 30-passenger dense case into a stress-test region, where service rate, terminal onboard passengers, and no-interaction loops should be interpreted together with mean wait."
    )
    return findings


def build_recommendations() -> list[str]:
    return [
        "Do not claim general superiority from one passenger-count result alone. The scenario sweep should be framed as load-dependent performance.",
        "The next main experiment should use multiple random seeds per load at cap=4. Include 14/18/22/30/40/50 or a controlled random passenger-count range so the policy sees sparse, medium, dense, and stress-test states during training.",
        "Retrain a mixed-demand policy using randomized passenger count or randomized scenario selection at every episode. Include sparse scenarios so the policy learns when not to chase low-value or future stops, and include 40/50 so it learns completion pressure under high load.",
        "Use the current 30-passenger-trained ONNX as a specialized high-load policy. Train a second mixed-load ONNX and compare both against Greedy 1 and Vanilla across all loads.",
        "Capacity sweep is useful after the demand-load sweep, not before it. Evaluate cap=4/6/8/10 with the same test scenarios and report load ratio alongside raw capacity.",
    ]


def build_pdf(
    output: Path,
    metrics_df: pd.DataFrame,
    source_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    figure_paths: list[Path],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Title"], fontName="Times-Bold", fontSize=17, leading=20, alignment=TA_CENTER)
    subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"], fontName="Times-Roman", fontSize=8.5, leading=10, alignment=TA_CENTER, textColor=colors.HexColor("#444444"), spaceAfter=7)
    heading_style = ParagraphStyle("Heading", parent=styles["Heading2"], fontName="Times-Bold", fontSize=12, leading=14, spaceBefore=7, spaceAfter=5)
    body_style = ParagraphStyle("Body", parent=styles["BodyText"], fontName="Times-Roman", fontSize=9, leading=11, alignment=TA_JUSTIFY, spaceAfter=5)
    bullet_style = ParagraphStyle("Bullet", parent=body_style, leftIndent=10, firstLineIndent=-6, bulletIndent=0, alignment=TA_LEFT)
    caption_style = ParagraphStyle("Caption", parent=body_style, fontName="Times-Italic", fontSize=8, leading=9, alignment=TA_CENTER, textColor=colors.HexColor("#333333"))
    table_header = ParagraphStyle("TableHeader", parent=body_style, fontName="Times-Bold", fontSize=7.3, leading=8.2)
    table_cell = ParagraphStyle("TableCell", parent=body_style, fontName="Times-Roman", fontSize=7.1, leading=8.0)

    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=13 * mm,
        leftMargin=13 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
        title="DRT Scenario Sweep Policy Evaluation",
    )
    story = []
    story.append(paragraph("DRT Scenario Sweep Policy Evaluation", title_style))
    story.append(paragraph("ONNX Inference vs Greedy 1 vs Vanilla Sequential across 14/18/22/30/40/50 passengers", subtitle_style))
    story.append(paragraph(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} from Matrix Teleport episode CSV exports.", subtitle_style))

    story.append(paragraph("Abstract", heading_style))
    story.append(
        paragraph(
            "This report evaluates whether the trained next-stop ONNX policy remains competitive outside the original 30-passenger training-like scenario. "
            "The comparison uses the same exported CSV schema for ONNX inference, Greedy 1 nearest-feasible routing, and Vanilla Sequential routing across "
            "14, 18, 22, 30, 40, and 50 passenger scenarios.",
            body_style,
        )
    )

    story.append(paragraph("Data Sources", heading_style))
    source_table = Table(
        wrap(source_rows(source_df, metrics_df), table_header, table_cell),
        colWidths=[15 * mm, 28 * mm, 50 * mm, 17 * mm, 17 * mm, 50 * mm],
        repeatRows=1,
    )
    source_table.setStyle(table_style())
    story.append(source_table)

    story.append(paragraph("Key Findings", heading_style))
    for finding in build_findings(agg_df):
        story.append(Paragraph(f"- {finding}", bullet_style))

    story.append(PageBreak())
    story.append(paragraph("Scenario-Level Metric Summary", heading_style))
    metric_table = Table(
        wrap(scenario_metric_rows(agg_df), table_header, table_cell),
        colWidths=[14 * mm, 28 * mm, 16 * mm, 24 * mm, 24 * mm, 24 * mm, 25 * mm, 16 * mm, 17 * mm],
        repeatRows=1,
    )
    metric_table.setStyle(table_style(colors.HexColor("#DCE6F2")))
    story.append(metric_table)

    story.append(PageBreak())
    story.append(paragraph("Figures", heading_style))
    story.append(paragraph("Figures 1-5 visualize the load-dependent behavior. The key result is whether ONNX remains competitive as demand increases from sparse loads to 40/50-passenger stress-test loads.", body_style))
    figure_captions = [
        "Scenario-load passenger service metrics.",
        "Completion and route-efficiency diagnostics.",
        "ONNX percentage change against Greedy and Vanilla baselines.",
        "Wait-ride frontier under sparse, medium, and high demand.",
        "Best policy by scenario and metric.",
    ]
    for idx, fig_path in enumerate(figure_paths[:3], start=1):
        story.append(scaled_image(fig_path, 175 * mm))
        story.append(paragraph(f"Figure {idx}. {figure_captions[idx - 1]}", caption_style))

    story.append(PageBreak())
    for idx, fig_path in enumerate(figure_paths[3:], start=4):
        story.append(scaled_image(fig_path, 175 * mm))
        story.append(paragraph(f"Figure {idx}. {figure_captions[idx - 1]}", caption_style))

    story.append(PageBreak())
    story.append(paragraph("Recommended Next Experiments", heading_style))
    for recommendation in build_recommendations():
        story.append(Paragraph(f"- {recommendation}", bullet_style))
    story.append(Spacer(1, 5))
    story.append(
        paragraph(
            "Training recommendation: keep the current reward structure with the no-interaction penalty, but train a new mixed-demand policy with episode-level "
            "scenario randomization over 14/18/22/30/40/50 passenger CSVs and several new random seeds per passenger count. Keep bus capacity fixed at 4 for this next run. "
            "Only after the mixed-load policy is stable should capacity 6/8/10 be used as a zero-shot sensitivity test or as a separate retraining study.",
            body_style,
        )
    )
    story.append(
        paragraph(
            "Interpretation recommendation: present the current ONNX as a high-load specialist and Greedy 1 as a strong non-learning baseline. A paper-strength claim "
            "should be based on where the learned policy beats Greedy under high operational load, and where the mixed-demand retrained policy closes the sparse-demand gap.",
            body_style,
        )
    )

    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)


def page_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Times-Roman", 7)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(13 * mm, 8 * mm, "DRT scenario-sweep policy comparison")
    canvas.drawRightString(A4[0] - 13 * mm, 8 * mm, f"Page {doc.page}")
    canvas.restoreState()


def save_outputs(metrics_df: pd.DataFrame, passenger_df: pd.DataFrame, agg_df: pd.DataFrame, args: argparse.Namespace) -> None:
    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)
    agg_df.to_csv(args.summary_csv, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(args.run_metrics_csv, index=False, encoding="utf-8-sig")
    passenger_df.to_csv(args.passenger_csv, index=False, encoding="utf-8-sig")


def main() -> int:
    args = parse_args()
    metrics_df, route_df, passenger_df, source_df = load_records(Path(args.exports_root))
    agg_df = aggregate(metrics_df)
    figures_dir = Path(args.figures_dir)
    figure_paths = make_figures(metrics_df, agg_df, figures_dir)
    save_outputs(metrics_df, passenger_df, agg_df, args)
    build_pdf(Path(args.output), metrics_df, source_df, agg_df, figure_paths)
    print(f"PDF: {args.output}")
    print(f"Figures: {figures_dir}")
    print(f"Summary CSV: {args.summary_csv}")
    for scenario in SCENARIOS:
        print(f"Scenario {scenario}")
        for policy in POLICY_ORDER:
            n = to_int(agg_value(agg_df, scenario, policy, "service_rate", "n"))
            print(
                f"  {policy}: n={n}, sr={fmt(agg_value(agg_df, scenario, policy, 'service_rate'), 3)}, "
                f"wait={fmt(agg_value(agg_df, scenario, policy, 'mean_wait_seconds'), 1)}s, "
                f"ride={fmt(agg_value(agg_df, scenario, policy, 'mean_ride_completed_seconds'), 1)}s, "
                f"noInt={fmt(agg_value(agg_df, scenario, policy, 'no_interaction_count'), 1)}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
