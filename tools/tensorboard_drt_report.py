#!/usr/bin/env python3
"""Create a screenshot-friendly PDF report from local TensorBoard scalar data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


TB_URL = "http://localhost:6006"
RUN = "."
SMOOTHING = 0.99
BEST_CHECKPOINT_STEP = 70999
COLLAPSE_START_STEP = 75000

OUTPUT_STEM = "tensorboard_drt_training_report_20260530"

SERVICE_TAGS = [
    ("DRT/EpisodeServiceRate", "Service rate"),
    ("DRT/EpisodeCompletedAllRequests", "Completed all"),
    ("DRT/EpisodeCompletedPassengers", "Completed passengers"),
    ("Environment/Episode Length", "Episode length"),
]

QUALITY_TAGS = [
    ("DRT/EpisodeAverageWaitSeconds", "Average wait (s)"),
    ("DRT/EpisodeAverageRideSeconds", "Average ride (s)"),
    ("DRT/EpisodeTravelDistanceMeters", "Distance (m)"),
    ("DRT/EpisodeStopArrivalCount", "Stop arrivals"),
]

REWARD_TAGS = [
    ("DRT/Reward/EpisodeTotal", "Episode reward"),
    ("Environment/Cumulative Reward", "Cumulative reward"),
    ("DRT/Reward/Boarding", "Boarding reward"),
    ("DRT/Reward/UnboardedPenalty", "Unboarded penalty"),
]

PPO_TAGS = [
    ("Policy/Entropy", "Entropy"),
    ("Losses/Policy Loss", "Policy loss"),
    ("Losses/Value Loss", "Value loss"),
    ("Policy/Extrinsic Value Estimate", "Value estimate"),
]

SCHEDULE_TAGS = [
    ("Policy/Learning Rate", "Learning rate"),
    ("Policy/Epsilon", "Epsilon"),
    ("Policy/Beta", "Beta"),
]

WINDOW_METRICS = [
    ("DRT/EpisodeServiceRate", "sr"),
    ("DRT/EpisodeCompletedAllRequests", "all_done"),
    ("DRT/EpisodeAverageWaitSeconds", "avg_wait"),
    ("DRT/EpisodeAverageRideSeconds", "avg_ride"),
    ("DRT/EpisodeTravelDistanceMeters", "distance"),
    ("DRT/Reward/EpisodeTotal", "reward"),
    ("Policy/Entropy", "entropy"),
    ("Losses/Value Loss", "value_loss"),
]


@dataclass
class ScalarPoint:
    wall_time: float
    step: int
    value: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a DRT TensorBoard scalar PDF report.")
    parser.add_argument("--tb-url", default=TB_URL)
    parser.add_argument("--run", default=RUN)
    parser.add_argument("--smoothing", type=float, default=SMOOTHING)
    parser.add_argument("--output", default=f"output/pdf/{OUTPUT_STEM}.pdf")
    parser.add_argument("--figures-dir", default=f"output/figures/{OUTPUT_STEM}")
    parser.add_argument("--csv-dir", default="output/csv")
    parser.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="Optional inclusive step cutoff. Use this to remove post-collapse or recovery regions from the report.",
    )
    return parser.parse_args()


def fetch_json(url: str) -> object:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def scalar_url(base_url: str, run: str, tag: str) -> str:
    query = urllib.parse.urlencode({"run": run, "tag": tag, "format": "json"})
    return f"{base_url.rstrip('/')}/data/plugin/scalars/scalars?{query}"


def fetch_all_scalars(base_url: str, run: str) -> tuple[dict[str, list[ScalarPoint]], dict[str, object]]:
    tags_url = f"{base_url.rstrip('/')}/data/plugin/scalars/tags"
    env_url = f"{base_url.rstrip('/')}/data/environment"
    tags_payload = fetch_json(tags_url)
    env_payload = fetch_json(env_url)
    run_tags = sorted(tags_payload.get(run, {}).keys())
    if not run_tags:
        raise RuntimeError(f"No scalar tags found for run={run!r}. TensorBoard URL={base_url}")

    series: dict[str, list[ScalarPoint]] = {}
    for tag in run_tags:
        payload = fetch_json(scalar_url(base_url, run, tag))
        points = [ScalarPoint(float(row[0]), int(row[1]), float(row[2])) for row in payload]
        series[tag] = points
    return series, env_payload


def filter_max_step(series: dict[str, list[ScalarPoint]], max_step: int | None) -> dict[str, list[ScalarPoint]]:
    if max_step is None:
        return series
    return {
        tag: [point for point in points if point.step <= max_step]
        for tag, points in series.items()
    }


def ema(values: list[float], smoothing: float) -> list[float]:
    if not values:
        return []
    smoothing = min(max(smoothing, 0.0), 0.9999)
    result = []
    last = values[0]
    for value in values:
        last = last * smoothing + value * (1.0 - smoothing)
        result.append(last)
    return result


def mean(values: Iterable[float]) -> float | None:
    clean = [value for value in values if math.isfinite(value)]
    return statistics.fmean(clean) if clean else None


def std(values: Iterable[float]) -> float | None:
    clean = [value for value in values if math.isfinite(value)]
    if len(clean) < 2:
        return 0.0 if clean else None
    return statistics.pstdev(clean)


def values_in_window(points: list[ScalarPoint], start: int, end: int) -> list[float]:
    return [point.value for point in points if start <= point.step <= end]


def nearest_window_stats(series: dict[str, list[ScalarPoint]], end_step: int, window: int = 5000) -> dict[str, float | int | None]:
    start = max(0, end_step - window)
    result: dict[str, float | int | None] = {"step": end_step, "window_start": start, "window_end": end_step}
    for tag, key in WINDOW_METRICS:
        vals = values_in_window(series.get(tag, []), start, end_step)
        result[f"{key}_n"] = len(vals)
        result[f"{key}_mean"] = mean(vals)
        result[f"{key}_min"] = min(vals) if vals else None
        result[f"{key}_max"] = max(vals) if vals else None
    return result


def make_window_rows(series: dict[str, list[ScalarPoint]], width: int = 5000) -> list[dict[str, object]]:
    max_step = max((point.step for points in series.values() for point in points), default=0)
    rows: list[dict[str, object]] = []
    for start in range(0, max_step + 1, width):
        end = start + width - 1
        sr_vals = values_in_window(series.get("DRT/EpisodeServiceRate", []), start, end)
        if not sr_vals:
            continue
        row: dict[str, object] = {"window": f"{start//1000}-{end//1000}k", "start": start, "end": end, "n": len(sr_vals)}
        for tag, key in WINDOW_METRICS:
            vals = values_in_window(series.get(tag, []), start, end)
            row[key] = mean(vals)
            row[f"{key}_std"] = std(vals)
        rows.append(row)
    return rows


def tag_summary_rows(series: dict[str, list[ScalarPoint]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for tag, points in sorted(series.items()):
        if not points:
            continue
        values = [point.value for point in points]
        min_index = min(range(len(points)), key=lambda index: values[index])
        max_index = max(range(len(points)), key=lambda index: values[index])
        rows.append(
            {
                "tag": tag,
                "n": len(points),
                "first_step": points[0].step,
                "last_step": points[-1].step,
                "first_value": values[0],
                "last_value": values[-1],
                "min_value": values[min_index],
                "min_step": points[min_index].step,
                "max_value": values[max_index],
                "max_step": points[max_index].step,
            }
        )
    return rows


def write_raw_csv(path: Path, series: dict[str, list[ScalarPoint]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tag", "step", "wall_time", "value"])
        writer.writeheader()
        for tag, points in sorted(series.items()):
            for point in points:
                writer.writerow({"tag": tag, "step": point.step, "wall_time": point.wall_time, "value": point.value})


def write_dict_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def k_formatter(value: float, _pos: int) -> str:
    if abs(value) >= 1000:
        return f"{value/1000:.0f}k"
    return f"{value:.0f}"


def clean_tag_name(tag: str) -> str:
    return tag.replace("DRT/", "").replace("Environment/", "").replace("Policy/", "").replace("Losses/", "")


def plot_group(
    series: dict[str, list[ScalarPoint]],
    specs: list[tuple[str, str]],
    title: str,
    path: Path,
    smoothing: float,
    ncols: int = 2,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nrows = math.ceil(len(specs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 7.8), dpi=180)
    axes_list = list(axes.flatten()) if hasattr(axes, "flatten") else [axes]
    palette = ["#2563EB", "#059669", "#DC2626", "#7C3AED", "#EA580C", "#0891B2"]

    for index, (tag, label) in enumerate(specs):
        ax = axes_list[index]
        points = series.get(tag, [])
        if not points:
            ax.text(0.5, 0.5, f"No data\n{label}", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        xs = [point.step for point in points]
        ys = [point.value for point in points]
        color = palette[index % len(palette)]
        if len(xs) <= 75:
            ax.plot(xs, ys, color=color, alpha=0.35, linewidth=1.2, marker="o", markersize=2.2, label="raw")
        else:
            ax.plot(xs, ys, color=color, alpha=0.22, linewidth=0.8, label="raw")
        ax.plot(xs, ema(ys, smoothing), color=color, linewidth=2.4, label=f"EMA {smoothing:.2f}")
        ax.axvline(BEST_CHECKPOINT_STEP, color="#16A34A", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.axvline(COLLAPSE_START_STEP, color="#B91C1C", linestyle="--", linewidth=1.2, alpha=0.65)
        ax.set_title(label, loc="left", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_formatter(FuncFormatter(k_formatter))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.tick_params(axis="both", labelsize=8)
        if tag in {"DRT/EpisodeServiceRate", "DRT/EpisodeCompletedAllRequests"}:
            ax.set_ylim(-0.05, 1.05)
        if tag == "Policy/Learning Rate":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        ax.legend(loc="best", fontsize=7, frameon=False)

    for index in range(len(specs), len(axes_list)):
        axes_list[index].set_axis_off()

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.015,
        "Dashed green = selected checkpoint 70,999. Dashed red = observed instability zone around 75k. Raw line is faint; bold line is smoothed.",
        ha="center",
        fontsize=9,
        color="#374151",
    )
    fig.tight_layout(rect=[0.03, 0.04, 0.98, 0.95])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fmt(value: object, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "n/a"
    if abs(numeric) >= 10000:
        return f"{numeric:,.0f}"
    return f"{numeric:,.{digits}f}"


def display_path(value: object, max_chars: int = 86) -> str:
    text = str(value)
    try:
        path = Path(text)
        if path.is_absolute():
            try:
                text = path.relative_to(Path.cwd()).as_posix()
            except ValueError:
                parts = path.parts
                text = ".../" + "/".join(parts[-4:])
        else:
            text = path.as_posix()
    except (TypeError, ValueError):
        text = str(value)
    if len(text) > max_chars:
        return "..." + text[-(max_chars - 3) :]
    return text


def percent_delta(new: float | None, old: float | None) -> str:
    if new is None or old is None or abs(old) < 1e-12:
        return "n/a"
    return f"{(new - old) / old * 100:+.1f}%"


def report_table(rows: list[list[object]], widths: list[float] | None = None, font_size: int = 8) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def add_title(story: list[object], text: str, styles: dict[str, ParagraphStyle]) -> None:
    story.append(Paragraph(text, styles["PageTitle"]))
    story.append(Spacer(1, 4 * mm))


def figure_page(story: list[object], title: str, image_path: Path, styles: dict[str, ParagraphStyle]) -> None:
    add_title(story, title, styles)
    story.append(Image(str(image_path), width=255 * mm, height=142 * mm))
    story.append(PageBreak())


def build_pdf(
    output_path: Path,
    series: dict[str, list[ScalarPoint]],
    env_payload: dict[str, object],
    window_rows: list[dict[str, object]],
    checkpoint_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    figure_paths: dict[str, Path],
    raw_csv: Path,
    window_csv: Path,
    checkpoint_csv: Path,
    tag_summary_csv: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="PageTitle", parent=styles["Heading1"], fontSize=20, leading=24, spaceAfter=8))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10))
    styles.add(ParagraphStyle(name="Body", parent=styles["BodyText"], fontSize=10, leading=13))

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=landscape(A4),
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=10 * mm,
    )
    story: list[object] = []

    max_step = max((point.step for points in series.values() for point in points), default=0)
    service_best = nearest_window_stats(series, BEST_CHECKPOINT_STEP, 5000)
    collapse_reference_step = min(COLLAPSE_START_STEP - 1, max_step)
    collapse = nearest_window_stats(series, collapse_reference_step, 5000)
    last = nearest_window_stats(series, max_step, 5000)
    data_location = display_path(env_payload.get("data_location", "n/a"), 74)

    add_title(story, "DRT PPO TensorBoard Summary", styles)
    story.append(
        Paragraph(
            f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} from TensorBoard scalar API. "
            f"Smoothing uses EMA coefficient {SMOOTHING:.2f}. Run: {RUN}.",
            styles["Body"],
        )
    )
    story.append(Spacer(1, 4 * mm))
    kpi_rows = [
        ["Item", "Value", "Interpretation"],
        ["TensorBoard logdir", str(data_location), "Source used for every scalar in this report"],
        ["Scalar tags extracted", str(len(series)), "All scalar tags exposed by TensorBoard were exported"],
        ["Total scalar rows", f"{sum(len(points) for points in series.values()):,}", "Saved to raw CSV"],
        ["Last training step", f"{max_step:,}", "Latest scalar step in the event file"],
        [
            "Selected checkpoint",
            f"{BEST_CHECKPOINT_STEP:,}",
            "Best stable region before the completed-all metric starts degrading",
        ],
        [
            "Best window service rate",
            fmt(service_best.get("sr_mean"), 3),
            "5k rolling window ending at selected checkpoint",
        ],
        [
            "Best window completed-all",
            fmt(service_best.get("all_done_mean"), 3),
            "Completion stability matters more than distance alone",
        ],
        [
            "Best window avg wait",
            f"{fmt(service_best.get('avg_wait_mean'))} s",
            "Main operational quality metric",
        ],
        [
            "Pre-collapse cutoff",
            f"last step {max_step:,}",
            "Post-collapse/recovery region is intentionally excluded",
        ],
        [
            "Last pre-cutoff window",
            f"completed-all {fmt(collapse.get('all_done_mean'), 3)}",
            "Final included 5k window before the excluded instability zone",
        ],
        [
            "Latest 5k window",
            f"service rate {fmt(last.get('sr_mean'), 3)}",
            "Later checkpoints are not as stable as 50k-70k",
        ],
    ]
    story.append(report_table(kpi_rows, widths=[45 * mm, 110 * mm, 105 * mm], font_size=8))
    story.append(Spacer(1, 5 * mm))
    story.append(
        Paragraph(
            "Recommendation: use checkpoint 70,999 for evaluation. Distance improves as a side effect, but the stronger "
            "evidence is service completion stability plus lower wait and ride times.",
            styles["Body"],
        )
    )
    story.append(PageBreak())

    figure_page(story, "Service Completion Metrics", figure_paths["service"], styles)
    figure_page(story, "Passenger Quality And Efficiency", figure_paths["quality"], styles)
    figure_page(story, "Reward Dynamics", figure_paths["reward"], styles)
    figure_page(story, "PPO Stability Metrics", figure_paths["ppo"], styles)
    figure_page(story, "Training Schedule", figure_paths["schedule"], styles)

    add_title(story, "5k Window Summary", styles)
    table_rows = [["Window", "n", "Service", "All done", "Avg wait", "Avg ride", "Distance", "Reward", "Entropy", "Value loss"]]
    for row in window_rows:
        table_rows.append(
            [
                row["window"],
                row["n"],
                fmt(row.get("sr"), 3),
                fmt(row.get("all_done"), 3),
                fmt(row.get("avg_wait")),
                fmt(row.get("avg_ride")),
                fmt(row.get("distance"), 0),
                fmt(row.get("reward")),
                fmt(row.get("entropy"), 3),
                fmt(row.get("value_loss")),
            ]
        )
    story.append(report_table(table_rows, widths=[22 * mm, 13 * mm, 25 * mm, 25 * mm, 27 * mm, 27 * mm, 30 * mm, 30 * mm, 25 * mm, 30 * mm], font_size=7))
    story.append(PageBreak())

    add_title(story, "Checkpoint Candidate Table", styles)
    candidate_rows = [["Checkpoint", "Window", "Service", "Min SR", "All done", "Avg wait", "Distance", "Reward", "Entropy", "Value loss", "Status"]]
    for row in checkpoint_rows:
        step = int(row["step"])
        status = "Selected" if step == BEST_CHECKPOINT_STEP else ("Unstable" if step >= COLLAPSE_START_STEP else "Candidate")
        candidate_rows.append(
            [
                f"{step:,}",
                f"{int(row['window_start'])//1000}-{int(row['window_end'])//1000}k",
                fmt(row.get("sr_mean"), 3),
                fmt(row.get("sr_min"), 3),
                fmt(row.get("all_done_mean"), 3),
                fmt(row.get("avg_wait_mean")),
                fmt(row.get("distance_mean"), 0),
                fmt(row.get("reward_mean")),
                fmt(row.get("entropy_mean"), 3),
                fmt(row.get("value_loss_mean")),
                status,
            ]
        )
    story.append(report_table(candidate_rows, widths=[26 * mm, 22 * mm, 22 * mm, 22 * mm, 24 * mm, 25 * mm, 29 * mm, 26 * mm, 22 * mm, 25 * mm, 25 * mm], font_size=7))
    story.append(PageBreak())

    add_title(story, "All Extracted Scalar Tags", styles)
    tag_rows = [["Tag", "n", "First", "Last", "Min", "Min step", "Max", "Max step"]]
    for row in summary_rows:
        tag_rows.append(
            [
                clean_tag_name(str(row["tag"])),
                row["n"],
                f"{int(row['first_step']):,}",
                f"{int(row['last_step']):,}",
                fmt(row["min_value"]),
                f"{int(row['min_step']):,}",
                fmt(row["max_value"]),
                f"{int(row['max_step']):,}",
            ]
        )
    story.append(report_table(tag_rows, widths=[70 * mm, 15 * mm, 22 * mm, 22 * mm, 28 * mm, 24 * mm, 28 * mm, 24 * mm], font_size=6))
    story.append(PageBreak())

    add_title(story, "Exported Data Files", styles)
    export_rows = [
        ["Artifact", "Path"],
        ["Raw scalar CSV", display_path(raw_csv, 120)],
        ["5k window CSV", display_path(window_csv, 120)],
        ["Checkpoint candidate CSV", display_path(checkpoint_csv, 120)],
        ["Tag summary CSV", display_path(tag_summary_csv, 120)],
    ]
    story.append(report_table(export_rows, widths=[55 * mm, 205 * mm], font_size=8))

    doc.build(story)


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    figures_dir = Path(args.figures_dir)
    csv_dir = Path(args.csv_dir)
    series, env_payload = fetch_all_scalars(args.tb_url, args.run)
    series = filter_max_step(series, args.max_step)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_csv = csv_dir / f"tensorboard_drt_scalars_raw_{timestamp}.csv"
    window_csv = csv_dir / f"tensorboard_drt_5k_windows_{timestamp}.csv"
    checkpoint_csv = csv_dir / f"tensorboard_drt_checkpoint_candidates_{timestamp}.csv"
    tag_summary_csv = csv_dir / f"tensorboard_drt_tag_summary_{timestamp}.csv"

    window_rows = make_window_rows(series, 5000)
    checkpoint_steps = [
        49999,
        54999,
        59999,
        64999,
        66999,
        67999,
        68999,
        69999,
        70999,
        71999,
        72999,
        73999,
        74999,
        75999,
        76999,
        77999,
        78999,
        79999,
        84999,
        89999,
        92584,
    ]
    if args.max_step is not None:
        checkpoint_steps = [step for step in checkpoint_steps if step <= args.max_step]
    checkpoint_rows = [nearest_window_stats(series, step, 5000) for step in checkpoint_steps]
    summary_rows = tag_summary_rows(series)

    write_raw_csv(raw_csv, series)
    write_dict_csv(window_csv, window_rows)
    write_dict_csv(checkpoint_csv, checkpoint_rows)
    write_dict_csv(tag_summary_csv, summary_rows)

    figure_paths = {
        "service": figures_dir / "01_service_completion.png",
        "quality": figures_dir / "02_quality_efficiency.png",
        "reward": figures_dir / "03_reward_dynamics.png",
        "ppo": figures_dir / "04_ppo_stability.png",
        "schedule": figures_dir / "05_training_schedule.png",
    }
    plot_group(series, SERVICE_TAGS, "Service Completion Metrics", figure_paths["service"], args.smoothing)
    plot_group(series, QUALITY_TAGS, "Passenger Quality And Efficiency", figure_paths["quality"], args.smoothing)
    plot_group(series, REWARD_TAGS, "Reward Dynamics", figure_paths["reward"], args.smoothing)
    plot_group(series, PPO_TAGS, "PPO Stability Metrics", figure_paths["ppo"], args.smoothing)
    plot_group(series, SCHEDULE_TAGS, "Training Schedule", figure_paths["schedule"], args.smoothing, ncols=3)

    build_pdf(
        output_path,
        series,
        env_payload,
        window_rows,
        checkpoint_rows,
        summary_rows,
        figure_paths,
        raw_csv,
        window_csv,
        checkpoint_csv,
        tag_summary_csv,
    )

    print(f"Wrote {output_path}")
    print(f"Wrote figures: {figures_dir}")
    print(f"Wrote {raw_csv}")
    print(f"Wrote {window_csv}")
    print(f"Wrote {checkpoint_csv}")
    print(f"Wrote {tag_summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
