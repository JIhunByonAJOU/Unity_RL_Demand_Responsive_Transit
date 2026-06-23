#!/usr/bin/env python3
"""Export all TensorBoard scalar series as clean PNG and CSV files."""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from tensorboard.backend.event_processing import event_accumulator


DEFAULT_MARKS = ("Early=60000", "Middle=600000", "Late=1200000")
ORANGE = "#F97316"


@dataclass(frozen=True)
class ScalarPoint:
    wall_time: float
    step: int
    value: float


@dataclass(frozen=True)
class Mark:
    label: str
    step: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export every scalar in a TensorBoard event file or run directory."
    )
    parser.add_argument("event_path", help="Path to an events.out.tfevents file or TensorBoard run directory.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for exported files. Defaults to <event parent>/scalar_export.",
    )
    parser.add_argument(
        "--mark",
        action="append",
        default=None,
        help="Vertical marker in LABEL=STEP form. Repeat for multiple markers.",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.95,
        help="EMA smoothing coefficient for the bold line. Use 0 for no smoothing.",
    )
    parser.add_argument(
        "--hide-raw",
        action="store_true",
        help="Hide the unsmoothed raw scalar line behind the EMA line.",
    )
    parser.add_argument("--show-raw", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def parse_marks(raw_marks: list[str] | None) -> list[Mark]:
    marks: list[Mark] = []
    for raw in raw_marks or list(DEFAULT_MARKS):
        if "=" not in raw:
            raise ValueError(f"Marker must use LABEL=STEP format: {raw}")
        label, step_text = raw.split("=", 1)
        label = label.strip()
        step = int(step_text.replace(",", "").strip())
        if not label:
            raise ValueError(f"Marker label is empty: {raw}")
        marks.append(Mark(label, step))
    return marks


def sanitize_filename(value: str) -> str:
    value = value.replace("/", "__")
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value.strip("._") or "scalar"


def k_formatter(value: float, _pos: int) -> str:
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def load_scalars(event_path: Path) -> dict[str, list[ScalarPoint]]:
    accumulator = event_accumulator.EventAccumulator(
        str(event_path),
        size_guidance={
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 0,
            event_accumulator.COMPRESSED_HISTOGRAMS: 0,
            event_accumulator.IMAGES: 0,
            event_accumulator.AUDIO: 0,
            event_accumulator.TENSORS: 0,
        },
    )
    accumulator.Reload()
    series: dict[str, list[ScalarPoint]] = {}
    for tag in sorted(accumulator.Tags().get("scalars", [])):
        series[tag] = [
            ScalarPoint(float(point.wall_time), int(point.step), float(point.value))
            for point in accumulator.Scalars(tag)
        ]
    return series


def ema(values: list[float], smoothing: float) -> list[float]:
    if not values:
        return []
    smoothing = min(max(smoothing, 0.0), 0.9999)
    if smoothing <= 0.0:
        return values[:]
    result: list[float] = []
    last = values[0]
    for value in values:
        if math.isfinite(value):
            last = last * smoothing + value * (1.0 - smoothing)
        result.append(last)
    return result


def nearest_index(points: list[ScalarPoint], target_step: int) -> int:
    return min(range(len(points)), key=lambda index: (abs(points[index].step - target_step), points[index].step))


def safe_percent_delta(new: float | None, old: float | None) -> float | None:
    if new is None or old is None or not math.isfinite(old) or abs(old) < 1e-12:
        return None
    return (new - old) / old * 100.0


def write_raw_csv(path: Path, series: dict[str, list[ScalarPoint]], smoothed: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tag", "step", "wall_time", "raw_value", "smoothed_value"])
        writer.writeheader()
        for tag, points in sorted(series.items()):
            smooth_values = smoothed[tag]
            for point, smooth_value in zip(points, smooth_values):
                writer.writerow(
                    {
                        "tag": tag,
                        "step": point.step,
                        "wall_time": point.wall_time,
                        "raw_value": point.value,
                        "smoothed_value": smooth_value,
                    }
                )


def write_phase_csvs(
    long_path: Path,
    wide_path: Path,
    series: dict[str, list[ScalarPoint]],
    smoothed: dict[str, list[float]],
    marks: list[Mark],
) -> None:
    long_path.parent.mkdir(parents=True, exist_ok=True)
    long_rows: list[dict[str, object]] = []
    wide_rows: list[dict[str, object]] = []

    for tag, points in sorted(series.items()):
        row: dict[str, object] = {
            "tag": tag,
            "n": len(points),
            "first_step": points[0].step,
            "last_step": points[-1].step,
        }
        phase_raw_values: dict[str, float] = {}
        phase_smooth_values: dict[str, float] = {}
        for mark in marks:
            index = nearest_index(points, mark.step)
            point = points[index]
            smooth_value = smoothed[tag][index]
            key = sanitize_filename(mark.label).lower()
            phase_raw_values[key] = point.value
            phase_smooth_values[key] = smooth_value
            row[f"{key}_target_step"] = mark.step
            row[f"{key}_nearest_step"] = point.step
            row[f"{key}_step_delta"] = point.step - mark.step
            row[f"{key}_raw_value"] = point.value
            row[f"{key}_smoothed_value"] = smooth_value
            long_rows.append(
                {
                    "tag": tag,
                    "phase": mark.label,
                    "target_step": mark.step,
                    "nearest_step": point.step,
                    "step_delta": point.step - mark.step,
                    "raw_value": point.value,
                    "smoothed_value": smooth_value,
                }
            )

        if len(marks) >= 2:
            first_key = sanitize_filename(marks[0].label).lower()
            last_key = sanitize_filename(marks[-1].label).lower()
            row["first_to_last_raw_delta"] = phase_raw_values[last_key] - phase_raw_values[first_key]
            row["first_to_last_raw_delta_percent"] = safe_percent_delta(
                phase_raw_values[last_key], phase_raw_values[first_key]
            )
            row["first_to_last_smoothed_delta"] = phase_smooth_values[last_key] - phase_smooth_values[first_key]
            row["first_to_last_smoothed_delta_percent"] = safe_percent_delta(
                phase_smooth_values[last_key], phase_smooth_values[first_key]
            )
            for previous, current in zip(marks, marks[1:]):
                prev_key = sanitize_filename(previous.label).lower()
                cur_key = sanitize_filename(current.label).lower()
                label = f"{prev_key}_to_{cur_key}"
                row[f"{label}_raw_delta"] = phase_raw_values[cur_key] - phase_raw_values[prev_key]
                row[f"{label}_raw_delta_percent"] = safe_percent_delta(
                    phase_raw_values[cur_key], phase_raw_values[prev_key]
                )
                row[f"{label}_smoothed_delta"] = phase_smooth_values[cur_key] - phase_smooth_values[prev_key]
                row[f"{label}_smoothed_delta_percent"] = safe_percent_delta(
                    phase_smooth_values[cur_key], phase_smooth_values[prev_key]
                )
        wide_rows.append(row)

    with long_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "tag",
            "phase",
            "target_step",
            "nearest_step",
            "step_delta",
            "raw_value",
            "smoothed_value",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(long_rows)

    with wide_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(wide_rows[0].keys()) if wide_rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(wide_rows)


def write_summary_csv(path: Path, series: dict[str, list[ScalarPoint]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for tag, points in sorted(series.items()):
        values = [point.value for point in points]
        min_index = min(range(len(values)), key=values.__getitem__)
        max_index = max(range(len(values)), key=values.__getitem__)
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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def plot_scalar(
    tag: str,
    points: list[ScalarPoint],
    smooth_values: list[float],
    marks: list[Mark],
    output_path: Path,
    smoothing: float,
    show_raw: bool,
    dpi: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    steps = [point.step for point in points]
    values = [point.value for point in points]

    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=dpi, facecolor="white")
    ax.set_facecolor("white")
    if show_raw:
        ax.plot(steps, values, color="#9CA3AF", linewidth=0.55, alpha=0.28, label="raw")
    if smoothing > 0.0:
        ax.plot(steps, smooth_values, color="#2563EB", linewidth=2.1, label=f"EMA {smoothing:.2f}")
    else:
        ax.plot(steps, values, color="#2563EB", linewidth=1.6, label="raw")

    for mark in marks:
        ax.axvline(mark.step, color=ORANGE, linestyle="--", linewidth=1.35, alpha=0.82)
        ax.text(
            mark.step,
            0.98,
            f"{mark.label}\n{k_formatter(mark.step, 0)}",
            transform=ax.get_xaxis_transform(),
            ha="right",
            va="top",
            rotation=90,
            color=ORANGE,
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )

    ax.set_title(tag, loc="left", fontsize=13, fontweight="bold", color="#111827")
    ax.set_xlabel("Training step", fontsize=10, color="#374151")
    ax.set_ylabel("Value", fontsize=10, color="#374151")
    ax.grid(True, color="#E5E7EB", linewidth=0.8)
    ax.xaxis.set_major_formatter(FuncFormatter(k_formatter))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.tick_params(axis="both", labelsize=9, colors="#374151")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.legend(loc="best", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    event_path = Path(args.event_path).expanduser().resolve()
    if not event_path.exists():
        raise FileNotFoundError(event_path)

    marks = parse_marks(args.mark)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else event_path.parent / "scalar_export"
    figures_dir = output_dir / "figures"
    csv_dir = output_dir / "csv"

    series = load_scalars(event_path)
    if not series:
        raise RuntimeError(f"No scalar tags found in {event_path}")

    smoothed = {tag: ema([point.value for point in points], args.smoothing) for tag, points in series.items()}

    write_raw_csv(csv_dir / "all_scalars_raw_and_smoothed.csv", series, smoothed)
    write_phase_csvs(
        csv_dir / "phase_values_long.csv",
        csv_dir / "phase_values_wide.csv",
        series,
        smoothed,
        marks,
    )
    write_summary_csv(csv_dir / "scalar_summary.csv", series)

    for index, (tag, points) in enumerate(sorted(series.items()), start=1):
        filename = f"{index:02d}_{sanitize_filename(tag)}.png"
        plot_scalar(
            tag,
            points,
            smoothed[tag],
            marks,
            figures_dir / filename,
            args.smoothing,
            not args.hide_raw,
            args.dpi,
        )

    print(f"Loaded scalar tags: {len(series)}")
    print(f"Wrote figures: {figures_dir}")
    print(f"Wrote CSV files: {csv_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
