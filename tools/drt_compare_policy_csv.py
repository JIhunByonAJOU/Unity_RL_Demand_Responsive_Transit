#!/usr/bin/env python3
"""Build a PDF comparison report from DRT episode CSV exports.

The script is intentionally read-only for Unity project data. It scans exported
CSV files, aggregates Matrix Teleport runs by next-stop policy, and writes a PDF
report for Vanilla Sequential vs ONNX Inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


TARGET_POLICIES = ("Vanilla Sequential", "ONNX Inference")
REPORT_METRICS = [
    "episode_distance_meters",
    "episode_time_seconds",
    "service_rate",
    "completed_passengers",
    "completed_all_requests",
    "average_wait_seconds",
    "p95_wait_seconds",
    "max_wait_seconds",
    "average_ride_seconds",
    "route_leg_count",
    "reward",
]
LOWER_IS_BETTER = {
    "episode_distance_meters",
    "episode_time_seconds",
    "average_wait_seconds",
    "p95_wait_seconds",
    "max_wait_seconds",
    "average_ride_seconds",
    "route_leg_count",
}
HIGHER_IS_BETTER = {
    "service_rate",
    "completed_passengers",
    "completed_all_requests",
    "reward",
}

REWARD_TOLERANCE_SECONDS = 0.05
BOARDING_REWARD_WEIGHT = 1.0
DROPOFF_REWARD_WEIGHT = 1.0
UNBOARDED_PASSENGER_PENALTY_WEIGHT = 1.0
ACCEPTABLE_WAIT_SECONDS = 600.0
ACCEPTABLE_WAIT_REWARD_MULTIPLIER = 10.0
MINIMUM_NETWORK_AVERAGE_REWARD = 0.01


@dataclass
class EpisodeRecord:
    path: Path
    values: dict[str, str]
    policy: str
    scenario_id: str
    timestamp: datetime | None
    json_data: dict[str, object] | None = None
    episode_csv_rows: dict[str, list[dict[str, str]]] | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare DRT Matrix Teleport policy CSV exports.")
    parser.add_argument(
        "--exports",
        nargs="+",
        default=["DRT_Episode_Exports", "DRT_Inference_Exports"],
        help="Export directories to scan. Defaults to DRT_Episode_Exports and DRT_Inference_Exports.",
    )
    parser.add_argument(
        "--output",
        default="output/pdf/drt_matrix_policy_comparison.pdf",
        help="PDF path to write.",
    )
    parser.add_argument(
        "--scenario-id",
        default=None,
        help="Scenario id to compare. If omitted, the best common scenario is selected.",
    )
    parser.add_argument(
        "--runs-per-policy",
        type=int,
        default=10,
        help="Number of latest runs to use per policy.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if fewer than runs-per-policy are available for either policy.",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include timeout/partial episodes. By default only completed_all_requests=1 episodes are selected.",
    )
    parser.add_argument(
        "--matrix-csv",
        default="Assets/DRT/Resources/drt_stop_travel_time_matrix.csv",
        help="Travel-time matrix CSV used to reconstruct the reward scale.",
    )
    parser.add_argument(
        "--selected-runs-csv",
        default=None,
        help="Optional path for the selected run-level metrics CSV.",
    )
    parser.add_argument(
        "--aggregate-csv",
        default=None,
        help="Optional path for the aggregate comparison metrics CSV.",
    )
    parser.add_argument(
        "--markdown-output",
        default=None,
        help="Optional path for a Markdown comparison report.",
    )
    return parser.parse_args()


def normalize_token(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def display_policy(raw: str | None) -> str | None:
    token = normalize_token(raw)
    if token in {"vanillasequential", "vanilla", "sequential"}:
        return "Vanilla Sequential"
    if token in {"onnxinference", "inference", "inferenceonly"}:
        return "ONNX Inference"
    if token in {"mlagentstraining", "training", "default"}:
        return "ML-Agents Training"
    return raw.strip() if raw else None


def is_matrix_teleport(values: dict[str, str]) -> bool:
    raw = values.get("travel_execution_mode") or values.get("travel_mode") or values.get("mode") or values.get("execution_mode")
    return normalize_token(raw) in {"matrixteleport", "matrix"}


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_summary(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        return values

    header = [cell.strip() for cell in rows[0]]
    if len(header) >= 2 and {"metric", "key", "name"} & {header[0].lower()}:
        for row in rows[1:]:
            if len(row) >= 2 and row[0].strip():
                values[row[0].strip()] = row[1].strip()
        return values

    for row in rows:
        if len(row) >= 2 and row[0].strip():
            values[row[0].strip()] = row[1].strip()
    return values


def scalar_to_summary_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def read_episode_json(path: Path) -> tuple[dict[str, str], dict[str, object]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        return {}, {}
    summary = data.get("summary")
    if not isinstance(summary, dict):
        return {}, data
    return {str(key): scalar_to_summary_value(value) for key, value in summary.items()}, data


def read_episode_section_csv(path: Path) -> tuple[dict[str, str], dict[str, list[dict[str, str]]]]:
    values: dict[str, str] = {}
    sections: dict[str, list[dict[str, str]]] = {"route_legs": [], "passengers": []}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))

    index = 0
    while index < len(rows):
        row = rows[index]
        if not row or not any(cell.strip() for cell in row):
            index += 1
            continue

        header = [cell.strip() for cell in row]
        if header[:3] == ["section", "key", "value"]:
            index += 1
            while index < len(rows):
                current = rows[index]
                if not current or not any(cell.strip() for cell in current):
                    index += 1
                    break
                if len(current) >= 3 and current[0].strip() == "summary":
                    values[current[1].strip()] = current[2].strip()
                index += 1
            continue

        if header and header[0] == "section":
            section_name = ""
            if len(header) > 1 and header[1] == "leg_index":
                section_name = "route_legs"
            elif len(header) > 1 and header[1] == "passenger_id":
                section_name = "passengers"

            if section_name:
                index += 1
                while index < len(rows):
                    current = rows[index]
                    if not current or not any(cell.strip() for cell in current):
                        index += 1
                        break
                    padded = current + [""] * max(0, len(header) - len(current))
                    record = {header[col]: padded[col] for col in range(len(header))}
                    sections[section_name].append(record)
                    index += 1
                continue

        index += 1

    return values, sections


def read_dict_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def dict_rows(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


def first_numeric(row: dict[str, str], candidates: Iterable[str]) -> float | None:
    normalized = {normalize_token(key): value for key, value in row.items()}
    for candidate in candidates:
        value = safe_float(normalized.get(normalize_token(candidate)))
        if value is not None:
            return value
    return None


def percentile(values: list[float], pct: float) -> float | None:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    rank = (len(clean) - 1) * pct
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return clean[int(rank)]
    return clean[lower] * (upper - rank) + clean[upper] * (rank - lower)


def load_network_average_reward_minutes(path: str | Path | None) -> float | None:
    if path is None:
        return None
    matrix_path = Path(path)
    if not matrix_path.exists():
        return None
    values: list[float] = []
    with matrix_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_index, row in enumerate(csv.reader(handle)):
            for col_index, cell in enumerate(row):
                if row_index == col_index:
                    continue
                seconds = safe_float(cell)
                if seconds is not None and seconds > 0:
                    values.append(seconds / 60.0)
    if not values:
        return None
    return max(MINIMUM_NETWORK_AVERAGE_REWARD, statistics.fmean(values))


def timestamp_from_name(path: Path) -> datetime | None:
    candidates = [path.name, *reversed(path.parts)]
    for candidate in candidates:
        match = re.search(r"(\d{8})_(\d{6})", candidate)
        if not match:
            continue
        try:
            return datetime.strptime("_".join(match.groups()), "%Y%m%d_%H%M%S")
        except ValueError:
            continue

    for candidate in candidates:
        match = re.search(r"(\d{4})_(\d{6})", candidate)
        if not match:
            continue
        try:
            year = datetime.fromtimestamp(path.stat().st_mtime).year if path.exists() else datetime.now().year
            return datetime.strptime(f"{year}{match.group(1)}_{match.group(2)}", "%Y%m%d_%H%M%S")
        except (OSError, ValueError):
            continue

    return None


def companion_path(summary_path: Path, suffix: str) -> Path:
    name = summary_path.name
    if name.endswith("_summary.csv"):
        return summary_path.with_name(name[: -len("_summary.csv")] + suffix)
    return summary_path.with_name(summary_path.stem + suffix)


def is_same_time(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return False
    return abs(left - right) <= REWARD_TOLERANCE_SECONDS


def reconstruct_reward(
    passenger_rows: list[dict[str, str]],
    route_rows: list[dict[str, str]],
    network_average_reward_minutes: float | None,
) -> float | None:
    if not passenger_rows or not route_rows:
        return None

    network_reward = network_average_reward_minutes
    if network_reward is None:
        leg_minutes = [
            value / 60.0
            for row in route_rows
            for value in [first_numeric(row, ("travel_time_seconds", "duration_seconds", "leg_time_seconds", "time_seconds"))]
            if value is not None and value > 0
        ]
        if not leg_minutes:
            return None
        network_reward = max(MINIMUM_NETWORK_AVERAGE_REWARD, statistics.fmean(leg_minutes))

    passengers: list[dict[str, float | None]] = []
    for row in passenger_rows:
        passengers.append(
            {
                "request": first_numeric(row, ("request_time_seconds", "request_time", "request")),
                "pickup": first_numeric(row, ("pickup_time_seconds", "pickup_time", "pickup")),
                "dropoff": first_numeric(row, ("dropoff_time_seconds", "dropoff_time", "dropoff")),
            }
        )

    total_reward = 0.0
    for row in route_rows:
        completed = first_numeric(row, ("completed",))
        if completed is not None and completed <= 0:
            continue
        arrival_time = first_numeric(row, ("arrival_time_seconds", "arrival_time", "time_seconds"))
        if arrival_time is None:
            continue

        boarding_reward = 0.0
        dropoff_reward = 0.0
        unboarded_penalty = 0.0

        for passenger in passengers:
            request_time = passenger["request"]
            pickup_time = passenger["pickup"]
            dropoff_time = passenger["dropoff"]

            if request_time is None:
                continue

            if is_same_time(pickup_time, arrival_time):
                wait_seconds = max(0.0, pickup_time - request_time)
                multiplier = ACCEPTABLE_WAIT_REWARD_MULTIPLIER if wait_seconds <= ACCEPTABLE_WAIT_SECONDS else 1.0
                boarding_reward += BOARDING_REWARD_WEIGHT * network_reward * multiplier

            if is_same_time(dropoff_time, arrival_time):
                dropoff_reward += DROPOFF_REWARD_WEIGHT * network_reward

            has_not_boarded = pickup_time is None or pickup_time > arrival_time + REWARD_TOLERANCE_SECONDS
            if request_time <= arrival_time and has_not_boarded:
                unboarded_penalty += max(0.0, arrival_time - request_time) / 60.0

        total_reward += boarding_reward + dropoff_reward - unboarded_penalty * UNBOARDED_PASSENGER_PENALTY_WEIGHT

    return total_reward


def enrich_metrics(record: EpisodeRecord, network_average_reward_minutes: float | None) -> None:
    for key, raw in record.values.items():
        value = safe_float(raw)
        if value is not None:
            record.metrics[key] = value

    reward_keys = [key for key in record.metrics if "reward" in normalize_token(key)]
    if reward_keys and "reward" not in record.metrics:
        record.metrics["reward"] = record.metrics[reward_keys[0]]

    passenger_rows: list[dict[str, object]]
    route_rows: list[dict[str, object]]
    if record.episode_csv_rows is not None:
        passenger_rows = record.episode_csv_rows.get("passengers", [])
        route_rows = record.episode_csv_rows.get("route_legs", [])
    elif record.json_data is not None:
        passenger_rows = dict_rows(record.json_data.get("passengers", []))
        route_rows = dict_rows(record.json_data.get("route_legs", []))
    else:
        passengers_path = companion_path(record.path, "_passengers.csv")
        passenger_rows = read_dict_rows(passengers_path)
        route_path = companion_path(record.path, "_route_legs.csv")
        route_rows = read_dict_rows(route_path)

    if passenger_rows:
        waits: list[float] = []
        rides: list[float] = []
        for row in passenger_rows:
            wait = first_numeric(
                row,
                ("wait_seconds", "waiting_seconds", "wait_time_seconds", "wait_time", "wait"),
            )
            ride = first_numeric(
                row,
                ("ride_seconds", "ride_time_seconds", "in_vehicle_seconds", "ride_time", "ride"),
            )
            if wait is not None:
                waits.append(wait)
            if ride is not None:
                rides.append(ride)
        if waits:
            record.metrics.setdefault("average_wait_seconds", statistics.fmean(waits))
            p95_wait = percentile(waits, 0.95)
            if p95_wait is not None:
                record.metrics["p95_wait_seconds"] = p95_wait
            record.metrics["max_wait_seconds"] = max(waits)
        if rides:
            record.metrics.setdefault("average_ride_seconds", statistics.fmean(rides))
        record.metrics["passenger_csv_rows"] = float(len(passenger_rows))
    elif record.json_data is None and record.episode_csv_rows is None:
        passengers_path = companion_path(record.path, "_passengers.csv")
        record.notes.append(f"Missing passenger companion CSV: {passengers_path.name}")

    if route_rows:
        distances: list[float] = []
        durations: list[float] = []
        for row in route_rows:
            distance = first_numeric(
                row,
                ("distance_meters", "leg_distance_meters", "distance", "travel_distance_meters"),
            )
            duration = first_numeric(
                row,
                ("travel_time_seconds", "duration_seconds", "leg_time_seconds", "time_seconds"),
            )
            if distance is not None:
                distances.append(distance)
            if duration is not None:
                durations.append(duration)
        record.metrics["route_leg_count"] = float(len(route_rows))
        if distances:
            record.metrics.setdefault("episode_distance_meters", sum(distances))
        if durations:
            record.metrics.setdefault("episode_time_seconds", sum(durations))
    elif record.json_data is None and record.episode_csv_rows is None:
        route_path = companion_path(record.path, "_route_legs.csv")
        record.notes.append(f"Missing route companion CSV: {route_path.name}")

    if "reward" not in record.metrics:
        reconstructed_reward = reconstruct_reward(passenger_rows, route_rows, network_average_reward_minutes)
        if reconstructed_reward is not None:
            record.metrics["reward"] = reconstructed_reward
            record.metrics["reward_reconstructed_from_csv"] = 1.0
        else:
            record.notes.append("Reward unavailable: no reward field and reconstruction inputs were incomplete.")


def discover_records(export_dirs: list[str], network_average_reward_minutes: float | None) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    for export_dir in export_dirs:
        root = Path(export_dir)
        if not root.exists():
            continue
        for episode_path in sorted(root.rglob("*_episode.csv")):
            try:
                values, section_rows = read_episode_section_csv(episode_path)
            except OSError:
                continue
            if not is_matrix_teleport(values):
                continue
            policy = display_policy(values.get("next_stop_policy") or values.get("policy"))
            if policy not in TARGET_POLICIES:
                continue
            scenario_id = values.get("scenario_id") or values.get("scenario") or "unknown"
            record = EpisodeRecord(
                path=episode_path,
                values=values,
                policy=policy,
                scenario_id=str(scenario_id),
                timestamp=timestamp_from_name(episode_path),
                episode_csv_rows=section_rows,
            )
            enrich_metrics(record, network_average_reward_minutes)
            records.append(record)

        for episode_path in sorted(root.rglob("*_episode.json")):
            try:
                values, json_data = read_episode_json(episode_path)
            except (OSError, json.JSONDecodeError):
                continue
            if not is_matrix_teleport(values):
                continue
            policy = display_policy(values.get("next_stop_policy") or values.get("policy"))
            if policy not in TARGET_POLICIES:
                continue
            scenario_id = values.get("scenario_id") or values.get("scenario") or "unknown"
            record = EpisodeRecord(
                path=episode_path,
                values=values,
                policy=policy,
                scenario_id=str(scenario_id),
                timestamp=timestamp_from_name(episode_path),
                json_data=json_data,
            )
            enrich_metrics(record, network_average_reward_minutes)
            records.append(record)

        for summary_path in sorted(root.rglob("*_summary.csv")):
            values = read_summary(summary_path)
            if not is_matrix_teleport(values):
                continue
            policy = display_policy(values.get("next_stop_policy") or values.get("policy"))
            if policy not in TARGET_POLICIES:
                continue
            scenario_id = values.get("scenario_id") or values.get("scenario") or "unknown"
            record = EpisodeRecord(
                path=summary_path,
                values=values,
                policy=policy,
                scenario_id=str(scenario_id),
                timestamp=timestamp_from_name(summary_path),
            )
            enrich_metrics(record, network_average_reward_minutes)
            records.append(record)
    return records


def choose_scenario(records: list[EpisodeRecord], requested: str | None) -> str | None:
    if requested is not None:
        return requested
    scenarios = sorted({record.scenario_id for record in records})
    if not scenarios:
        return None

    def score(scenario: str) -> tuple[int, int]:
        counts = {
            policy: sum(1 for record in records if record.scenario_id == scenario and record.policy == policy)
            for policy in TARGET_POLICIES
        }
        return (min(counts.values()), sum(counts.values()))

    return max(scenarios, key=score)


def is_completed_episode(record: EpisodeRecord) -> bool:
    value = record.values.get("completed_all_requests")
    if value is None:
        return True
    return str(value).strip() in {"1", "true", "True", "TRUE"}


def select_latest(
    records: list[EpisodeRecord],
    scenario_id: str | None,
    runs_per_policy: int,
    include_incomplete: bool,
) -> dict[str, list[EpisodeRecord]]:
    selected: dict[str, list[EpisodeRecord]] = {}
    for policy in TARGET_POLICIES:
        candidates = [
            record
            for record in records
            if record.policy == policy and (scenario_id is None or record.scenario_id == scenario_id)
        ]
        if not include_incomplete:
            candidates = [record for record in candidates if is_completed_episode(record)]
        candidates.sort(key=lambda record: (record.timestamp or datetime.fromtimestamp(record.path.stat().st_mtime), record.path.name))
        selected[policy] = candidates[-runs_per_policy:]
    return selected


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return None, None
    if len(clean) == 1:
        return clean[0], 0.0
    return statistics.fmean(clean), statistics.stdev(clean)


def aggregate(selected: dict[str, list[EpisodeRecord]], metric: str) -> dict[str, tuple[float | None, float | None, int]]:
    result: dict[str, tuple[float | None, float | None, int]] = {}
    for policy, records in selected.items():
        values = [record.metrics[metric] for record in records if metric in record.metrics]
        avg, std = mean_std(values)
        result[policy] = (avg, std, len(values))
    return result


def fmt_number(value: float | None, suffix: str = "", digits: int = 2) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:,.{digits}f}{suffix}"


def fmt_mean_std(avg: float | None, std: float | None, count: int, suffix: str = "", digits: int = 2) -> str:
    if avg is None:
        return "n/a"
    if count <= 1:
        return fmt_number(avg, suffix, digits)
    return f"{avg:,.{digits}f} +/- {std:,.{digits}f}{suffix}"


def metric_label(metric: str) -> str:
    labels = {
        "episode_distance_meters": "Total distance (m)",
        "episode_time_seconds": "Episode time (s)",
        "service_rate": "Service rate",
        "completed_passengers": "Completed passengers",
        "completed_all_requests": "Completed all requests",
        "average_wait_seconds": "Average wait (s)",
        "p95_wait_seconds": "P95 wait (s)",
        "max_wait_seconds": "Max wait (s)",
        "average_ride_seconds": "Average ride (s)",
        "route_leg_count": "Route legs",
        "reward": "Reward",
    }
    return labels.get(metric, metric)


def metric_suffix(metric: str) -> tuple[str, int]:
    if metric == "service_rate":
        return "", 3
    if metric in {"completed_all_requests", "completed_passengers", "route_leg_count"}:
        return "", 2
    return "", 2


def run_label(record: EpisodeRecord) -> str:
    timestamp = record.timestamp.strftime("%Y-%m-%d %H:%M:%S") if record.timestamp else "unknown time"
    episode = record.values.get("episode_index", "?")
    return f"ep{episode} {timestamp}"


def comparison_status(metric: str, vanilla: float | None, onnx: float | None) -> str:
    if vanilla is None or onnx is None:
        return "n/a"
    if metric in LOWER_IS_BETTER:
        if onnx < vanilla:
            return "ONNX better"
        if onnx > vanilla:
            return "Vanilla better"
        return "Tie"
    if metric in HIGHER_IS_BETTER:
        if onnx > vanilla:
            return "ONNX better"
        if onnx < vanilla:
            return "Vanilla better"
        return "Tie"
    return "Compare manually"


def make_table(data: list[list[object]], widths: list[float] | None = None, header_rows: int = 1) -> Table:
    table = Table(data, colWidths=widths, repeatRows=header_rows)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, header_rows - 1), colors.HexColor("#263238")),
                ("TEXTCOLOR", (0, 0), (-1, header_rows - 1), colors.white),
                ("FONTNAME", (0, 0), (-1, header_rows - 1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#B0BEC5")),
                ("ROWBACKGROUNDS", (0, header_rows), (-1, -1), [colors.white, colors.HexColor("#F6F8FA")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def write_selected_runs_csv(path: Path, selected: dict[str, list[EpisodeRecord]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "policy",
        "run_order",
        "episode_index",
        "timestamp",
        "scenario_id",
        "source_summary_csv",
        "finish_reason",
        "completed_all_requests_raw",
        *REPORT_METRICS,
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for policy in TARGET_POLICIES:
            for run_order, record in enumerate(selected.get(policy, []), start=1):
                row: dict[str, object] = {
                    "policy": policy,
                    "run_order": run_order,
                    "episode_index": record.values.get("episode_index", ""),
                    "timestamp": record.timestamp.isoformat(sep=" ") if record.timestamp else "",
                    "scenario_id": record.scenario_id,
                    "source_summary_csv": str(record.path),
                    "finish_reason": record.values.get("finish_reason", ""),
                    "completed_all_requests_raw": record.values.get("completed_all_requests", ""),
                }
                for metric in REPORT_METRICS:
                    row[metric] = record.metrics.get(metric, "")
                writer.writerow(row)


def write_aggregate_csv(path: Path, selected: dict[str, list[EpisodeRecord]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "metric",
        "label",
        "direction",
        "vanilla_mean",
        "vanilla_std",
        "vanilla_n",
        "onnx_mean",
        "onnx_std",
        "onnx_n",
        "onnx_minus_vanilla",
        "onnx_pct_change_vs_vanilla",
        "status",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metric in REPORT_METRICS:
            agg = aggregate(selected, metric)
            vanilla_avg, vanilla_std, vanilla_count = agg["Vanilla Sequential"]
            onnx_avg, onnx_std, onnx_count = agg["ONNX Inference"]
            delta = None if vanilla_avg is None or onnx_avg is None else onnx_avg - vanilla_avg
            pct_delta = None
            if delta is not None and vanilla_avg is not None and not math.isclose(vanilla_avg, 0.0):
                pct_delta = delta / vanilla_avg
            writer.writerow(
                {
                    "metric": metric,
                    "label": metric_label(metric),
                    "direction": "lower_is_better"
                    if metric in LOWER_IS_BETTER
                    else "higher_is_better"
                    if metric in HIGHER_IS_BETTER
                    else "manual",
                    "vanilla_mean": vanilla_avg if vanilla_avg is not None else "",
                    "vanilla_std": vanilla_std if vanilla_std is not None else "",
                    "vanilla_n": vanilla_count,
                    "onnx_mean": onnx_avg if onnx_avg is not None else "",
                    "onnx_std": onnx_std if onnx_std is not None else "",
                    "onnx_n": onnx_count,
                    "onnx_minus_vanilla": delta if delta is not None else "",
                    "onnx_pct_change_vs_vanilla": pct_delta if pct_delta is not None else "",
                    "status": comparison_status(metric, vanilla_avg, onnx_avg),
                }
            )


def markdown_metric_row(metric: str, selected: dict[str, list[EpisodeRecord]]) -> list[str]:
    agg = aggregate(selected, metric)
    vanilla_avg, vanilla_std, vanilla_count = agg["Vanilla Sequential"]
    onnx_avg, onnx_std, onnx_count = agg["ONNX Inference"]
    suffix, digits = metric_suffix(metric)
    delta = None if vanilla_avg is None or onnx_avg is None else onnx_avg - vanilla_avg
    pct_delta = ""
    if delta is not None and vanilla_avg is not None and not math.isclose(vanilla_avg, 0.0):
        pct_delta = f"{delta / vanilla_avg:+.1%}"
    return [
        metric_label(metric),
        fmt_mean_std(vanilla_avg, vanilla_std, vanilla_count, suffix, digits),
        fmt_mean_std(onnx_avg, onnx_std, onnx_count, suffix, digits),
        fmt_number(delta, suffix, digits),
        pct_delta,
        comparison_status(metric, vanilla_avg, onnx_avg),
    ]


def write_markdown_report(
    path: Path,
    selected: dict[str, list[EpisodeRecord]],
    scenario_id: str | None,
    runs_per_policy: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DRT Matrix Teleport Policy Comparison",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Scenario: {scenario_id or 'not selected'}",
        f"- Target runs per policy: {runs_per_policy}",
        f"- Vanilla selected runs: {len(selected.get('Vanilla Sequential', []))}",
        f"- ONNX selected runs: {len(selected.get('ONNX Inference', []))}",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Vanilla Sequential | ONNX Inference | ONNX - Vanilla | % change | Direction |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for metric in REPORT_METRICS:
        row = markdown_metric_row(metric, selected)
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Service completion is equal when both policies show completed_all_requests=1 and service_rate=1.0.",
            "- Lower distance, episode time, wait time, ride time, and route leg count are operational improvements.",
            "- Higher reward is better because the reconstructed reward follows the current paper reward terms.",
            "- Reward is reconstructed from route/passenger CSVs when Unity summary CSVs do not export a reward field.",
            "",
            "## Selected Run CSVs",
            "",
        ]
    )
    for policy in TARGET_POLICIES:
        lines.append(f"### {policy}")
        lines.append("")
        lines.append("| Run | Episode | Timestamp | Distance (m) | Avg wait (s) | Reward |")
        lines.append("|---:|---:|---|---:|---:|---:|")
        for run_order, record in enumerate(selected.get(policy, []), start=1):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(run_order),
                        record.values.get("episode_index", ""),
                        record.timestamp.strftime("%Y-%m-%d %H:%M:%S") if record.timestamp else "",
                        fmt_number(record.metrics.get("episode_distance_meters")),
                        fmt_number(record.metrics.get("average_wait_seconds")),
                        fmt_number(record.metrics.get("reward")),
                    ]
                )
                + " |"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_pdf(
    output_path: Path,
    export_dirs: list[str],
    records: list[EpisodeRecord],
    selected: dict[str, list[EpisodeRecord]],
    scenario_id: str | None,
    runs_per_policy: int,
    include_incomplete: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10))
    styles.add(ParagraphStyle(name="Warn", parent=styles["BodyText"], fontSize=9, leading=11, textColor=colors.HexColor("#8A4B00")))
    story: list[object] = []

    story.append(Paragraph("DRT Matrix Teleport Policy Comparison", styles["Title"]))
    story.append(
        Paragraph(
            f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} from Matrix Teleport CSV exports.",
            styles["Small"],
        )
    )
    story.append(Spacer(1, 6))

    availability_rows = [["Policy", "Matrix CSVs", "Completed CSVs", "Runs selected", "Target runs"]]
    for policy in TARGET_POLICIES:
        found = sum(1 for record in records if record.policy == policy and (scenario_id is None or record.scenario_id == scenario_id))
        completed_found = sum(
            1
            for record in records
            if record.policy == policy
            and (scenario_id is None or record.scenario_id == scenario_id)
            and is_completed_episode(record)
        )
        availability_rows.append([policy, str(found), str(completed_found), str(len(selected.get(policy, []))), str(runs_per_policy)])
    story.append(Paragraph("Data Availability", styles["Heading2"]))
    story.append(
        Paragraph(
            f"Export directories: {', '.join(export_dirs)}<br/>Scenario: {scenario_id or 'not selected'}",
            styles["Small"],
        )
    )
    story.append(make_table(availability_rows, widths=[43 * mm, 39 * mm, 34 * mm, 31 * mm, 31 * mm]))
    story.append(Spacer(1, 6))

    if not include_incomplete:
        story.append(
            Paragraph(
                "Selection excludes timeout/partial episodes and uses only completed_all_requests=1 runs.",
                styles["Small"],
            )
        )
        story.append(Spacer(1, 6))

    missing_10 = any(len(selected.get(policy, [])) < runs_per_policy for policy in TARGET_POLICIES)
    if not records:
        story.append(
            Paragraph(
                "No Matrix Teleport episode exports were found. This PDF is a readiness report; place the 10 Vanilla "
                "Sequential and 10 ONNX Inference *_episode.csv, *_episode.json, or *_summary.csv files in the export folder, then rerun the script.",
                styles["Warn"],
            )
        )
    elif missing_10:
        story.append(
            Paragraph(
                "The selected dataset has fewer than the requested 10 runs for at least one policy. Treat all "
                "comparisons below as incomplete and do not use them as statistical evidence yet.",
                styles["Warn"],
            )
        )
    story.append(Spacer(1, 8))

    metrics = REPORT_METRICS
    story.append(Paragraph("Aggregate Metrics", styles["Heading2"]))
    metric_rows = [["Metric", "Vanilla Sequential", "ONNX Inference", "ONNX - Vanilla", "Direction"]]
    for metric in metrics:
        agg = aggregate(selected, metric)
        vanilla_avg, vanilla_std, vanilla_count = agg["Vanilla Sequential"]
        onnx_avg, onnx_std, onnx_count = agg["ONNX Inference"]
        suffix, digits = metric_suffix(metric)
        delta = None if vanilla_avg is None or onnx_avg is None else onnx_avg - vanilla_avg
        metric_rows.append(
            [
                metric_label(metric),
                fmt_mean_std(vanilla_avg, vanilla_std, vanilla_count, suffix, digits),
                fmt_mean_std(onnx_avg, onnx_std, onnx_count, suffix, digits),
                fmt_number(delta, suffix, digits),
                comparison_status(metric, vanilla_avg, onnx_avg),
            ]
        )
    story.append(make_table(metric_rows, widths=[43 * mm, 38 * mm, 38 * mm, 31 * mm, 30 * mm]))
    story.append(Spacer(1, 8))

    reward_available = any("reward" in record.metrics for policy_records in selected.values() for record in policy_records)
    reward_reconstructed = any(
        record.metrics.get("reward_reconstructed_from_csv") == 1.0
        for policy_records in selected.values()
        for record in policy_records
    )
    if reward_reconstructed:
        story.append(
            Paragraph(
                "Reward is reconstructed from passenger and route CSVs because the episode summary does not export "
                "the ML-Agents episode reward directly. The reconstruction follows the current paper reward terms: "
                "boarding reward, dropoff reward, and unboarded passenger wait penalty.",
                styles["Small"],
            )
        )
        story.append(Spacer(1, 8))
    elif not reward_available:
        story.append(
            Paragraph(
                "Reward was requested, but no reward field was found in the exported CSV summaries or companion CSVs. "
                "The report leaves reward as n/a instead of inventing a proxy.",
                styles["Small"],
            )
        )
        story.append(Spacer(1, 8))

    story.append(PageBreak())
    story.append(Paragraph("Run-Level Summary", styles["Heading2"]))
    run_rows = [["Policy", "Run", "Distance (m)", "Reward", "Avg wait (s)", "Service rate", "Completed"]]
    for policy in TARGET_POLICIES:
        for record in selected.get(policy, []):
            completed = record.metrics.get("completed_passengers")
            total = record.metrics.get("total_passengers")
            completed_text = "n/a"
            if completed is not None and total is not None:
                completed_text = f"{completed:.0f}/{total:.0f}"
            elif completed is not None:
                completed_text = f"{completed:.0f}"
            run_rows.append(
                [
                    policy,
                    run_label(record),
                    fmt_number(record.metrics.get("episode_distance_meters")),
                    fmt_number(record.metrics.get("reward")),
                    fmt_number(record.metrics.get("average_wait_seconds")),
                    fmt_number(record.metrics.get("service_rate"), digits=3),
                    completed_text,
                ]
            )
    if len(run_rows) == 1:
        run_rows.append(["n/a", "No selected summary CSVs", "n/a", "n/a", "n/a", "n/a", "n/a"])
    story.append(make_table(run_rows, widths=[33 * mm, 39 * mm, 26 * mm, 23 * mm, 24 * mm, 24 * mm, 21 * mm]))

    notes = sorted({note for records_for_policy in selected.values() for record in records_for_policy for note in record.notes})
    if notes:
        story.append(PageBreak())
        story.append(Paragraph("Input Notes", styles["Heading2"]))
        note_rows = [["Note"]]
        note_rows.extend([[note] for note in notes[:80]])
        story.append(make_table(note_rows, widths=[178 * mm]))

    doc.build(story)


def main() -> int:
    args = parse_args()
    network_average_reward_minutes = load_network_average_reward_minutes(args.matrix_csv)
    records = discover_records(args.exports, network_average_reward_minutes)
    scenario_id = choose_scenario(records, args.scenario_id)
    selected = select_latest(records, scenario_id, args.runs_per_policy, args.include_incomplete)
    output_path = Path(args.output)
    build_pdf(output_path, args.exports, records, selected, scenario_id, args.runs_per_policy, args.include_incomplete)
    if args.selected_runs_csv:
        write_selected_runs_csv(Path(args.selected_runs_csv), selected)
    if args.aggregate_csv:
        write_aggregate_csv(Path(args.aggregate_csv), selected)
    if args.markdown_output:
        write_markdown_report(Path(args.markdown_output), selected, scenario_id, args.runs_per_policy)

    counts = {policy: len(selected.get(policy, [])) for policy in TARGET_POLICIES}
    print(f"Wrote {output_path}")
    if args.selected_runs_csv:
        print(f"Wrote {args.selected_runs_csv}")
    if args.aggregate_csv:
        print(f"Wrote {args.aggregate_csv}")
    if args.markdown_output:
        print(f"Wrote {args.markdown_output}")
    print(f"Selected scenario: {scenario_id or 'none'}")
    for policy in TARGET_POLICIES:
        print(f"{policy}: {counts[policy]} / {args.runs_per_policy} runs")

    if args.strict and any(count < args.runs_per_policy for count in counts.values()):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
