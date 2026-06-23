from __future__ import annotations

import csv
import math
import re
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
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
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path("DRT_Episode_Exports")
PDF_OUT = Path("output") / "pdf"
POLICIES = {
    "fifo": "FIFO",
    "inference": "ONNX",
    "vanilla": "Vanilla",
}
POLICY_ORDER = ["FIFO", "ONNX", "Vanilla"]
COLORS = {"FIFO": "#4C78A8", "ONNX": "#54A24B", "Vanilla": "#E45756"}
TENSORBOARD_TAGS = [
    "Environment/Cumulative Reward",
    "Environment/Episode Length",
    "DRT/EpisodeTimeAtDecision",
    "DRT/EpisodeDecisionCount",
    "DRT/EpisodeStopArrivalCount",
    "DRT/EpisodeCompletedAllRequests",
    "DRT/EpisodeServiceRate",
    "DRT/EpisodeCompletedPassengers",
    "DRT/EpisodeAverageWaitSeconds",
    "DRT/EpisodeAverageRideSeconds",
    "DRT/EpisodeTravelDistanceMeters",
    "DRT/Reward/EpisodeTotal",
    "DRT/Reward/Boarding",
    "DRT/Reward/Dropoff",
    "DRT/Reward/UnboardedPenalty",
    "DRT/Reward/NoInteractionPenalty",
    "Policy/Entropy",
    "Losses/Policy Loss",
    "Losses/Value Loss",
    "Policy/Learning Rate",
    "Policy/Epsilon",
    "Policy/Beta",
]

FONT_REGULAR = "MalgunGothic"
FONT_BOLD = "MalgunGothic-Bold"
FONT_REGULAR_PATH = Path(r"C:\Windows\Fonts\malgun.ttf")
FONT_BOLD_PATH = Path(r"C:\Windows\Fonts\malgunbd.ttf")


def register_fonts() -> None:
    if not FONT_REGULAR_PATH.exists() or not FONT_BOLD_PATH.exists():
        raise SystemExit("Malgun Gothic font files were not found under C:\\Windows\\Fonts.")
    pdfmetrics.registerFont(TTFont(FONT_REGULAR, str(FONT_REGULAR_PATH)))
    pdfmetrics.registerFont(TTFont(FONT_BOLD, str(FONT_BOLD_PATH)))
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


def to_float(value) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def to_int(value) -> int | float:
    try:
        if value is None or value == "":
            return float("nan")
        return int(float(value))
    except Exception:
        return float("nan")


def latest_run_dirs() -> dict[str, Path]:
    selected: dict[str, Path] = {}
    for mode_key in POLICIES:
        mode_root = ROOT / mode_key
        dirs = [p for p in mode_root.iterdir() if p.is_dir()] if mode_root.exists() else []
        if not dirs:
            raise SystemExit(f"No export directory found for {mode_key}: {mode_root}")
        selected[mode_key] = max(dirs, key=lambda p: p.stat().st_mtime)
    return selected


def latest_tensorboard_event() -> Path | None:
    roots = [
        Path("../ml-agents/results/drt_5rnd_new2_5m/DRTNextStopPPO"),
        Path("Assets/DRT/DRT onnx files/0613_5rnd_mnew_final"),
    ]
    candidates: list[Path] = []
    for root in roots:
        if root.exists():
            candidates.extend(
                p for p in root.glob("events.out.tfevents*") if p.is_file() and not p.name.endswith(".meta")
            )
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_training_config() -> dict[str, str]:
    config_path = Path("../ml-agents/config/ppo/drt_next_stop_ppo.yaml")
    values: dict[str, str] = {"config_path": display_path(config_path)}
    if not config_path.exists():
        return values
    text = config_path.read_text(encoding="utf-8")
    for key in [
        "batch_size",
        "buffer_size",
        "learning_rate",
        "beta",
        "epsilon",
        "lambd",
        "num_epoch",
        "gamma",
        "max_steps",
        "time_horizon",
        "summary_freq",
        "checkpoint_interval",
        "keep_checkpoints",
        "hidden_units",
        "num_layers",
    ]:
        match = re.search(rf"^\s*{re.escape(key)}:\s*([^\r\n#]+)", text, re.MULTILINE)
        if match:
            values[key] = match.group(1).strip()
    return values


def parse_scene_runtime_settings() -> dict[str, str]:
    scene_path = Path("Assets/Gley/TrafficSystem/Samples/TrafficExample.unity")
    values: dict[str, str] = {"scene_path": display_path(scene_path)}
    if not scene_path.exists():
        return values
    text = scene_path.read_text(encoding="utf-8")
    block_match = re.search(
        r"busStopsRoot:.*?episodeLengthSeconds:\s*([0-9.]+).*?"
        r"simulationSecondsPerRealSecond:\s*([0-9.]+).*?"
        r"stopWhenAllRequestsCompleted:\s*([0-9]+).*?"
        r"noMovementTimeoutRealSeconds:\s*([0-9.]+).*?"
        r"trafficBlockTimeoutRealSeconds:\s*([0-9.]+)",
        text,
        re.DOTALL,
    )
    if block_match:
        values["episode_length_seconds"] = block_match.group(1)
        values["simulation_seconds_per_real_second"] = block_match.group(2)
        values["stop_when_all_requests_completed"] = block_match.group(3)
        values["no_movement_timeout_real_seconds"] = block_match.group(4)
        values["traffic_block_timeout_real_seconds"] = block_match.group(5)
    max_step_match = re.search(r"agentParameters:.*?maxStep:\s*([0-9]+).*?MaxStep:\s*([0-9]+)", text, re.DOTALL)
    if max_step_match:
        values["agent_parameters_max_step"] = max_step_match.group(1)
        values["agent_max_step"] = max_step_match.group(2)
    return values


def load_tensorboard_scalars() -> tuple[pd.DataFrame, dict[str, str], pd.DataFrame]:
    event_path = latest_tensorboard_event()
    if event_path is None:
        return pd.DataFrame(), {}, pd.DataFrame()

    accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    frames: list[pd.DataFrame] = []
    inventory_rows: list[dict] = []

    for tag in TENSORBOARD_TAGS:
        if tag not in available:
            continue
        values = accumulator.Scalars(tag)
        if not values:
            continue
        frames.append(pd.DataFrame({"step": [v.step for v in values], tag: [v.value for v in values]}))
        inventory_rows.append(
            {
                "tag": tag,
                "samples": len(values),
                "first_step": int(values[0].step),
                "last_step": int(values[-1].step),
                "last_value": float(values[-1].value),
            }
        )

    if not frames:
        return pd.DataFrame(), {"event_path": display_path(event_path)}, pd.DataFrame(inventory_rows)

    tensorboard = frames[0]
    for frame in frames[1:]:
        tensorboard = tensorboard.merge(frame, on="step", how="outer")
    tensorboard = tensorboard.sort_values("step").reset_index(drop=True)
    meta = {
        "event_path": display_path(event_path),
        "run_folder": display_path(event_path.parent),
        "first_step": str(int(tensorboard["step"].min())),
        "last_step": str(int(tensorboard["step"].max())),
        "scalar_rows": str(len(tensorboard)),
        "scalar_tag_count": str(len(inventory_rows)),
    }
    return tensorboard, meta, pd.DataFrame(inventory_rows)


def build_tensorboard_windows(tensorboard: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    if tensorboard.empty:
        return pd.DataFrame(rows)
    windows = [10, 50, 100, 500]
    for window in windows:
        subset = tensorboard.tail(window)
        row = {
            "window": f"last {window}",
            "samples": int(len(subset)),
            "start_step": int(subset["step"].min()),
            "end_step": int(subset["step"].max()),
        }
        metric_map = {
            "service_rate": "DRT/EpisodeServiceRate",
            "full_completion_rate": "DRT/EpisodeCompletedAllRequests",
            "completed_passengers": "DRT/EpisodeCompletedPassengers",
            "wait_seconds": "DRT/EpisodeAverageWaitSeconds",
            "ride_seconds": "DRT/EpisodeAverageRideSeconds",
            "episode_time_at_decision": "DRT/EpisodeTimeAtDecision",
            "decision_count": "DRT/EpisodeDecisionCount",
            "travel_distance_meters": "DRT/EpisodeTravelDistanceMeters",
            "cumulative_reward": "Environment/Cumulative Reward",
            "episode_total_reward": "DRT/Reward/EpisodeTotal",
        }
        for out_col, tag in metric_map.items():
            row[out_col] = float(subset[tag].dropna().mean()) if tag in subset else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def best_tensorboard_reliability_window(tensorboard: pd.DataFrame, window: int = 100) -> dict[str, float | int | str]:
    if tensorboard.empty:
        return {}
    needed = [
        "step",
        "DRT/EpisodeServiceRate",
        "DRT/EpisodeCompletedAllRequests",
        "DRT/EpisodeCompletedPassengers",
        "DRT/EpisodeAverageWaitSeconds",
        "DRT/EpisodeAverageRideSeconds",
        "DRT/EpisodeTimeAtDecision",
    ]
    if any(col not in tensorboard.columns for col in needed):
        return {}
    data = tensorboard[needed].dropna().copy()
    if len(data) < window:
        return {}
    data["service_rate"] = data["DRT/EpisodeServiceRate"].rolling(window).mean()
    data["full_completion_rate"] = data["DRT/EpisodeCompletedAllRequests"].rolling(window).mean()
    data["completed_passengers"] = data["DRT/EpisodeCompletedPassengers"].rolling(window).mean()
    data["wait_seconds"] = data["DRT/EpisodeAverageWaitSeconds"].rolling(window).mean()
    data["ride_seconds"] = data["DRT/EpisodeAverageRideSeconds"].rolling(window).mean()
    data["episode_time_at_decision"] = data["DRT/EpisodeTimeAtDecision"].rolling(window).mean()
    candidates = data.dropna()
    reliable = candidates[candidates["service_rate"] >= 0.99]
    if len(reliable):
        best = reliable.sort_values(["wait_seconds", "ride_seconds"]).iloc[0]
        label = f"best {window}-sample window with service >= 99%"
    else:
        best = candidates.sort_values(["service_rate", "wait_seconds"], ascending=[False, True]).iloc[0]
        label = f"best {window}-sample service window; no service >= 99% window"
    return {
        "label": label,
        "window": window,
        "end_step": int(best["step"]),
        "service_rate": float(best["service_rate"]),
        "full_completion_rate": float(best["full_completion_rate"]),
        "completed_passengers": float(best["completed_passengers"]),
        "wait_seconds": float(best["wait_seconds"]),
        "ride_seconds": float(best["ride_seconds"]),
        "episode_time_at_decision": float(best["episode_time_at_decision"]),
    }


def parse_episode_name(path: Path) -> tuple[int | float, int | float]:
    match = re.search(r"scenario_(\d+)_ep(\d+)_episode\.csv$", path.name)
    if not match:
        return float("nan"), float("nan")
    return int(match.group(1)), int(match.group(2))


def parse_episode(path: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    summary: dict[str, str] = {}
    passengers: list[dict[str, str]] = []
    header: list[str] | None = None
    current_section: str | None = None

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if not row or all(cell == "" for cell in row):
                continue
            if row[0] == "metadata" and len(row) >= 3:
                summary["metadata_" + row[1]] = row[2]
            elif row[0] == "summary" and len(row) >= 3:
                summary[row[1]] = row[2]
            elif row[0] == "section":
                header = row
                current_section = (
                    "passenger" if len(row) > 1 and row[1] == "passenger_id" else None
                )
            elif row[0] == "passenger" and current_section == "passenger" and header:
                passengers.append(dict(zip(header, row)))
    return summary, passengers


def collect_data(selected_dirs: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    episode_rows: list[dict] = []
    passenger_rows: list[dict] = []

    for mode_key, policy in POLICIES.items():
        run_dir = selected_dirs[mode_key]
        for ep_path in sorted(run_dir.glob("*_episode.csv")):
            file_scenario, file_episode = parse_episode_name(ep_path)
            summary, passengers = parse_episode(ep_path)
            scenario = to_int(summary.get("scenario_id"))
            if not np.isfinite(scenario):
                scenario = file_scenario
            episode_index = to_int(summary.get("episode_index"))
            if not np.isfinite(episode_index):
                episode_index = file_episode

            episode_rows.append(
                {
                    "policy": policy,
                    "mode_key": mode_key,
                    "scenario": int(scenario),
                    "episode_index": int(episode_index),
                    "run_folder": str(run_dir),
                    "file_name": ep_path.name,
                    "finish_reason": summary.get("finish_reason", ""),
                    "completed_all_requests": to_float(summary.get("completed_all_requests")),
                    "episode_time_seconds": to_float(summary.get("episode_time_seconds")),
                    "episode_distance_meters": to_float(summary.get("episode_distance_meters")),
                    "total_passengers": to_float(summary.get("total_passengers")),
                    "completed_passengers": to_float(summary.get("completed_passengers")),
                    "service_rate": to_float(summary.get("service_rate")),
                    "average_wait_seconds": to_float(summary.get("average_wait_seconds")),
                    "average_ride_seconds": to_float(summary.get("average_ride_seconds")),
                }
            )

            for row in passengers:
                status = row.get("status", "")
                passenger_rows.append(
                    {
                        "policy": policy,
                        "mode_key": mode_key,
                        "scenario": int(scenario),
                        "episode_index": int(episode_index),
                        "passenger_id": to_int(row.get("passenger_id")),
                        "status": status,
                        "completed": status == "Completed",
                        "wait_time_seconds": to_float(row.get("wait_time_seconds")),
                        "ride_time_seconds": to_float(row.get("ride_time_seconds")),
                        "total_time_seconds": to_float(row.get("total_time_seconds")),
                    }
                )

    episodes = pd.DataFrame(episode_rows)
    passengers = pd.DataFrame(passenger_rows)
    episodes["policy"] = pd.Categorical(episodes["policy"], POLICY_ORDER, ordered=True)
    passengers["policy"] = pd.Categorical(passengers["policy"], POLICY_ORDER, ordered=True)
    return (
        episodes.sort_values(["scenario", "policy", "episode_index"]).reset_index(drop=True),
        passengers.sort_values(["scenario", "policy", "episode_index", "passenger_id"]).reset_index(
            drop=True
        ),
    )


def q90(series: pd.Series) -> float:
    clean = series.dropna()
    return float(clean.quantile(0.9)) if len(clean) else float("nan")


def std0(series: pd.Series) -> float:
    value = series.std(ddof=1)
    return 0.0 if pd.isna(value) else float(value)


def build_summaries(episodes: pd.DataFrame, passengers: pd.DataFrame):
    ep_group = episodes.groupby(["scenario", "policy"], observed=False)
    summary = (
        ep_group.agg(
            n_episodes=("episode_index", "count"),
            full_completion_rate=("completed_all_requests", "mean"),
            completed_runs=("completed_all_requests", "sum"),
            service_rate_mean=("service_rate", "mean"),
            service_rate_min=("service_rate", "min"),
            wait_mean=("average_wait_seconds", "mean"),
            wait_std=("average_wait_seconds", std0),
            ride_mean=("average_ride_seconds", "mean"),
            ride_std=("average_ride_seconds", std0),
            timeout_runs=("finish_reason", lambda s: int((s == "Episode time ended.").sum())),
            completed_passengers=("completed_passengers", "sum"),
            total_passengers=("total_passengers", "sum"),
            episode_time_mean=("episode_time_seconds", "mean"),
            episode_time_std=("episode_time_seconds", std0),
            episode_time_max=("episode_time_seconds", "max"),
        )
        .reset_index()
    )

    pax_group = passengers.groupby(["scenario", "policy"], observed=False)
    pax = (
        pax_group.agg(
            passenger_records=("passenger_id", "count"),
            passenger_completed=("completed", "sum"),
            passenger_wait_mean=("wait_time_seconds", "mean"),
            passenger_wait_p90=("wait_time_seconds", q90),
            passenger_ride_mean=("ride_time_seconds", "mean"),
            passenger_ride_p90=("ride_time_seconds", q90),
        )
        .reset_index()
    )

    summary = summary.merge(pax, on=["scenario", "policy"], how="left")
    summary["policy"] = pd.Categorical(summary["policy"], POLICY_ORDER, ordered=True)
    summary = summary.sort_values(["scenario", "policy"]).reset_index(drop=True)

    macro = (
        summary.groupby("policy", observed=False)
        .agg(
            scenarios=("scenario", "nunique"),
            total_episodes=("n_episodes", "sum"),
            completed_run_rate=("full_completion_rate", "mean"),
            service_rate=("service_rate_mean", "mean"),
            service_rate_min=("service_rate_min", "min"),
            wait_mean=("wait_mean", "mean"),
            wait_p90_mean=("passenger_wait_p90", "mean"),
            ride_mean=("ride_mean", "mean"),
            ride_p90_mean=("passenger_ride_p90", "mean"),
            episode_time_mean=("episode_time_mean", "mean"),
            episode_time_max=("episode_time_max", "max"),
            timeout_runs=("timeout_runs", "sum"),
        )
        .reset_index()
    )
    macro["policy"] = pd.Categorical(macro["policy"], POLICY_ORDER, ordered=True)
    macro = macro.sort_values("policy").reset_index(drop=True)

    decisions = []
    for scenario, group in summary.groupby("scenario", sort=True, observed=False):
        reliable = group[group["service_rate_mean"] >= 0.999]
        candidates = reliable if len(reliable) else group
        selected = candidates.sort_values(
            ["wait_mean", "ride_mean", "service_rate_mean"], ascending=[True, True, False]
        ).iloc[0]
        wait_only = group.sort_values(
            ["wait_mean", "ride_mean", "service_rate_mean"], ascending=[True, True, False]
        ).iloc[0]
        runner = (
            candidates[candidates["policy"] != selected["policy"]]
            .sort_values(["wait_mean", "ride_mean", "service_rate_mean"], ascending=[True, True, False])
            .head(1)
        )
        runner_wait = float(runner.iloc[0]["wait_mean"]) if len(runner) else float("nan")
        note = ""
        if wait_only["policy"] != selected["policy"]:
            note = (
                f"{wait_only['policy']} has lower served-passenger wait, "
                f"but service rate is {wait_only['service_rate_mean'] * 100:.1f}%."
            )
        elif selected["timeout_runs"] > 0:
            note = f"{int(selected['timeout_runs'])} timeout runs."
        else:
            note = "Selected by reliable service, then lower wait and ride."

        decisions.append(
            {
                "scenario": int(scenario),
                "selected_policy": str(selected["policy"]),
                "selected_wait": float(selected["wait_mean"]),
                "selected_ride": float(selected["ride_mean"]),
                "selected_service_rate": float(selected["service_rate_mean"]),
                "selected_full_rate": float(selected["full_completion_rate"]),
                "runner_up_wait": runner_wait,
                "wait_advantage_vs_runner": runner_wait - float(selected["wait_mean"])
                if np.isfinite(runner_wait)
                else float("nan"),
                "wait_only_policy": str(wait_only["policy"]),
                "wait_only_wait": float(wait_only["wait_mean"]),
                "wait_only_service_rate": float(wait_only["service_rate_mean"]),
                "note": note,
            }
        )
    decisions_df = pd.DataFrame(decisions)
    return summary, macro, decisions_df


def source_coverage(episodes: pd.DataFrame, selected_dirs: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for mode_key, policy in POLICIES.items():
        subset = episodes[episodes["mode_key"] == mode_key]
        rows.append(
            {
                "policy": policy,
                "export_folder": display_path(selected_dirs[mode_key]),
                "episodes": int(len(subset)),
                "completed_episodes": int((subset["finish_reason"] == "All passenger requests completed.").sum()),
                "timeout_episodes": int((subset["finish_reason"] == "Episode time ended.").sum()),
                "scenarios": ", ".join(str(s) for s in sorted(subset["scenario"].unique())),
            }
        )
    return pd.DataFrame(rows)


def ensure_dirs(stamp: str) -> tuple[Path, Path, Path]:
    result_dir = Path("output") / f"drt_wait_first_fifo_onnx_vanilla_{stamp}"
    fig_dir = result_dir / "figures"
    table_dir = result_dir / "tables"
    PDF_OUT.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    return result_dir, fig_dir, table_dir


def apply_hatch_for_service(ax, bars, data: pd.DataFrame) -> None:
    for bar, (_, row) in zip(bars, data.iterrows()):
        if row["service_rate_mean"] < 0.999:
            bar.set_hatch("//")
            bar.set_edgecolor("#9A1E1E")
            bar.set_linewidth(1.1)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(12, ax.get_ylim()[1] * 0.012),
                "SR<100",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#9A1E1E",
            )


def grouped_metric_bar(
    summary: pd.DataFrame,
    fig_dir: Path,
    metric: str,
    std_col: str,
    ylabel: str,
    title: str,
    filename: str,
) -> Path:
    scenarios = sorted(summary["scenario"].unique())
    x = np.arange(len(scenarios))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.8, 5.6), dpi=180)
    ymax = 0.0
    for offset, policy in enumerate(POLICY_ORDER):
        values = []
        errors = []
        subset_rows = []
        for scenario in scenarios:
            row = summary[(summary["scenario"] == scenario) & (summary["policy"] == policy)].iloc[0]
            values.append(float(row[metric]))
            errors.append(float(row[std_col]) if np.isfinite(row[std_col]) else 0.0)
            subset_rows.append(row)
        positions = x + (offset - 1) * width
        bars = ax.bar(
            positions,
            values,
            width=width,
            yerr=errors,
            capsize=3,
            label=policy,
            color=COLORS[policy],
            alpha=0.92,
        )
        ymax = max(ymax, max(values) + max(errors or [0]))
        apply_hatch_for_service(ax, bars, pd.DataFrame(subset_rows))

    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel("Scenario passenger count", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in scenarios])
    ax.grid(axis="y", color="#D8DDE6", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, loc="upper left", frameon=False)
    ax.set_ylim(0, ymax * 1.2)
    fig.text(
        0.01,
        0.01,
        "Error bars are episode-level standard deviation. Hatched bars indicate mean service rate below 100%.",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    path = fig_dir / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def service_figure(summary: pd.DataFrame, fig_dir: Path) -> Path:
    scenarios = sorted(summary["scenario"].unique())
    x = np.arange(len(scenarios))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.8, 5.4), dpi=180)
    for offset, policy in enumerate(POLICY_ORDER):
        subset = summary[summary["policy"] == policy].sort_values("scenario")
        positions = x + (offset - 1) * width
        service = subset["service_rate_mean"].to_numpy(dtype=float) * 100
        full = subset["full_completion_rate"].to_numpy(dtype=float) * 100
        bars = ax.bar(
            positions,
            service,
            width=width,
            label=f"{policy} service",
            color=COLORS[policy],
            alpha=0.88,
        )
        for bar, service_value, full_value in zip(bars, service, full):
            if service_value < 99.9:
                bar.set_hatch("//")
                bar.set_edgecolor("#9A1E1E")
                bar.set_linewidth(1.1)
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    service_value + 2.0,
                    f"SR {service_value:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#9A1E1E",
                )
            if full_value < 99.9:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    full_value + 3.0,
                    f"Full {full_value:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#222222",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
                )
        ax.scatter(
            positions,
            full,
            marker="D",
            s=28,
            color="#222222",
            zorder=4,
        )
    ax.set_title("Service Rate and Full-Completion Reliability", fontsize=15, fontweight="bold", pad=12)
    ax.set_ylabel("Rate (%)", fontsize=11)
    ax.set_xlabel("Scenario passenger count", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in scenarios])
    ax.set_ylim(0, 110)
    ax.grid(axis="y", color="#D8DDE6", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, loc="lower left", frameon=False)
    fig.text(
        0.01,
        0.01,
        "Bars: passenger service rate. Black diamonds: fraction of runs that completed all requests.",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    path = fig_dir / "fig04_service_reliability_by_scenario.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def episode_time_figure(summary: pd.DataFrame, fig_dir: Path, horizon_seconds: float | None) -> Path:
    scenarios = sorted(summary["scenario"].unique())
    x = np.arange(len(scenarios))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.8, 5.6), dpi=180)
    ymax = 0.0
    for offset, policy in enumerate(POLICY_ORDER):
        values = []
        errors = []
        for scenario in scenarios:
            row = summary[(summary["scenario"] == scenario) & (summary["policy"] == policy)].iloc[0]
            values.append(float(row["episode_time_mean"]))
            errors.append(float(row["episode_time_std"]) if np.isfinite(row["episode_time_std"]) else 0.0)
        positions = x + (offset - 1) * width
        ax.bar(
            positions,
            values,
            width=width,
            yerr=errors,
            capsize=3,
            label=policy,
            color=COLORS[policy],
            alpha=0.92,
        )
        ymax = max(ymax, max(values) + max(errors or [0]))

    if horizon_seconds and np.isfinite(horizon_seconds):
        ax.axhline(horizon_seconds, color="#222222", linewidth=1.1, linestyle="--")
        ax.text(
            len(scenarios) - 0.35,
            horizon_seconds + max(70, ymax * 0.015),
            f"configured horizon {horizon_seconds:.0f}s",
            ha="right",
            va="bottom",
            fontsize=8,
            color="#222222",
        )

    ax.set_title("Episode End Time by Scenario", fontsize=15, fontweight="bold", pad=12)
    ax.set_ylabel("Episode end time (s)", fontsize=11)
    ax.set_xlabel("Scenario passenger count", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in scenarios])
    ax.grid(axis="y", color="#D8DDE6", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, loc="upper left", frameon=False)
    ax.set_ylim(0, max(ymax * 1.18, (horizon_seconds or 0) * 1.12))
    fig.text(
        0.01,
        0.01,
        "The exported time can exceed the configured horizon because matrix travel advances episode time in leg-sized jumps before the timeout is recorded.",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    path = fig_dir / "fig03_episode_end_time_by_scenario.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def tradeoff_figure(summary: pd.DataFrame, decisions: pd.DataFrame, fig_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9.8, 6.1), dpi=180)
    markers = {"FIFO": "o", "ONNX": "s", "Vanilla": "^"}
    for policy in POLICY_ORDER:
        subset = summary[summary["policy"] == policy].sort_values("scenario")
        ax.scatter(
            subset["wait_mean"],
            subset["ride_mean"],
            s=72,
            marker=markers[policy],
            color=COLORS[policy],
            label=policy,
            edgecolor="#222222",
            linewidth=0.4,
            alpha=0.95,
        )
        for _, row in subset.iterrows():
            ax.text(
                row["wait_mean"] + 12,
                row["ride_mean"] + 2,
                str(int(row["scenario"])),
                fontsize=8,
                color="#222222",
            )
    selected = summary.merge(
        decisions[["scenario", "selected_policy"]],
        left_on=["scenario", "policy"],
        right_on=["scenario", "selected_policy"],
        how="inner",
    )
    ax.scatter(
        selected["wait_mean"],
        selected["ride_mean"],
        s=190,
        facecolors="none",
        edgecolors="#000000",
        linewidths=1.5,
        label="Selected",
    )
    ax.set_title("Wait-Ride Trade-off Under Service Constraint", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Mean wait time (s), lower is better", fontsize=11)
    ax.set_ylabel("Mean ride time (s), lower is better", fontsize=11)
    ax.grid(color="#D8DDE6", linewidth=0.8, alpha=0.9)
    ax.legend(ncol=4, loc="upper right", frameon=False)
    fig.text(
        0.01,
        0.01,
        "Point labels are scenario passenger counts. Black rings indicate the service-constrained wait-first selection.",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    path = fig_dir / "fig05_wait_ride_tradeoff_selected.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def tensorboard_trend_figure(tensorboard: pd.DataFrame, fig_dir: Path) -> Path | None:
    if tensorboard.empty:
        return None
    required = [
        "DRT/EpisodeAverageWaitSeconds",
        "DRT/EpisodeAverageRideSeconds",
        "DRT/EpisodeServiceRate",
        "DRT/EpisodeCompletedPassengers",
        "DRT/EpisodeTimeAtDecision",
        "Environment/Cumulative Reward",
    ]
    if any(col not in tensorboard.columns for col in required):
        return None

    data = tensorboard.copy()
    data["step_m"] = data["step"] / 1_000_000.0
    roll = data.set_index("step_m")[required].rolling(100, min_periods=5).mean().reset_index()
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.3), dpi=180)
    panels = [
        ("DRT/EpisodeAverageWaitSeconds", "Wait (s)", "#4C78A8"),
        ("DRT/EpisodeAverageRideSeconds", "Ride (s)", "#59A14F"),
        ("DRT/EpisodeServiceRate", "Service rate", "#E15759"),
        ("DRT/EpisodeCompletedPassengers", "Completed pax", "#B07AA1"),
        ("DRT/EpisodeTimeAtDecision", "Episode time at decision (s)", "#F28E2B"),
        ("Environment/Cumulative Reward", "Cumulative reward", "#222222"),
    ]
    for ax, (tag, ylabel, color) in zip(axes.flat, panels):
        y = roll[tag] * 100 if tag == "DRT/EpisodeServiceRate" else roll[tag]
        ax.plot(roll["step_m"], y, color=color, linewidth=1.4)
        ax.set_ylabel("Service (%)" if tag == "DRT/EpisodeServiceRate" else ylabel, fontsize=8.5)
        ax.set_xlabel("Training step (M)", fontsize=8.5)
        ax.grid(color="#D8DDE6", linewidth=0.7, alpha=0.9)
        ax.tick_params(labelsize=8)
        ax.set_title(ylabel, fontsize=9.2, fontweight="bold")
    fig.suptitle("TensorBoard Operational Scalars, 100-Sample Rolling Mean", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.text(
        0.01,
        0.01,
        "Training scalars are summary-frequency samples, not the same unit as the exported evaluation episodes.",
        fontsize=8,
        color="#444444",
    )
    path = fig_dir / "fig06_tensorboard_operational_trends.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def make_figures(
    summary: pd.DataFrame,
    decisions: pd.DataFrame,
    fig_dir: Path,
    tensorboard: pd.DataFrame,
    horizon_seconds: float | None,
) -> list[Path]:
    paths = [
        grouped_metric_bar(
            summary,
            fig_dir,
            metric="wait_mean",
            std_col="wait_std",
            ylabel="Mean wait time (s)",
            title="Primary Objective: Mean Wait Time by Scenario",
            filename="fig01_wait_time_by_scenario.png",
        ),
        grouped_metric_bar(
            summary,
            fig_dir,
            metric="ride_mean",
            std_col="ride_std",
            ylabel="Mean ride time (s)",
            title="Secondary Objective: Mean Ride Time by Scenario",
            filename="fig02_ride_time_by_scenario.png",
        ),
        episode_time_figure(summary, fig_dir, horizon_seconds),
        service_figure(summary, fig_dir),
        tradeoff_figure(summary, decisions, fig_dir),
    ]
    tb_path = tensorboard_trend_figure(tensorboard, fig_dir)
    if tb_path is not None:
        paths.append(tb_path)
    return paths


def fmt(value, digits: int = 1, suffix: str = "") -> str:
    if value is None or not np.isfinite(float(value)):
        return "-"
    return f"{float(value):,.{digits}f}{suffix}"


def pct(value, digits: int = 1) -> str:
    if value is None or not np.isfinite(float(value)):
        return "-"
    return f"{float(value) * 100:,.{digits}f}%"


def display_path(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(text).replace("\n", "<br/>"), style)


def cell(text, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(str(text)).replace("\n", "<br/>"), style)


def table_style(font_size: float = 7.2, header_bg: str = "#E8EDF4") -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111111")),
            ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
            ("FONTNAME", (0, 1), (-1, -1), FONT_REGULAR),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("LEADING", (0, 0), (-1, -1), font_size + 2.0),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#B9C0CA")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F9FB")]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3.2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3.2),
            ("TOPPADDING", (0, 0), (-1, -1), 3.0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3.0),
        ]
    )


def long_table(rows, col_widths=None, font_size: float = 7.2) -> LongTable:
    tbl = LongTable(rows, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(table_style(font_size=font_size))
    return tbl


def scaled_image(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as img:
        width_px, height_px = img.size
    ratio = min(max_width / width_px, max_height / height_px)
    return Image(str(path), width=width_px * ratio, height=height_px * ratio)


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont(FONT_REGULAR, 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(1.45 * cm, 1.0 * cm, "DRT wait-first policy evaluation")
    canvas.drawRightString(A4[0] - 1.45 * cm, 1.0 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build_source_table(source: pd.DataFrame, styles: dict[str, ParagraphStyle]):
    rows = [[cell("Policy", styles["th"]), cell("Export folder", styles["th"]), cell("Episodes", styles["th"]), cell("Completed", styles["th"]), cell("Timeout", styles["th"]), cell("Scenarios", styles["th"])]]
    for _, row in source.iterrows():
        rows.append(
            [
                cell(row["policy"], styles["td"]),
                cell(row["export_folder"], styles["td_small"]),
                cell(row["episodes"], styles["td"]),
                cell(row["completed_episodes"], styles["td"]),
                cell(row["timeout_episodes"], styles["td"]),
                cell(row["scenarios"], styles["td"]),
            ]
        )
    return rows


def build_macro_table(macro: pd.DataFrame, styles: dict[str, ParagraphStyle]):
    rows = [[cell(x, styles["th"]) for x in ["Policy", "Episodes", "Full-run", "Service", "Min SR", "Wait", "P90 Wait", "Ride", "End time", "Timeout"]]]
    for _, row in macro.iterrows():
        rows.append(
            [
                cell(row["policy"], styles["td"]),
                cell(int(row["total_episodes"]), styles["td"]),
                cell(pct(row["completed_run_rate"]), styles["td"]),
                cell(pct(row["service_rate"]), styles["td"]),
                cell(pct(row["service_rate_min"]), styles["td"]),
                cell(fmt(row["wait_mean"], 1, " s"), styles["td"]),
                cell(fmt(row["wait_p90_mean"], 1, " s"), styles["td"]),
                cell(fmt(row["ride_mean"], 1, " s"), styles["td"]),
                cell(fmt(row["episode_time_mean"], 1, " s"), styles["td"]),
                cell(int(row["timeout_runs"]), styles["td"]),
            ]
        )
    return rows


def build_runtime_table(
    runtime: dict[str, str],
    training_config: dict[str, str],
    tb_meta: dict[str, str],
    styles: dict[str, ParagraphStyle],
):
    rows = [[cell("Source", styles["th"]), cell("Field", styles["th"]), cell("Value", styles["th"]), cell("Meaning", styles["th"])]]
    runtime_items = [
        ("Scene", "episodeLengthSeconds", runtime.get("episode_length_seconds", "-"), "DRTBusController configured episode horizon."),
        ("Scene", "stopWhenAllRequestsCompleted", runtime.get("stop_when_all_requests_completed", "-"), "Episodes can finish early when every request is served."),
        ("Scene", "simulationSecondsPerRealSecond", runtime.get("simulation_seconds_per_real_second", "-"), "Episode clock multiplier for real-time physical simulation."),
        ("Scene", "noMovementTimeoutRealSeconds", runtime.get("no_movement_timeout_real_seconds", "-"), "Vehicle fault timeout in real seconds."),
        ("Scene", "trafficBlockTimeoutRealSeconds", runtime.get("traffic_block_timeout_real_seconds", "-"), "Traffic-block tolerance in real seconds."),
        ("Scene", "Agent MaxStep", runtime.get("agent_max_step", "-"), "ML-Agents max step in scene; 0 means no scene-side cap."),
        ("PPO YAML", "max_steps", training_config.get("max_steps", "-"), "Training budget configured in ml-agents."),
        ("PPO YAML", "time_horizon", training_config.get("time_horizon", "-"), "PPO trajectory horizon."),
        ("PPO YAML", "summary_freq", training_config.get("summary_freq", "-"), "TensorBoard scalar write interval."),
        ("PPO YAML", "checkpoint_interval", training_config.get("checkpoint_interval", "-"), "Checkpoint save interval."),
        ("TensorBoard", "last_step", tb_meta.get("last_step", "-"), "Latest scalar step included in this report."),
    ]
    for source, field, value, meaning in runtime_items:
        rows.append([cell(source, styles["td"]), cell(field, styles["td"]), cell(value, styles["td"]), cell(meaning, styles["td_small"])])
    return rows


def build_episode_time_table(summary: pd.DataFrame, horizon_seconds: float | None, styles: dict[str, ParagraphStyle]):
    rows = [[cell(x, styles["th"]) for x in ["Scen.", "Policy", "Runs", "Comp.", "T/O", "Mean end", "Max end", "Horizon use", "Finish note"]]]
    for _, row in summary.iterrows():
        if horizon_seconds and np.isfinite(horizon_seconds) and horizon_seconds > 0:
            horizon_use = float(row["episode_time_mean"]) / horizon_seconds
            horizon_text = pct(horizon_use)
        else:
            horizon_text = "-"
        note = "all completed" if int(row["timeout_runs"]) == 0 else f"{int(row['timeout_runs'])} timeout run(s)"
        rows.append(
            [
                cell(int(row["scenario"]), styles["td"]),
                cell(row["policy"], styles["td"]),
                cell(int(row["n_episodes"]), styles["td"]),
                cell(f"{int(row['completed_runs'])}/{int(row['n_episodes'])}", styles["td"]),
                cell(int(row["timeout_runs"]), styles["td"]),
                cell(fmt(row["episode_time_mean"], 1, " s"), styles["td"]),
                cell(fmt(row["episode_time_max"], 1, " s"), styles["td"]),
                cell(horizon_text, styles["td"]),
                cell(note, styles["td_small"]),
            ]
        )
    return rows


def build_tensorboard_window_table(windows: pd.DataFrame, styles: dict[str, ParagraphStyle]):
    rows = [[cell(x, styles["th"]) for x in ["Window", "Steps", "Service", "Full", "Completed", "Wait", "Ride", "Decision time", "Reward"]]]
    for _, row in windows.iterrows():
        rows.append(
            [
                cell(row["window"], styles["td"]),
                cell(f"{int(row['start_step']):,}-{int(row['end_step']):,}", styles["td_small"]),
                cell(pct(row["service_rate"]), styles["td"]),
                cell(pct(row["full_completion_rate"]), styles["td"]),
                cell(fmt(row["completed_passengers"], 1), styles["td"]),
                cell(fmt(row["wait_seconds"], 1, " s"), styles["td"]),
                cell(fmt(row["ride_seconds"], 1, " s"), styles["td"]),
                cell(fmt(row["episode_time_at_decision"], 1, " s"), styles["td"]),
                cell(fmt(row["cumulative_reward"], 1), styles["td"]),
            ]
        )
    return rows


def build_tensorboard_inventory_table(inventory: pd.DataFrame, styles: dict[str, ParagraphStyle]):
    important = [
        "DRT/EpisodeAverageWaitSeconds",
        "DRT/EpisodeAverageRideSeconds",
        "DRT/EpisodeServiceRate",
        "DRT/EpisodeCompletedPassengers",
        "DRT/EpisodeCompletedAllRequests",
        "DRT/EpisodeTimeAtDecision",
        "DRT/EpisodeDecisionCount",
        "DRT/EpisodeTravelDistanceMeters",
        "DRT/Reward/EpisodeTotal",
        "Environment/Cumulative Reward",
        "Environment/Episode Length",
    ]
    rows = [[cell(x, styles["th"]) for x in ["Tag", "Samples", "First step", "Last step", "Last value"]]]
    selected = inventory[inventory["tag"].isin(important)].copy()
    tag_order = {tag: i for i, tag in enumerate(important)}
    selected["order"] = selected["tag"].map(tag_order)
    for _, row in selected.sort_values("order").iterrows():
        rows.append(
            [
                cell(row["tag"], styles["td_small"]),
                cell(int(row["samples"]), styles["td"]),
                cell(f"{int(row['first_step']):,}", styles["td"]),
                cell(f"{int(row['last_step']):,}", styles["td"]),
                cell(fmt(row["last_value"], 3), styles["td"]),
            ]
        )
    return rows


def build_decision_table(decisions: pd.DataFrame, styles: dict[str, ParagraphStyle]):
    rows = [[cell(x, styles["th"]) for x in ["Scenario", "Selected", "Service", "Wait", "Ride", "Wait advantage", "Note"]]]
    for _, row in decisions.iterrows():
        rows.append(
            [
                cell(int(row["scenario"]), styles["td"]),
                cell(row["selected_policy"], styles["td"]),
                cell(pct(row["selected_service_rate"]), styles["td"]),
                cell(fmt(row["selected_wait"], 1, " s"), styles["td"]),
                cell(fmt(row["selected_ride"], 1, " s"), styles["td"]),
                cell(fmt(row["wait_advantage_vs_runner"], 1, " s"), styles["td"]),
                cell(row["note"], styles["td_small"]),
            ]
        )
    return rows


def build_scenario_table(summary: pd.DataFrame, styles: dict[str, ParagraphStyle]):
    rows = [
        [
            cell(x, styles["th"])
            for x in [
                "Scen.",
                "Policy",
                "Runs",
                "Full",
                "Service",
                "Wait mean",
                "P90 wait",
                "Ride mean",
                "Timeout",
            ]
        ]
    ]
    for _, row in summary.iterrows():
        rows.append(
            [
                cell(int(row["scenario"]), styles["td"]),
                cell(row["policy"], styles["td"]),
                cell(int(row["n_episodes"]), styles["td"]),
                cell(pct(row["full_completion_rate"]), styles["td"]),
                cell(pct(row["service_rate_mean"]), styles["td"]),
                cell(fmt(row["wait_mean"], 1, " s"), styles["td"]),
                cell(fmt(row["passenger_wait_p90"], 1, " s"), styles["td"]),
                cell(fmt(row["ride_mean"], 1, " s"), styles["td"]),
                cell(int(row["timeout_runs"]), styles["td"]),
            ]
        )
    return rows


def write_summary_md(
    result_dir: Path,
    source: pd.DataFrame,
    macro: pd.DataFrame,
    decisions: pd.DataFrame,
    summary: pd.DataFrame,
    runtime: dict[str, str],
    training_config: dict[str, str],
    tb_meta: dict[str, str],
    tb_windows: pd.DataFrame,
    tb_inventory: pd.DataFrame,
    pdf_path: Path,
) -> None:
    lines = [
        "# DRT FIFO / ONNX / Vanilla wait-first report",
        "",
        f"- PDF: `{display_path(pdf_path)}`",
        "- Objective: minimize mean wait time first, minimize ride time second, keep service rate as high as possible.",
        "",
        "## Source coverage",
        source.to_markdown(index=False),
        "",
        "## Runtime and TensorBoard context",
        pd.DataFrame(
            [
                {"source": "scene", "field": k, "value": v}
                for k, v in runtime.items()
            ]
            + [{"source": "ppo_yaml", "field": k, "value": v} for k, v in training_config.items()]
            + [{"source": "tensorboard", "field": k, "value": v} for k, v in tb_meta.items()]
        ).to_markdown(index=False),
        "",
        "## TensorBoard windows",
        tb_windows.to_markdown(index=False),
        "",
        "## TensorBoard key tags",
        tb_inventory.to_markdown(index=False),
        "",
        "## Macro scenario-equal summary",
        macro.to_markdown(index=False),
        "",
        "## Scenario decisions",
        decisions.to_markdown(index=False),
        "",
        "## Scenario policy summary",
        summary.to_markdown(index=False),
        "",
    ]
    (result_dir / "summary_kr.md").write_text("\n".join(lines), encoding="utf-8")


def build_pdf(
    pdf_path: Path,
    result_dir: Path,
    fig_paths: list[Path],
    source: pd.DataFrame,
    macro: pd.DataFrame,
    decisions: pd.DataFrame,
    summary: pd.DataFrame,
    runtime: dict[str, str],
    training_config: dict[str, str],
    tb_meta: dict[str, str],
    tb_windows: pd.DataFrame,
    tb_inventory: pd.DataFrame,
    best_tb_window: dict[str, float | int | str],
) -> None:
    styles = {
        "title": ParagraphStyle(
            "title",
            fontName=FONT_BOLD,
            fontSize=18,
            leading=23,
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            fontName=FONT_REGULAR,
            fontSize=10,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#444444"),
            spaceAfter=18,
        ),
        "h1": ParagraphStyle(
            "h1",
            fontName=FONT_BOLD,
            fontSize=12.5,
            leading=16,
            spaceBefore=12,
            spaceAfter=7,
        ),
        "h2": ParagraphStyle(
            "h2",
            fontName=FONT_BOLD,
            fontSize=10.5,
            leading=14,
            spaceBefore=8,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "body",
            fontName=FONT_REGULAR,
            fontSize=9.0,
            leading=13.4,
            alignment=TA_JUSTIFY,
            spaceAfter=5,
        ),
        "caption": ParagraphStyle(
            "caption",
            fontName=FONT_REGULAR,
            fontSize=8.0,
            leading=11,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#333333"),
            spaceAfter=8,
        ),
        "th": ParagraphStyle("th", fontName=FONT_BOLD, fontSize=7.0, leading=9.5, alignment=TA_CENTER),
        "td": ParagraphStyle("td", fontName=FONT_REGULAR, fontSize=7.0, leading=9.5, alignment=TA_CENTER),
        "td_small": ParagraphStyle("td_small", fontName=FONT_REGULAR, fontSize=6.2, leading=8.2, alignment=TA_LEFT),
    }

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=1.45 * cm,
        leftMargin=1.45 * cm,
        topMargin=1.35 * cm,
        bottomMargin=1.55 * cm,
        title="DRT FIFO ONNX Vanilla Wait-First Policy Evaluation",
        author="Codex",
    )
    story = []
    story.append(p("DRT Policy Evaluation Report: FIFO vs ONNX vs Vanilla", styles["title"]))
    story.append(
        p(
            "Wait-First Objective with Ride-Time Tie Break, Service Reliability, and TensorBoard Context",
            styles["subtitle"],
        )
    )
    story.append(p(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} KST", styles["subtitle"]))

    story.append(p("Abstract", styles["h1"]))
    story.append(
        p(
            "본 보고서는 Greedy 정책을 제외하고 FIFO, ONNX PPO inference, Vanilla sequential 정책만 비교한다. "
            "평가 목적은 평균 wait time 최소화가 1순위이고, 평균 ride time 최소화가 2순위이며, 서비스율은 가능한 한 100%에 가깝게 유지하는 것이다. "
            "동일 scenario 내 여러 episode는 먼저 평균을 내고, scenario별 winner는 service_rate 100% 정책을 우선 feasible set으로 둔 뒤 wait, ride 순서로 결정했다. "
            "추가로 scene의 episode horizon, CSV의 episode end time 및 finish_reason, TensorBoard custom scalar를 함께 분석하여 timeout과 학습 추세가 최종 해석에 반영되도록 했다.",
            styles["body"],
        )
    )
    story.append(p("Index Terms - DRT, PPO inference, FIFO dispatch, wait time, ride time, service rate.", styles["body"]))

    story.append(p("I. Data and Evaluation Objective", styles["h1"]))
    story.append(
        p(
            "분석은 각 정책의 최신 export 폴더에서 *_episode.csv 파일을 직접 읽어 수행했다. "
            "summary 행은 episode-level QoS 평균을 제공하고, passenger 행은 wait 및 ride 분포 검증에 사용했다. "
            "아래 source table은 이번 PDF에 포함된 실제 CSV coverage이다.",
            styles["body"],
        )
    )
    story.append(p("Table I. Source coverage used for this report.", styles["caption"]))
    story.append(
        long_table(
            build_source_table(source, styles),
            col_widths=[1.7 * cm, 8.2 * cm, 1.45 * cm, 1.55 * cm, 1.35 * cm, 2.4 * cm],
            font_size=6.8,
        )
    )
    story.append(Spacer(1, 7))
    story.append(
        p(
            "Ranking rule: for each scenario, policies with mean service_rate >= 99.9% are treated as reliable candidates. "
            "The selected policy is the reliable candidate with the lowest mean wait time; if wait is tied, lower mean ride time is preferred. "
            "If a non-reliable policy has lower served-passenger wait, it is reported as a wait-only result but not selected.",
            styles["body"],
        )
    )

    story.append(p("II. Episode Termination and Training Context", styles["h1"]))
    horizon_value = to_float(runtime.get("episode_length_seconds"))
    horizon_text = fmt(horizon_value, 0, " s") if np.isfinite(horizon_value) else "-"
    story.append(
        p(
            f"현재 scene 기준 DRTBusController의 episodeLengthSeconds는 {horizon_text}이고, stopWhenAllRequestsCompleted가 켜져 있으므로 모든 승객이 완료되면 horizon 이전에도 episode가 종료된다. "
            "반대로 timeout episode는 matrix travel이 한 leg 단위로 episode clock을 이동시킨 뒤 종료 판정이 기록되기 때문에 exported episode_time_seconds가 configured horizon을 초과할 수 있다. "
            "따라서 finish_reason과 episode_time_seconds는 반드시 같이 봐야 한다.",
            styles["body"],
        )
    )
    story.append(p("Table II. Runtime and TensorBoard configuration relevant to episode termination.", styles["caption"]))
    story.append(
        long_table(
            build_runtime_table(runtime, training_config, tb_meta, styles),
            col_widths=[2.3 * cm, 4.2 * cm, 2.6 * cm, 7.3 * cm],
            font_size=6.55,
        )
    )
    story.append(Spacer(1, 7))
    story.append(p("Table III. Episode end-time and finish-reason summary from evaluation CSVs.", styles["caption"]))
    story.append(
        long_table(
            build_episode_time_table(summary, horizon_value, styles),
            col_widths=[1.0 * cm, 1.45 * cm, 0.95 * cm, 1.45 * cm, 1.1 * cm, 1.65 * cm, 1.65 * cm, 1.55 * cm, 4.5 * cm],
            font_size=6.35,
        )
    )

    if not tb_windows.empty:
        story.append(PageBreak())
        story.append(p("III. TensorBoard Training Scalars", styles["h1"]))
        story.append(
            p(
                f"TensorBoard source는 {tb_meta.get('run_folder', '-')}이며, 이번 보고서는 step {tb_meta.get('first_step', '-')}부터 {tb_meta.get('last_step', '-')}까지의 scalar를 읽었다. "
                "이 값들은 ML-Agents summary_freq 간격의 학습 scalar이며, 아래 평가 CSV처럼 고정된 inference episode 집합은 아니다. "
                "그래도 학습 중 정책이 어떤 운영 상태를 보였는지 확인하는 데 중요하다.",
                styles["body"],
            )
        )
        if best_tb_window:
            story.append(
                p(
                    f"100-sample rolling 기준으로 service>=99%를 만족하는 구간은 없었다. 가장 높은 service window는 step {int(best_tb_window['end_step']):,} 부근이며, "
                    f"service {best_tb_window['service_rate'] * 100:.1f}%, full-completion {best_tb_window['full_completion_rate'] * 100:.1f}%, "
                    f"completed passengers {best_tb_window['completed_passengers']:.1f}, wait {best_tb_window['wait_seconds']:.1f}s, ride {best_tb_window['ride_seconds']:.1f}s였다. "
                    "즉 TensorBoard 관점에서도 high-load completion reliability가 완전히 안정화되었다고 보기는 어렵다.",
                    styles["body"],
                )
            )
        story.append(p("Table IV. TensorBoard operational windows from the latest training scalar stream.", styles["caption"]))
        story.append(
            long_table(
                build_tensorboard_window_table(tb_windows, styles),
                col_widths=[1.55 * cm, 2.65 * cm, 1.35 * cm, 1.25 * cm, 1.45 * cm, 1.55 * cm, 1.55 * cm, 2.05 * cm, 1.55 * cm],
                font_size=6.35,
            )
        )
        story.append(Spacer(1, 7))
        story.append(p("Table V. Key TensorBoard scalar tags used for interpretation.", styles["caption"]))
        story.append(
            long_table(
                build_tensorboard_inventory_table(tb_inventory, styles),
                col_widths=[6.2 * cm, 1.55 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm],
                font_size=6.35,
            )
        )

    story.append(PageBreak())
    story.append(p("IV. Aggregate Results", styles["h1"]))
    onnx_macro = macro[macro["policy"] == "ONNX"].iloc[0]
    fifo_macro = macro[macro["policy"] == "FIFO"].iloc[0]
    vanilla_macro = macro[macro["policy"] == "Vanilla"].iloc[0]
    story.append(
        p(
            f"Scenario-equal macro average shows ONNX has the lowest mean wait ({onnx_macro['wait_mean']:.1f} s), "
            f"followed by FIFO ({fifo_macro['wait_mean']:.1f} s) and Vanilla ({vanilla_macro['wait_mean']:.1f} s). "
            f"However, ONNX service reliability is weaker because scenario 50 contains timeout runs, giving macro service {onnx_macro['service_rate'] * 100:.1f}% "
            f"and minimum scenario service {onnx_macro['service_rate_min'] * 100:.1f}%. "
            f"FIFO keeps 100.0% service across all scenarios and is therefore the conservative single-policy choice under a strict service constraint.",
            styles["body"],
        )
    )
    story.append(p("Table VI. Scenario-equal macro summary. Lower wait, ride, and end time are better.", styles["caption"]))
    story.append(
        long_table(
            build_macro_table(macro, styles),
            col_widths=[1.55 * cm, 1.35 * cm, 1.45 * cm, 1.35 * cm, 1.35 * cm, 1.5 * cm, 1.6 * cm, 1.45 * cm, 1.55 * cm, 1.1 * cm],
            font_size=6.8,
        )
    )

    story.append(p("V. Scenario-Level Policy Selection", styles["h1"]))
    story.append(
        p(
            "Scenario별 결론은 단일 평균값보다 중요하다. 승객 수가 낮은 scenario 14에서는 FIFO가 wait와 ride 모두 가장 낮았다. "
            "Scenario 18, 22, 30, 40에서는 ONNX가 100% service를 유지하면서 wait를 가장 낮췄다. "
            "Scenario 50은 ONNX의 served-passenger wait가 낮게 보이는 구간이 있지만 9개 run 중 5개가 episode timeout으로 끝났으므로, "
            "서비스 신뢰성 기준에서는 FIFO가 최종 선택이다.",
            styles["body"],
        )
    )
    story.append(p("Table VII. Service-constrained wait-first scenario decisions.", styles["caption"]))
    story.append(
        long_table(
            build_decision_table(decisions, styles),
            col_widths=[1.45 * cm, 1.8 * cm, 1.55 * cm, 1.85 * cm, 1.75 * cm, 2.0 * cm, 6.4 * cm],
            font_size=6.8,
        )
    )
    story.append(Spacer(1, 7))
    story.append(p("Table VIII. Full scenario-policy QoS table.", styles["caption"]))
    story.append(
        long_table(
            build_scenario_table(summary, styles),
            col_widths=[1.15 * cm, 1.55 * cm, 1.05 * cm, 1.4 * cm, 1.45 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm, 1.25 * cm],
            font_size=6.55,
        )
    )

    story.append(PageBreak())
    story.append(p("VI. Figures", styles["h1"]))
    figure_captions = {
        "fig01_wait_time_by_scenario.png": "Fig. 1. Primary objective comparison. ONNX is wait-best in scenarios 18, 22, 30, and 40; FIFO is wait-best in scenario 14 and the reliable choice in scenario 50.",
        "fig02_ride_time_by_scenario.png": "Fig. 2. Secondary objective comparison. Vanilla has competitive ride time in several middle scenarios, but its wait penalty is too large for the requested objective.",
        "fig03_episode_end_time_by_scenario.png": "Fig. 3. Episode end time. ONNX scenario 50 has timeout behavior, and exported episode time can exceed the configured horizon because timeout is recorded after leg-level time advancement.",
        "fig04_service_reliability_by_scenario.png": "Fig. 4. Service reliability. ONNX scenario 50 is the only material reliability failure, with timeout runs and service below 100%.",
        "fig05_wait_ride_tradeoff_selected.png": "Fig. 5. Wait-ride trade-off. Black rings mark the selected policy after applying the service constraint.",
        "fig06_tensorboard_operational_trends.png": "Fig. 6. TensorBoard operational scalar trends. The high-load completion and service metrics remain unstable near the latest training segment.",
    }
    for fig_path in fig_paths:
        story.append(KeepTogether([scaled_image(fig_path, 17.4 * cm, 9.2 * cm), p(figure_captions[fig_path.name], styles["caption"])]))

    story.append(PageBreak())
    story.append(p("VII. Discussion", styles["h1"]))
    story.append(p("A. Wait time as the dominant objective", styles["h2"]))
    story.append(
        p(
            "ONNX는 중간 이상 수요에서 wait time을 줄이는 효과가 뚜렷하다. "
            "FIFO 대비 ONNX의 평균 wait 감소율은 scenario 18에서 약 17.5%, scenario 22에서 10.6%, scenario 30에서 3.7%, scenario 40에서 16.5%이며, 이 네 구간에서는 모두 100% service를 유지했다. "
            "다만 개선 폭은 균일하지 않다. Scenario 30은 wait 이득이 작고 ride time은 FIFO보다 길기 때문에, 강한 승리라기보다 wait-first 기준의 제한적 우위로 해석하는 편이 맞다.",
            styles["body"],
        )
    )
    story.append(p("B. Ride time as the secondary objective", styles["h2"]))
    story.append(
        p(
            "Ride time만 보면 Vanilla가 일부 구간에서 경쟁력이 있다. "
            "예를 들어 Vanilla는 scenario 18과 22에서 ride-best이지만, 같은 구간의 mean wait는 ONNX보다 각각 36.3%, 38.8% 높다. "
            "이번 목표는 wait 감소가 최우선이므로 이 ride 이득만으로 Vanilla를 선택할 수 없다. Scenario 50에서는 FIFO가 100% service를 유지하면서 reliable ride time도 가장 낮기 때문에 secondary objective까지 같이 만족한다.",
            styles["body"],
        )
    )
    story.append(p("C. Service-rate reliability", styles["h2"]))
    story.append(
        p(
            "가장 큰 운영 리스크는 ONNX scenario 50이다. "
            "완료된 승객 기준 평균 wait만 보면 ONNX가 좋아 보일 수 있지만, 실제로는 9개 run 중 5개가 timeout으로 끝났고 평균 service rate는 91.6%까지 떨어졌다. "
            "DRT 운영에서는 미탑승 또는 미하차 승객이 ride time 증가보다 더 큰 실패이므로, ONNX timeout 문제가 해결되어 재평가되기 전까지 scenario 50의 권장 정책은 FIFO로 유지해야 한다.",
            styles["body"],
        )
    )
    story.append(p("D. Episode end time and TensorBoard evidence", styles["h2"]))
    story.append(
        p(
            "Episode 종료 시간은 단순 보조 지표가 아니라 reliability 해석의 핵심이다. "
            "Evaluation CSV에서 ONNX scenario 50은 평균 wait가 낮아 보이지만 timeout run이 포함되고 max episode_time_seconds가 configured horizon을 넘는다. "
            "TensorBoard의 latest scalar도 마지막 구간에서 service와 completed-all 비율이 완전히 안정적이지 않음을 보여준다. "
            "따라서 ONNX의 wait 이득은 100% service를 유지한 scenario 18, 22, 30, 40에 한정해서 해석해야 한다.",
            styles["body"],
        )
    )
    story.append(p("E. Recommended policy use", styles["h2"]))
    story.append(
        p(
            "Scenario-aware switching이 가능하다면 scenario 14와 50은 FIFO, scenario 18, 22, 30, 40은 ONNX를 쓰는 구성이 가장 직접적으로 목표에 맞는다. "
            "단일 정책만 배포해야 한다면 FIFO가 보수적인 선택이다. FIFO는 이번 export에서 모든 scenario의 completion과 service가 100%이고, wait도 Vanilla보다 크게 낮다. "
            "ONNX를 단일 정책으로 쓰려면 scenario 50 timeout을 먼저 제거하거나, 배포 범위에서 해당 고부하 case를 제외해야 한다.",
            styles["body"],
        )
    )

    story.append(p("VIII. Conclusion", styles["h1"]))
    story.append(
        p(
            "제시된 목표 기준에서 Vanilla는 최종 선택 대상이 아니다. "
            "ONNX는 6개 scenario 중 4개에서 가장 강한 wait-time 감소를 보였지만, scenario 50의 reliability 실패 때문에 무조건적인 ONNX 추천은 불가능하다. "
            "실제 결론은 hybrid policy decision이다. ONNX가 100% service를 유지하는 구간에서는 ONNX를 쓰고, ONNX reliability가 무너지거나 FIFO가 이미 더 빠른 구간에서는 FIFO를 쓰는 것이 가장 타당하다. "
            "이 방식이 평균 wait 숫자 뒤에 service-rate 실패를 숨기지 않는 wait-first 결론이다.",
            styles["body"],
        )
    )
    story.append(p("Artifacts", styles["h1"]))
    story.append(
        p(
            f"Detailed CSV tables and PNG figures were written to {display_path(result_dir)}. The final PDF is {display_path(pdf_path)}.",
            styles["body"],
        )
    )

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    register_fonts()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    selected_dirs = latest_run_dirs()
    result_dir, fig_dir, table_dir = ensure_dirs(stamp)

    episodes, passengers = collect_data(selected_dirs)
    summary, macro, decisions = build_summaries(episodes, passengers)
    source = source_coverage(episodes, selected_dirs)
    runtime = parse_scene_runtime_settings()
    training_config = parse_training_config()
    tensorboard, tb_meta, tb_inventory = load_tensorboard_scalars()
    tb_windows = build_tensorboard_windows(tensorboard)
    best_tb_window = best_tensorboard_reliability_window(tensorboard, 100)
    horizon_seconds = to_float(runtime.get("episode_length_seconds"))
    if not np.isfinite(horizon_seconds):
        horizon_seconds = None

    episodes.to_csv(table_dir / "episode_level_metrics.csv", index=False, encoding="utf-8-sig")
    passengers.to_csv(table_dir / "passenger_level_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(table_dir / "scenario_policy_summary_wait_first.csv", index=False, encoding="utf-8-sig")
    macro.to_csv(table_dir / "policy_macro_summary_wait_first.csv", index=False, encoding="utf-8-sig")
    decisions.to_csv(table_dir / "scenario_decisions_wait_first.csv", index=False, encoding="utf-8-sig")
    source.to_csv(table_dir / "source_coverage.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([runtime]).to_csv(table_dir / "runtime_scene_settings.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([training_config]).to_csv(table_dir / "training_config_summary.csv", index=False, encoding="utf-8-sig")
    if not tensorboard.empty:
        tensorboard.to_csv(table_dir / "tensorboard_scalar_timeseries.csv", index=False, encoding="utf-8-sig")
    tb_windows.to_csv(table_dir / "tensorboard_window_summary.csv", index=False, encoding="utf-8-sig")
    tb_inventory.to_csv(table_dir / "tensorboard_scalar_inventory.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([best_tb_window]).to_csv(table_dir / "tensorboard_best_window.csv", index=False, encoding="utf-8-sig")

    fig_paths = make_figures(summary, decisions, fig_dir, tensorboard, horizon_seconds)
    pdf_path = PDF_OUT / f"drt_fifo_onnx_vanilla_wait_first_report_{stamp}.pdf"
    build_pdf(
        pdf_path,
        result_dir,
        fig_paths,
        source,
        macro,
        decisions,
        summary,
        runtime,
        training_config,
        tb_meta,
        tb_windows,
        tb_inventory,
        best_tb_window,
    )
    write_summary_md(
        result_dir,
        source,
        macro,
        decisions,
        summary,
        runtime,
        training_config,
        tb_meta,
        tb_windows,
        tb_inventory,
        pdf_path,
    )
    print(pdf_path)
    print(result_dir)
    for path in fig_paths:
        print(path)


if __name__ == "__main__":
    main()
