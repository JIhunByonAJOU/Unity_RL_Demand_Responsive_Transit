# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
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
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    TableStyle,
)


SOURCE_PDF = Path("output/pdf/DRT_compare_30Pas_new.pdf")
SOURCE_TABLE_DIR = Path("output/drt_compare_30Pas_new/tables")
OUTPUT_ROOT = Path("output/drt_compare_30Pas_new_professional")
FIG_DIR = OUTPUT_ROOT / "figures"
TABLE_DIR = OUTPUT_ROOT / "tables"
PDF_PATH = Path("output/pdf/DRT_compare_30Pas_new_professional_report.pdf")

POLICY_ORDER = ["Vanilla", "FIFO", "ONNX"]
POLICY_COLORS = {
    "Vanilla": "#B8B8B8",
    "FIFO": "#666666",
    "ONNX": "#1F5F99",
}
GREEN = "#2E7D5B"
RUST = "#B75B45"
GRID = "#D7DEE8"

FONT_REGULAR = "MalgunGothic"
FONT_BOLD = "MalgunGothic-Bold"
FONT_REGULAR_PATH = Path(r"C:\Windows\Fonts\malgun.ttf")
FONT_BOLD_PATH = Path(r"C:\Windows\Fonts\malgunbd.ttf")


def register_fonts() -> tuple[str, str]:
    if FONT_REGULAR_PATH.exists() and FONT_BOLD_PATH.exists():
        pdfmetrics.registerFont(TTFont(FONT_REGULAR, str(FONT_REGULAR_PATH)))
        pdfmetrics.registerFont(TTFont(FONT_BOLD, str(FONT_BOLD_PATH)))
        plt.rcParams["font.family"] = "Malgun Gothic"
        return FONT_REGULAR, FONT_BOLD
    plt.rcParams["font.family"] = "DejaVu Sans"
    return "Helvetica", "Helvetica-Bold"


def setup_matplotlib() -> None:
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["font.size"] = 10


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {
        "summary": SOURCE_TABLE_DIR / "policy_analysis_summary_ko.csv",
        "passengers": SOURCE_TABLE_DIR / "passenger_kpi_metrics.csv",
        "episodes": SOURCE_TABLE_DIR / "episode_kpi_metrics.csv",
        "legs": SOURCE_TABLE_DIR / "route_leg_kpi_metrics.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise SystemExit("Missing source table files: " + ", ".join(missing))

    summary = pd.read_csv(required["summary"]).sort_values("policy_order").reset_index(drop=True)
    passengers = pd.read_csv(required["passengers"]).sort_values(["policy_order", "episode_index", "passenger_id"])
    episodes = pd.read_csv(required["episodes"]).sort_values(["policy_order", "episode_index"])
    legs = pd.read_csv(required["legs"]).sort_values(["policy_order", "episode_index", "leg_index"])
    return summary, passengers, episodes, legs


def finite(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def pct(value: float, digits: int = 1) -> str:
    if math.isnan(value):
        return "-"
    return f"{value * 100.0:,.{digits}f}%"


def num(value: float, digits: int = 1, suffix: str = "") -> str:
    if math.isnan(value):
        return "-"
    return f"{value:,.{digits}f}{suffix}"


def delta_text(delta: float, digits: int = 1, suffix: str = "") -> str:
    if math.isnan(delta):
        return "-"
    return f"{delta:+,.{digits}f}{suffix}"


def best_policy(frame: pd.DataFrame, value_col: str, higher: bool = False) -> str:
    values = frame.set_index("policy")[value_col].astype(float)
    return values.idxmax() if higher else values.idxmin()


def ordered(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.set_index("policy").loc[POLICY_ORDER].reset_index()


def add_percentile_tables(
    passengers: pd.DataFrame,
    episodes: pd.DataFrame,
    legs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for policy in POLICY_ORDER:
        data = passengers[passengers["policy"] == policy]
        for label, col in [
            ("wait", "wait_time_seconds"),
            ("ride", "ride_time_seconds"),
            ("total_service", "total_service_time_seconds"),
        ]:
            series = finite(data[col])
            quantiles = series.quantile([0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0])
            rows.append(
                {
                    "policy": policy,
                    "metric": label,
                    "n": int(series.shape[0]),
                    "mean": float(series.mean()),
                    "std": float(series.std(ddof=0)),
                    "min": float(quantiles.loc[0.0]),
                    "p25": float(quantiles.loc[0.25]),
                    "p50": float(quantiles.loc[0.5]),
                    "p75": float(quantiles.loc[0.75]),
                    "p90": float(quantiles.loc[0.9]),
                    "p95": float(quantiles.loc[0.95]),
                    "max": float(quantiles.loc[1.0]),
                }
            )
    percentiles = pd.DataFrame(rows)

    thresholds = [300, 600, 900, 1200, 1800, 2400]
    threshold_rows: list[dict[str, object]] = []
    for policy in POLICY_ORDER:
        data = passengers[passengers["policy"] == policy]
        for threshold in thresholds:
            threshold_rows.append(
                {
                    "policy": policy,
                    "threshold_seconds": threshold,
                    "wait_share": float((data["wait_time_seconds"] <= threshold).mean()),
                    "total_service_share": float((data["total_service_time_seconds"] <= threshold).mean()),
                }
            )
    threshold_frame = pd.DataFrame(threshold_rows)

    load_rows: list[dict[str, object]] = []
    for policy in POLICY_ORDER:
        data = legs[legs["policy"] == policy].copy()
        episodes_n = max(1, int(episodes[episodes["policy"] == policy]["episode_index"].nunique()))
        data["load_bin"] = pd.cut(
            data["onboard_during_leg"].astype(float),
            bins=[-1, 0, 1, 2, 3, 999],
            labels=["0", "1", "2", "3", "4+"],
        )
        grouped = data.groupby("load_bin", observed=False)["leg_distance_meters"].sum()
        total = float(grouped.sum())
        for load_bin, meters in grouped.items():
            km_total = float(meters) / 1000.0
            load_rows.append(
                {
                    "policy": policy,
                    "load_bin": str(load_bin),
                    "distance_km_total": km_total,
                    "distance_km_per_episode": km_total / episodes_n,
                    "distance_share": km_total / (total / 1000.0) if total else float("nan"),
                }
            )
    load_frame = pd.DataFrame(load_rows)

    stop_rows: list[dict[str, object]] = []
    for policy in POLICY_ORDER:
        data = legs[legs["policy"] == policy]
        episodes_n = max(1, int(episodes[episodes["policy"] == policy]["episode_index"].nunique()))
        events = data.groupby("arrived_stop_id", as_index=False)[["boarded_count", "dropped_off_count"]].sum()
        for _, row in events.iterrows():
            stop_rows.append(
                {
                    "policy": policy,
                    "stop_id": int(row["arrived_stop_id"]),
                    "boarded_per_episode": float(row["boarded_count"]) / episodes_n,
                    "dropped_per_episode": float(row["dropped_off_count"]) / episodes_n,
                }
            )
    stop_frame = pd.DataFrame(stop_rows)

    episode_variability = (
        episodes.groupby("policy", as_index=False)
        .agg(
            episodes=("episode_index", "nunique"),
            wait_mean_sd=("average_wait_seconds", lambda s: float(s.std(ddof=0))),
            ride_mean_sd=("average_ride_seconds", lambda s: float(s.std(ddof=0))),
            episode_time_sd=("episode_time_seconds", lambda s: float(s.std(ddof=0))),
            distance_sd=("episode_distance_meters", lambda s: float(s.std(ddof=0) / 1000.0)),
        )
        .pipe(ordered)
    )
    return percentiles, threshold_frame, load_frame, stop_frame, episode_variability


def normalized_scorecard(
    summary: pd.DataFrame,
    percentiles: pd.DataFrame,
    episodes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pvt = percentiles.pivot(index="policy", columns="metric")
    enriched = summary.set_index("policy").copy()
    for metric in ["wait", "ride", "total_service"]:
        for q in ["p50", "p90", "p95", "max"]:
            enriched[f"{metric}_{q}"] = pvt[(q, metric)]
    enriched["episode_time_cv"] = (
        episodes.groupby("policy")["episode_time_seconds"].std(ddof=0)
        / episodes.groupby("policy")["episode_time_seconds"].mean()
    ).fillna(0.0)
    enriched = enriched.loc[POLICY_ORDER]

    specs = [
        ("서비스율", "service_rate", "higher", "Reliability"),
        ("평균 대기시간", "wait_mean_seconds", "lower", "Passenger time"),
        ("중앙 대기시간", "wait_p50", "lower", "Passenger time"),
        ("p90 대기시간", "wait_p90", "lower", "Tail/fairness"),
        ("최대 대기시간", "wait_max_seconds", "lower", "Tail/fairness"),
        ("대기 Gini", "wait_gini", "lower", "Tail/fairness"),
        ("평균 탑승시간", "ride_mean_seconds", "lower", "Passenger time"),
        ("중앙 탑승시간", "ride_p50", "lower", "Passenger time"),
        ("p90 탑승시간", "ride_p90", "lower", "Tail/fairness"),
        ("최대 탑승시간", "ride_max_seconds", "lower", "Tail/fairness"),
        ("탑승 Gini", "ride_gini", "lower", "Tail/fairness"),
        ("총 서비스 중앙값", "total_service_p50", "lower", "Passenger time"),
        ("총 주행거리", "total_distance_km", "lower", "Operating efficiency"),
        ("에피소드 시간", "episode_time_minutes", "lower", "Operating efficiency"),
        ("km/완료승객", "distance_per_completed_km", "lower", "Operating efficiency"),
        ("분/완료승객", "time_per_completed_minutes", "lower", "Operating efficiency"),
        ("Loaded distance", "loaded_distance_ratio", "higher", "Operating efficiency"),
        ("운행시간 변동계수", "episode_time_cv", "lower", "Robustness"),
    ]

    rows: list[dict[str, object]] = []
    for metric_label, col, direction, category in specs:
        values = enriched[col].astype(float).to_numpy()
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        if math.isclose(vmin, vmax):
            scores = np.repeat(0.5, len(values))
        elif direction == "higher":
            scores = (values - vmin) / (vmax - vmin)
        else:
            scores = (vmax - values) / (vmax - vmin)
        for policy, raw, score in zip(POLICY_ORDER, values, scores):
            rows.append(
                {
                    "metric": metric_label,
                    "category": category,
                    "direction": direction,
                    "policy": policy,
                    "raw_value": float(raw),
                    "score": float(score),
                }
            )
    scorecard = pd.DataFrame(rows)
    category_scores = (
        scorecard.groupby(["category", "policy"], as_index=False)["score"]
        .mean()
        .pivot(index="policy", columns="category", values="score")
        .loc[POLICY_ORDER]
        .reset_index()
    )
    category_scores["Overall_equal_weight"] = category_scores.drop(columns=["policy"]).mean(axis=1)
    return scorecard, category_scores


def save_analysis_tables(
    percentiles: pd.DataFrame,
    thresholds: pd.DataFrame,
    load_frame: pd.DataFrame,
    stop_frame: pd.DataFrame,
    variability: pd.DataFrame,
    scorecard: pd.DataFrame,
    category_scores: pd.DataFrame,
) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    percentiles.to_csv(TABLE_DIR / "passenger_time_percentiles.csv", index=False, encoding="utf-8-sig")
    thresholds.to_csv(TABLE_DIR / "service_threshold_coverage.csv", index=False, encoding="utf-8-sig")
    load_frame.to_csv(TABLE_DIR / "route_load_distance_share.csv", index=False, encoding="utf-8-sig")
    stop_frame.to_csv(TABLE_DIR / "stop_event_profile.csv", index=False, encoding="utf-8-sig")
    variability.to_csv(TABLE_DIR / "episode_variability.csv", index=False, encoding="utf-8-sig")
    scorecard.to_csv(TABLE_DIR / "professional_scorecard.csv", index=False, encoding="utf-8-sig")
    category_scores.to_csv(TABLE_DIR / "category_scores.csv", index=False, encoding="utf-8-sig")


def clean_axes(ax, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.7, alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#3A3A3A")
    ax.spines["bottom"].set_color("#3A3A3A")
    ax.tick_params(labelsize=9, colors="#222222")


def ecdf(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    data = np.sort(finite(values).to_numpy())
    if data.size == 0:
        return np.array([]), np.array([])
    y = np.arange(1, data.size + 1) / data.size
    return data, y


def policy_boxplot(ax, passengers: pd.DataFrame, value_col: str, title: str) -> None:
    data = [
        finite(passengers[passengers["policy"] == policy][value_col]).to_numpy()
        for policy in POLICY_ORDER
    ]
    box = ax.boxplot(
        data,
        tick_labels=POLICY_ORDER,
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "#111111",
            "markersize": 4.5,
        },
        medianprops={"color": "#111111", "linewidth": 1.3},
        whiskerprops={"color": "#333333", "linewidth": 1.0},
        capprops={"color": "#333333", "linewidth": 1.0},
        flierprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "#777777",
            "markersize": 2.4,
            "alpha": 0.65,
        },
    )
    for patch, policy in zip(box["boxes"], POLICY_ORDER):
        patch.set_facecolor(POLICY_COLORS[policy])
        patch.set_edgecolor("#222222")
        patch.set_alpha(0.78)

    ax.set_title(title, fontsize=12.5, fontweight="bold")
    ax.set_ylabel("초")
    clean_axes(ax, "y")
    ax.text(
        0.98,
        0.96,
        "상자=IQR, 선=중앙값, 원=평균",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.6,
        color="#333333",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#B6BEC8", "linewidth": 0.5},
    )


def build_fig_direct_kpi_comparison(summary: pd.DataFrame, passengers: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.8), dpi=190)
    fig.suptitle("핵심 KPI 직접 비교", fontsize=16, fontweight="bold")

    policy_boxplot(axes[0, 0], passengers, "wait_time_seconds", "대기시간 분포 상자수염그림")
    policy_boxplot(axes[0, 1], passengers, "ride_time_seconds", "탑승시간 분포 상자수염그림")

    data = summary.set_index("policy").loc[POLICY_ORDER]
    x = np.arange(len(POLICY_ORDER))
    width = 0.34
    wait_mean = data["wait_mean_seconds"].to_numpy(dtype=float)
    ride_mean = data["ride_mean_seconds"].to_numpy(dtype=float)
    wait_median = data["wait_median_seconds"].to_numpy(dtype=float)
    ride_median = data["ride_median_seconds"].to_numpy(dtype=float)

    wait_bars = axes[1, 0].bar(
        x - width / 2,
        wait_mean,
        width=width,
        label="대기 평균",
        color="#4F81BD",
        edgecolor="#222222",
        linewidth=0.35,
    )
    ride_bars = axes[1, 0].bar(
        x + width / 2,
        ride_mean,
        width=width,
        label="탑승 평균",
        color="#9E9E9E",
        edgecolor="#222222",
        linewidth=0.35,
    )
    axes[1, 0].scatter(x - width / 2, wait_median, marker="D", s=32, color="#0B3D66", label="대기 중앙값", zorder=4)
    axes[1, 0].scatter(x + width / 2, ride_median, marker="D", s=32, color="#3B3B3B", label="탑승 중앙값", zorder=4)
    axes[1, 0].bar_label(wait_bars, labels=[f"{value:.0f}" for value in wait_mean], fontsize=7, padding=2)
    axes[1, 0].bar_label(ride_bars, labels=[f"{value:.0f}" for value in ride_mean], fontsize=7, padding=2)
    axes[1, 0].set_xticks(x, POLICY_ORDER)
    axes[1, 0].set_title("대기시간과 탑승시간 평균/중앙값 비교", fontsize=12.5, fontweight="bold")
    axes[1, 0].set_ylabel("초")
    axes[1, 0].legend(frameon=False, fontsize=8, ncols=2, loc="upper right")
    clean_axes(axes[1, 0], "y")

    distance_values = data["total_distance_km"].to_numpy(dtype=float)
    distance_bars = axes[1, 1].bar(
        x,
        distance_values,
        color=[POLICY_COLORS[policy] for policy in POLICY_ORDER],
        edgecolor="#222222",
        linewidth=0.4,
        width=0.58,
    )
    axes[1, 1].bar_label(distance_bars, labels=[f"{value:.1f} km" for value in distance_values], fontsize=8, padding=3)
    for xi, policy in zip(x, POLICY_ORDER):
        km_per_passenger = float(data.loc[policy, "distance_per_completed_km"])
        axes[1, 1].text(
            xi,
            max(distance_values) * 0.06,
            f"{km_per_passenger:.2f} km/승객",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#222222",
        )
    axes[1, 1].set_xticks(x, POLICY_ORDER)
    axes[1, 1].set_ylim(0, max(distance_values) * 1.18)
    axes[1, 1].set_title("총 주행거리 직접 비교", fontsize=12.5, fontweight="bold")
    axes[1, 1].set_ylabel("km")
    clean_axes(axes[1, 1], "y")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = FIG_DIR / "fig00_direct_kpi_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def build_fig_distribution(passengers: pd.DataFrame, thresholds: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.4), dpi=190)
    fig.suptitle("승객 시간 분포와 임계시간 도달률", fontsize=16, fontweight="bold")

    for policy in POLICY_ORDER:
        data = passengers[passengers["policy"] == policy]
        x, y = ecdf(data["wait_time_seconds"])
        axes[0, 0].plot(x, y, linewidth=2.2, label=policy, color=POLICY_COLORS[policy])
        x, y = ecdf(data["ride_time_seconds"])
        axes[0, 1].plot(x, y, linewidth=2.2, label=policy, color=POLICY_COLORS[policy])

    for ax, title, xlabel in [
        (axes[0, 0], "대기시간 ECDF", "대기시간 (초)"),
        (axes[0, 1], "탑승시간 ECDF", "탑승시간 (초)"),
    ]:
        for threshold in [600, 1200, 1800, 2400]:
            ax.axvline(threshold, color="#B0B8C2", linewidth=0.8, linestyle=":")
        ax.set_title(title, fontsize=12.5, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("누적 승객 비율")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value * 100:.0f}%"))
        clean_axes(ax, "both")
    axes[0, 1].legend(frameon=False, loc="lower right", fontsize=9)

    selected_thresholds = [600, 1200, 1800, 2400]
    width = 0.22
    x = np.arange(len(selected_thresholds))
    for idx, policy in enumerate(POLICY_ORDER):
        data = thresholds[thresholds["policy"] == policy].set_index("threshold_seconds")
        axes[1, 0].bar(
            x + (idx - 1) * width,
            data.loc[selected_thresholds, "wait_share"].to_numpy(),
            width=width,
            label=policy,
            color=POLICY_COLORS[policy],
        )
        axes[1, 1].bar(
            x + (idx - 1) * width,
            data.loc[selected_thresholds, "total_service_share"].to_numpy(),
            width=width,
            label=policy,
            color=POLICY_COLORS[policy],
        )
    for ax, title in [
        (axes[1, 0], "대기시간 임계 도달률"),
        (axes[1, 1], "총 서비스시간 임계 도달률"),
    ]:
        ax.set_xticks(x, [f"{t // 60}분" for t in selected_thresholds])
        ax.set_ylim(0, 1.03)
        ax.set_title(title, fontsize=12.5, fontweight="bold")
        ax.set_ylabel("승객 비율")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value * 100:.0f}%"))
        clean_axes(ax, "y")
        for container in ax.containers:
            ax.bar_label(container, labels=[f"{bar.get_height() * 100:.0f}%" for bar in container], fontsize=7, padding=2)
    axes[1, 1].legend(frameon=False, loc="upper left", fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = FIG_DIR / "fig01_passenger_time_distribution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def percentile_matrix(percentiles: pd.DataFrame, metric: str, qs: list[str]) -> pd.DataFrame:
    data = percentiles[percentiles["metric"] == metric].set_index("policy").loc[POLICY_ORDER]
    return data[qs].reset_index()


def grouped_percentile_bars(ax, matrix: pd.DataFrame, title: str, ylabel: str) -> None:
    qs = [col for col in matrix.columns if col != "policy"]
    x = np.arange(len(qs))
    width = 0.22
    for idx, policy in enumerate(POLICY_ORDER):
        values = matrix[matrix["policy"] == policy][qs].iloc[0].to_numpy(dtype=float)
        bars = ax.bar(
            x + (idx - 1) * width,
            values,
            width=width,
            label=policy,
            color=POLICY_COLORS[policy],
        )
        ax.bar_label(bars, labels=[f"{value:.0f}" for value in values], fontsize=7, padding=2, rotation=90)
    ax.set_xticks(x, [q.upper() for q in qs])
    ax.set_title(title, fontsize=12.5, fontweight="bold")
    ax.set_ylabel(ylabel)
    clean_axes(ax, "y")


def improvement_pct(onnx: float, baseline: float, direction: str) -> float:
    if baseline == 0.0 or math.isnan(baseline) or math.isnan(onnx):
        return float("nan")
    if direction == "higher":
        return (onnx - baseline) / baseline * 100.0
    return (baseline - onnx) / baseline * 100.0


def build_fig_tail_and_delta(summary: pd.DataFrame, percentiles: pd.DataFrame) -> Path:
    fig = plt.figure(figsize=(12.6, 10.3), dpi=190)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25], hspace=0.38, wspace=0.18)
    axes_top = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    axes_bottom = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    fig.suptitle("중앙값, 꼬리 위험, ONNX 기준선 대비 변화", fontsize=16, fontweight="bold")

    grouped_percentile_bars(
        axes_top[0],
        percentile_matrix(percentiles, "wait", ["p50", "p90", "p95", "max"]),
        "대기시간 tail profile",
        "초",
    )
    grouped_percentile_bars(
        axes_top[1],
        percentile_matrix(percentiles, "ride", ["p50", "p90", "p95", "max"]),
        "탑승시간 tail profile",
        "초",
    )
    axes_top[1].legend(frameon=False, loc="upper left", fontsize=9)

    metric_specs = [
        ("대기 평균", "wait_mean_seconds", "lower"),
        ("대기 중앙값", "wait_median_seconds", "lower"),
        ("대기 p90", "wait_p90", "lower"),
        ("대기 최대", "wait_max_seconds", "lower"),
        ("대기 Gini", "wait_gini", "lower"),
        ("탑승 평균", "ride_mean_seconds", "lower"),
        ("탑승 중앙값", "ride_p50", "lower"),
        ("탑승 p90", "ride_p90", "lower"),
        ("탑승 최대", "ride_max_seconds", "lower"),
        ("총 km", "total_distance_km", "lower"),
        ("에피소드 시간", "episode_time_minutes", "lower"),
        ("Loaded ratio", "loaded_distance_ratio", "higher"),
    ]
    pvt = percentiles.pivot(index="policy", columns="metric")
    enriched = summary.set_index("policy").copy()
    enriched["wait_p90"] = pvt[("p90", "wait")]
    enriched["ride_p50"] = pvt[("p50", "ride")]
    enriched["ride_p90"] = pvt[("p90", "ride")]
    enriched = enriched.loc[POLICY_ORDER]
    onnx = enriched.loc["ONNX"]

    for ax, baseline in zip(axes_bottom, ["FIFO", "Vanilla"]):
        base = enriched.loc[baseline]
        values = [
            improvement_pct(float(onnx[col]), float(base[col]), direction)
            for _label, col, direction in metric_specs
        ]
        labels = [label for label, _col, _direction in metric_specs]
        y = np.arange(len(labels))
        colors_for_values = [GREEN if value >= 0 else RUST for value in values]
        ax.barh(y, values, color=colors_for_values, edgecolor="#222222", linewidth=0.35)
        ax.axvline(0, color="#1D1D1D", linewidth=0.85)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_title(f"ONNX vs {baseline}", fontsize=12.5, fontweight="bold")
        ax.set_xlabel("개선율 (%), 음수는 악화")
        clean_axes(ax, "x")
        values_min = min(values + [0])
        values_max = max(values + [0])
        pad = max(4.0, (values_max - values_min) * 0.12)
        ax.set_xlim(values_min - pad, values_max + pad)
        for yi, value in zip(y, values):
            if abs(value) >= 18.0:
                xpos = value - 1.4 if value > 0 else value + 1.4
                ha = "right" if value > 0 else "left"
                label_color = "white"
            else:
                xpos = value + 1.3 if value >= 0 else value - 1.3
                ha = "left" if value >= 0 else "right"
                label_color = "#222222"
            ax.text(
                xpos,
                yi,
                f"{value:+.1f}%",
                ha=ha,
                va="center",
                fontsize=8,
                color=label_color,
            )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = FIG_DIR / "fig02_tail_delta_analysis.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def scatter_panel(
    ax,
    summary: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    x_pct: bool = False,
) -> None:
    data = summary.set_index("policy").loc[POLICY_ORDER]
    for policy in POLICY_ORDER:
        row = data.loc[policy]
        ax.scatter(
            float(row[x_col]),
            float(row[y_col]),
            s=150,
            color=POLICY_COLORS[policy],
            edgecolor="#222222",
            linewidth=0.5,
            zorder=3,
        )
        ax.annotate(policy, (float(row[x_col]), float(row[y_col])), xytext=(6, 5), textcoords="offset points", fontsize=8.5)
    ax.set_title(title, fontsize=12.5, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if x_pct:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value * 100:.0f}%"))
    ax.margins(x=0.12, y=0.15)
    clean_axes(ax, "both")


def build_fig_efficiency_and_route(summary: pd.DataFrame, load_frame: pd.DataFrame) -> Path:
    fig = plt.figure(figsize=(12.6, 9.0), dpi=190)
    gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.22)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    fig.suptitle("운영 효율, 서비스 품질, 경로 적재 구조", fontsize=16, fontweight="bold")

    scatter_panel(
        axes[0],
        summary,
        "distance_per_completed_km",
        "wait_mean_seconds",
        "완료 승객당 km와 평균 대기시간",
        "km/완료승객",
        "평균 대기시간 (초)",
    )
    scatter_panel(
        axes[1],
        summary,
        "loaded_distance_ratio",
        "ride_median_seconds",
        "Loaded ratio와 중앙 탑승시간",
        "Loaded distance ratio",
        "중앙 탑승시간 (초)",
        x_pct=True,
    )

    load_bins = ["0", "1", "2", "3", "4+"]
    bottom = np.zeros(len(POLICY_ORDER))
    x = np.arange(len(POLICY_ORDER))
    bin_colors = ["#DDE4EE", "#9BC5DE", "#6BA5C8", "#3F82B0", "#1F5F99"]
    for load_bin, color in zip(load_bins, bin_colors):
        values = (
            load_frame[load_frame["load_bin"] == load_bin]
            .set_index("policy")
            .loc[POLICY_ORDER, "distance_share"]
            .to_numpy(dtype=float)
        )
        axes[2].bar(x, values, bottom=bottom, label=f"{load_bin}명", color=color, edgecolor="white", linewidth=0.4)
        bottom += values
    axes[2].set_xticks(x, POLICY_ORDER)
    axes[2].set_ylim(0, 1)
    axes[2].set_title("탑승 인원별 주행거리 비중", fontsize=12.5, fontweight="bold")
    axes[2].set_ylabel("주행거리 비중")
    axes[2].yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value * 100:.0f}%"))
    axes[2].legend(frameon=False, ncols=5, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
    clean_axes(axes[2], "y")

    metrics = [
        ("total_distance_km", "총 km"),
        ("episode_time_minutes", "운행분"),
        ("distance_per_completed_km", "km/승객"),
        ("time_per_completed_minutes", "분/승객"),
    ]
    norm_rows = []
    data = summary.set_index("policy").loc[POLICY_ORDER]
    for col, label in metrics:
        values = data[col].astype(float)
        norm = values / values.max()
        for policy, value in norm.items():
            norm_rows.append({"policy": policy, "metric": label, "value": value})
    norm_df = pd.DataFrame(norm_rows)
    width = 0.2
    x = np.arange(len(metrics))
    for idx, policy in enumerate(POLICY_ORDER):
        values = norm_df[norm_df["policy"] == policy]["value"].to_numpy()
        axes[3].bar(x + (idx - 1) * width, values, width=width, color=POLICY_COLORS[policy], label=policy)
    axes[3].set_xticks(x, [label for _col, label in metrics])
    axes[3].set_title("운영 비용 정규화 비교", fontsize=12.5, fontweight="bold")
    axes[3].set_ylabel("각 지표 최대값 대비 비율")
    axes[3].legend(frameon=False, fontsize=8)
    clean_axes(axes[3], "y")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = FIG_DIR / "fig03_efficiency_route_structure.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def build_fig_scorecard(scorecard: pd.DataFrame, category_scores: pd.DataFrame) -> Path:
    metric_order = scorecard["metric"].drop_duplicates().tolist()
    matrix = scorecard.pivot(index="metric", columns="policy", values="score").loc[metric_order, POLICY_ORDER]

    fig = plt.figure(figsize=(12.2, 8.0), dpi=190)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.28)
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    fig.suptitle("정규화 KPI scorecard와 범주별 판단", fontsize=16, fontweight="bold")

    im = ax_heat.imshow(matrix.values, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax_heat.set_xticks(np.arange(len(POLICY_ORDER)), POLICY_ORDER)
    ax_heat.set_yticks(np.arange(len(metric_order)), metric_order)
    ax_heat.tick_params(axis="both", labelsize=8.5)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.values[i, j]
            ax_heat.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7.4,
                color="white" if value > 0.58 else "#222222",
            )
    ax_heat.set_title("개별 KPI 점수 (1에 가까울수록 우수)", fontsize=12.5, fontweight="bold")
    ax_heat.set_xticks(np.arange(-0.5, len(POLICY_ORDER), 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, len(metric_order), 1), minor=True)
    ax_heat.grid(which="minor", color="white", linewidth=0.7)
    ax_heat.tick_params(which="minor", bottom=False, left=False)
    fig.colorbar(im, ax=ax_heat, fraction=0.045, pad=0.02)

    category_order = [
        "Operating efficiency",
        "Passenger time",
        "Reliability",
        "Robustness",
        "Tail/fairness",
        "Overall_equal_weight",
    ]
    category_labels = {
        "Operating efficiency": "운영 효율",
        "Passenger time": "승객 시간",
        "Reliability": "신뢰성",
        "Robustness": "변동성",
        "Tail/fairness": "꼬리/공정성",
        "Overall_equal_weight": "종합",
    }
    categories = [category for category in category_order if category in category_scores.columns]
    y = np.arange(len(categories))
    height = 0.22
    for idx, policy in enumerate(POLICY_ORDER):
        values = category_scores[category_scores["policy"] == policy][categories].iloc[0].to_numpy(dtype=float)
        bars = ax_bar.barh(
            y + (idx - 1) * height,
            values,
            height=height,
            color=POLICY_COLORS[policy],
            label=policy,
        )
        for bar, value in zip(bars, values):
            ax_bar.text(
                min(value + 0.025, 1.04),
                bar.get_y() + bar.get_height() / 2.0,
                f"{value:.2f}",
                va="center",
                ha="left",
                fontsize=7.3,
                color="#222222",
            )
    ax_bar.set_yticks(y, [category_labels[category] for category in categories])
    ax_bar.set_xlim(0, 1.12)
    ax_bar.invert_yaxis()
    ax_bar.set_title("범주별 평균 점수", fontsize=12.5, fontweight="bold")
    ax_bar.set_xlabel("점수")
    ax_bar.legend(frameon=False, fontsize=8, loc="lower right")
    clean_axes(ax_bar, "x")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = FIG_DIR / "fig04_scorecard_category.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def build_figures(
    summary: pd.DataFrame,
    passengers: pd.DataFrame,
    percentiles: pd.DataFrame,
    thresholds: pd.DataFrame,
    load_frame: pd.DataFrame,
    scorecard: pd.DataFrame,
    category_scores: pd.DataFrame,
) -> dict[str, Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "direct": build_fig_direct_kpi_comparison(summary, passengers),
        "distribution": build_fig_distribution(passengers, thresholds),
        "tail_delta": build_fig_tail_and_delta(summary, percentiles),
        "efficiency": build_fig_efficiency_and_route(summary, load_frame),
        "scorecard": build_fig_scorecard(scorecard, category_scores),
    }


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(text), style)


def cell(text: object, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(str(text)), style)


def table_style(font: str, font_bold: str, size: float = 7.0) -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9EEF5")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111111")),
            ("FONTNAME", (0, 0), (-1, 0), font_bold),
            ("FONTNAME", (0, 1), (-1, -1), font),
            ("FONTSIZE", (0, 0), (-1, -1), size),
            ("LEADING", (0, 0), (-1, -1), size + 1.35),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#AAB2BE")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FC")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
    )


def styled_table(rows: list[list[object]], font: str, font_bold: str, widths=None, size: float = 7.0) -> LongTable:
    table = LongTable(rows, colWidths=widths, repeatRows=1)
    table.setStyle(table_style(font, font_bold, size))
    return table


def scaled_image(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as image:
        width, height = image.size
    ratio = min(max_width / width, max_height / height)
    return Image(str(path), width=width * ratio, height=height * ratio)


def source_rows(
    summary: pd.DataFrame,
    passengers: pd.DataFrame,
    episodes: pd.DataFrame,
    legs: pd.DataFrame,
    style: ParagraphStyle,
) -> list[list[object]]:
    rows = [[cell("항목", style), cell("내용", style)]]
    rows.extend(
        [
            [cell("기존 PDF", style), cell(SOURCE_PDF.as_posix(), style)],
            [cell("사용 테이블", style), cell(SOURCE_TABLE_DIR.as_posix(), style)],
            [
                cell("분석 단위", style),
                cell(
                    f"episode {len(episodes)}개, passenger {len(passengers)}개, route leg {len(legs)}개",
                    style,
                ),
            ],
            [
                cell("정책 반복 수", style),
                cell(
                    ", ".join(
                        f"{policy}: {int(summary[summary['policy'] == policy]['episodes'].iloc[0])}회"
                        for policy in POLICY_ORDER
                    ),
                    style,
                ),
            ],
            [cell("수요", style), cell("asset:drt_scenario_30_new_2000, 정책별 30명 완료 기준", style)],
            [cell("주의", style), cell("Vanilla는 1회, FIFO는 4회 반복이 동일값이라 통계적 유의성보다 시나리오 사례 비교로 해석", style)],
        ]
    )
    return rows


def kpi_rows(summary: pd.DataFrame, percentiles: pd.DataFrame, style: ParagraphStyle) -> list[list[object]]:
    wait_p = percentiles[percentiles["metric"] == "wait"].set_index("policy")
    ride_p = percentiles[percentiles["metric"] == "ride"].set_index("policy")
    total_p = percentiles[percentiles["metric"] == "total_service"].set_index("policy")
    rows = [
        [
            cell("정책", style),
            cell("반복", style),
            cell("SR", style),
            cell("대기 평균", style),
            cell("대기 p50", style),
            cell("대기 p90", style),
            cell("대기 max", style),
            cell("탑승 평균", style),
            cell("탑승 p50", style),
            cell("탑승 p90", style),
            cell("총서비스 p50", style),
            cell("총 km", style),
            cell("시간", style),
            cell("Loaded", style),
        ]
    ]
    for policy in POLICY_ORDER:
        row = summary[summary["policy"] == policy].iloc[0]
        rows.append(
            [
                cell(policy, style),
                cell(int(row["episodes"]), style),
                cell(pct(float(row["service_rate"])), style),
                cell(num(float(row["wait_mean_seconds"]), 1, " s"), style),
                cell(num(float(wait_p.loc[policy, "p50"]), 1, " s"), style),
                cell(num(float(wait_p.loc[policy, "p90"]), 1, " s"), style),
                cell(num(float(wait_p.loc[policy, "max"]), 1, " s"), style),
                cell(num(float(row["ride_mean_seconds"]), 1, " s"), style),
                cell(num(float(ride_p.loc[policy, "p50"]), 1, " s"), style),
                cell(num(float(ride_p.loc[policy, "p90"]), 1, " s"), style),
                cell(num(float(total_p.loc[policy, "p50"]), 1, " s"), style),
                cell(num(float(row["total_distance_km"]), 1, " km"), style),
                cell(num(float(row["episode_time_minutes"]), 1, " min"), style),
                cell(pct(float(row["loaded_distance_ratio"])), style),
            ]
        )
    return rows


def wins_rows(summary: pd.DataFrame, percentiles: pd.DataFrame, style: ParagraphStyle) -> list[list[object]]:
    wait = percentiles[percentiles["metric"] == "wait"].set_index("policy").loc[POLICY_ORDER]
    ride = percentiles[percentiles["metric"] == "ride"].set_index("policy").loc[POLICY_ORDER]
    total = percentiles[percentiles["metric"] == "total_service"].set_index("policy").loc[POLICY_ORDER]
    summary_idx = summary.set_index("policy").loc[POLICY_ORDER]
    rows = [[cell("판단 기준", style), cell("우수 정책", style), cell("수치", style), cell("해석", style)]]
    specs = [
        ("평균 대기시간", summary_idx["wait_mean_seconds"], "낮을수록", "평균 체감 서비스 품질"),
        ("중앙 대기시간", wait["p50"], "낮을수록", "일반 승객 경험"),
        ("p90 대기시간", wait["p90"], "낮을수록", "상위 10% 꼬리 위험"),
        ("최대 대기시간", wait["max"], "낮을수록", "최악 승객 경험"),
        ("평균 탑승시간", summary_idx["ride_mean_seconds"], "낮을수록", "우회/배차 품질"),
        ("중앙 탑승시간", ride["p50"], "낮을수록", "전형적 탑승 경험"),
        ("총 서비스 중앙값", total["p50"], "낮을수록", "대기+탑승 종합 경험"),
        ("총 주행거리", summary_idx["total_distance_km"], "낮을수록", "차량 운영 비용"),
        ("Loaded ratio", summary_idx["loaded_distance_ratio"], "높을수록", "공차 운행 억제"),
        ("대기 Gini", summary_idx["wait_gini"], "낮을수록", "대기시간 공정성"),
    ]
    for label, values, direction, meaning in specs:
        best = values.idxmax() if direction == "높을수록" else values.idxmin()
        value = float(values.loc[best])
        suffix = "%" if label == "Loaded ratio" else (" km" if label == "총 주행거리" else (" s" if "시간" in label or "중앙값" in label else ""))
        formatted = pct(value) if suffix == "%" else num(value, 1 if suffix != "" else 3, suffix)
        rows.append([cell(label, style), cell(best, style), cell(formatted, style), cell(f"{direction}. {meaning}", style)])
    return rows


def delta_summary(summary: pd.DataFrame) -> dict[str, float]:
    data = summary.set_index("policy")
    onnx = data.loc["ONNX"]
    fifo = data.loc["FIFO"]
    vanilla = data.loc["Vanilla"]
    return {
        "wait_mean_fifo": float(fifo["wait_mean_seconds"] - onnx["wait_mean_seconds"]),
        "wait_median_fifo": float(fifo["wait_median_seconds"] - onnx["wait_median_seconds"]),
        "ride_mean_fifo": float(fifo["ride_mean_seconds"] - onnx["ride_mean_seconds"]),
        "ride_median_fifo": float(fifo["ride_median_seconds"] - onnx["ride_median_seconds"]),
        "wait_mean_vanilla": float(vanilla["wait_mean_seconds"] - onnx["wait_mean_seconds"]),
        "wait_median_vanilla": float(vanilla["wait_median_seconds"] - onnx["wait_median_seconds"]),
        "distance_vanilla": float(vanilla["total_distance_km"] - onnx["total_distance_km"]),
        "distance_fifo": float(fifo["total_distance_km"] - onnx["total_distance_km"]),
        "wait_max_fifo_loss": float(onnx["wait_max_seconds"] - fifo["wait_max_seconds"]),
        "wait_gini_fifo_loss": float(onnx["wait_gini"] - fifo["wait_gini"]),
    }


def footer(font: str):
    def draw(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont(font, 8)
        canvas.setFillColor(colors.HexColor("#56606A"))
        canvas.drawString(1.25 * cm, 0.92 * cm, "DRT 30new professional analysis")
        canvas.drawRightString(A4[0] - 1.25 * cm, 0.92 * cm, f"{doc.page}")
        canvas.restoreState()

    return draw


def build_pdf(
    summary: pd.DataFrame,
    passengers: pd.DataFrame,
    episodes: pd.DataFrame,
    legs: pd.DataFrame,
    percentiles: pd.DataFrame,
    variability: pd.DataFrame,
    figures: dict[str, Path],
    font: str,
    font_bold: str,
) -> None:
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=1.15 * cm,
        rightMargin=1.15 * cm,
        topMargin=1.0 * cm,
        bottomMargin=1.35 * cm,
    )
    styles = {
        "title": ParagraphStyle("title", fontName=font_bold, fontSize=18, leading=22, alignment=TA_CENTER, spaceAfter=6),
        "subtitle": ParagraphStyle(
            "subtitle",
            fontName=font,
            fontSize=9.2,
            leading=12,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4C5560"),
            spaceAfter=8,
        ),
        "h1": ParagraphStyle("h1", fontName=font_bold, fontSize=12.5, leading=15.5, alignment=TA_LEFT, spaceBefore=8, spaceAfter=4),
        "h2": ParagraphStyle("h2", fontName=font_bold, fontSize=10.2, leading=13, alignment=TA_LEFT, spaceBefore=5, spaceAfter=3),
        "body": ParagraphStyle("body", fontName=font, fontSize=8.9, leading=12.4, alignment=TA_LEFT, textColor=colors.HexColor("#222222"), spaceAfter=5),
        "cell": ParagraphStyle("cell", fontName=font, fontSize=6.15, leading=7.7, alignment=TA_LEFT),
        "small": ParagraphStyle("small", fontName=font, fontSize=7.4, leading=10, alignment=TA_LEFT, textColor=colors.HexColor("#3D4752")),
    }
    cell_style = styles["cell"]
    data = summary.set_index("policy").loc[POLICY_ORDER]
    onnx = data.loc["ONNX"]
    fifo = data.loc["FIFO"]
    vanilla = data.loc["Vanilla"]
    deltas = delta_summary(summary)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    story: list = []
    story.append(para("DRT 30new 정책 비교 심화 분석 보고서", styles["title"]))
    story.append(para("ONNX 추론 정책, FIFO 기준선, Vanilla 순차 정책의 승객 시간 품질과 운영 효율 비교", styles["subtitle"]))
    story.append(para(f"생성 시각: {generated} / 기존 PDF와 중간 CSV 테이블 재분석", styles["subtitle"]))

    story.append(para("분석 근거", styles["h1"]))
    story.append(styled_table(source_rows(summary, passengers, episodes, legs, cell_style), font, font_bold, widths=[3.2 * cm, 14.2 * cm], size=6.25))

    story.append(para("핵심 결론", styles["h1"]))
    bullets = [
        "세 정책 모두 서비스율 100%라서 완료 여부보다 승객 시간 품질, 꼬리 위험, 차량 운영 효율이 실제 비교 기준이다.",
        f"ONNX는 평균 대기시간 {onnx['wait_mean_seconds']:.1f}초, 중앙 대기시간 {onnx['wait_median_seconds']:.1f}초로 가장 낮다. FIFO 대비 평균 {deltas['wait_mean_fifo']:.1f}초, 중앙값 {deltas['wait_median_fifo']:.1f}초를 줄였다.",
        f"ONNX는 평균 탑승시간 {onnx['ride_mean_seconds']:.1f}초, 중앙 탑승시간 {onnx['ride_median_seconds']:.1f}초로 가장 낮다. FIFO 대비 평균 {deltas['ride_mean_fifo']:.1f}초, 중앙값 {deltas['ride_median_fifo']:.1f}초 감소다.",
        f"운영 효율도 ONNX가 총 {onnx['total_distance_km']:.2f} km, {onnx['episode_time_minutes']:.2f}분으로 FIFO와 거의 같거나 조금 낮다. Vanilla 대비 주행거리는 {deltas['distance_vanilla']:.2f} km 감소한다.",
        f"단, FIFO는 최대 대기시간이 ONNX보다 {deltas['wait_max_fifo_loss']:.1f}초 낮고 대기 Gini도 {deltas['wait_gini_fifo_loss']:.3f} 낮다. 따라서 ONNX의 주장은 평균/중앙 서비스 품질 개선으로 제한하는 것이 안전하다.",
    ]
    for bullet in bullets:
        story.append(para(f"- {bullet}", styles["body"]))

    story.append(para("KPI 요약표", styles["h1"]))
    story.append(
        styled_table(
            kpi_rows(summary, percentiles, cell_style),
            font,
            font_bold,
            widths=[
                1.25 * cm,
                0.75 * cm,
                1.0 * cm,
                1.4 * cm,
                1.35 * cm,
                1.35 * cm,
                1.35 * cm,
                1.4 * cm,
                1.35 * cm,
                1.35 * cm,
                1.55 * cm,
                1.25 * cm,
                1.25 * cm,
                1.2 * cm,
            ],
            size=5.35,
        )
    )

    story.append(PageBreak())
    story.append(para("대기/탑승/주행거리 직접 비교", styles["h1"]))
    story.append(
        para(
            "아래 그림은 정책별 대기시간과 탑승시간을 각각 상자수염그림으로 분리하고, 평균/중앙값 및 총 주행거리를 같은 페이지에서 직접 비교한 것이다.",
            styles["small"],
        )
    )
    story.append(scaled_image(figures["direct"], 18.0 * cm, 16.4 * cm))

    story.append(PageBreak())
    story.append(para("정책별 우수 지표와 해석", styles["h1"]))
    story.append(styled_table(wins_rows(summary, percentiles, cell_style), font, font_bold, widths=[3.1 * cm, 2.2 * cm, 2.6 * cm, 9.1 * cm], size=6.3))
    story.append(Spacer(1, 0.25 * cm))
    story.append(para("승객 시간 분포", styles["h1"]))
    story.append(scaled_image(figures["distribution"], 18.0 * cm, 16.0 * cm))

    story.append(PageBreak())
    story.append(para("꼬리 위험과 기준선 대비 ONNX 변화", styles["h1"]))
    story.append(scaled_image(figures["tail_delta"], 18.0 * cm, 22.0 * cm))

    story.append(PageBreak())
    story.append(para("운영 효율과 경로 적재 구조", styles["h1"]))
    story.append(scaled_image(figures["efficiency"], 18.0 * cm, 18.0 * cm))
    story.append(Spacer(1, 0.15 * cm))
    story.append(
        para(
            "해석: ONNX와 FIFO는 주행거리와 에피소드 시간이 거의 같은 수준이다. 차이는 승객 시간 분포에서 발생하며, ONNX는 중앙 승객 경험을 더 낮추지만 FIFO는 최대 대기시간과 공정성 지표에서 더 보수적이다.",
            styles["small"],
        )
    )

    story.append(PageBreak())
    story.append(para("종합 Scorecard", styles["h1"]))
    story.append(scaled_image(figures["scorecard"], 18.0 * cm, 14.2 * cm))
    story.append(para("반복 실행 변동성", styles["h1"]))
    rows = [[cell("정책", cell_style), cell("반복", cell_style), cell("대기평균 SD", cell_style), cell("탑승평균 SD", cell_style), cell("시간 SD", cell_style), cell("거리 SD", cell_style)]]
    for _, row in variability.iterrows():
        rows.append(
            [
                cell(row["policy"], cell_style),
                cell(int(row["episodes"]), cell_style),
                cell(num(float(row["wait_mean_sd"]), 1, " s"), cell_style),
                cell(num(float(row["ride_mean_sd"]), 1, " s"), cell_style),
                cell(num(float(row["episode_time_sd"]), 1, " s"), cell_style),
                cell(num(float(row["distance_sd"]), 2, " km"), cell_style),
            ]
        )
    story.append(styled_table(rows, font, font_bold, widths=[2.2 * cm, 1.5 * cm, 3.0 * cm, 3.0 * cm, 3.0 * cm, 3.0 * cm], size=6.4))

    story.append(PageBreak())
    story.append(para("논문/발표용 권장 서술", styles["h1"]))
    story.append(
        para(
            "30명 new 수요 시나리오에서 ONNX 정책은 Vanilla 및 FIFO와 동일하게 모든 승객을 처리하면서 평균 및 중앙 대기시간을 가장 낮게 유지하였다. 특히 FIFO 대비 중앙 대기시간과 중앙 탑승시간이 각각 275.9초, 64.7초 감소하여, 학습 기반 정책이 단순 선착순 기준선보다 일반 승객의 체감 시간을 더 적극적으로 줄였음을 보인다.",
            styles["body"],
        )
    )
    story.append(
        para(
            "다만 FIFO는 최대 대기시간과 대기 Gini에서 ONNX보다 안정적인 값을 보인다. 따라서 본 결과는 ONNX가 모든 KPI에서 전면적으로 우월하다는 결론이 아니라, 평균/중앙 승객 시간과 차량 운영 효율을 개선하는 대신 일부 꼬리 위험과 공정성 지표에서는 trade-off가 남는 결과로 해석하는 것이 타당하다.",
            styles["body"],
        )
    )
    story.append(para("후속 비교 권장", styles["h1"]))
    followups = [
        "동일 KPI 체계를 여러 random seed와 14/18/22/40/50 수요량에 반복 적용한다.",
        "평균값 옆에 p90, p95, 최대값, Gini를 함께 제시해 평균 개선이 꼬리 위험 개선으로 오해되지 않게 한다.",
        "Vanilla 1회, FIFO 반복 동일값이라는 현재 데이터 한계를 명시하고, 논문 본문에서는 사례 비교 또는 ablation 성격으로 표현한다.",
        "정책 선택 기준을 보상값보다 서비스율, 평균/중앙 대기시간, 꼬리 위험, 운행 비용의 다목적 KPI로 설명한다.",
    ]
    for item in followups:
        story.append(para(f"- {item}", styles["body"]))

    doc.build(story, onFirstPage=footer(font), onLaterPages=footer(font))


def write_markdown_summary(summary: pd.DataFrame) -> Path:
    data = summary.set_index("policy").loc[POLICY_ORDER]
    onnx = data.loc["ONNX"]
    fifo = data.loc["FIFO"]
    vanilla = data.loc["Vanilla"]
    deltas = delta_summary(summary)
    text = f"""# DRT 30new Professional Analysis Summary

- Output PDF: `{PDF_PATH.as_posix()}`
- Source PDF: `{SOURCE_PDF.as_posix()}`
- Source tables: `{SOURCE_TABLE_DIR.as_posix()}`

## Main finding

ONNX is the strongest policy for mean/median passenger time in the 30new scenario.
It reaches 100% service rate like FIFO and Vanilla, with mean wait {onnx['wait_mean_seconds']:.1f}s and median wait {onnx['wait_median_seconds']:.1f}s.
Compared with FIFO, ONNX reduces mean wait by {deltas['wait_mean_fifo']:.1f}s and median wait by {deltas['wait_median_fifo']:.1f}s.

## Caveat

FIFO remains stronger on worst-case waiting and wait inequality.
ONNX maximum wait is {onnx['wait_max_seconds']:.1f}s, while FIFO is {fifo['wait_max_seconds']:.1f}s.
This should be framed as a passenger-time improvement with tail/fairness trade-off, not universal dominance.

## Efficiency

ONNX total distance is {onnx['total_distance_km']:.2f} km, FIFO is {fifo['total_distance_km']:.2f} km, and Vanilla is {vanilla['total_distance_km']:.2f} km.
"""
    path = OUTPUT_ROOT / "DRT_compare_30Pas_new_professional_summary.md"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    font, font_bold = register_fonts()
    setup_matplotlib()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary, passengers, episodes, legs = load_inputs()
    percentiles, thresholds, load_frame, stop_frame, variability = add_percentile_tables(passengers, episodes, legs)
    scorecard, category_scores = normalized_scorecard(summary, percentiles, episodes)
    save_analysis_tables(percentiles, thresholds, load_frame, stop_frame, variability, scorecard, category_scores)
    figures = build_figures(summary, passengers, percentiles, thresholds, load_frame, scorecard, category_scores)
    build_pdf(summary, passengers, episodes, legs, percentiles, variability, figures, font, font_bold)
    markdown_path = write_markdown_summary(summary)
    print(PDF_PATH)
    print(OUTPUT_ROOT)
    print(markdown_path)


if __name__ == "__main__":
    main()
