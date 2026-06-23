# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path("output/drt_compare_30Pas_new/tables")
OUT = Path("output/ppt_drt_30passenger_compare")
POLICY_ORDER = ["ONNX", "FIFO", "Vanilla"]
COLORS = {"ONNX": "#1F5F99", "FIFO": "#6F7782", "Vanilla": "#B8BDC4"}


def configure_fonts() -> None:
    regular = Path(r"C:\Windows\Fonts\malgun.ttf")
    bold = Path(r"C:\Windows\Fonts\malgunbd.ttf")
    for path in [regular, bold]:
        if path.exists():
            fm.fontManager.addfont(str(path))
    if regular.exists():
        plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def load_summary() -> pd.DataFrame:
    summary = pd.read_csv(BASE / "policy_analysis_summary_ko.csv")
    return summary.set_index("policy").loc[POLICY_ORDER].reset_index()


def reduction_pct(onnx_value: float, baseline_value: float) -> float:
    return (baseline_value - onnx_value) / baseline_value * 100.0


def change_note(summary: pd.DataFrame, metric: str) -> str:
    data = summary.set_index("policy")
    onnx = float(data.loc["ONNX", metric])
    fifo = float(data.loc["FIFO", metric])
    vanilla = float(data.loc["Vanilla", metric])
    fifo_pct = reduction_pct(onnx, fifo)
    vanilla_pct = reduction_pct(onnx, vanilla)
    return f"ONNX: FIFO 대비 {fifo_pct:.1f}% 감소, Vanilla 대비 {vanilla_pct:.1f}% 감소"


def draw_single_metric_bar(
    summary: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    filename: str,
    suffix: str,
) -> None:
    values = summary.set_index("policy").loc[POLICY_ORDER, metric].astype(float)
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=180)
    bars = ax.bar(POLICY_ORDER, values.to_numpy(), color=[COLORS[p] for p in POLICY_ORDER], width=0.56)
    max_value = float(values.max())

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max_value * 0.035,
            f"{value:.1f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=14)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#D6DCE5", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max_value * 1.24)
    ax.text(
        0.98,
        -0.16,
        change_note(summary, metric),
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=10,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / filename, bbox_inches="tight")
    plt.close(fig)


def draw_summary_grid(summary: pd.DataFrame) -> None:
    specs = [
        ("평균 대기시간", "wait_mean_seconds", "초", "평균 대기시간 (초)"),
        ("평균 탑승시간", "ride_mean_seconds", "초", "평균 탑승시간 (초)"),
        ("총 주행거리", "total_distance_km", "km", "총 주행거리 (km)"),
        ("종료시간", "episode_time_minutes", "분", "종료시간 (분)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.4), dpi=180)
    fig.suptitle("30명 시나리오 알고리즘 비교", fontsize=21, fontweight="bold", y=0.98)
    data = summary.set_index("policy")

    for ax, (title, metric, suffix, ylabel) in zip(axes.ravel(), specs):
        values = data.loc[POLICY_ORDER, metric].astype(float)
        bars = ax.bar(POLICY_ORDER, values.to_numpy(), color=[COLORS[p] for p in POLICY_ORDER], width=0.58)
        max_value = float(values.max())
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max_value * 0.035,
                f"{value:.1f}{suffix}",
                ha="center",
                va="bottom",
                fontsize=10.5,
                fontweight="bold",
            )
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#D6DCE5", linewidth=0.75)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, max_value * 1.24)
        ax.text(
            0.98,
            -0.22,
            change_note(summary, metric),
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=8.5,
            color="#4B5563",
        )

    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(OUT / "drt_30pas_summary_kpis.png", bbox_inches="tight")
    plt.close(fig)


def write_values(summary: pd.DataFrame) -> None:
    data = summary.set_index("policy")
    rows = []
    for label, metric, unit in [
        ("평균 대기시간", "wait_mean_seconds", "s"),
        ("평균 탑승시간", "ride_mean_seconds", "s"),
        ("총 주행거리", "total_distance_km", "km"),
        ("종료시간", "episode_time_minutes", "min"),
    ]:
        onnx = float(data.loc["ONNX", metric])
        fifo = float(data.loc["FIFO", metric])
        vanilla = float(data.loc["Vanilla", metric])
        rows.append(
            {
                "metric": label,
                "unit": unit,
                "ONNX": onnx,
                "FIFO": fifo,
                "Vanilla": vanilla,
                "ONNX_vs_FIFO_reduction_pct": reduction_pct(onnx, fifo),
                "ONNX_vs_Vanilla_reduction_pct": reduction_pct(onnx, vanilla),
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "drt_30pas_kpi_values.csv", index=False, encoding="utf-8-sig")

    text = f"""# 30명 시나리오 PPT 핵심 값

- 평균 대기시간: ONNX {data.loc['ONNX', 'wait_mean_seconds']:.1f}s, FIFO {data.loc['FIFO', 'wait_mean_seconds']:.1f}s, Vanilla {data.loc['Vanilla', 'wait_mean_seconds']:.1f}s
- 평균 탑승시간: ONNX {data.loc['ONNX', 'ride_mean_seconds']:.1f}s, FIFO {data.loc['FIFO', 'ride_mean_seconds']:.1f}s, Vanilla {data.loc['Vanilla', 'ride_mean_seconds']:.1f}s
- 총 주행거리: ONNX {data.loc['ONNX', 'total_distance_km']:.2f}km, FIFO {data.loc['FIFO', 'total_distance_km']:.2f}km, Vanilla {data.loc['Vanilla', 'total_distance_km']:.2f}km
- 종료시간: ONNX {data.loc['ONNX', 'episode_time_minutes']:.2f}분, FIFO {data.loc['FIFO', 'episode_time_minutes']:.2f}분, Vanilla {data.loc['Vanilla', 'episode_time_minutes']:.2f}분

한 줄 결론:
30명 시나리오에서 ONNX는 FIFO 및 고정노선 대비 평균 대기시간, 평균 탑승시간, 주행거리, 종료시간을 모두 낮췄다.
"""
    (OUT / "drt_30pas_ppt_text.md").write_text(text, encoding="utf-8")


def main() -> None:
    configure_fonts()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = load_summary()
    write_values(summary)
    draw_summary_grid(summary)
    draw_single_metric_bar(
        summary,
        "wait_mean_seconds",
        "30명 시나리오 평균 대기시간",
        "평균 대기시간 (초)",
        "drt_30pas_average_wait.png",
        "s",
    )
    draw_single_metric_bar(
        summary,
        "ride_mean_seconds",
        "30명 시나리오 평균 탑승시간",
        "평균 탑승시간 (초)",
        "drt_30pas_average_ride.png",
        "s",
    )
    draw_single_metric_bar(
        summary,
        "total_distance_km",
        "30명 시나리오 총 주행거리",
        "총 주행거리 (km)",
        "drt_30pas_total_distance.png",
        "km",
    )
    draw_single_metric_bar(
        summary,
        "episode_time_minutes",
        "30명 시나리오 종료시간",
        "종료시간 (분)",
        "drt_30pas_end_time.png",
        "분",
    )
    print(OUT)


if __name__ == "__main__":
    main()
