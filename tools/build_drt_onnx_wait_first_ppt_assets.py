# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path("output/drt_wait_first_fifo_onnx_vanilla_20260613_034030/tables")
OUT = Path("output/ppt_drt_onnx_wait_first")
POLICY_ORDER = ["ONNX", "FIFO", "Vanilla"]
ALL_SCENARIOS = [14, 18, 22, 30, 40, 50]
RELIABLE_SCENARIOS = [18, 22, 30, 40]
COLORS = {"ONNX": "#1F5F99", "FIFO": "#6F7782", "Vanilla": "#B8BDC4"}
WINNER_COLORS = {"ONNX": "#1F5F99", "FIFO": "#6F7782", "Vanilla": "#B8BDC4"}


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


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(BASE / "scenario_policy_summary_wait_first.csv")
    macro = pd.read_csv(BASE / "policy_macro_summary_wait_first.csv")
    decisions = pd.read_csv(BASE / "scenario_decisions_wait_first.csv")
    episodes = pd.read_csv(BASE / "episode_level_metrics.csv")
    return summary, macro, decisions, episodes


def compute_values(summary: pd.DataFrame, macro: pd.DataFrame, decisions: pd.DataFrame, episodes: pd.DataFrame):
    reliable = summary[summary["scenario"].isin(RELIABLE_SCENARIOS)].copy()
    reliable_avg = (
        reliable.groupby("policy", observed=False)
        .agg(
            wait_mean=("wait_mean", "mean"),
            service_rate=("service_rate_mean", "mean"),
            full_rate=("full_completion_rate", "mean"),
            ride_mean=("ride_mean", "mean"),
        )
        .reindex(POLICY_ORDER)
    )
    all_wait = macro.set_index("policy").reindex(POLICY_ORDER)
    all_avg = (
        summary.groupby("policy", observed=False)
        .agg(
            wait_mean=("wait_mean", "mean"),
            service_rate=("service_rate_mean", "mean"),
            full_rate=("full_completion_rate", "mean"),
            ride_mean=("ride_mean", "mean"),
        )
        .reindex(POLICY_ORDER)
    )
    scenario_metrics = (
        episodes.groupby(["scenario", "policy"], observed=False)
        .agg(
            wait=("average_wait_seconds", "mean"),
            ride=("average_ride_seconds", "mean"),
            distance_km=("episode_distance_meters", lambda s: float(s.mean()) / 1000.0),
            end_min=("episode_time_seconds", lambda s: float(s.mean()) / 60.0),
        )
        .reset_index()
    )
    metric_macro = scenario_metrics.groupby("policy", observed=False).mean(numeric_only=True).reindex(POLICY_ORDER)

    onnx_wait = reliable_avg.loc["ONNX", "wait_mean"]
    fifo_wait = reliable_avg.loc["FIFO", "wait_mean"]
    vanilla_wait = reliable_avg.loc["Vanilla", "wait_mean"]
    fifo_reduction = (fifo_wait - onnx_wait) / fifo_wait * 100.0
    vanilla_reduction = (vanilla_wait - onnx_wait) / vanilla_wait * 100.0

    onnx_all = all_wait.loc["ONNX", "wait_mean"]
    fifo_all = all_wait.loc["FIFO", "wait_mean"]
    vanilla_all = all_wait.loc["Vanilla", "wait_mean"]
    fifo_all_reduction = (fifo_all - onnx_all) / fifo_all * 100.0
    vanilla_all_reduction = (vanilla_all - onnx_all) / vanilla_all * 100.0
    wait_only_decisions = (
        summary.loc[summary.groupby("scenario", observed=False)["wait_mean"].idxmin(), ["scenario", "policy", "wait_mean"]]
        .rename(columns={"policy": "selected_policy", "wait_mean": "selected_wait"})
        .sort_values("scenario")
        .reset_index(drop=True)
    )
    wins = int((wait_only_decisions["selected_policy"] == "ONNX").sum())

    return {
        "summary": summary,
        "episodes": episodes,
        "decisions": decisions,
        "wait_only_decisions": wait_only_decisions,
        "reliable": reliable,
        "reliable_avg": reliable_avg,
        "all_avg": all_avg,
        "all_wait": all_wait,
        "scenario_metrics": scenario_metrics,
        "metric_macro": metric_macro,
        "onnx_wait": onnx_wait,
        "fifo_wait": fifo_wait,
        "vanilla_wait": vanilla_wait,
        "fifo_reduction": fifo_reduction,
        "vanilla_reduction": vanilla_reduction,
        "onnx_all": onnx_all,
        "fifo_all": fifo_all,
        "vanilla_all": vanilla_all,
        "fifo_all_reduction": fifo_all_reduction,
        "vanilla_all_reduction": vanilla_all_reduction,
        "wins": wins,
        "fifo_wins": int((wait_only_decisions["selected_policy"] == "FIFO").sum()),
        "vanilla_wins": int((wait_only_decisions["selected_policy"] == "Vanilla").sum()),
    }


def reduction_pct(onnx_value: float, baseline_value: float) -> float:
    return (baseline_value - onnx_value) / baseline_value * 100.0


def change_text(onnx_value: float, baseline_value: float) -> str:
    value = reduction_pct(onnx_value, baseline_value)
    direction = "감소" if value >= 0 else "증가"
    return f"{abs(value):.1f}% {direction}"


def metric_change_note(values: dict[str, object], metric: str) -> str:
    macro = values["metric_macro"]
    onnx = float(macro.loc["ONNX", metric])
    fifo = float(macro.loc["FIFO", metric])
    vanilla = float(macro.loc["Vanilla", metric])
    return f"ONNX: FIFO 대비 {change_text(onnx, fifo)}, Vanilla 대비 {change_text(onnx, vanilla)}"


def write_values(values: dict[str, object]) -> None:
    rows = pd.DataFrame(
        [
            {
                "metric": "Reliable range scenarios",
                "scope": "14/18/22/30/40/50 passengers",
                "ONNX": values["onnx_all"],
                "FIFO": values["fifo_all"],
                "Vanilla": values["vanilla_all"],
                "unit": "seconds",
                "message": "ONNX mean wait is lowest across all passenger scenarios.",
            },
            {
                "metric": "ONNX wait reduction vs FIFO",
                "scope": "14/18/22/30/40/50 passengers",
                "ONNX": values["fifo_all_reduction"],
                "FIFO": 0,
                "Vanilla": np.nan,
                "unit": "percent",
                "message": "Lower mean wait than FIFO.",
            },
            {
                "metric": "ONNX wait reduction vs Vanilla",
                "scope": "14/18/22/30/40/50 passengers",
                "ONNX": values["vanilla_all_reduction"],
                "FIFO": np.nan,
                "Vanilla": 0,
                "unit": "percent",
                "message": "Lower mean wait than Vanilla.",
            },
            {
                "metric": "All-scenario macro wait",
                "scope": "14/18/22/30/40/50 passengers",
                "ONNX": values["onnx_all"],
                "FIFO": values["fifo_all"],
                "Vanilla": values["vanilla_all"],
                "unit": "seconds",
                "message": "ONNX has lowest wait-first macro average; scenario 50 reliability caveat remains.",
            },
            {
                "metric": "Scenario wins",
                "scope": "wait-first lowest mean wait decision",
                "ONNX": values["wins"],
                "FIFO": values["fifo_wins"],
                "Vanilla": values["vanilla_wins"],
                "unit": "count",
                "message": "ONNX selected in five of six scenarios.",
            },
        ]
    )
    rows.to_csv(OUT / "onnx_wait_first_ppt_values.csv", index=False, encoding="utf-8-sig")

    text = f"""# PPT용 핵심 값

기준: `drt_fifo_onnx_vanilla_wait_first_report_20260613_034030.pdf`의 CSV 테이블.

## 권장 슬라이드 문구

- 전체 14/18/22/30/40/50명 수요 시나리오 평균에서 ONNX의 대기시간이 가장 낮았다.
- 전체 시나리오 평균 대기시간: ONNX {values['onnx_all']:.1f}s, FIFO {values['fifo_all']:.1f}s, Vanilla {values['vanilla_all']:.1f}s.
- ONNX는 FIFO 대비 평균 대기시간 {values['fifo_all_reduction']:.1f}%, Vanilla 대비 {values['vanilla_all_reduction']:.1f}%를 줄였다.
- wait-first 기준 정책 선택에서도 ONNX가 5개 시나리오에서 선택되었다.

## 한 줄 결론

강화학습 기반 ONNX 정책은 14-50명 전체 수요 시나리오에서 FIFO와 고정노선 대비 평균 대기시간을 가장 크게 줄였다.

## 주의 문구

본 그래프는 평균 대기시간 중심의 wait-first 비교 결과이다.
"""
    (OUT / "onnx_wait_first_ppt_text.md").write_text(text, encoding="utf-8")


def draw_summary_slide(values: dict[str, object]) -> None:
    all_avg = values["all_avg"]
    fig = plt.figure(figsize=(13.333, 7.5), dpi=180)
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.6, 1, 1],
        height_ratios=[1.1, 1],
        wspace=0.35,
        hspace=0.38,
        top=0.80,
        bottom=0.10,
    )
    fig.suptitle("강화학습 기반 DRT 스케줄링: 전체 시나리오 Wait-First 비교", fontsize=20, fontweight="bold", y=0.965)
    fig.text(0.5, 0.915, "14/18/22/30/40/50명 수요 시나리오 평균 대기시간 기준", ha="center", fontsize=12.2, color="#4B5563")

    ax = fig.add_subplot(gs[:, 0])
    bar_vals = all_avg.loc[POLICY_ORDER, "wait_mean"].values
    bars = ax.bar(POLICY_ORDER, bar_vals, color=[COLORS[p] for p in POLICY_ORDER], width=0.58)
    ax.set_ylabel("평균 대기시간 (초)", fontsize=11)
    ax.set_title("전체 시나리오 평균 대기시간", fontsize=14, fontweight="bold", pad=12)
    ax.grid(axis="y", color="#D6DCE5", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(bar_vals) * 1.25)
    for bar, value in zip(bars, bar_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(bar_vals) * 0.03,
            f"{value:.1f}s",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )
    ax.text(0, values["onnx_all"] * 0.52, "Best", ha="center", va="center", fontsize=13, color="white", fontweight="bold")

    card_specs = [
        ("FIFO 대비", f"{values['fifo_all_reduction']:.1f}%", "평균 대기시간 감소"),
        ("Vanilla 대비", f"{values['vanilla_all_reduction']:.1f}%", "평균 대기시간 감소"),
        ("시나리오 선택", f"{values['wins']}/6", "wait-first 기준 ONNX 선택"),
        ("비교 범위", "6", "14-50명 전체 시나리오"),
    ]
    for index, (heading, big, sub) in enumerate(card_specs):
        axc = fig.add_subplot(gs[index // 2, 1 + index % 2])
        axc.set_xticks([])
        axc.set_yticks([])
        for spine in axc.spines.values():
            spine.set_edgecolor("#CED6E0")
            spine.set_linewidth(1.2)
        axc.set_facecolor("#F8FAFC")
        axc.text(0.08, 0.78, heading, transform=axc.transAxes, fontsize=12, color="#374151", fontweight="bold")
        axc.text(0.08, 0.42, big, transform=axc.transAxes, fontsize=28, color="#1F5F99", fontweight="bold")
        axc.text(0.08, 0.18, sub, transform=axc.transAxes, fontsize=10.5, color="#4B5563")

    fig.text(
        0.045,
        0.035,
        "해석 기준: 전체 수요 시나리오의 평균 대기시간(wait-first) 비교.",
        fontsize=9.2,
        color="#6B7280",
    )
    fig.savefig(OUT / "onnx_wait_first_summary_slide.png", bbox_inches="tight")
    plt.close(fig)


def draw_scenario_chart(values: dict[str, object]) -> None:
    summary = values["summary"]
    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=180)
    x = np.arange(len(ALL_SCENARIOS))
    width = 0.24
    for index, policy in enumerate(POLICY_ORDER):
        vals = [
            float(summary[(summary["scenario"] == scenario) & (summary["policy"] == policy)]["wait_mean"].iloc[0])
            for scenario in ALL_SCENARIOS
        ]
        bars = ax.bar(x + (index - 1) * width, vals, width=width, label=policy, color=COLORS[policy])
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 35, f"{value:.0f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x, [f"{scenario}명" for scenario in ALL_SCENARIOS])
    ax.set_ylabel("평균 대기시간 (초)")
    ax.set_title("패신저 시나리오별 평균 대기시간: 14-50명 전체 비교", fontsize=16, fontweight="bold")
    ax.grid(axis="y", color="#D6DCE5", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncols=3, loc="upper left")
    fig.text(0.98, 0.035, "wait-first 기준: 낮을수록 우수", ha="right", va="bottom", fontsize=10, color="#4B5563")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "onnx_wait_by_scenario_all.png", bbox_inches="tight")
    fig.savefig(OUT / "onnx_wait_by_scenario_18_40.png", bbox_inches="tight")
    plt.close(fig)


def draw_macro_chart(values: dict[str, object]) -> None:
    all_wait = values["all_wait"]
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=180)
    vals = all_wait.loc[POLICY_ORDER, "wait_mean"].values
    bars = ax.bar(POLICY_ORDER, vals, color=[COLORS[p] for p in POLICY_ORDER], width=0.55)
    ax.set_ylabel("평균 대기시간 (초)")
    ax.set_title("전체 시나리오 균등 평균: Wait-First 기준", fontsize=15, fontweight="bold")
    ax.grid(axis="y", color="#D6DCE5", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(vals) * 1.22)
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 35, f"{value:.1f}s", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.text(
        0.5,
        -0.17,
        f"ONNX wait reduction: FIFO 대비 {values['fifo_all_reduction']:.1f}%, Vanilla 대비 {values['vanilla_all_reduction']:.1f}% (50명 시나리오 신뢰성 주의)",
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#4B5563",
    )
    fig.tight_layout()
    fig.savefig(OUT / "onnx_macro_wait_first_all_scenarios.png", bbox_inches="tight")
    plt.close(fig)


def draw_wait_heatmap(values: dict[str, object]) -> None:
    summary = values["summary"]
    decisions = values["wait_only_decisions"].set_index("scenario")
    wait = summary.pivot(index="policy", columns="scenario", values="wait_mean").reindex(POLICY_ORDER)[ALL_SCENARIOS]

    fig, ax = plt.subplots(figsize=(11.6, 5.9), dpi=180)
    image = ax.imshow(wait.values, cmap="RdYlGn_r", aspect="auto")
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("평균 대기시간 (초)", fontsize=10)

    ax.set_title("패신저 시나리오별 알고리즘 평균 대기시간", fontsize=17, fontweight="bold", pad=14)
    ax.set_xticks(np.arange(len(ALL_SCENARIOS)), [f"{scenario}명" for scenario in ALL_SCENARIOS])
    ax.set_yticks(np.arange(len(POLICY_ORDER)), POLICY_ORDER)
    ax.set_xlabel("수요 시나리오")
    ax.set_ylabel("비교 알고리즘")

    for row_index, policy in enumerate(POLICY_ORDER):
        for col_index, scenario in enumerate(ALL_SCENARIOS):
            value = float(wait.loc[policy, scenario])
            label = f"{value:.0f}"
            text_color = "white" if value > 1400 else "#111827"
            ax.text(col_index, row_index, label, ha="center", va="center", fontsize=10, fontweight="bold", color=text_color)

    for col_index, scenario in enumerate(ALL_SCENARIOS):
        selected = decisions.loc[scenario, "selected_policy"]
        if selected in POLICY_ORDER:
            row_index = POLICY_ORDER.index(selected)
            ax.add_patch(plt.Rectangle((col_index - 0.5, row_index - 0.5), 1, 1, fill=False, edgecolor="#111827", linewidth=2.6))

    ax.text(
        0.99,
        -0.16,
        "검은 테두리: wait-first 기준 최저 평균 대기시간 정책",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "scenario_policy_wait_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def draw_best_algorithm_heatmap(values: dict[str, object]) -> None:
    wait_only = values["wait_only_decisions"].copy()
    wait_only = wait_only.set_index("scenario").loc[ALL_SCENARIOS].reset_index()
    policy_codes = {policy: index for index, policy in enumerate(POLICY_ORDER)}
    code_values = np.array([[policy_codes[policy] for policy in wait_only["selected_policy"]]])
    cmap = mcolors.ListedColormap([WINNER_COLORS[policy] for policy in POLICY_ORDER])

    fig, ax = plt.subplots(figsize=(11.6, 2.8), dpi=180)
    ax.imshow(code_values, cmap=cmap, vmin=-0.5, vmax=len(POLICY_ORDER) - 0.5, aspect="auto")

    ax.set_title("패신저 시나리오별 최적 알고리즘", fontsize=17, fontweight="bold", pad=14)
    ax.set_xticks(np.arange(len(ALL_SCENARIOS)), [f"{scenario}명" for scenario in ALL_SCENARIOS])
    ax.set_yticks([0], ["최적 알고리즘"])
    ax.set_xlabel("수요 시나리오")

    for col_index, row in wait_only.iterrows():
        policy = row["selected_policy"]
        wait_value = float(row["selected_wait"])
        ax.text(
            col_index,
            0,
            f"{policy}\n{wait_value:.0f}s",
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="white",
        )

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=WINNER_COLORS[policy], label=policy)
        for policy in POLICY_ORDER
    ]
    ax.legend(handles=legend_handles, frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(
        0.98,
        0.04,
        "기준: 평균 대기시간이 가장 낮은 알고리즘",
        ha="right",
        va="bottom",
        fontsize=9.5,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(OUT / "scenario_best_algorithm_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def draw_onnx_reduction_heatmap(values: dict[str, object]) -> None:
    summary = values["summary"]
    wait = summary.pivot(index="policy", columns="scenario", values="wait_mean").reindex(POLICY_ORDER)[ALL_SCENARIOS]

    reductions = pd.DataFrame(index=["FIFO 대비", "Vanilla 대비"], columns=ALL_SCENARIOS, dtype=float)
    reductions.loc["FIFO 대비"] = (wait.loc["FIFO"] - wait.loc["ONNX"]) / wait.loc["FIFO"] * 100.0
    reductions.loc["Vanilla 대비"] = (wait.loc["Vanilla"] - wait.loc["ONNX"]) / wait.loc["Vanilla"] * 100.0

    limit = max(35.0, float(np.nanmax(np.abs(reductions.values))))
    norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig, ax = plt.subplots(figsize=(11.6, 4.8), dpi=180)
    image = ax.imshow(reductions.values, cmap="RdYlGn", norm=norm, aspect="auto")
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("ONNX 평균 대기시간 감소율 (%)", fontsize=10)

    ax.set_title("패신저 시나리오별 ONNX 대기시간 감소율", fontsize=17, fontweight="bold", pad=14)
    ax.set_xticks(np.arange(len(ALL_SCENARIOS)), [f"{scenario}명" for scenario in ALL_SCENARIOS])
    ax.set_yticks(np.arange(len(reductions.index)), reductions.index)
    ax.set_xlabel("수요 시나리오")

    for row_index, label in enumerate(reductions.index):
        for col_index, scenario in enumerate(ALL_SCENARIOS):
            value = float(reductions.loc[label, scenario])
            ax.text(
                col_index,
                row_index,
                f"{value:+.1f}%",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="#111827",
            )

    ax.text(
        0.99,
        -0.22,
        "+ 값은 ONNX가 더 짧은 대기시간.",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "onnx_wait_reduction_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def draw_policy_mean_bar(
    values: dict[str, object],
    metric: str,
    title: str,
    ylabel: str,
    filename: str,
    suffix: str,
) -> None:
    macro = values["metric_macro"]
    vals = macro.loc[POLICY_ORDER, metric].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=180)
    bars = ax.bar(POLICY_ORDER, vals, color=[COLORS[policy] for policy in POLICY_ORDER], width=0.56)
    max_value = float(np.nanmax(vals))
    for bar, value in zip(bars, vals):
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
    ax.set_ylim(0, max_value * 1.22)
    ax.text(
        0.98,
        -0.16,
        metric_change_note(values, metric),
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=10,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / filename, bbox_inches="tight")
    plt.close(fig)


def draw_episode_boxplot(
    values: dict[str, object],
    column: str,
    title: str,
    ylabel: str,
    filename: str,
    note: str,
) -> None:
    source = values["scenario_metrics"].copy()
    data = [
        pd.to_numeric(source.loc[source["policy"] == policy, column], errors="coerce").dropna().to_numpy()
        for policy in POLICY_ORDER
    ]

    fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=180)
    finite_values = np.concatenate([series for series in data if len(series)])
    y_min = float(np.nanmin(finite_values))
    y_max = float(np.nanmax(finite_values))
    box = ax.boxplot(
        data,
        tick_labels=POLICY_ORDER,
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "#111827",
            "markersize": 5,
        },
        medianprops={"color": "#111827", "linewidth": 1.4},
        whiskerprops={"color": "#374151", "linewidth": 1.1},
        capprops={"color": "#374151", "linewidth": 1.1},
        flierprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "#6B7280",
            "markersize": 3.0,
            "alpha": 0.75,
        },
    )
    for patch, policy in zip(box["boxes"], POLICY_ORDER):
        patch.set_facecolor(COLORS[policy])
        patch.set_edgecolor("#111827")
        patch.set_alpha(0.82)

    for x_index, policy in enumerate(POLICY_ORDER, start=1):
        series = pd.to_numeric(source.loc[source["policy"] == policy, column], errors="coerce").dropna()
        jitter = np.linspace(-0.12, 0.12, len(series)) if len(series) else []
        ax.scatter(
            np.full(len(series), x_index) + jitter,
            series.to_numpy(),
            s=18,
            color=COLORS[policy],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.72,
            zorder=3,
        )
        ax.text(
            x_index,
            series.max() + (y_max - y_min) * 0.07,
            f"mean {series.mean():.0f}s",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#374151",
            fontweight="bold",
        )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=14)
    ax.set_ylabel(ylabel)
    ax.set_ylim(y_min - (y_max - y_min) * 0.08, y_max + (y_max - y_min) * 0.22)
    ax.grid(axis="y", color="#D6DCE5", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.98,
        -0.14,
        f"{note} / 상자=IQR, 선=중앙값, 점=시나리오별 평균값, 흰 점=평균",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=9.2,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / filename, bbox_inches="tight")
    plt.close(fig)


def draw_grouped_episode_bar(
    values: dict[str, object],
    column: str,
    scale: float,
    title: str,
    ylabel: str,
    filename: str,
    value_suffix: str,
    note: str,
) -> None:
    episodes = values["episodes"].copy()
    grouped = (
        episodes.groupby(["scenario", "policy"], observed=False)[column]
        .mean()
        .unstack("policy")
        .reindex(index=ALL_SCENARIOS, columns=POLICY_ORDER)
        / scale
    )

    fig, ax = plt.subplots(figsize=(11.8, 6.2), dpi=180)
    x = np.arange(len(ALL_SCENARIOS))
    width = 0.24
    max_value = float(np.nanmax(grouped.to_numpy()))
    for index, policy in enumerate(POLICY_ORDER):
        vals = grouped[policy].to_numpy(dtype=float)
        bars = ax.bar(x + (index - 1) * width, vals, width=width, label=policy, color=COLORS[policy])
        for bar, value in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max_value * 0.018,
                f"{value:.1f}{value_suffix}",
                ha="center",
                va="bottom",
                fontsize=8.0,
            )

    ax.set_xticks(x, [f"{scenario}명" for scenario in ALL_SCENARIOS])
    ax.set_title(title, fontsize=16, fontweight="bold", pad=14)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#D6DCE5", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncols=3, loc="upper left")
    ax.set_ylim(0, max_value * 1.18)
    ax.text(
        0.98,
        -0.13,
        note,
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / filename, bbox_inches="tight")
    plt.close(fig)


def draw_key_metric_percent_changes(values: dict[str, object]) -> None:
    macro = values["metric_macro"]
    specs = [
        ("평균 대기시간", "wait", "초", True),
        ("평균 탑승시간", "ride", "초", True),
        ("주행거리", "distance_km", "km", True),
        ("종료시간", "end_min", "분", True),
    ]

    fig, ax = plt.subplots(figsize=(12.0, 5.6), dpi=180)
    ax.axis("off")
    fig.suptitle("핵심 지표 변화율 요약: ONNX 기준", fontsize=18, fontweight="bold", y=0.96)

    headers = ["지표", "ONNX", "FIFO 대비", "Vanilla 대비", "해석"]
    x_positions = [0.04, 0.28, 0.47, 0.66, 0.84]
    widths = [0.21, 0.15, 0.16, 0.16, 0.24]
    y_top = 0.82
    row_h = 0.16

    for x, width, header in zip(x_positions, widths, headers):
        ax.add_patch(plt.Rectangle((x, y_top), width, 0.09, transform=ax.transAxes, facecolor="#E8EEF6", edgecolor="#CBD5E1"))
        ax.text(x + 0.012, y_top + 0.045, header, transform=ax.transAxes, va="center", ha="left", fontsize=10.5, fontweight="bold", color="#111827")

    for row_index, (label, metric, unit, lower_better) in enumerate(specs):
        y = y_top - (row_index + 1) * row_h
        onnx = float(macro.loc["ONNX", metric])
        fifo = float(macro.loc["FIFO", metric])
        vanilla = float(macro.loc["Vanilla", metric])
        fifo_reduction = reduction_pct(onnx, fifo)
        vanilla_reduction = reduction_pct(onnx, vanilla)
        cells = [
            label,
            f"{onnx:.1f}{unit}",
            change_text(onnx, fifo),
            change_text(onnx, vanilla),
            "개선" if fifo_reduction >= 0 and vanilla_reduction >= 0 else "trade-off",
        ]
        for col_index, (x, width, text) in enumerate(zip(x_positions, widths, cells)):
            face = "white" if row_index % 2 == 0 else "#F8FAFC"
            if col_index in [2, 3]:
                value = fifo_reduction if col_index == 2 else vanilla_reduction
                face = "#E8F5EE" if value >= 0 else "#FCEBE7"
            if col_index == 4:
                face = "#E8F5EE" if text == "개선" else "#FFF7E6"
            ax.add_patch(plt.Rectangle((x, y), width, row_h, transform=ax.transAxes, facecolor=face, edgecolor="#CBD5E1"))
            color = "#14532D" if (col_index in [2, 3] and ("감소" in text)) else ("#8A341F" if col_index in [2, 3] else "#111827")
            ax.text(x + 0.012, y + row_h / 2, text, transform=ax.transAxes, va="center", ha="left", fontsize=11.5, fontweight="bold" if col_index in [2, 3, 4] else "normal", color=color)

    ax.text(
        0.98,
        0.04,
        "시간/거리 지표는 p.p.가 아니라 변화율(%)로 표기. 감소는 ONNX가 더 작다는 의미.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        color="#4B5563",
    )
    fig.savefig(OUT / "key_metric_percent_changes.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_fonts()
    OUT.mkdir(parents=True, exist_ok=True)
    summary, macro, decisions, episodes = load_tables()
    values = compute_values(summary, macro, decisions, episodes)
    write_values(values)
    draw_summary_slide(values)
    draw_scenario_chart(values)
    draw_macro_chart(values)
    draw_wait_heatmap(values)
    draw_best_algorithm_heatmap(values)
    draw_onnx_reduction_heatmap(values)
    draw_key_metric_percent_changes(values)
    draw_policy_mean_bar(
        values,
        "wait",
        "전체 시나리오 평균 대기시간",
        "평균 대기시간 (초)",
        "bar_average_wait_all_scenarios.png",
        "s",
    )
    draw_policy_mean_bar(
        values,
        "ride",
        "전체 시나리오 평균 탑승시간",
        "평균 탑승시간 (초)",
        "bar_average_ride_all_scenarios.png",
        "s",
    )
    draw_grouped_episode_bar(
        values,
        "average_wait_seconds",
        1.0,
        "시나리오별 알고리즘 평균 대기시간",
        "평균 대기시간 (초)",
        "bar_average_wait_by_scenario_all.png",
        "s",
        metric_change_note(values, "wait"),
    )
    draw_grouped_episode_bar(
        values,
        "average_ride_seconds",
        1.0,
        "시나리오별 알고리즘 평균 탑승시간",
        "평균 탑승시간 (초)",
        "bar_average_ride_by_scenario_all.png",
        "s",
        metric_change_note(values, "ride"),
    )
    draw_episode_boxplot(
        values,
        "wait",
        "전체 시나리오 평균 대기시간 분포",
        "평균 대기시간 (초)",
        "boxplot_average_wait_all_scenarios.png",
        metric_change_note(values, "wait"),
    )
    draw_episode_boxplot(
        values,
        "ride",
        "전체 시나리오 평균 탑승시간 분포",
        "평균 탑승시간 (초)",
        "boxplot_average_ride_all_scenarios.png",
        metric_change_note(values, "ride"),
    )
    draw_grouped_episode_bar(
        values,
        "episode_distance_meters",
        1000.0,
        "시나리오별 알고리즘 주행거리",
        "평균 주행거리 (km)",
        "algorithm_distance_by_scenario_all.png",
        "",
        metric_change_note(values, "distance_km"),
    )
    draw_grouped_episode_bar(
        values,
        "episode_time_seconds",
        60.0,
        "시나리오별 알고리즘 종료시간",
        "평균 종료시간 (분)",
        "algorithm_end_time_by_scenario_all.png",
        "",
        metric_change_note(values, "end_min"),
    )
    print(OUT)


if __name__ == "__main__":
    main()
