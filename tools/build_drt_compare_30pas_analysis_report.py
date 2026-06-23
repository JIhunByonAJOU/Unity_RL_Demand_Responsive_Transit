from __future__ import annotations

import math
import sys
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
from reportlab.platypus import (
    Image,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    TableStyle,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_drt_compare_30pas_kpi_report import (
    OUTPUT_ROOT,
    PDF_PATH,
    POLICY_COLORS,
    POLICY_ORDER,
    build_delta_table,
    load_data,
    register_fonts,
)


FIG_DIR = OUTPUT_ROOT / "figures"
TABLE_DIR = OUTPUT_ROOT / "tables"
TITLE = "DRT 30new 정책 비교 평가"
SUBTITLE = "30명 new 수요 시나리오: ONNX 추론 정책, FIFO 기준선, Vanilla 순차 정책 비교"


def fmt(value: object, digits: int = 1) -> str:
    try:
        value_f = float(value)
    except Exception:
        return "-"
    if math.isnan(value_f):
        return "-"
    return f"{value_f:,.{digits}f}"


def fmt_pct(value: object, digits: int = 1) -> str:
    try:
        value_f = float(value)
    except Exception:
        return "-"
    if math.isnan(value_f):
        return "-"
    return f"{value_f * 100.0:,.{digits}f}%"


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(text), style)


def cell(text: object, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(str(text)), style)


def select(summary: pd.DataFrame, policy: str) -> pd.Series:
    return summary[summary["policy"] == policy].iloc[0]


def pct_change(new: float, base: float) -> float:
    if base == 0.0 or math.isnan(base) or math.isnan(new):
        return float("nan")
    return (new - base) / base * 100.0


def best_policy(summary: pd.DataFrame, col: str, lower_better: bool = True) -> str:
    values = [(policy, float(select(summary, policy)[col])) for policy in POLICY_ORDER]
    values = [(policy, value) for policy, value in values if not math.isnan(value)]
    if not values:
        return "-"
    return min(values, key=lambda item: item[1])[0] if lower_better else max(values, key=lambda item: item[1])[0]


def save_analysis_tables(summary: pd.DataFrame, deltas: pd.DataFrame, scores: pd.DataFrame) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLE_DIR / "policy_analysis_summary_ko.csv", index=False, encoding="utf-8-sig")
    deltas.to_csv(TABLE_DIR / "onnx_analysis_delta_vs_baselines.csv", index=False, encoding="utf-8-sig")
    scores.to_csv(TABLE_DIR / "policy_normalized_scorecard.csv", index=False, encoding="utf-8-sig")


def clean_axes(ax, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color="#D9D9D9", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(axis="both", labelsize=9)


def policy_values(summary: pd.DataFrame, col: str) -> list[float]:
    return [float(select(summary, policy)[col]) for policy in POLICY_ORDER]


def line_panel(ax, summary: pd.DataFrame, cols: list[tuple[str, str]], title: str, ylabel: str) -> None:
    x = np.arange(len(POLICY_ORDER))
    markers = ["o", "s", "^"]
    colors_for_lines = ["#0B5CAD", "#4F81BD", "#7A7A7A"]
    for idx, (col, label) in enumerate(cols):
        values = policy_values(summary, col)
        ax.plot(
            x,
            values,
            marker=markers[idx % len(markers)],
            linewidth=1.9,
            markersize=6,
            label=label,
            color=colors_for_lines[idx % len(colors_for_lines)],
        )
    ax.set_xticks(x, POLICY_ORDER)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=10)
    clean_axes(ax, grid_axis="y")
    ax.legend(frameon=True, fontsize=8, loc="best")


def bar_panel(ax, summary: pd.DataFrame, col: str, title: str, ylabel: str, pct: bool = False) -> None:
    values = policy_values(summary, col)
    x = np.arange(len(POLICY_ORDER))
    ax.bar(x, values, width=0.58, color=[POLICY_COLORS[p] for p in POLICY_ORDER], edgecolor="#222222", linewidth=0.45)
    ax.set_xticks(x, POLICY_ORDER)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=10)
    clean_axes(ax, grid_axis="y")
    for xi, value in zip(x, values):
        label = f"{value * 100.0:.1f}%" if pct else f"{value:.1f}"
        ax.text(xi, value, label, ha="center", va="bottom", fontsize=8)


def box_panel(ax, passengers: pd.DataFrame, value_col: str, title: str) -> None:
    data = []
    for policy in POLICY_ORDER:
        series = passengers.loc[passengers["policy"] == policy, value_col].dropna().astype(float)
        data.append(series.to_numpy())

    box = ax.boxplot(
        data,
        tick_labels=POLICY_ORDER,
        widths=0.54,
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "#FFFFFF",
            "markeredgecolor": "#111111",
            "markersize": 4.5,
        },
        medianprops={"color": "#111111", "linewidth": 1.35},
        whiskerprops={"color": "#333333", "linewidth": 1.0},
        capprops={"color": "#333333", "linewidth": 1.0},
        flierprops={
            "marker": "o",
            "markerfacecolor": "#FFFFFF",
            "markeredgecolor": "#777777",
            "markersize": 2.4,
            "alpha": 0.65,
        },
    )
    for patch, policy in zip(box["boxes"], POLICY_ORDER):
        patch.set_facecolor(POLICY_COLORS[policy])
        patch.set_edgecolor("#222222")
        patch.set_alpha(0.72)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("초", fontsize=10)
    clean_axes(ax, grid_axis="y")
    ax.text(
        0.98,
        0.96,
        "상자=IQR, 선=중앙값, 원=평균",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.7,
        color="#333333",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#FFFFFF", "edgecolor": "#BBBBBB", "linewidth": 0.5},
    )


def build_fig1_service_quality(summary: pd.DataFrame, passengers: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.0), dpi=190)
    fig.suptitle("그림 1. 승객 시간 분포 상자수염그림 및 서비스 진단", fontsize=17, fontweight="bold")
    box_panel(axes[0, 0], passengers, "wait_time_seconds", "대기 시간 분포")
    completed_passengers = passengers[passengers["status"] == "Completed"]
    box_panel(axes[0, 1], completed_passengers, "ride_time_seconds", "탑승 시간 분포")
    bar_panel(axes[1, 0], summary, "service_rate", "서비스 완료율", "비율", pct=True)
    line_panel(
        axes[1, 1],
        summary,
        [("wait_gini", "대기 Gini"), ("ride_gini", "탑승 Gini")],
        "시간 불평등",
        "Gini",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = FIG_DIR / "fig_paper_01_service_quality.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def delta_percent(summary: pd.DataFrame, baseline_policy: str, col: str, lower_better: bool = True) -> float:
    onnx = float(select(summary, "ONNX")[col])
    base = float(select(summary, baseline_policy)[col])
    if base == 0.0:
        return float("nan")
    if lower_better:
        return (onnx - base) / base * 100.0
    return (onnx - base) / base * 100.0


def build_fig2_deltas(summary: pd.DataFrame) -> Path:
    metrics = [
        ("대기 평균", "wait_mean_seconds"),
        ("대기 중앙값", "wait_median_seconds"),
        ("탑승 평균", "ride_mean_seconds"),
        ("탑승 중앙값", "ride_median_seconds"),
        ("총 주행거리", "total_distance_km"),
        ("에피소드 시간", "episode_time_minutes"),
        ("km/완료승객", "distance_per_completed_km"),
        ("분/완료승객", "time_per_completed_minutes"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), dpi=190, sharey=True)
    fig.suptitle("그림 2. ONNX의 기준선 대비 변화율", fontsize=17, fontweight="bold")
    for ax, baseline in zip(axes, ["FIFO", "Vanilla"]):
        vals = [delta_percent(summary, baseline, col, lower_better=True) for _, col in metrics]
        y = np.arange(len(metrics))
        colors = ["#0B5CAD" if v <= 0 else "#B85C38" for v in vals]
        ax.barh(y, vals, color=colors, edgecolor="#222222", linewidth=0.45)
        ax.axvline(0, color="#111111", linewidth=0.8)
        ax.set_yticks(y, [label for label, _ in metrics])
        ax.invert_yaxis()
        ax.set_title(f"ONNX vs {baseline}", fontsize=13, fontweight="bold")
        ax.set_xlabel("% 변화; 음수는 개선", fontsize=10)
        clean_axes(ax, grid_axis="x")
        for yi, value in zip(y, vals):
            if abs(value) >= 6.0:
                ax.text(value / 2.0, yi, f"{value:+.1f}%", va="center", ha="center", fontsize=8, color="white")
            else:
                ax.text(
                    value + (0.8 if value >= 0 else -0.8),
                    yi,
                    f"{value:+.1f}%",
                    va="center",
                    ha="left" if value >= 0 else "right",
                    fontsize=8,
                    color="#222222",
                )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    path = FIG_DIR / "fig_paper_02_onnx_deltas.png"
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
    y_pct: bool = False,
) -> None:
    label_offsets = {
        "Vanilla": (5, 5, "left"),
        "FIFO": (-18, 12, "right"),
        "ONNX": (18, -14, "left"),
    }
    for policy in POLICY_ORDER:
        row = select(summary, policy)
        x = float(row[x_col])
        y = float(row[y_col])
        size = 120.0 + float(row["episode_time_minutes"]) * 1.8
        ax.scatter(
            x,
            y,
            s=size,
            color=POLICY_COLORS[policy],
            edgecolor="#222222",
            linewidth=0.55,
            alpha=0.92,
        )
        dx, dy, ha = label_offsets.get(policy, (5, 5, "left"))
        ax.annotate(
            policy,
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.3,
            fontweight="bold",
            ha=ha,
        )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if x_pct:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value * 100:.0f}%"))
    if y_pct:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value * 100:.0f}%"))
    ax.margins(x=0.08, y=0.12)
    clean_axes(ax, grid_axis="both")


def build_fig3_efficiency(summary: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.0), dpi=190)
    fig.suptitle("그림 3. 운영 효율과 서비스 품질의 trade-off", fontsize=17, fontweight="bold")
    scatter_panel(
        axes[0, 0],
        summary,
        "total_distance_km",
        "wait_mean_seconds",
        "총 주행거리와 평균 대기시간",
        "총 주행거리 (km)",
        "평균 대기시간 (초)",
    )
    scatter_panel(
        axes[0, 1],
        summary,
        "episode_time_minutes",
        "wait_median_seconds",
        "에피소드 시간과 중앙 대기시간",
        "에피소드 시간 (분)",
        "중앙 대기시간 (초)",
    )
    scatter_panel(
        axes[1, 0],
        summary,
        "distance_per_completed_km",
        "time_per_completed_minutes",
        "완료 승객당 운행 비용",
        "km/완료 승객",
        "분/완료 승객",
    )
    scatter_panel(
        axes[1, 1],
        summary,
        "loaded_distance_ratio",
        "ride_median_seconds",
        "적재 주행 비율과 중앙 탑승시간",
        "Loaded distance ratio",
        "중앙 탑승시간 (초)",
        x_pct=True,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = FIG_DIR / "fig_paper_03_efficiency.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def normalized_scores(summary: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("서비스율", "service_rate", "higher"),
        ("대기 평균", "wait_mean_seconds", "lower"),
        ("대기 중앙값", "wait_median_seconds", "lower"),
        ("대기 최대", "wait_max_seconds", "lower"),
        ("대기 Gini", "wait_gini", "lower"),
        ("탑승 평균", "ride_mean_seconds", "lower"),
        ("탑승 중앙값", "ride_median_seconds", "lower"),
        ("탑승 최대", "ride_max_seconds", "lower"),
        ("탑승 Gini", "ride_gini", "lower"),
        ("총 주행거리", "total_distance_km", "lower"),
        ("에피소드 시간", "episode_time_minutes", "lower"),
        ("km/완료승객", "distance_per_completed_km", "lower"),
        ("분/완료승객", "time_per_completed_minutes", "lower"),
        ("Loaded 비율", "loaded_distance_ratio", "higher"),
        ("방문 스탑 수", "visited_stop_count", "lower"),
    ]
    rows = []
    for label, col, direction in specs:
        values = np.array(policy_values(summary, col), dtype=float)
        vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
        if math.isclose(vmin, vmax):
            scores = np.repeat(0.5, len(values))
        elif direction == "higher":
            scores = (values - vmin) / (vmax - vmin)
        else:
            scores = (vmax - values) / (vmax - vmin)
        for policy, raw, score in zip(POLICY_ORDER, values, scores):
            rows.append({"metric": label, "policy": policy, "raw_value": raw, "score": float(score)})
    return pd.DataFrame(rows)


def build_fig4_scorecard(scores: pd.DataFrame) -> Path:
    metric_order = scores["metric"].drop_duplicates().tolist()
    matrix = scores.pivot(index="metric", columns="policy", values="score").loc[metric_order, POLICY_ORDER]
    fig, ax = plt.subplots(figsize=(9.2, 6.0), dpi=190)
    im = ax.imshow(matrix.values, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_title("그림 4. 정규화 KPI scorecard", fontsize=17, fontweight="bold", pad=16)
    ax.set_xticks(np.arange(len(POLICY_ORDER)), POLICY_ORDER)
    ax.set_yticks(np.arange(len(metric_order)), metric_order)
    ax.tick_params(axis="both", labelsize=9)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.values[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color="white" if value > 0.58 else "#222222")
    ax.set_xticks(np.arange(-0.5, len(POLICY_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(metric_order), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    path = FIG_DIR / "fig_paper_04_scorecard.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def build_figures(summary: pd.DataFrame, passengers: pd.DataFrame, scores: pd.DataFrame) -> dict[str, Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "fig1": build_fig1_service_quality(summary, passengers),
        "fig2": build_fig2_deltas(summary),
        "fig3": build_fig3_efficiency(summary),
        "fig4": build_fig4_scorecard(scores),
    }


def table_style(font: str, font_bold: str, size: float = 6.4) -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9EDF3")),
            ("FONTNAME", (0, 0), (-1, 0), font_bold),
            ("FONTNAME", (0, 1), (-1, -1), font),
            ("FONTSIZE", (0, 0), (-1, -1), size),
            ("LEADING", (0, 0), (-1, -1), size + 1.4),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#A8A8A8")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F7F7")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
    )


def styled_table(rows, font: str, font_bold: str, widths=None, size: float = 6.4) -> LongTable:
    table = LongTable(rows, colWidths=widths, repeatRows=1)
    table.setStyle(table_style(font, font_bold, size=size))
    return table


def image(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as img:
        width_px, height_px = img.size
    ratio = min(max_width / width_px, max_height / height_px)
    return Image(str(path), width=width_px * ratio, height=height_px * ratio)


def source_rows(summary: pd.DataFrame, style: ParagraphStyle):
    rows = [[cell("정책", style), cell("선택된 CSV 폴더", style), cell("반복", style), cell("완료", style), cell("수요 소스", style)]]
    for _, row in summary.iterrows():
        rows.append(
            [
                cell(str(row["policy"]), style),
                cell(Path(str(row["run_folder"])).as_posix(), style),
                cell(int(row["episodes"]), style),
                cell(f"{int(row['finished_by_completion'])}/{int(row['episodes'])}", style),
                cell("asset:drt_scenario_30_new_2000", style),
            ]
        )
    return rows


def summary_rows(summary: pd.DataFrame, style: ParagraphStyle):
    rows = [
        [
            cell("정책", style),
            cell("SR", style),
            cell("대기 평균", style),
            cell("대기 중앙", style),
            cell("대기 최대", style),
            cell("대기 Gini", style),
            cell("탑승 평균", style),
            cell("탑승 중앙", style),
            cell("탑승 최대", style),
            cell("탑승 Gini", style),
            cell("총 km", style),
            cell("시간", style),
            cell("km/완료", style),
            cell("분/완료", style),
            cell("Loaded", style),
            cell("스탑", style),
        ]
    ]
    for _, row in summary.iterrows():
        rows.append(
            [
                cell(row["policy"], style),
                cell(fmt_pct(row["service_rate"]), style),
                cell(f"{fmt(row['wait_mean_seconds'])} s", style),
                cell(f"{fmt(row['wait_median_seconds'])} s", style),
                cell(f"{fmt(row['wait_max_seconds'])} s", style),
                cell(fmt(row["wait_gini"], 3), style),
                cell(f"{fmt(row['ride_mean_seconds'])} s", style),
                cell(f"{fmt(row['ride_median_seconds'])} s", style),
                cell(f"{fmt(row['ride_max_seconds'])} s", style),
                cell(fmt(row["ride_gini"], 3), style),
                cell(f"{fmt(row['total_distance_km'])} km", style),
                cell(f"{fmt(row['episode_time_minutes'])} min", style),
                cell(fmt(row["distance_per_completed_km"], 2), style),
                cell(fmt(row["time_per_completed_minutes"], 2), style),
                cell(fmt_pct(row["loaded_distance_ratio"]), style),
                cell(fmt(row["visited_stop_count"], 0), style),
            ]
        )
    return rows


def footer(font: str):
    def draw(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont(font, 8)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(1.25 * cm, 1.0 * cm, "DRT 30new 정책 비교")
        canvas.drawRightString(A4[0] - 1.25 * cm, 1.0 * cm, f"{doc.page}쪽")
        canvas.restoreState()

    return draw


def build_pdf(summary: pd.DataFrame, figures: dict[str, Path], font: str, font_bold: str) -> None:
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=1.1 * cm,
        leftMargin=1.1 * cm,
        topMargin=1.05 * cm,
        bottomMargin=1.35 * cm,
    )
    styles = {
        "title": ParagraphStyle("title", fontName=font_bold, fontSize=18, leading=22, alignment=TA_CENTER, spaceAfter=7),
        "subtitle": ParagraphStyle("subtitle", fontName=font, fontSize=9.2, leading=12.5, alignment=TA_CENTER, textColor=colors.HexColor("#444444"), spaceAfter=8),
        "h1": ParagraphStyle("h1", fontName=font_bold, fontSize=12.4, leading=15, alignment=TA_LEFT, spaceBefore=7, spaceAfter=4),
        "body": ParagraphStyle("body", fontName=font, fontSize=8.9, leading=12.3, alignment=TA_LEFT, textColor=colors.HexColor("#222222"), spaceAfter=5),
        "cell": ParagraphStyle("cell", fontName=font, fontSize=5.75, leading=7.1, alignment=TA_LEFT),
    }
    cstyle = styles["cell"]

    vanilla = select(summary, "Vanilla")
    fifo = select(summary, "FIFO")
    onnx = select(summary, "ONNX")
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    wait_mean_gain_fifo = float(fifo["wait_mean_seconds"] - onnx["wait_mean_seconds"])
    wait_mean_gain_vanilla = float(vanilla["wait_mean_seconds"] - onnx["wait_mean_seconds"])
    wait_median_gain_fifo = float(fifo["wait_median_seconds"] - onnx["wait_median_seconds"])
    wait_median_gain_vanilla = float(vanilla["wait_median_seconds"] - onnx["wait_median_seconds"])
    fifo_dist_delta_pct = pct_change(float(onnx["total_distance_km"]), float(fifo["total_distance_km"]))
    vanilla_dist_delta_pct = pct_change(float(onnx["total_distance_km"]), float(vanilla["total_distance_km"]))
    ride_mean_best = best_policy(summary, "ride_mean_seconds", lower_better=True)
    ride_median_best = best_policy(summary, "ride_median_seconds", lower_better=True)
    episode_best = best_policy(summary, "episode_time_minutes", lower_better=True)
    loaded_best = best_policy(summary, "loaded_distance_ratio", lower_better=False)

    if ride_mean_best == "ONNX":
        ride_sentence = (
            f"탑승 시간에서는 ONNX가 평균 {onnx['ride_mean_seconds']:.1f}초와 중앙값 "
            f"{onnx['ride_median_seconds']:.1f}초로 가장 낮다."
        )
    else:
        ride_sentence = (
            f"탑승 평균은 {ride_mean_best}가 가장 낮지만, ONNX는 {onnx['ride_mean_seconds']:.1f}초로 거의 같은 수준이며 "
            f"탑승 중앙값은 {onnx['ride_median_seconds']:.1f}초로 가장 낮다."
        )

    if abs(fifo_dist_delta_pct) <= 2.0:
        distance_sentence = (
            f"ONNX의 총 주행거리는 FIFO와 거의 같은 수준이다({onnx['total_distance_km']:.1f} km vs "
            f"{fifo['total_distance_km']:.1f} km)."
        )
    else:
        distance_sentence = (
            f"운영 비용 측면에서는 ONNX의 총 주행거리가 FIFO보다 {fifo_dist_delta_pct:+.1f}% 길다"
            f"({onnx['total_distance_km']:.1f} km vs {fifo['total_distance_km']:.1f} km). "
            f"다만 Vanilla와는 {vanilla_dist_delta_pct:+.1f}% 차이로 거의 같은 거리에서 대기시간을 크게 줄였다."
        )

    story = []
    story.append(para(TITLE, styles["title"]))
    story.append(para(SUBTITLE, styles["subtitle"]))
    story.append(para(f"생성 시각: {generated} / Matrix Teleport episode CSV exports 기반", styles["subtitle"]))
    story.append(para("초록", styles["h1"]))
    story.append(
        para(
            "본 보고서는 30명 new 수요 시나리오에서 ONNX 추론 정책을 FIFO 기준선 및 Vanilla 순차 정책과 비교한다. "
            "평가는 선택된 Matrix Teleport episode CSV export를 기반으로 하며, 서비스율, 대기시간, 탑승시간, 총 주행거리, "
            "완료 승객당 운행 비용, loaded distance ratio, 시간 불평등, 방문 스탑 수를 동일한 KPI 체계로 정리하였다. "
            "대기시간과 탑승시간은 서로 다른 의사결정 품질을 나타내므로 표와 그림에서 분리하여 해석한다.",
            styles["body"],
        )
    )
    story.append(para("핵심 결과", styles["h1"]))
    bullets = [
        "세 정책 모두 서비스율 100%를 달성했기 때문에, 단순 완료 여부보다 승객 시간 품질과 운행 효율이 핵심 비교 기준이다.",
        f"ONNX는 평균 대기시간 {onnx['wait_mean_seconds']:.1f}초, 중앙 대기시간 {onnx['wait_median_seconds']:.1f}초로 가장 낮다. FIFO 대비 평균 {wait_mean_gain_fifo:.1f}초, 중앙값 {wait_median_gain_fifo:.1f}초를 줄였고, Vanilla 대비 평균 {wait_mean_gain_vanilla:.1f}초, 중앙값 {wait_median_gain_vanilla:.1f}초를 줄였다.",
        ride_sentence,
        distance_sentence,
        f"FIFO는 총 주행거리와 에피소드 시간이 가장 낮은 정책이며, 최악 대기시간과 불평등 지표에서도 더 보수적인 결과를 보인다. 따라서 ONNX의 우세는 평균/중앙 승객시간 중심으로 주장하고, 꼬리 위험과 운행거리 증가는 한계로 함께 제시하는 것이 타당하다.",
        f"Loaded distance ratio는 {loaded_best}가 가장 높고, 에피소드 시간은 {episode_best}가 가장 짧다. 이 두 지표를 함께 보면 정책별로 승객 시간 개선과 차량 운행 비용 사이의 trade-off가 존재한다.",
    ]
    for bullet in bullets:
        story.append(para(f"- {bullet}", styles["body"]))
    story.append(para("결론", styles["h1"]))
    story.append(
        para(
            "30new 단일 시나리오에서 ONNX는 전 승객을 처리하면서 평균 및 중앙 대기시간을 가장 크게 낮춘 정책이다. "
            "따라서 논문에서는 ONNX를 '승객의 평균 체감 서비스 품질을 개선한 학습 기반 정책'으로 서술하는 것이 가장 설득력 있다. "
            "다만 FIFO가 총 주행거리, 에피소드 시간, 최악값 및 불평등 지표에서 강점을 보이므로, ONNX가 모든 KPI에서 우월하다고 쓰기보다는 "
            "대기시간 중심 성능 우세와 운영 비용 trade-off를 함께 제시해야 한다.",
            styles["body"],
        )
    )
    story.append(para("사용 CSV 소스", styles["h1"]))
    story.append(styled_table(source_rows(summary, cstyle), font, font_bold, widths=[2.0 * cm, 6.9 * cm, 1.4 * cm, 1.5 * cm, 4.3 * cm], size=5.9))

    story.append(PageBreak())
    story.append(para("KPI 요약표", styles["h1"]))
    story.append(styled_table(summary_rows(summary, cstyle), font, font_bold, widths=[1.3 * cm, 1.1 * cm, 1.25 * cm, 1.25 * cm, 1.25 * cm, 1.0 * cm, 1.25 * cm, 1.25 * cm, 1.25 * cm, 1.0 * cm, 1.15 * cm, 1.15 * cm, 1.0 * cm, 1.0 * cm, 1.05 * cm, 0.8 * cm], size=5.15))
    story.append(Spacer(1, 0.2 * cm))
    story.append(image(figures["fig1"], 18.2 * cm, 15.6 * cm))

    story.append(PageBreak())
    story.append(para("그림 분석", styles["h1"]))
    story.append(image(figures["fig2"], 18.2 * cm, 8.1 * cm))
    story.append(Spacer(1, 0.35 * cm))
    story.append(image(figures["fig3"], 18.2 * cm, 8.8 * cm))

    story.append(PageBreak())
    story.append(image(figures["fig4"], 17.2 * cm, 11.2 * cm))
    story.append(Spacer(1, 0.2 * cm))
    story.append(para("논문 본문 권장 서술", styles["h1"]))
    story.append(
        para(
            "30명 new 수요 시나리오에서 ONNX 정책은 Vanilla 및 FIFO와 동일하게 모든 승객을 완료했으며, 평균 및 중앙 대기시간을 가장 낮게 유지하였다. "
            "특히 FIFO 대비 중앙 대기시간과 중앙 탑승시간이 개선되어, 학습 정책이 단순 선착순 기준선보다 승객 체감 시간을 더 적극적으로 줄였음을 보인다. "
            "반면 FIFO는 총 주행거리와 최악 대기시간에서 더 안정적인 값을 보였으므로, 본 결과는 ONNX의 전면적 우위가 아니라 평균 서비스 품질 개선과 운영 비용 사이의 trade-off로 해석하는 것이 적절하다.",
            styles["body"],
        )
    )
    story.append(para("후속 비교 권장 사항", styles["h1"]))
    story.append(
        para(
            "동일 KPI 체계를 30new의 여러 random seed에 반복 적용한 뒤, 14/18/22/40/50 수요량으로 확장하는 것이 좋다. "
            "보고서에서는 대기시간과 탑승시간을 계속 분리하고, 평균/중앙값 옆에 최대값과 Gini를 함께 제시해야 평균 개선이 꼬리 위험 개선으로 오해되지 않는다.",
            styles["body"],
        )
    )

    doc.build(story, onFirstPage=footer(font), onLaterPages=footer(font))


def main() -> None:
    font, font_bold = register_fonts()
    _, passengers, _, summary, _ = load_data()
    deltas = build_delta_table(summary)
    scores = normalized_scores(summary)
    save_analysis_tables(summary, deltas, scores)
    figures = build_figures(summary, passengers, scores)
    build_pdf(summary, figures, font, font_bold)
    print(PDF_PATH)
    print(TABLE_DIR / "policy_analysis_summary_ko.csv")
    print(TABLE_DIR / "policy_normalized_scorecard.csv")


if __name__ == "__main__":
    main()
