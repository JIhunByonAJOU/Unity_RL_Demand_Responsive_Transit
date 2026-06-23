from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
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


POLICY_ORDER = ["Vanilla", "Greedy", "ONNX"]
POLICY_LABELS = {
    "Vanilla": "Vanilla Sequential",
    "Greedy": "Greedy Nearest Feasible",
    "ONNX": "ONNX PPO Inference",
}


def latest_result_dir() -> Path:
    dirs = sorted(Path("output").glob("drt_result_report_*"), key=lambda p: p.stat().st_mtime)
    if not dirs:
        raise SystemExit("No output/drt_result_report_* directory found.")
    return dirs[-1]


def fmt(value, digits=1, suffix=""):
    if pd.isna(value):
        return "-"
    return f"{float(value):,.{digits}f}{suffix}"


def fmt_pct(value, digits=1):
    if pd.isna(value):
        return "-"
    return f"{float(value) * 100:,.{digits}f}%"


def fmt_signed(value, digits=1, suffix="%"):
    if pd.isna(value):
        return "-"
    return f"{float(value):+,.{digits}f}{suffix}"


def para(text: str, style):
    return Paragraph(text.replace("&", "&amp;"), style)


def table_style(header_bg=colors.HexColor("#E8EDF4"), font_size=7.2):
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), header_bg),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111111")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("LEADING", (0, 0), (-1, -1), font_size + 1.3),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#B8B8B8")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F7F7")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
    )


def styled_table(data, col_widths=None, font_size=7.2, repeat_rows=1):
    tbl = LongTable(data, colWidths=col_widths, repeatRows=repeat_rows)
    tbl.setStyle(table_style(font_size=font_size))
    return tbl


def scaled_image(path: Path, max_width, max_height):
    with PILImage.open(path) as img:
        width_px, height_px = img.size
    ratio = min(max_width / width_px, max_height / height_px)
    return Image(str(path), width=width_px * ratio, height=height_px * ratio)


def split_figures(fig_paths):
    main = [
        "fig01_average_wait_by_scenario.png",
        "fig02_service_rate_by_scenario.png",
        "fig03_completion_reliability_by_scenario.png",
        "fig04_episode_time_by_scenario.png",
        "fig05_route_distance_by_scenario.png",
        "fig06_distance_per_completed_passenger.png",
        "fig07_empty_distance_share_by_scenario.png",
        "fig08_mean_onboard_load_by_scenario.png",
        "fig09_passenger_wait_distribution.png",
        "fig10_wait_distance_tradeoff.png",
        "fig11_finish_reason_distribution.png",
    ]
    heat = [
        "fig12_edge_frequency_vanilla.png",
        "fig12_edge_frequency_greedy.png",
        "fig12_edge_frequency_onnx.png",
    ]
    by_name = {p.name: p for p in fig_paths}
    return [by_name[n] for n in main if n in by_name], [by_name[n] for n in heat if n in by_name]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(1.5 * cm, 1.0 * cm, "DRT extended-horizon policy evaluation")
    canvas.drawRightString(A4[0] - 1.5 * cm, 1.0 * cm, f"Page {doc.page}")
    canvas.restoreState()


def load_data(result_dir: Path):
    tables = result_dir / "tables"
    data = {
        "episode": pd.read_csv(tables / "episode_level_metrics.csv"),
        "summary": pd.read_csv(tables / "scenario_policy_summary.csv"),
        "macro": pd.read_csv(tables / "policy_macro_summary.csv"),
        "deltas": pd.read_csv(tables / "pairwise_policy_deltas.csv"),
        "finish": pd.read_csv(tables / "finish_reason_counts.csv"),
        "winners": pd.read_csv(tables / "scenario_winners_service_then_wait.csv"),
        "primary": pd.read_csv(tables / "table_primary_results.csv"),
    }
    for key in ("summary", "macro", "primary"):
        if "policy" in data[key].columns:
            data[key]["policy"] = pd.Categorical(data[key]["policy"], POLICY_ORDER, ordered=True)
            sort_cols = ["policy"] if key == "macro" else ["scenario", "policy"]
            data[key] = data[key].sort_values(sort_cols).reset_index(drop=True)
    return data


def build_source_table(ep: pd.DataFrame):
    rows = [["Policy", "Selected export folder", "Episodes", "Completed", "Timed out"]]
    finish_counts = ep.groupby(["policy", "finish_reason"]).size().unstack(fill_value=0)
    run_folders = ep.groupby(["policy", "run_folder"]).size().reset_index(name="episodes")
    for policy in POLICY_ORDER:
        policy_runs = run_folders[run_folders["policy"] == policy]
        folder = ", ".join(policy_runs["run_folder"].astype(str).tolist())
        total = int((ep["policy"] == policy).sum())
        completed = int(finish_counts.loc[policy].get("All passenger requests completed.", 0))
        timed = int(finish_counts.loc[policy].get("Episode time ended.", 0))
        rows.append([POLICY_LABELS[policy], folder, str(total), str(completed), str(timed)])
    return rows


def build_inventory_table(primary: pd.DataFrame):
    rows = [["Scenario", "Policy", "Runs", "Full", "Service rate", "Mean wait"]]
    for _, row in primary.iterrows():
        rows.append(
            [
                str(int(row["scenario"])),
                str(row["policy"]),
                str(int(row["n_episodes"])),
                str(row["completion_runs"]),
                f"{float(row['service_rate_pct']):.1f}%",
                f"{float(row['average_wait_seconds_mean']):,.1f} s",
            ]
        )
    return rows


def build_qos_table(summary: pd.DataFrame):
    rows = [["Scen.", "Policy", "SR", "Wait", "P90 wait", "Ride", "Time", "Distance"]]
    for _, row in summary.iterrows():
        rows.append(
            [
                str(int(row["scenario"])),
                str(row["policy"]),
                fmt_pct(row["service_rate_mean"]),
                f"{fmt(row['average_wait_seconds_mean'])} s",
                f"{fmt(row['wait_p90_seconds_mean'])} s",
                f"{fmt(row['average_ride_seconds_mean'])} s",
                f"{fmt(row['episode_time_minutes_mean'])} min",
                f"{fmt(row['episode_distance_km_mean'])} km",
            ]
        )
    return rows


def build_macro_table(macro: pd.DataFrame):
    rows = [["Policy", "Episodes", "Full", "SR", "Wait", "Ride", "Dist.", "km/pass.", "Empty"]]
    macro_sorted = macro.copy()
    macro_sorted["policy"] = pd.Categorical(macro_sorted["policy"], POLICY_ORDER, ordered=True)
    for _, row in macro_sorted.sort_values("policy").iterrows():
        rows.append(
            [
                str(row["policy"]),
                str(int(row["total_episodes"])),
                fmt_pct(row["completion_rate_macro"]),
                fmt_pct(row["service_rate_macro"]),
                f"{fmt(row['average_wait_seconds_macro'])} s",
                f"{fmt(row['average_ride_seconds_macro'])} s",
                f"{fmt(row['episode_distance_km_macro'])} km",
                f"{fmt(row['dist_per_completed_macro'], 2)}",
                fmt_pct(row["empty_distance_rate_macro"]),
            ]
        )
    return rows


def build_delta_table(deltas: pd.DataFrame, target: str, baseline: str):
    rows = [["Scenario", "d SR", "d Full", "Wait red.", "Ride red.", "Dist. red.", "Time red."]]
    selected = deltas[(deltas["target"] == target) & (deltas["baseline"] == baseline)]
    for _, row in selected.sort_values("scenario").iterrows():
        rows.append(
            [
                str(int(row["scenario"])),
                fmt_signed(row["service_rate_delta_pctpt"], suffix=" pp"),
                fmt_signed(row["completion_rate_delta_pctpt"], suffix=" pp"),
                fmt_signed(row["wait_reduction_pct"]),
                fmt_signed(row["ride_reduction_pct"]),
                fmt_signed(row["distance_reduction_pct"]),
                fmt_signed(row["time_reduction_pct"]),
            ]
        )
    return rows


def build_route_table(summary: pd.DataFrame):
    rows = [["Scen.", "Policy", "km/comp.", "Empty", "Mean load", "Legs/comp.", "Edge reuse"]]
    for _, row in summary.iterrows():
        rows.append(
            [
                str(int(row["scenario"])),
                str(row["policy"]),
                fmt(row["distance_km_per_completed_passenger_mean"], 2),
                fmt_pct(row["empty_distance_rate_mean"]),
                fmt(row["mean_onboard_load_mean"], 2),
                fmt(row["legs_per_completed_passenger_mean"], 2),
                fmt(row["edge_reuse_factor_mean"], 2),
            ]
        )
    return rows


def scenario_narratives(summary: pd.DataFrame, winners: pd.DataFrame):
    narratives = []
    for scenario in sorted(summary["scenario"].unique()):
        rows = summary[summary["scenario"] == scenario].set_index("policy")
        winner = winners[winners["scenario"] == scenario]["best_policy_service_then_wait"].iloc[0]
        g = rows.loc["Greedy"]
        o = rows.loc["ONNX"]
        v = rows.loc["Vanilla"]
        if scenario == 40:
            note = (
                "ONNX is the scenario winner on wait while preserving SR=1.0, "
                "but Greedy remains close and has shorter ride, time, and distance."
            )
        elif scenario == 50:
            note = (
                "ONNX reports a wait close to Greedy, but its SR=91.6% and only 4/9 full-completion runs "
                "make Greedy the correct service-first selection."
            )
        else:
            note = "Greedy dominates the service-first ranking and also gives the lowest mean wait."
        narratives.append(
            f"Scenario {int(scenario)}: winner={winner}. "
            f"Waits are Greedy {g['average_wait_seconds_mean']:.1f} s, "
            f"ONNX {o['average_wait_seconds_mean']:.1f} s, "
            f"Vanilla {v['average_wait_seconds_mean']:.1f} s. {note}"
        )
    return narratives


def build_pdf(result_dir: Path, output_pdf: Path):
    data = load_data(result_dir)
    ep = data["episode"]
    summary = data["summary"]
    macro = data["macro"]
    deltas = data["deltas"]
    finish = data["finish"]
    winners = data["winners"]
    primary = data["primary"]

    figures = sorted((result_dir / "figures").glob("*.png"))
    main_figs, heat_figs = split_figures(figures)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        leftMargin=1.55 * cm,
        rightMargin=1.55 * cm,
        topMargin=1.55 * cm,
        bottomMargin=1.7 * cm,
        title="DRT Extended-Horizon Policy Evaluation",
    )
    width = doc.width
    height = doc.height

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
        fontSize=21,
        leading=25,
        spaceAfter=8,
    )
    subtitle = ParagraphStyle(
        "SubtitleCustom",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontName="Helvetica",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#444444"),
        spaceAfter=8,
    )
    h1 = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=17,
        spaceBefore=9,
        spaceAfter=5,
    )
    h2 = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=14,
        spaceBefore=7,
        spaceAfter=4,
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.2,
        leading=12.3,
        alignment=TA_JUSTIFY,
        spaceAfter=5,
    )
    bullet = ParagraphStyle(
        "Bullet",
        parent=body,
        leftIndent=12,
        firstLineIndent=-7,
        alignment=TA_LEFT,
    )
    caption = ParagraphStyle(
        "Caption",
        parent=styles["BodyText"],
        fontName="Helvetica-Oblique",
        fontSize=8.1,
        leading=10,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#444444"),
        spaceBefore=2,
        spaceAfter=8,
    )

    macro_idx = macro.set_index("policy")
    total_episodes = len(ep)
    timeout_total = int((ep["finish_reason"] == "Episode time ended.").sum())
    onnx_50 = summary[(summary["policy"] == "ONNX") & (summary["scenario"] == 50)].iloc[0]
    greedy_macro = macro_idx.loc["Greedy"]
    vanilla_macro = macro_idx.loc["Vanilla"]
    onnx_macro = macro_idx.loc["ONNX"]
    winner_text = ", ".join(
        [
            f"{int(row.scenario)}:{row.best_policy_service_then_wait}"
            for row in winners.itertuples(index=False)
        ]
    )

    story = []
    story.append(Paragraph("DRT Extended-Horizon Scenario Policy Evaluation", title))
    story.append(
        Paragraph(
            "ONNX PPO Inference vs Greedy Nearest Feasible vs Vanilla Sequential",
            subtitle,
        )
    )
    story.append(
        Paragraph(
            f"Generated {datetime.now():%Y-%m-%d %H:%M:%S} from latest Matrix Teleport episode CSV exports.",
            subtitle,
        )
    )
    story.append(Spacer(1, 8))
    story.append(Paragraph("Abstract", h1))
    story.append(
        Paragraph(
            "This report evaluates three DRT next-stop dispatch policies across 14, 18, 22, 30, 40, and 50 passenger demand scenarios. "
            "The analysis uses the latest export folder for each policy mode and averages repeated runs within each scenario before comparing policies. "
            "The primary ranking rule is service-rate first, average passenger waiting time second, and route efficiency third. "
            f"The evaluated dataset contains {total_episodes} episodes, {int(ep['completed_passengers'].sum())} completed passenger services, and "
            f"{timeout_total} time-ended episodes. Greedy is the best overall policy under the primary rule, with macro service rate 100.0%, "
            f"macro completion 100.0%, and macro mean wait {greedy_macro['average_wait_seconds_macro']:.1f} s. ONNX wins scenario 40 on mean wait, "
            f"but its 50-passenger result is weakened by SR={onnx_50['service_rate_mean']*100:.1f}% and only "
            f"{onnx_50['completed_all_requests_mean']*100:.1f}% full-completion runs.",
            body,
        )
    )
    story.append(
        Paragraph(
            "<b>Index Terms</b>--Demand-responsive transit, dispatch policy, service rate, passenger waiting time, route efficiency, ML-Agents, ONNX inference.",
            body,
        )
    )
    story.append(Paragraph("Key Findings", h1))
    key_findings = [
        f"Greedy is the overall winner: service-rate-first winners by scenario are {winner_text}.",
        f"Greedy reduces macro mean wait from Vanilla's {vanilla_macro['average_wait_seconds_macro']:.1f} s to {greedy_macro['average_wait_seconds_macro']:.1f} s, a 34.5% macro reduction.",
        f"ONNX reduces wait relative to Vanilla in every scenario, but does not consistently beat Greedy; its 50-passenger case has {timeout_total} time-ended runs, all under ONNX scenario 50.",
        f"Scenario 40 is the strongest ONNX cell: ONNX wait is 1164.4 s versus Greedy 1197.4 s at SR=1.0.",
        "The main paper claim should therefore be framed as: learned routing is promising, but the current deployable benchmark is still Greedy under service reliability constraints.",
    ]
    for item in key_findings:
        story.append(Paragraph("- " + item, bullet))

    story.append(Paragraph("Selected CSV Sources", h1))
    story.append(styled_table(build_source_table(ep), [4.4 * cm, 5.2 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm], font_size=7.8))

    story.append(PageBreak())
    story.append(Paragraph("I. Experimental Protocol and Metric Definitions", h1))
    story.append(
        Paragraph(
            "Each episode CSV contains a summary section, a route-leg section, and a passenger section. "
            "The summary section is used for service rate, completion, episode time, route distance, average wait, and average ride. "
            "The passenger section is used for P50/P90/max wait and ride distributions. The route-leg section is used for distance-normalized and load-normalized route efficiency.",
            body,
        )
    )
    story.append(
        Paragraph(
            "For scenario s and policy p, the reported scenario value is the arithmetic mean over all runs in the selected export folder: "
            "M(p,s) = (1/N) sum_i M_i(p,s). Macro averages are then computed over the six scenario means, not over raw episodes, so policies with more repeated runs do not dominate the conclusion.",
            body,
        )
    )
    metric_rows = [
        ["Metric", "Definition", "Interpretation"],
        ["Service rate", "completed_passengers / total_passengers", "Primary reliability criterion."],
        ["Full completion", "completed_all_requests", "Whether the run finished all requests."],
        ["Mean wait", "mean pickup_time - request_time for completed passengers", "Primary QoS metric after reliability."],
        ["P90 wait", "90th percentile passenger wait", "Tail waiting burden."],
        ["km/completed pax", "episode_distance_km / completed_passengers", "Distance cost of service."],
        ["Empty-distance share", "distance with no onboard passenger / total distance", "Deadheading estimate from route legs."],
        ["Mean onboard load", "passenger-km / route-km", "Vehicle utilization along route legs."],
    ]
    story.append(styled_table(metric_rows, [3.2 * cm, 7.8 * cm, 6.2 * cm], font_size=7.5))

    story.append(Paragraph("II. Run Inventory and Termination Reliability", h1))
    story.append(
        Paragraph(
            "Table I reports the exact run counts selected from the latest export folders. Vanilla and Greedy completed every selected run. "
            "ONNX completed all requests in scenarios 14 through 40, but five of the nine ONNX 50-passenger runs still ended by episode time. "
            "That is why the 50-passenger ONNX wait value is not used as a direct winner even though its served-passenger wait is close to Greedy.",
            body,
        )
    )
    story.append(Paragraph("Table I. Scenario-level run inventory.", caption))
    story.append(styled_table(build_inventory_table(primary), [1.55 * cm, 3.4 * cm, 1.55 * cm, 1.9 * cm, 2.2 * cm, 2.8 * cm], font_size=7.1))

    story.append(PageBreak())
    story.append(Paragraph("III. Primary Quality-of-Service Results", h1))
    story.append(
        Paragraph(
            "Table II is the central result table. Service rate is evaluated before wait. This matters in scenario 50 because ONNX reports a mean wait of 1650.6 s, slightly below Greedy's 1655.7 s, but ONNX service rate is only 91.6% while Greedy remains at 100.0%. "
            "For a DRT service report, that makes Greedy the correct 50-passenger winner.",
            body,
        )
    )
    story.append(Paragraph("Table II. Primary service metrics by scenario.", caption))
    story.append(styled_table(build_qos_table(summary), [1.15 * cm, 2.3 * cm, 1.6 * cm, 2.25 * cm, 2.35 * cm, 2.15 * cm, 2.2 * cm, 2.3 * cm], font_size=6.35))
    story.append(Spacer(1, 5))
    story.append(Paragraph("Scenario-by-scenario interpretation", h2))
    for text in scenario_narratives(summary, winners):
        story.append(Paragraph("- " + text, bullet))

    story.append(PageBreak())
    story.append(Paragraph("IV. Macro-Average Policy Comparison", h1))
    story.append(
        Paragraph(
            "Table III averages scenario-level means across the six demand scenarios. Greedy is best on the operational metrics that should drive model selection: full-completion rate, service rate, mean wait, ride time, route distance, and distance per completed passenger. "
            "ONNX shows the lowest empty-distance percentage, but that does not compensate for weaker high-load completion and longer total route distance.",
            body,
        )
    )
    story.append(Paragraph("Table III. Macro-average results across scenario means.", caption))
    story.append(styled_table(build_macro_table(macro), [3.0 * cm, 1.7 * cm, 1.5 * cm, 1.5 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 1.9 * cm, 1.6 * cm], font_size=6.9))

    story.append(Paragraph("V. Pairwise Delta Analysis", h1))
    story.append(
        Paragraph(
            "Tables IV-A and IV-B show relative improvement over Vanilla. Positive wait, ride, distance, and time reductions mean the target policy is better than Vanilla. "
            "ONNX improves wait relative to Vanilla in all scenarios, but it often increases distance and time. Greedy improves wait and usually route cost while maintaining service reliability.",
            body,
        )
    )
    story.append(Paragraph("Table IV-A. ONNX relative to Vanilla.", caption))
    story.append(styled_table(build_delta_table(deltas, "ONNX", "Vanilla"), [1.6 * cm] * 7, font_size=6.7))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Table IV-B. Greedy relative to Vanilla.", caption))
    story.append(styled_table(build_delta_table(deltas, "Greedy", "Vanilla"), [1.6 * cm] * 7, font_size=6.7))

    story.append(PageBreak())
    story.append(Paragraph("VI. Route Efficiency and Dispatch Behavior", h1))
    story.append(
        Paragraph(
            "The route-efficiency table explains why policies with similar wait values can still differ operationally. Vanilla repeatedly circulates through the fixed stop sequence; this is reliable, but it creates long detours and large route distances as demand grows. "
            "Greedy reduces those detours through nearest-feasible dispatch. ONNX uses non-sequential routing and has low empty-distance share, but its total distance and km per completed passenger are not yet competitive at the macro level.",
            body,
        )
    )
    story.append(Paragraph("Table V. Route-efficiency metrics.", caption))
    story.append(styled_table(build_route_table(summary), [1.1 * cm, 2.2 * cm, 2.0 * cm, 1.7 * cm, 1.8 * cm, 2.0 * cm, 1.8 * cm], font_size=6.55))

    story.append(Paragraph("VII. Figures", h1))
    story.append(
        Paragraph(
            "Figures 1 through 11 summarize the main service and route metrics. The edge-frequency heatmaps are shown afterward as Fig. 12(a)-(c). "
            "All figures are generated directly from the current CSV-derived summary tables.",
            body,
        )
    )

    captions = {
        "fig01_average_wait_by_scenario.png": "Fig. 1. Mean served-passenger waiting time.",
        "fig02_service_rate_by_scenario.png": "Fig. 2. Mean service rate.",
        "fig03_completion_reliability_by_scenario.png": "Fig. 3. All-request completion reliability.",
        "fig04_episode_time_by_scenario.png": "Fig. 4. Episode duration.",
        "fig05_route_distance_by_scenario.png": "Fig. 5. Route distance.",
        "fig06_distance_per_completed_passenger.png": "Fig. 6. Distance per completed passenger.",
        "fig07_empty_distance_share_by_scenario.png": "Fig. 7. Empty-distance share.",
        "fig08_mean_onboard_load_by_scenario.png": "Fig. 8. Mean onboard load.",
        "fig09_passenger_wait_distribution.png": "Fig. 9. Passenger-level wait distribution.",
        "fig10_wait_distance_tradeoff.png": "Fig. 10. Wait-distance trade-off.",
        "fig11_finish_reason_distribution.png": "Fig. 11. Finish reason distribution.",
        "fig12_edge_frequency_vanilla.png": "Fig. 12(a). Vanilla route edge frequency.",
        "fig12_edge_frequency_greedy.png": "Fig. 12(b). Greedy route edge frequency.",
        "fig12_edge_frequency_onnx.png": "Fig. 12(c). ONNX route edge frequency.",
    }

    for i in range(0, len(main_figs), 2):
        story.append(PageBreak())
        for fig in main_figs[i : i + 2]:
            story.append(KeepTogether([scaled_image(fig, width, 8.2 * cm), Paragraph(captions[fig.name], caption)]))

    story.append(PageBreak())
    story.append(Paragraph("Route Edge-Frequency Heatmaps", h1))
    for fig in heat_figs:
        story.append(KeepTogether([scaled_image(fig, width * 0.72, 7.8 * cm), Paragraph(captions[fig.name], caption)]))

    story.append(PageBreak())
    story.append(Paragraph("VIII. Discussion", h1))
    story.append(
        Paragraph(
            "The extended horizon changes the interpretation of Vanilla and Greedy: both policies now complete every selected run, including the 50-passenger scenario. "
            "This removes the previous timeout-driven weakness for those baselines. ONNX also improves over older timeout-heavy runs, but the high-load limit is still visible in scenario 50. "
            "The result is not that ONNX fails everywhere: it beats Vanilla wait in every scenario and is the best wait policy in scenario 40. The result is that Greedy remains the strongest deployable policy because it combines complete service reliability with the lowest macro wait and route cost.",
            body,
        )
    )
    story.append(
        Paragraph(
            "For the IEEE paper, the claim should be phrased conservatively. A defensible statement is: learned next-stop inference can reduce waiting time compared with a sequential baseline and can match or exceed Greedy in selected dense scenarios, but the evaluated checkpoint does not yet dominate Greedy across the full demand sweep. "
            "The strongest evidence for the current system is the Greedy-vs-Vanilla improvement and the ONNX scenario-40 case, not a global ONNX victory.",
            body,
        )
    )

    story.append(Paragraph("IX. Threats to Validity", h1))
    limitations = [
        "The experiment uses MatrixTeleport, so travel dynamics reflect matrix-based travel-time abstraction rather than full physical driving.",
        "Repeated-run counts are unbalanced across policies and scenarios; macro averages are therefore computed over scenario means.",
        "Average wait is computed over completed passengers. When service rate is below 1.0, unserved passengers must be interpreted through service rate and finish reason.",
        "Greedy and Vanilla are deterministic under these matrix settings, which explains zero standard deviation in many scenario cells.",
        "The ONNX policy should be re-evaluated after additional high-demand training or reward shaping focused on completion reliability.",
    ]
    for item in limitations:
        story.append(Paragraph("- " + item, bullet))

    story.append(Paragraph("X. Conclusion", h1))
    story.append(
        Paragraph(
            "The latest CSV evidence supports Greedy Nearest Feasible as the best current policy under the chosen operational criterion: service rate first, passenger wait second, and route efficiency third. "
            "ONNX is not discarded; it is best characterized as an adaptive policy that improves substantially over Vanilla but still requires additional robustness work at the 50-passenger load. "
            "For reporting, Greedy should be presented as the current deployable benchmark, Vanilla as the deterministic sequential baseline, and ONNX as a learned policy with scenario-specific gains and unresolved high-load completion risk.",
            body,
        )
    )

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result_dir = args.result_dir or latest_result_dir()
    if args.output:
        output_pdf = args.output
    else:
        suffix = result_dir.name.replace("drt_result_report_", "")
        output_pdf = Path("output/pdf") / f"drt_extended_horizon_policy_evaluation_{suffix}.pdf"
    build_pdf(result_dir, output_pdf)
    print(output_pdf.resolve())


if __name__ == "__main__":
    main()
