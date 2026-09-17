"""Generate the Stage IV knowledge-base PDFs.

Why this file exists
---------------------
Stage IV requires a RAG knowledge base (manuals / SOPs / safety guidelines).
Inspection at the start of Stage IV confirmed no such documents exist
anywhere in the project (see docs/HANDOFF.md). This script creates the
minimum realistic set of three PDFs needed to demonstrate the RAG
requirement, and nothing more (see knowledge_base/SOURCE.md).

Grounding, not hard-coding
--------------------------
Every numeric operating threshold below is *computed* from
``config.MACHINE_TYPE_PROFILES`` (the same parameters Stage I has used since
the synthetic sensor generator was written), not typed in as arbitrary
numbers. Re-running this script is deterministic (no randomness) and will
regenerate byte-identical PDFs.

Run:
    python -m src.rag.build_knowledge_base_pdfs
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
)
from reportlab.lib import colors

from config.config import MACHINE_TYPE_PROFILES, PROJECT_ROOT

KB_DIR = PROJECT_ROOT / "knowledge_base"

# Warning threshold = mean + 2*std ; Danger threshold = mean + 3.5*std.
# This is a standard engineering rule-of-thumb (2-sigma = elevated /
# out-of-normal-range, 3.5-sigma = rare/critical), applied consistently to
# every machine type and every sensor channel so the document is internally
# consistent rather than hand-tuned per line.
WARNING_SIGMA = 2.0
DANGER_SIGMA = 3.5


def _thresholds(profile: dict, key_prefix: str) -> tuple[float, float, float]:
    mean = profile[f"{key_prefix}_mean"]
    std = profile[f"{key_prefix}_std"]
    return mean, mean + WARNING_SIGMA * std, mean + DANGER_SIGMA * std


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="DocTitle", fontSize=18, leading=22,
                               spaceAfter=14, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name="Section", fontSize=13, leading=16,
                               spaceBefore=14, spaceAfter=8,
                               fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name="Body", fontSize=10, leading=14,
                               spaceAfter=6))
    return styles


def build_machine_operation_manual() -> Path:
    """Per-machine-type operating ranges and thresholds."""
    path = KB_DIR / "machine_operation_manual.pdf"
    styles = _styles()
    doc = SimpleDocTemplate(str(path), pagesize=LETTER,
                             topMargin=0.8 * inch, bottomMargin=0.8 * inch)
    story = [
        Paragraph("Machine Operation Manual — AI Factory Fleet", styles["DocTitle"]),
        Paragraph(
            "This manual defines normal operating ranges and out-of-range "
            "thresholds for every machine type in the fleet (CNC, Press, "
            "Injection Molder). Warning thresholds indicate a reading that "
            "requires operator attention; danger thresholds indicate a "
            "reading that requires immediate action (see the Preventive "
            "Maintenance SOP and Safety &amp; Emergency Guidelines documents "
            "for required responses).",
            styles["Body"],
        ),
        Spacer(1, 8),
    ]

    channel_labels = {
        "temperature_k": ("Temperature", "K"),
        "vibration_mm_s": ("Vibration", "mm/s"),
        "pressure_bar": ("Pressure", "bar"),
        "torque_nm": ("Torque", "Nm"),
    }

    for idx, (machine_type, profile) in enumerate(MACHINE_TYPE_PROFILES.items()):
        if idx > 0:
            story.append(PageBreak())
        display_name = machine_type.replace("_", " ")
        story.append(Paragraph(f"Section {idx + 1}: {display_name}", styles["Section"]))
        story.append(Paragraph(
            f"Rated power draw: {profile['rated_power_w']} W. Readings below "
            f"are per-15-minute sensor sample.",
            styles["Body"],
        ))

        rows = [["Sensor", "Normal (mean)", "Warning threshold", "Danger threshold"]]
        for key, (label, unit) in channel_labels.items():
            mean, warn, danger = _thresholds(profile, key)
            rows.append([
                label,
                f"{mean:.1f} {unit}",
                f"\u2265 {warn:.1f} {unit}",
                f"\u2265 {danger:.1f} {unit}",
            ])

        table = Table(rows, colWidths=[1.4 * inch, 1.6 * inch, 1.8 * inch, 1.8 * inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b2b2b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
        ]))
        story.append(table)
        story.append(Spacer(1, 8))

        vib_mean, vib_warn, vib_danger = _thresholds(profile, "vibration_mm_s")
        temp_mean, temp_warn, temp_danger = _thresholds(profile, "temperature_k")
        pres_mean, pres_warn, pres_danger = _thresholds(profile, "pressure_bar")

        story.append(Paragraph(
            f"Vibration guidance: normal running vibration for a "
            f"{display_name} is approximately {vib_mean:.1f} mm/s. A "
            f"sustained reading at or above {vib_warn:.1f} mm/s is a "
            f"WARNING condition and should be logged and monitored. A "
            f"sustained reading at or above {vib_danger:.1f} mm/s is a "
            f"DANGER condition — vibration this high is dangerously "
            f"elevated and indicates likely bearing wear, misalignment, or "
            f"an imminent mechanical failure.",
            styles["Body"],
        ))
        story.append(Paragraph(
            f"Temperature guidance: normal operating temperature is "
            f"approximately {temp_mean:.1f} K. {temp_warn:.1f} K or above is "
            f"a WARNING condition; {temp_danger:.1f} K or above is a DANGER "
            f"condition indicating possible overheating or lubrication "
            f"failure.",
            styles["Body"],
        ))
        story.append(Paragraph(
            f"Pressure guidance: normal operating pressure is approximately "
            f"{pres_mean:.1f} bar. {pres_warn:.1f} bar or above is a WARNING "
            f"condition; {pres_danger:.1f} bar or above is a DANGER "
            f"condition indicating a possible seal or valve fault.",
            styles["Body"],
        ))

    doc.build(story)
    return path


def build_preventive_maintenance_sop() -> Path:
    path = KB_DIR / "preventive_maintenance_sop.pdf"
    styles = _styles()
    doc = SimpleDocTemplate(str(path), pagesize=LETTER,
                             topMargin=0.8 * inch, bottomMargin=0.8 * inch)
    story = [
        Paragraph("Preventive Maintenance SOP", styles["DocTitle"]),
        Paragraph(
            "This SOP defines the routine maintenance schedule and the "
            "required operator response when a machine sensor reading "
            "crosses a WARNING or DANGER threshold as defined in the "
            "Machine Operation Manual.",
            styles["Body"],
        ),
    ]

    story.append(Paragraph("Section 1: Routine Maintenance Schedule", styles["Section"]))
    story.append(Paragraph(
        "Every machine in the fleet receives a scheduled inspection every "
        "168 operating hours (approximately weekly under continuous 3-shift "
        "operation). The inspection covers lubrication, belt/bearing wear, "
        "sensor calibration, and a visual check for leaks or loose "
        "fasteners. Inspection results are logged in the maintenance "
        "record with machine ID, timestamp, and inspector ID.",
        styles["Body"],
    ))

    story.append(Paragraph("Section 2: Response to a WARNING-level reading", styles["Section"]))
    story.append(Paragraph(
        "If any sensor channel (temperature, vibration, pressure, or "
        "torque) reaches its WARNING threshold and remains there for more "
        "than one full shift, the operator must: (1) log the reading and "
        "machine ID in the maintenance system, (2) increase monitoring "
        "frequency for that machine to once per hour, and (3) notify the "
        "shift supervisor. The machine may continue running under "
        "increased monitoring; a WARNING reading alone does not require "
        "stopping the machine.",
        styles["Body"],
    ))

    story.append(Paragraph(
        "Section 3: Response to a DANGER-level Reading",
        styles["Section"],
    ))
    story.append(Paragraph(
        "If any sensor channel reaches its DANGER threshold, the operator "
        "must treat this as a priority maintenance event, not a routine "
        "log entry. Specifically for dangerously high vibration: (1) "
        "reduce the machine's load or production rate immediately to lower "
        "mechanical stress, (2) schedule the machine for preventive "
        "maintenance within the current shift rather than waiting for the "
        "next scheduled inspection, (3) do not wait for a full failure "
        "before intervening — a sustained DANGER-level vibration reading "
        "is a strong precursor to bearing or shaft failure, and (4) record "
        "the event, the action taken, and the responsible supervisor in "
        "the maintenance log. If the DANGER-level reading is accompanied "
        "by unusual noise, visible smoke, or a burning smell, follow the "
        "emergency stop procedure in the Safety &amp; Emergency Guidelines "
        "document instead of continuing to operate the machine at reduced "
        "load.",
        styles["Body"],
    ))

    story.append(Paragraph("Section 4: Escalation", styles["Section"]))
    story.append(Paragraph(
        "Any DANGER-level event that recurs on the same machine within 72 "
        "hours after a maintenance action must be escalated to the "
        "maintenance engineering team for root-cause investigation, rather "
        "than repeating the same corrective action.",
        styles["Body"],
    ))

    doc.build(story)
    return path


def build_safety_emergency_guidelines() -> Path:
    path = KB_DIR / "safety_emergency_guidelines.pdf"
    styles = _styles()
    doc = SimpleDocTemplate(str(path), pagesize=LETTER,
                             topMargin=0.8 * inch, bottomMargin=0.8 * inch)
    story = [
        Paragraph("Safety &amp; Emergency Operating Guidelines", styles["DocTitle"]),
        Paragraph(
            "This document defines emergency procedures for the factory "
            "floor. It applies in addition to, not instead of, the "
            "Preventive Maintenance SOP.",
            styles["Body"],
        ),
    ]

    story.append(Paragraph("Section 1: Emergency Stop Procedure", styles["Section"]))
    story.append(Paragraph(
        "Any operator may trigger an emergency stop on any machine showing "
        "visible smoke, a burning smell, unusual grinding or screeching "
        "noise, or any sensor reading at DANGER level combined with a "
        "physical warning sign. Press the red emergency stop button at the "
        "machine control panel, then notify the shift supervisor "
        "immediately. Do not attempt to restart the machine until it has "
        "been inspected and cleared by a qualified maintenance technician.",
        styles["Body"],
    ))

    story.append(Paragraph("Section 2: Lockout-Tagout (LOTO)", styles["Section"]))
    story.append(Paragraph(
        "Before any maintenance technician performs physical work on a "
        "machine that has been stopped for a DANGER-level event, the "
        "machine's energy source must be isolated and tagged "
        "(lockout-tagout) so it cannot be re-energized until the work is "
        "complete and the tag is removed by the technician who applied it.",
        styles["Body"],
    ))

    story.append(Paragraph("Section 3: Personal Protective Equipment", styles["Section"]))
    story.append(Paragraph(
        "Safety glasses and hearing protection are required within two "
        "meters of any running CNC or Press machine. Heat-resistant gloves "
        "are required when working near an Injection Molder due to "
        "elevated operating temperature.",
        styles["Body"],
    ))

    story.append(Paragraph("Section 4: Incident Reporting", styles["Section"]))
    story.append(Paragraph(
        "Every emergency stop event must be recorded within 24 hours with: "
        "machine ID, timestamp, operator name, description of the "
        "triggering condition, and the maintenance action taken. This "
        "record feeds into the maintenance-note dataset used for incident "
        "classification.",
        styles["Body"],
    ))

    doc.build(story)
    return path


def main() -> None:
    KB_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        build_machine_operation_manual(),
        build_preventive_maintenance_sop(),
        build_safety_emergency_guidelines(),
    ]
    for p in paths:
        print(f"Wrote {p} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
