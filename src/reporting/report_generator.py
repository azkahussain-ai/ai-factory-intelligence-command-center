"""Stage IX — PDF report generation.

Assembles a downloadable report purely from real outputs already produced
by the pipeline (Stage V orchestrator result + an optional Stage VIII HITL
audit record) - no value here is computed or invented; every field is read
from the dict already produced upstream. Uses reportlab, already a project
dependency (see src/rag/build_knowledge_base_pdfs.py for the existing
usage pattern this follows).
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage,
)

from config.config import PROJECT_ROOT

styles = getSampleStyleSheet()
_H1 = ParagraphStyle("H1", parent=styles["Heading1"], spaceAfter=10)
_H2 = ParagraphStyle("H2", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6)
_BODY = styles["BodyText"]


def _kv_table(rows: list[tuple[str, str]]) -> Table:
    t = Table(rows, colWidths=[1.8 * inch, 4.5 * inch])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#444444")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def generate_report(ai_output: dict, hitl_record: dict | None = None) -> bytes:
    """Build a PDF incident/decision report from a real orchestrator result
    (and, if the supervisor has decided, the real HITL audit record).
    Returns the PDF as bytes, ready for a Streamlit download button.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=LETTER,
                             topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    story = []

    context = ai_output.get("factory_context", {})
    machine_id = context.get("machine_id", "unknown")
    story.append(Paragraph("AI Factory Intelligence Command Center", _H1))
    story.append(Paragraph(f"Machine Decision Report — {machine_id}", styles["Heading3"]))
    story.append(Paragraph(f"Generated: {context.get('generated_at', 'n/a')}", _BODY))
    story.append(Spacer(1, 10))

    pm = ai_output.get("predictive_maintenance_result", {})
    story.append(Paragraph("Predictive Maintenance (Stage II model, live inference)", _H2))
    if pm.get("status") == "ok":
        story.append(_kv_table([
            ("Model", pm.get("model_name", "n/a")),
            ("Failure probability", f"{pm.get('failure_probability', 0):.2%}"),
            ("Risk level", pm.get("risk_level", "n/a")),
            ("Top signals", ", ".join(pm.get("important_signals", []))),
        ]))
    else:
        story.append(Paragraph(f"Unavailable: {pm.get('reason', pm.get('status'))}", _BODY))

    vision = ai_output.get("vision_result", {})
    story.append(Paragraph("Vision Inspection (Stage III CNN)", _H2))
    if vision.get("status") == "ok":
        story.append(_kv_table([
            ("Defect detected", str(vision.get("defect_detected"))),
            ("Confidence", f"{vision.get('confidence', 0):.2%}"),
        ]))
    else:
        story.append(Paragraph(f"Unavailable: {vision.get('reason', vision.get('status'))}", _BODY))

    xai = ai_output.get("xai_result", {})
    story.append(Paragraph("Explainable AI (Stage VI)", _H2))
    story.append(Paragraph(xai.get("human_summary", "No explanation available."), _BODY))
    top_features = xai.get("feature_explanations", [])[:5]
    if top_features:
        rows = [("Feature", "Contribution", "Direction")]
        rows += [(f["feature"], f"{f['contribution']:+.4f}", f["direction"]) for f in top_features]
        t = Table(rows, colWidths=[2.5 * inch, 1.5 * inch, 2 * inch])
        t.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ]))
        story.append(Spacer(1, 4))
        story.append(t)
    overlay_path = xai.get("vision_explanation", {}).get("overlay_path")
    if overlay_path and (PROJECT_ROOT / overlay_path).exists():
        story.append(Spacer(1, 6))
        story.append(RLImage(str(PROJECT_ROOT / overlay_path), width=1.4 * inch, height=1.4 * inch))

    rag = ai_output.get("rag_result", {})
    story.append(Paragraph("Knowledge / RAG Evidence (Stage IV)", _H2))
    if rag.get("status") == "ok":
        story.append(Paragraph(f"Q: {rag.get('question', '')}", _BODY))
        story.append(Paragraph(rag.get("answer", "")[:800], _BODY))
        for s in rag.get("sources", [])[:3]:
            story.append(Paragraph(
                f"&bull; {s.get('document')} (p.{s.get('page')}, {s.get('section')})",
                ParagraphStyle("src", parent=_BODY, fontSize=8, textColor=colors.grey)))
    else:
        story.append(Paragraph(f"Unavailable: {rag.get('reason', rag.get('status'))}", _BODY))

    twin = ai_output.get("digital_twin_result", {})
    story.append(Paragraph("Digital Twin What-If Simulation (Stage VII)", _H2))
    if twin.get("status") == "ok":
        rows = [("Scenario", "Net value (USD)", "Failure risk")]
        for s in twin.get("scenarios", []):
            rows.append((s["scenario"], f"${s['estimated_net_value_usd']:,.2f}",
                         f"{s['cumulative_failure_risk']:.2%}"))
        t = Table(rows, colWidths=[2.2 * inch, 2 * inch, 1.8 * inch])
        t.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ]))
        story.append(t)
        story.append(Paragraph(f"Recommended: {twin.get('recommended_scenario')} "
                                f"({twin.get('recommendation_basis')})", _BODY))
    else:
        story.append(Paragraph(f"Unavailable: {twin.get('reason', twin.get('status'))}", _BODY))

    decision = ai_output.get("decision", {})
    story.append(Paragraph("Multi-Agent AI Recommendation (Stage V)", _H2))
    story.append(_kv_table([
        ("Priority", decision.get("priority", "n/a")),
        ("Recommendation", decision.get("recommendation", "n/a")),
    ]))
    for line in decision.get("reasoning", []):
        story.append(Paragraph(f"&bull; {line}", ParagraphStyle("r", parent=_BODY, fontSize=8)))

    story.append(Paragraph("Human Supervisor Decision (Stage VIII)", _H2))
    if hitl_record:
        story.append(_kv_table([
            ("Decision", hitl_record.get("human_decision", "n/a")),
            ("Final action", hitl_record.get("final_action") or "(none — rejected)"),
            ("Comment", hitl_record.get("human_comment") or "-"),
            ("Supervisor", hitl_record.get("supervisor") or "-"),
            ("Timestamp", hitl_record.get("timestamp", "n/a")),
            ("Decision ID", hitl_record.get("decision_id", "n/a")),
        ]))
    else:
        story.append(Paragraph("No human decision has been recorded yet for this analysis.", _BODY))

    doc.build(story)
    return buf.getvalue()
