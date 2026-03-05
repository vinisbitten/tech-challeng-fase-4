from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm


def generate_pdf_report(analysis: dict, output_path: str) -> str:
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle("CustomTitle", parent=styles["Title"], fontSize=16)
    story.append(Paragraph("Relatorio de Analise de Video Clinico", title_style))
    story.append(Paragraph("Sistema de Monitoramento - Saude da Mulher", styles["Normal"]))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph(f"<b>Arquivo:</b> {analysis.get('video', 'N/A')}", styles["Normal"]))
    story.append(Paragraph(f"<b>Analisado em:</b> {analysis.get('analyzed_at', 'N/A')}", styles["Normal"]))
    story.append(Paragraph(f"<b>Duracao:</b> {analysis.get('duration_seconds', 0)}s", styles["Normal"]))
    story.append(Paragraph(f"<b>Nivel de Risco:</b> {analysis.get('risk_level', 'N/A')}", styles["Normal"]))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("<b>Deteccoes por Categoria</b>", styles["Heading2"]))
    det_data = [["Categoria", "Ocorrencias"]] + [
        [k, str(v)] for k, v in analysis.get("detections_by_class", {}).items()
    ]
    if len(det_data) > 1:
        t = Table(det_data, colWidths=[10 * cm, 5 * cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4A90D9")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ]))
        story.append(t)
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("<b>Alertas Detectados</b>", styles["Heading2"]))
    alerts = analysis.get("alerts", [])
    if alerts:
        alert_data = [["Timestamp", "Tipo", "Severidade", "Confianca"]] + [
            [f"{a['timestamp']}s", a["type"], a["severity"], f"{a['confidence']*100:.1f}%"]
            for a in alerts
        ]
        t = Table(alert_data, colWidths=[3 * cm, 6 * cm, 4 * cm, 3 * cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9534F")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(t)
    else:
        story.append(Paragraph("Nenhum alerta critico detectado.", styles["Normal"]))

    doc.build(story)
    return output_path


# Alias de compatibilidade
generate_report = generate_pdf_report
