"""
Module for exporting survival analyses to PDF.
Support Kaplan-Meier, Cox regression, Log-Rank, and Weibull.
"""

import io
import json
import re
from pathlib import Path
import pandas as pd
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import plotly.graph_objects as go
import plotly.io as pio
from translations import get_translation, get_display_text


BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "assets" / "uco_escudo_old.jpg"
LOGO_RIGHT_PATH = BASE_DIR / "assets" / "logo_espc.png"


class SurvivalAnalysisPDFExporter:
    """Generate survival analysis PDF reports."""
    
    def __init__(self, filename, language='en', has_graph=False, report_name=None, header_datetime=None):
        """Initialize exporter settings and prepare the list of PDF elements."""
        self.filename = filename
        self.language = language
        self.has_graph = has_graph  # Track whether landscape orientation is needed.
        self.report_name = (report_name or Path(filename).stem).strip()
        self.header_datetime = header_datetime or datetime.now()
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        self.elements = []
        
    def _setup_custom_styles(self):
        """Configure custom PDF styles using Times New Roman."""
        # Custom title.
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=6,
            spaceBefore=2,
            alignment=1,  # Center
            fontName='Times-Bold',
            leading=28
        ))
        
        # Subtitle.
        self.styles.add(ParagraphStyle(
            name='Subtitle',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=colors.HexColor('#555555'),
            spaceAfter=20,
            spaceBefore=10,
            alignment=1,  # Center
            fontName='Times-Roman',
            leading=16
        ))
        
        # Centered, bold section title.
        self.styles.add(ParagraphStyle(
            name='SectionTitle',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=8,
            spaceBefore=12,
            fontName='Times-Bold',
            bold=True,
            alignment=1,  # Center
            leading=16
        ))
        
        # Body text with indentation and spacing.
        self.styles.add(ParagraphStyle(
            name='NormalText',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=12,
            spaceBefore=6,
            fontName='Times-Roman',
            alignment=4,  # Justified
            leading=16,  # Line spacing.
            firstLineIndent=18,  # 0.25-inch indentation.
            leftIndent=18,
            rightIndent=18
        ))
        
        # Interpretation/conclusion text with formatted paragraphs.
        self.styles.add(ParagraphStyle(
            name='InterpretationText',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            spaceBefore=0,
            fontName='Times-Roman',
            alignment=4,  # Justified
            leading=17,  # Generous line spacing.
            firstLineIndent=18,  # First-line indentation.
            leftIndent=18,  # Left margin.
            rightIndent=18  # Right margin.
        ))

        self.styles.add(ParagraphStyle(
            name='InterpretationHeadingText',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=4,
            spaceBefore=6,
            fontName='Times-Bold',
            alignment=0,
            leading=14,
            leftIndent=18,
            rightIndent=18
        ))

        self.styles.add(ParagraphStyle(
            name='InterpretationMainHeadingText',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=3,
            spaceBefore=7,
            fontName='Times-Bold',
            alignment=0,
            leading=14,
            leftIndent=18,
            rightIndent=18
        ))

        self.styles.add(ParagraphStyle(
            name='InterpretationUnderlineText',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=2,
            spaceBefore=2,
            fontName='Times-Roman',
            alignment=0,
            leading=14,
            leftIndent=18,
            rightIndent=18
        ))
        
        # Analysis description/introduction text.
        self.styles.add(ParagraphStyle(
            name='DescriptionText',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=12,
            spaceBefore=6,
            fontName='Times-Roman',
            alignment=4,  # Justified
            leading=16,
            firstLineIndent=18,
            leftIndent=18,
            rightIndent=18,
            textColor=colors.HexColor('#34495e')
        ))

        self.styles.add(ParagraphStyle(
            name='TableCell',
            parent=self.styles['Normal'],
            fontSize=8,
            leading=9.5,
            alignment=1,
            fontName='Times-Roman',
            textColor=colors.HexColor('#1f1f1f')
        ))

        self.styles.add(ParagraphStyle(
            name='TableCellLeft',
            parent=self.styles['TableCell'],
            alignment=0
        ))

    def _is_english(self):
        """Check whether the report should be generated in English."""
        return self.language == 'en'

    def _format_header_datetime(self):
        """Format the date and time for the top header."""
        return self.header_datetime.strftime('%d/%m/%Y %H:%M')
    
    def _add_header(self, title, subtitle=""):
        """Add the cover/header to the document."""
        self.elements.append(Spacer(1, 0.5*inch))
        
        title_para = Paragraph(title, self.styles['CustomTitle'])
        self.elements.append(title_para)
        
        if subtitle:
            subtitle_para = Paragraph(subtitle, self.styles['NormalText'])
            self.elements.append(subtitle_para)
        
        self.elements.append(Spacer(1, 0.3*inch))

    def _add_paragraph_block(self, text, style_name='InterpretationText'):
        """Add a text block preserving paragraphs and internal line breaks."""
        if not text:
            return

        paragraphs = [part.strip() for part in re.split(r'\n\s*\n+', str(text).strip()) if part.strip()]
        underline_targets = (
            'follow up medio',
            'follow up mediano',
            'nivel de eventos',
            'seguimiento',
            'variable de edad'
        )

        def _clean_text(value):
            """Normalize text for comparing internal headings without symbols."""
            return re.sub(r'[^a-z0-9 ]+', '', value.lower()).strip()

        def _looks_like_subheading(value):
            """Detect whether a paragraph should be displayed as a PDF subheading."""
            cleaned = _clean_text(value)
            if not cleaned:
                return False
            if cleaned.startswith(underline_targets):
                return True
            if re.match(r'^(?:[-•*]|\d+(?:\.\d+)*)\s+', value.strip()):
                return True
            words = cleaned.split()
            return (
                len(words) <= 5 and
                len(cleaned) <= 45 and
                not re.search(r'[.;!?]$', value.strip())
            )

        for index, paragraph_text in enumerate(paragraphs):
            normalized_text = ' '.join(line.strip() for line in paragraph_text.splitlines() if line.strip())
            numbered_heading_match = re.match(r'^(\d+(?:\.\d+)*)\s+([^:]+?):\s*(.*)$', normalized_text)
            if numbered_heading_match:
                number_part = numbered_heading_match.group(1)
                title_part = numbered_heading_match.group(2).strip()
                remainder_part = numbered_heading_match.group(3).strip()
                rendered_parts = [f"<b>{number_part}. {title_part}</b>:"]
                if remainder_part:
                    rendered_parts.append(remainder_part)
                self.elements.append(Paragraph(' '.join(rendered_parts), self.styles['InterpretationText']))
            elif _looks_like_subheading(normalized_text):
                self.elements.append(Paragraph(f"<u>{normalized_text}</u>", self.styles['InterpretationUnderlineText']))
                if index < len(paragraphs) - 1:
                    self.elements.append(Spacer(1, 0.02 * inch))
            else:
                self.elements.append(Paragraph(normalized_text, self.styles[style_name]))
                if index < len(paragraphs) - 1:
                    self.elements.append(Spacer(1, 0.04 * inch))
    
    def _add_section_title(self, title):
        """Add a section title."""
        # Fixed space to visually separate the page header from the title.
        self.elements.append(Spacer(1, 0.14 * inch))
        self.elements.append(Paragraph(title, self.styles['SectionTitle']))
        underline = Table([[""]], colWidths=[7.6 * inch])
        underline.setStyle(TableStyle([
            ('LINEBELOW', (0, 0), (-1, -1), 1, colors.HexColor('#1f77b4')),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ]))
        self.elements.append(underline)
        self.elements.append(Spacer(1, 0.18*inch))
    
    def add_summary_section(self, n_patients, n_events, follow_up_mean, follow_up_median):
        """Add a professionally styled general summary section without overlap."""
        self._add_section_title("1. SUMMARY" if self._is_english() else "1. RESUMEN GENERAL")
        
        summary_data = [
            ["Metric" if self._is_english() else "Métrica", "Value" if self._is_english() else "Valor"],
            ["Number of students" if self._is_english() else "Número de estudiantes", str(n_patients)],
            ["Number of events" if self._is_english() else "Número de eventos", str(n_events)],
            ["Event rate (%)" if self._is_english() else "Tasa de eventos (%)", f"{(n_events/n_patients*100):.1f}%" if n_patients > 0 else "N/A"],
            ["Average follow-up (months)" if self._is_english() else "Follow-up medio (meses)", f"{follow_up_mean:.2f}" if follow_up_mean > 0 else "N/A"],
            ["Median follow-up (months)" if self._is_english() else "Follow-up mediano (meses)", f"{follow_up_median:.2f}" if follow_up_median > 0 else "N/A"]
        ]
        
        table = Table(summary_data, colWidths=[4.5*inch, 2.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f0f0')),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('TOPPADDING', (0, 1), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 10),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f8f8')])
        ]))

        self.elements.append(table)
        self.elements.append(Spacer(1, 0.18*inch))

        event_rate = f"{(n_events/n_patients*100):.1f}%" if n_patients > 0 else "N/A"
        summary_note = (
            f"This page summarizes the academic cohort used in the exported analysis: {n_patients} students, "
            f"{n_events} observed events and an event rate of {event_rate}. Each selected export option is kept "
            "on a separate page to make the report easier to review and share."
            if self._is_english() else
            f"Esta página resume la cohorte académica utilizada en el análisis exportado: {n_patients} estudiantes, "
            f"{n_events} eventos observados y una tasa de eventos del {event_rate}. Cada opción seleccionada se "
            "mantiene en una página independiente para facilitar la revisión y la lectura del informe."
        )
        self.elements.append(Paragraph(summary_note, self.styles['DescriptionText']))
        self.elements.append(Spacer(1, 0.3*inch))
    
    def add_kaplan_meier_section(self, variable_name="", survival_table_data=None):
        """Add the Kaplan-Meier section: 2. RESULTS TABLE."""
        subtitle = f" - {variable_name}" if variable_name else " - General"
        self._add_section_title(("2. RESULTS TABLE" if self._is_english() else "2. TABLA DE RESULTADOS") + subtitle)
        
        # Description.
        intro_text = (
            "Kaplan-Meier analysis is a non-parametric statistical method that estimates the survival function from time-to-event data. This section presents descriptive survival statistics."
            if self._is_english() else
            "El análisis de Kaplan-Meier es un método estadístico no paramétrico que estima la función de supervivencia a partir de datos de tiempo hasta evento. Esta sección presenta las estadísticas descriptivas de supervivencia."
        )
        self.elements.append(Paragraph(intro_text, self.styles['DescriptionText']))
        self.elements.append(Spacer(1, 0.15*inch))
        
        # Survival table, if available.
        if survival_table_data is not None and isinstance(survival_table_data, pd.DataFrame) and len(survival_table_data) > 0:
            self.elements.append(Spacer(1, 0.1*inch))
            
            # Convert the DataFrame to a list of lists for the table.
            table_data = [[get_display_text(self.language, column) for column in survival_table_data.columns]]
            for row in survival_table_data.values:
                table_data.append([get_display_text(self.language, str(val)) for val in row])
            
            # Calculate column widths dynamically.
            num_cols = len(survival_table_data.columns)
            total_width = 8.2*inch
            col_widths = [total_width / num_cols] * num_cols
            
            table = Table(table_data, colWidths=col_widths)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f5f5f5')),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')])
            ]))

            self.elements.append(table)
            self.elements.append(Spacer(1, 0.2*inch))
            print(f"""KM table added: {len(table_data)} rows""")
        else:
            self.elements.append(Paragraph(
                ("Survival Table<br/>" if self._is_english() else "Tabla de Supervivencia<br/>")
                + ("Survival parameters are computed from the data analysis." if self._is_english() else "Los parámetros de supervivencia se calculan a partir del análisis de los datos."),
                self.styles['NormalText']
            ))
        
        self.elements.append(Spacer(1, 0.3*inch))
    
    def add_cox_regression_section(self, coefficients_table=None):
        """Add the Cox regression section: 2. RESULTS TABLE."""
        self._add_section_title("2. RESULTS TABLE" if self._is_english() else "2. TABLA DE RESULTADOS")
        
        # Description.
        intro_text = (
            "Cox Regression is a semiparametric model used to assess the effect of multiple covariates on time-to-event. Model coefficients represent the change in log-hazard per unit change in the covariate. exp(coefficient) is the Hazard Ratio (HR), which interprets relative risk."
            if self._is_english() else
            "La Regresión de Cox es un modelo semiparamétrico que permite evaluar el efecto de múltiples covariables en el tiempo hasta evento. Los coeficientes del modelo representan el cambio en el log-hazard por unidad de cambio en la covariable. El exp(coeficiente) es el Hazard Ratio (HR), que interpreta el riesgo relativo."
        )
        self.elements.append(Paragraph(intro_text, self.styles['DescriptionText']))
        self.elements.append(Spacer(1, 0.15*inch))
        
        # Coefficient table.
        if coefficients_table is not None and isinstance(coefficients_table, pd.DataFrame) and len(coefficients_table) > 0:
            self.elements.append(Spacer(1, 0.1*inch))
            
            # Convert the DataFrame to a list of lists for the table.
            table_data = [[get_display_text(self.language, column) for column in coefficients_table.columns]]
            for row in coefficients_table.values:
                # Round numbers to take up less space.
                row_str = []
                for val in row:
                    if isinstance(val, float):
                        row_str.append(f"{val:.4f}")
                    else:
                        row_str.append(get_display_text(self.language, str(val)))
                table_data.append(row_str)
            
            # Calculate suitable column widths.
            num_cols = len(coefficients_table.columns)
            # Total available width.
            total_width = 8.2*inch
            # Distribute widths proportionally, widening the first column for names.
            col_widths = [2.5*inch] + [(total_width - 2.5*inch) / (num_cols - 1)] * (num_cols - 1) if num_cols > 1 else [total_width]
            
            table = Table(table_data, colWidths=col_widths)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f9f9f9')),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')])
            ]))

            self.elements.append(table)
            self.elements.append(Spacer(1, 0.2*inch))
            print(f"""Cox table added: {len(table_data)} rows""")
        else:
            self.elements.append(Paragraph(
                ("Cox Model Coefficients<br/>" if self._is_english() else "Coeficientes del Modelo Cox<br/>")
                + ("Coefficients are estimated by partial maximum likelihood." if self._is_english() else "Los coeficientes se estiman mediante máxima verosimilitud parcial."),
                self.styles['NormalText']
            ))
        
        self.elements.append(Spacer(1, 0.3*inch))
    
    def add_log_rank_section(self, test_results=None):
        """Add the Log-Rank test section: 2. RESULTS TABLE."""
        self._add_section_title("2. RESULTS TABLE" if self._is_english() else "2. TABLA DE RESULTADOS")
        
        # Description.
        intro_text = (
            "The Log-Rank Test is a non-parametric statistical test that compares survival curves between two or more groups. The null hypothesis states that there is no difference in survival functions between groups. A p-value < 0.05 indicates significant differences."
            if self._is_english() else
            "El Test de Log-Rank es una prueba estadística no paramétrica que compara las curvas de supervivencia entre dos o más grupos. La hipótesis nula es que no hay diferencia en las funciones de supervivencia entre grupos. Un p-valor < 0.05 indica diferencias significativas."
        )
        self.elements.append(Paragraph(intro_text, self.styles['DescriptionText']))
        self.elements.append(Spacer(1, 0.15*inch))
        
        if test_results is not None:
            # Dictionary input.
            if isinstance(test_results, dict):
                test_data = [
                    ["Statistic" if self._is_english() else "Estadístico", "Value" if self._is_english() else "Valor"],
                    ["Chi-square" if self._is_english() else "Chi-cuadrado", f"{test_results.get('test_statistic', 'N/A'):.4f}"],
                    ["p-value" if self._is_english() else "p-valor", f"{test_results.get('p_value', 'N/A'):.4f}"],
                    ["Degrees of freedom" if self._is_english() else "Grados de libertad", str(test_results.get('df', 'N/A'))]
                ]
                
                table = Table(test_data, colWidths=[4.4*inch, 3.8*inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27ae60')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
                    ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
                    ('FONTSIZE', (0, 0), (-1, 0), 11),
                    ('FONTSIZE', (0, 1), (-1, -1), 10),
                    ('TOPPADDING', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                    ('LEFTPADDING', (0, 0), (-1, -1), 8),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f8f0')),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f8f0')])
                ]))

                self.elements.append(table)
                self.elements.append(Spacer(1, 0.2*inch))
                print(f"""Log-Rank table added (dict)""")
                
                # Interpretation.
                p_val = test_results.get('p_value', 1)
                if p_val < 0.05:
                    interpretation = "There is a SIGNIFICANT difference between groups (p < 0.05). The null hypothesis is rejected."if self._is_english() else "Hay diferencia SIGNIFICATIVA entre grupos (p < 0.05). Se rechaza la hipótesis nula."
                    color = '#27ae60'
                else:
                    interpretation = "There is NO significant difference between groups (p ≥ 0.05). The null hypothesis is not rejected."if self._is_english() else "NO hay diferencia significativa entre grupos (p ≥ 0.05). No se rechaza la hipótesis nula."
                    color = '#e67e22'
                
                interp_label = "Interpretation" if self._is_english() else "Interpretación"
                interp_para = Paragraph(
                    f"<font color='{color}'><b>{interp_label}: {interpretation}</b></font>",
                    self.styles['NormalText']
                )
                self.elements.append(interp_para)
            
            # DataFrame input.
            elif isinstance(test_results, pd.DataFrame) and len(test_results) > 0:
                # Convert the DataFrame to a list of lists for the table.
                table_data = [[get_display_text(self.language, column) for column in test_results.columns]]
                for row in test_results.values:
                    # Round numbers to take up less space.
                    row_str = []
                    for val in row:
                        if isinstance(val, float):
                            row_str.append(f"{val:.4f}")
                        else:
                            row_str.append(get_display_text(self.language, str(val))[:16 if self._is_english() else 15])  # Keep the English H0 decision complete.
                    table_data.append(row_str)
                
                # Calculate column widths with improved distribution.
                num_cols = len(test_results.columns)
                total_width = 8.2*inch
                # Widen the first column for variable names.
                col_widths = [2.5*inch] + [(total_width - 2.5*inch) / (num_cols - 1)] * (num_cols - 1) if num_cols > 1 else [total_width]
                
                # Create the table.
                table = Table(table_data, colWidths=col_widths, repeatRows=1)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27ae60')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
                    ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
                    ('FONTSIZE', (0, 0), (-1, 0), 8),
                    ('FONTSIZE', (0, 1), (-1, -1), 7),
                    ('TOPPADDING', (0, 0), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('LEFTPADDING', (0, 0), (-1, -1), 4),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f8f0')),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f8f0')])
                ]))

                self.elements.append(table)
                self.elements.append(Spacer(1, 0.2*inch))
                print(f"""Log-Rank table added: {len(table_data)} rows x {num_cols} cols""")
        else:
            self.elements.append(Paragraph(
                ("Log-Rank Test Results<br/>" if self._is_english() else "Resultados del Test de Log-Rank<br/>")
                + ("The Log-Rank Test results are shown in the table above." if self._is_english() else "Los resultados del Test de Log-Rank se presentan en la tabla anterior."),
                self.styles['NormalText']
            ))
        
        self.elements.append(Spacer(1, 0.3*inch))

    def add_weibull_section(self, weibull_table=None, section_title=None):
        """Add the Weibull section: 2. RESULTS TABLE."""
        self._add_section_title(section_title or ("2. RESULTS TABLE" if self._is_english() else "2. TABLA DE RESULTADOS"))

        intro_text = (
            "The Weibull model is a parametric survival model that estimates how the risk changes over time using a shape and a scale parameter. The table below summarizes the fitted model and its main statistics, and the graph compares the fitted curve against the empirical Kaplan-Meier survival curve."
            if self._is_english() else
            "El modelo Weibull es un modelo paramétrico de supervivencia que estima cómo cambia el riesgo con el tiempo usando un parámetro de forma y otro de escala. La tabla siguiente resume el ajuste y sus estadísticas principales, y la gráfica compara la curva ajustada con la curva empírica de Kaplan-Meier."
        )
        self.elements.append(Paragraph(intro_text, self.styles['DescriptionText']))
        self.elements.append(Spacer(1, 0.15*inch))

        if weibull_table is not None and isinstance(weibull_table, pd.DataFrame) and len(weibull_table) > 0:
            self.elements.append(Spacer(1, 0.1*inch))

            table_data = [[get_display_text(self.language, column) for column in weibull_table.columns]]
            for row in weibull_table.values:
                row_str = []
                for val in row:
                    if isinstance(val, float):
                        row_str.append(f"{val:.4f}")
                    else:
                        row_str.append(get_display_text(self.language, str(val)))
                table_data.append(row_str)

            num_cols = len(weibull_table.columns)
            total_width = 8.2 * inch
            col_widths = [2.5 * inch] + [(total_width - 2.5 * inch) / (num_cols - 1)] * (num_cols - 1) if num_cols > 1 else [total_width]

            table = Table(table_data, colWidths=col_widths)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#fdf3f2')),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fdf3f2')])
            ]))

            self.elements.append(table)
            self.elements.append(Spacer(1, 0.2 * inch))
            print(f"""Weibull table added: {len(table_data)} rows""")
        else:
            self.elements.append(Paragraph(
                ("Weibull Fit Summary<br/>" if self._is_english() else "Resumen del Ajuste Weibull<br/>")
                + ("The Weibull fit is summarized in the table above." if self._is_english() else "El ajuste Weibull se resume en la tabla anterior."),
                self.styles['NormalText']
            ))

        self.elements.append(Spacer(1, 0.3 * inch))

    def add_exponential_section(self, exponential_table=None, section_title=None):
        """Add the Exponential section: 2. RESULTS TABLE."""
        self._add_section_title(section_title or ("2. RESULTS TABLE" if self._is_english() else "2. TABLA DE RESULTADOS"))

        intro_text = (
            "The Exponential model is a parametric survival model with constant hazard over time. It is a special case of Weibull when the shape parameter k = 1. The table below summarizes the fitted model and its main statistics, and the graph compares the fitted curve against the empirical Kaplan-Meier survival curve."
            if self._is_english() else
            "El modelo Exponencial es un modelo paramétrico de supervivencia con hazard constante en el tiempo. Es un caso particular del Weibull cuando el parámetro de forma k = 1. La tabla siguiente resume el ajuste y sus estadísticas principales, y la gráfica compara la curva ajustada con la curva empírica de Kaplan-Meier."
        )
        self.elements.append(Paragraph(intro_text, self.styles['DescriptionText']))
        self.elements.append(Spacer(1, 0.15 * inch))

        if exponential_table is not None and isinstance(exponential_table, pd.DataFrame) and len(exponential_table) > 0:
            self.elements.append(Spacer(1, 0.1 * inch))

            def _cell(value, header=False, left=False):
                """Format a table cell with the appropriate style."""
                style = self.styles['TableCellLeft'] if left else self.styles['TableCell']
                text = str(value).replace("\n", "<br/>")
                if header:
                    text = f"<b>{text}</b>"
                return Paragraph(text, style)

            table_data = [[_cell(get_display_text(self.language, column), header=True, left=index == 0) for index, column in enumerate(exponential_table.columns)]]
            for row in exponential_table.values:
                row_str = []
                for index, val in enumerate(row):
                    if isinstance(val, float):
                        cell_value = f"{val:.4f}"
                    else:
                        cell_value = get_display_text(self.language, str(val))
                    row_str.append(_cell(cell_value, left=index == 0))
                table_data.append(row_str)

            num_cols = len(exponential_table.columns)
            total_width = 8.0 * inch
            if num_cols == 3:
                col_widths = [2.05 * inch, 1.25 * inch, 4.70 * inch]
            elif num_cols > 1:
                col_widths = [2.2 * inch] + [(total_width - 2.2 * inch) / (num_cols - 1)] * (num_cols - 1)
            else:
                col_widths = [total_width]

            table = Table(table_data, colWidths=col_widths, repeatRows=1)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e67e22')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
                ('FONTSIZE', (0, 0), (-1, 0), 8.5),
                ('FONTSIZE', (0, 1), (-1, -1), 7.6),
                ('LEADING', (0, 0), (-1, -1), 8.5),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#fef5ed')),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fef5ed')])
            ]))

            self.elements.append(table)
            self.elements.append(Spacer(1, 0.2 * inch))
            print(f"""Exponential table added: {len(table_data)} rows""")
        else:
            self.elements.append(Paragraph(
                ("Exponential Fit Summary<br/>" if self._is_english() else "Resumen del Ajuste Exponencial<br/>")
                + ("The Exponential fit is summarized in the table above." if self._is_english() else "El ajuste Exponencial se resume en la tabla anterior."),
                self.styles['NormalText']
            ))

        self.elements.append(Spacer(1, 0.3 * inch))

    def add_rsf_section(self, rsf_table=None):
        """Add the Random Survival Forest section: 2. RESULTS TABLE."""
        self._add_section_title("2. RESULTS TABLE" if self._is_english() else "2. TABLA DE RESULTADOS")

        intro_text = (
            "Random Survival Forest is an ensemble survival model that builds many decision trees from bootstrap samples and combines their predictions to estimate survival and risk. The table below summarizes the fitted model, its concordance, and the most influential variable."
            if self._is_english() else
            "Random Survival Forest es un modelo de supervivencia en ensamblado que construye muchos árboles de decisión a partir de muestras bootstrap y combina sus predicciones para estimar la supervivencia y el riesgo. La tabla siguiente resume el modelo ajustado, su concordancia y la variable más influyente."
        )
        self.elements.append(Paragraph(intro_text, self.styles['DescriptionText']))
        self.elements.append(Spacer(1, 0.15 * inch))

        if rsf_table is not None and isinstance(rsf_table, pd.DataFrame) and len(rsf_table) > 0:
            self.elements.append(Spacer(1, 0.1 * inch))

            table_data = [[get_display_text(self.language, column) for column in rsf_table.columns]]
            for row in rsf_table.values:
                row_str = []
                for val in row:
                    if isinstance(val, float):
                        row_str.append(f"{val:.4f}")
                    else:
                        row_str.append(get_display_text(self.language, str(val)))
                table_data.append(row_str)

            num_cols = len(rsf_table.columns)
            total_width = 8.2 * inch
            col_widths = [2.5 * inch] + [(total_width - 2.5 * inch) / (num_cols - 1)] * (num_cols - 1) if num_cols > 1 else [total_width]

            table = Table(table_data, colWidths=col_widths, repeatRows=1)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2980b9')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f5f9fc')),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f9fc')])
            ]))

            self.elements.append(table)
            self.elements.append(Spacer(1, 0.2 * inch))
            print(f"""RSF table added: {len(table_data)} rows""")
        else:
            self.elements.append(Paragraph(
                ("RSF Model Summary<br/>" if self._is_english() else "Resumen del modelo RSF<br/>")
                + ("The RSF summary is shown in the table above." if self._is_english() else "El resumen RSF se muestra en la tabla anterior."),
                self.styles['NormalText']
            ))

        self.elements.append(Spacer(1, 0.3 * inch))
    
    def add_conclusions_section(self, conclusions_text=None):
        """Add the conclusions section: 4. CONCLUSION."""
        self._add_section_title("4. CONCLUSION" if self._is_english() else "4. CONCLUSIÓN")
        
        if conclusions_text:
            self._add_paragraph_block(conclusions_text, 'InterpretationText')
        else:
            default_conclusion = (
                "This report presents a complete survival analysis of the provided data. The results are based on robust statistical methods. Specialist review is recommended for clinical interpretation."
                if self._is_english() else
                "Este informe presenta un análisis completo de supervivencia de los datos proporcionados. Los resultados se basan en métodos estadísticos robustos. Se recomienda la consulta con especialistas para la interpretación clínica de los resultados."
            )
            self._add_paragraph_block(default_conclusion, 'InterpretationText')
        
        self.elements.append(Spacer(1, 0.3*inch))
    
    def add_graph_section(self, title="3. GRÁFICA"):
        """Start the chart section."""
        self._add_section_title(title)
        self.elements.append(Spacer(1, 0.2*inch))
    
    def _plotly_to_image(self, fig):
        """Convert a Plotly figure to a high-resolution image (BytesIO)."""
        try:
            # Some callbacks return a complete dcc.Graph. Extract its
            # internal figure to export the same chart the user sees.
            if hasattr(fig, 'figure'):
                fig = fig.figure

            if isinstance(fig, dict):
                fig = go.Figure(fig)

            img_bytes = pio.to_image(fig, format='png', width=1400, height=700, scale=2)
            return io.BytesIO(img_bytes)
        except Exception as e:
            print(f"""Error converting chart: {str(e)}""")
            return None
    
    def add_plotly_figure(self, fig, caption=""):
        """Add a large Plotly figure to the PDF in landscape orientation."""
        if fig is None:
            return
        
        try:
            # Some callbacks return a complete dcc.Graph. Extract its
            # internal figure to export the same chart the user sees.
            if hasattr(fig, 'figure'):
                fig = fig.figure

            # Convert a JSON figure string to a Plotly object.
            if isinstance(fig, str):
                try:
                    fig_dict = json.loads(fig)
                    fig = go.Figure(fig_dict)
                except:
                    print(f"""Could not convert JSON to a Plotly figure""")
                    return
            if isinstance(fig, dict):
                fig = go.Figure(fig)
            
            # Convert with LARGE dimensions for better resolution.
            # 1400x700 (2:1 aspect ratio) for good resolution.
            img_bytes = pio.to_image(fig, format='png', width=1400, height=700, scale=2)
            img_io = io.BytesIO(img_bytes)
            
            # PDF size according to orientation.
            if self.has_graph:
                # LANDSCAPE: much more space available.
                # With 0.3-inch margins on each side, available width is approximately 9.4 inches.
                # Use 9.0 x 4.5 inches for maximum visibility.
                img = Image(img_io, width=9.0*inch, height=4.5*inch)
            else:
                # PORTRAIT: standard space.
                img = Image(img_io, width=8.0*inch, height=4.0*inch)
            
            # Add with minimal spacing.
            self.elements.append(Spacer(1, 0.1*inch))
            self.elements.append(img)
            self.elements.append(Spacer(1, 0.15*inch))
            
            if caption:
                caption_para = Paragraph(f"<i><b>{caption}</b></i>", self.styles['NormalText'])
                self.elements.append(caption_para)
                self.elements.append(Spacer(1, 0.1*inch))
            
            print(f"""Plotly figure added to PDF in LANDSCAPE (9.0" x 4.5")""")
        except Exception as e:
            print(f"""Error adding Plotly figure: {str(e)}""")
    
    def generate(self, title="SURVIVAL ANALYSIS REPORT"):
        """Generate the final PDF in landscape orientation if charts are present."""
        # Select orientation based on whether charts are present.
        if self.has_graph:
            pagesize = landscape(letter)
            # Landscape margins.
            right_margin = 0.3*inch
            left_margin = 0.3*inch
            top_margin = 0.5*inch
            bottom_margin = 0.5*inch
        else:
            pagesize = letter
            # Portrait margins.
            right_margin = 0.35*inch
            left_margin = 0.35*inch
            top_margin = 0.6*inch
            bottom_margin = 0.6*inch
        
        doc = SimpleDocTemplate(
            self.filename,
            pagesize=pagesize,
            rightMargin=right_margin,
            leftMargin=left_margin,
            topMargin=top_margin,
            bottomMargin=bottom_margin
        )
        
        # Build the PDF with the existing self.elements.
        def _draw_page_decorations(canvas_obj, doc_obj):
            """Draw logos, header, top line, and footer on every page."""
            canvas_obj.saveState()
            page_width, page_height = doc_obj.pagesize

            # Top line.
            canvas_obj.setStrokeColor(colors.HexColor('#d9e5f3'))
            canvas_obj.setLineWidth(0.75)
            canvas_obj.line(doc_obj.leftMargin, page_height - 0.66 * inch, page_width - doc_obj.rightMargin, page_height - 0.66 * inch)

            # Logos.
            logo_width = 0.48 * inch
            logo_height = 0.48 * inch
            if LOGO_PATH.exists():
                try:
                    canvas_obj.drawImage(
                        ImageReader(str(LOGO_PATH)),
                        doc_obj.leftMargin,
                        page_height - 0.58 * inch,
                        width=logo_width,
                        height=logo_height,
                        preserveAspectRatio=True,
                        mask='auto'
                    )
                except Exception:
                    pass

            if LOGO_RIGHT_PATH.exists():
                try:
                    canvas_obj.drawImage(
                        ImageReader(str(LOGO_RIGHT_PATH)),
                        page_width - doc_obj.rightMargin - logo_width,
                        page_height - 0.58 * inch,
                        width=logo_width,
                        height=logo_height,
                        preserveAspectRatio=True,
                        mask='auto'
                    )
                except Exception:
                    pass

            # Text between logos on the first page only: "name: date time".
            if canvas_obj.getPageNumber() == 1:
                canvas_obj.setFont('Times-Bold', 10.5)
                canvas_obj.setFillColor(colors.HexColor('#1f77b4'))
                header_text = f"{self.report_name}: {self._format_header_datetime()}"
                canvas_obj.drawCentredString(page_width / 2, page_height - 0.28 * inch, header_text)

            # Footer.
            canvas_obj.setStrokeColor(colors.HexColor('#d9e5f3'))
            canvas_obj.setLineWidth(0.8)
            canvas_obj.line(doc_obj.leftMargin, 0.55 * inch, page_width - doc_obj.rightMargin, 0.55 * inch)
            canvas_obj.setFont('Times-Roman', 9)
            canvas_obj.setFillColor(colors.HexColor('#666666'))
            footer_label = "Generated" if self._is_english() else "Generado"
            page_label = "Page" if self._is_english() else "Página"
            canvas_obj.drawString(doc_obj.leftMargin, 0.34 * inch, f"{footer_label}: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
            canvas_obj.drawRightString(page_width - doc_obj.rightMargin, 0.34 * inch, f"{page_label} {canvas_obj.getPageNumber()}")

            canvas_obj.restoreState()

        doc.build(self.elements, onFirstPage=_draw_page_decorations, onLaterPages=_draw_page_decorations)
        print(f"""[OK] PDF generated successfully: {self.filename}""")


# Utility function for quick export.
def export_survival_analysis_to_pdf(
    filename,
    title="SURVIVAL ANALYSIS REPORT",
    report_name=None,
    header_datetime=None,
    include_summary=False,
    include_km=False,
    km_figure=None,
    km_table=None,
    include_cox=False,
    cox_table=None,
    forest_figure=None,
    include_logrank=False,
    logrank_figure=None,
    logrank_results=None,
    include_weibull=False,
    weibull_figure=None,
    weibull_table=None,
    include_exponential=False,
    exponential_figure=None,
    exponential_table=None,
    include_rsf=False,
    rsf_figure=None,
    rsf_importance_figure=None,
    rsf_table=None,
    include_rsf_profile=False,
    rsf_profile_figure=None,
    rsf_profile_text="",
    include_ai_interpretation=False,
    ai_text="",
    summary_stats=None,
    language='en',
    landscape_tables=None,
    include_graph=None
):
    """
    Helper for exporting an analysis to PDF.
    Include only selected content (include_summary, include_km, include_cox,
    include_ai_interpretation).

    Args:
        include_summary: Whether to show the general summary section.
        include_km: Whether to show Kaplan-Meier analysis.
        include_cox: Whether to show Cox regression analysis.
        include_ai_interpretation: Whether to show the AI interpretation.
        landscape_tables: Table names requiring landscape orientation,
            e.g. ['logrank', 'cox'].
        include_graph: Force orientation (None=auto-detect, True/False=explicit).
    """
    # Detect charts to use landscape orientation.
    if include_graph is None:
        # Auto-detect: landscape if ANY figure exists.
        has_graph = (include_km and km_figure is not None) or \
                    (include_cox and forest_figure is not None) or \
                    (include_logrank and logrank_figure is not None) or \
                    (include_weibull and weibull_figure is not None) or \
                    (include_exponential and exponential_figure is not None) or \
                    (include_rsf and (rsf_figure is not None or rsf_importance_figure is not None)) or \
                    (include_rsf_profile and rsf_profile_figure is not None)
    else:
        # Use the explicit value (True = force landscape, False = force portrait).
        has_graph = include_graph
    
    exporter = SurvivalAnalysisPDFExporter(
        filename,
        language,
        has_graph=has_graph,
        report_name=report_name,
        header_datetime=header_datetime
    )
    if landscape_tables is None:
        landscape_tables = []
    
    # SIMPLE HEADER (title and date only), MINIMAL SPACING.
    exporter.elements.append(Spacer(1, 0.18*inch))
    title_para = Paragraph(title, exporter.styles['CustomTitle'])
    exporter.elements.append(title_para)
    exporter.elements.append(Spacer(1, 0.16*inch))
    
    print(f"""[PDF BUILDER] Initial elements (header): {len(exporter.elements)}""")
    
    # CONDITIONAL STRUCTURE: Add only selected content.
    numero_seccion = 1
    primera_seccion = True

    def _start_new_section():
        """Insert a page break before a new section when appropriate."""
        nonlocal primera_seccion
        if primera_seccion:
            primera_seccion = False
            return
        exporter.elements.append(PageBreak())
    
    # 1. GENERAL SUMMARY.
    if summary_stats:
        _start_new_section()
        exporter.add_summary_section(
            summary_stats.get('n_patients', 0),
            summary_stats.get('n_events', 0),
            summary_stats.get('follow_up_mean', 0),
            summary_stats.get('follow_up_median', 0)
        )
        numero_seccion += 1
    
    # 2. RESULTS TABLE (omitted for Kaplan-Meier).
    tabla_agregada = False
    # Omit the KM table entirely; it is not offered in the user options.
    if include_cox:
        _start_new_section()
        exporter.add_cox_regression_section(cox_table)
        tabla_agregada = True
    elif include_logrank:
        _start_new_section()
        exporter.add_log_rank_section(logrank_results)
        tabla_agregada = True
    elif include_weibull:
        _start_new_section()
        exporter.add_weibull_section(weibull_table)
        tabla_agregada = True
    elif include_exponential:
        _start_new_section()
        exporter.add_exponential_section(exponential_table)
        tabla_agregada = True
    elif include_rsf:
        _start_new_section()
        exporter.add_rsf_section(rsf_table)
        tabla_agregada = True
    
    if tabla_agregada:
        numero_seccion += 1
    
    # 3. CHART.
    grafica_agregada = False
    if include_km and km_figure is not None:
        _start_new_section()
        exporter.add_graph_section(f"{numero_seccion}. GRAPH" if exporter._is_english() else f"{numero_seccion}. GRÁFICA")
        exporter.add_plotly_figure(km_figure, "Kaplan-Meier survival curve" if language == 'en' else "Curva de supervivencia Kaplan-Meier")
        grafica_agregada = True
    elif include_cox and forest_figure is not None:
        _start_new_section()
        exporter.add_graph_section(f"{numero_seccion}. GRAPH" if exporter._is_english() else f"{numero_seccion}. GRÁFICA")
        exporter.add_plotly_figure(forest_figure, "Hazard Ratio forest plot" if language == 'en' else "Forest plot de Hazard Ratios")
        grafica_agregada = True
    elif include_logrank and logrank_figure is not None:
        _start_new_section()
        exporter.add_graph_section(f"{numero_seccion}. GRAPH" if exporter._is_english() else f"{numero_seccion}. GRÁFICA")
        exporter.add_plotly_figure(logrank_figure, "Survival curves by groups (Log-Rank Test)" if language == 'en' else "Curvas de supervivencia por grupos (Log-Rank Test)")
        grafica_agregada = True
    elif include_weibull and weibull_figure is not None:
        _start_new_section()
        exporter.add_graph_section(f"{numero_seccion}. GRAPH" if exporter._is_english() else f"{numero_seccion}. GRÁFICA")
        exporter.add_plotly_figure(weibull_figure, "Weibull fitted curve" if language == 'en' else "Curva ajustada Weibull")
        grafica_agregada = True
    elif include_exponential and exponential_figure is not None:
        _start_new_section()
        exporter.add_graph_section(f"{numero_seccion}. GRAPH" if exporter._is_english() else f"{numero_seccion}. GRÁFICA")
        exporter.add_plotly_figure(exponential_figure, "Exponential fitted curve" if language == 'en' else "Curva ajustada Exponencial")
        grafica_agregada = True
    elif include_rsf and rsf_figure is not None:
        _start_new_section()
        exporter.add_graph_section(f"{numero_seccion}. GRAPH" if exporter._is_english() else f"{numero_seccion}. GRÁFICA")
        exporter.add_plotly_figure(rsf_figure, "Random Survival Forest fitted curves" if language == 'en' else "Curvas estimadas del Random Survival Forest")
        grafica_agregada = True
        if rsf_importance_figure is not None:
            numero_seccion += 1
            _start_new_section()
            exporter.add_graph_section(f"{numero_seccion}. GRAPH" if exporter._is_english() else f"{numero_seccion}. GRÁFICA")
            exporter.add_plotly_figure(rsf_importance_figure, "Variable importance" if language == 'en' else "Importancia de variables")
            grafica_agregada = True
    elif include_rsf and rsf_importance_figure is not None:
        _start_new_section()
        exporter.add_graph_section(f"{numero_seccion}. GRAPH" if exporter._is_english() else f"{numero_seccion}. GRÁFICA")
        exporter.add_plotly_figure(rsf_importance_figure, "Variable importance" if language == 'en' else "Importancia de variables")
        grafica_agregada = True

    if include_rsf_profile and rsf_profile_figure is not None:
        if grafica_agregada:
            numero_seccion += 1
        _start_new_section()
        exporter.add_graph_section(f"{numero_seccion}. GRAPH" if exporter._is_english() else f"{numero_seccion}. GRÁFICA")
        exporter.add_plotly_figure(
            rsf_profile_figure,
            "Simulated individual profile" if language == 'en' else "Perfil individual simulado"
        )
        if rsf_profile_text:
            exporter._add_paragraph_block(rsf_profile_text, 'InterpretationText')
        grafica_agregada = True
    
    if grafica_agregada:
        numero_seccion += 1
    
    # 4. CONCLUSION: ONLY if selected in the options.
    if include_ai_interpretation and ai_text:
        _start_new_section()
        exporter.elements.append(Spacer(1, 0.14 * inch))
        exporter._add_section_title(f"{numero_seccion}. CONCLUSION" if exporter._is_english() else f"{numero_seccion}. CONCLUSIÓN")
        exporter._add_paragraph_block(ai_text, 'InterpretationText')
    
    print(f"""[PDF BUILDER] Elements before generate(): {len(exporter.elements)}""")
    
    # Generate the PDF.
    exporter.generate(title)


def export_weibull_exponential_combined_pdf(
    filename,
    title="WEIBULL + EXPONENTIAL REPORT: COMBINED",
    report_name=None,
    header_datetime=None,
    include_summary=True,
    include_table=True,
    include_graph=True,
    include_ai_interpretation=False,
    include_weibull=True,
    include_exponential=True,
    weibull_table=None,
    weibull_figure=None,
    exponential_table=None,
    exponential_figure=None,
    ai_text="",
    summary_stats=None,
    language='en'
):
    """Generate a combined report with Weibull and Exponential sections."""
    has_graph = include_graph and (
        (include_weibull and weibull_figure is not None) or
        (include_exponential and exponential_figure is not None)
    )

    exporter = SurvivalAnalysisPDFExporter(
        filename,
        language,
        has_graph=has_graph,
        report_name=report_name,
        header_datetime=header_datetime
    )

    exporter.elements.append(Spacer(1, 0.18 * inch))
    exporter.elements.append(Paragraph(title, exporter.styles['CustomTitle']))
    exporter.elements.append(Spacer(1, 0.16 * inch))

    section_number = 1
    first_section = True

    def _start_new_section():
        """Insert a page break between combined report sections."""
        nonlocal first_section
        if first_section:
            first_section = False
            return
        exporter.elements.append(PageBreak())

    if include_summary and summary_stats:
        _start_new_section()
        exporter.add_summary_section(
            summary_stats.get('n_patients', 0),
            summary_stats.get('n_events', 0),
            summary_stats.get('follow_up_mean', 0),
            summary_stats.get('follow_up_median', 0)
        )

        # Fixed text explaining the methodological relationship between the two models.
        relation_text = (
            "Methodological note: the Exponential model is a special case of the Weibull model when the shape parameter is k = 1. "
            "Therefore, both techniques are directly comparable: if Weibull estimates k close to 1, Exponential can be considered an adequate simplification; "
            "if k differs clearly from 1, Weibull usually offers a more flexible representation of the risk pattern over time."
            if exporter._is_english() else
            "Nota metodológica: el modelo Exponencial es un caso particular del modelo Weibull cuando el parámetro de forma es k = 1. "
            "Por ello, ambas técnicas son directamente comparables: si Weibull estima un k cercano a 1, Exponencial puede considerarse una simplificación adecuada; "
            "si k se aleja claramente de 1, Weibull suele ofrecer una representación más flexible del patrón de riesgo en el tiempo."
        )
        exporter.elements.append(Paragraph(relation_text, exporter.styles['DescriptionText']))
        exporter.elements.append(Spacer(1, 0.12 * inch))

        section_number += 1

    if include_weibull:
        if include_table:
            _start_new_section()
            exporter.add_weibull_section(
                weibull_table,
                section_title=(
                    f"{section_number}. WEIBULL RESULTS" if exporter._is_english() else f"{section_number}. RESULTADOS WEIBULL"
                )
            )
            section_number += 1

        if include_graph and weibull_figure is not None:
            _start_new_section()
            exporter.add_graph_section(
                f"{section_number}. WEIBULL GRAPH" if exporter._is_english() else f"{section_number}. GRÁFICA WEIBULL"
            )
            exporter.add_plotly_figure(
                weibull_figure,
                "Weibull fitted curve" if exporter._is_english() else "Curva ajustada Weibull"
            )
            section_number += 1

    if include_exponential:
        if include_table:
            _start_new_section()
            exporter.add_exponential_section(
                exponential_table,
                section_title=(
                    f"{section_number}. EXPONENTIAL RESULTS" if exporter._is_english() else f"{section_number}. RESULTADOS EXPONENCIAL"
                )
            )
            section_number += 1

        if include_graph and exponential_figure is not None:
            _start_new_section()
            exporter.add_graph_section(
                f"{section_number}. EXPONENTIAL GRAPH" if exporter._is_english() else f"{section_number}. GRÁFICA EXPONENCIAL"
            )
            exporter.add_plotly_figure(
                exponential_figure,
                "Exponential fitted curve" if exporter._is_english() else "Curva ajustada Exponencial"
            )
            section_number += 1

    if include_ai_interpretation and ai_text:
        _start_new_section()
        exporter.elements.append(Spacer(1, 0.14 * inch))
        exporter._add_section_title(
            f"{section_number}. CONCLUSION" if exporter._is_english() else f"{section_number}. CONCLUSIÓN"
        )
        exporter._add_paragraph_block(ai_text, 'InterpretationText')

    exporter.generate(title)
