"""
Callbacks for exporting analyses to PDF.
Generate professional PDFs using actual analysis data.
"""

from dash import callback_context, dcc, html, Output, Input, State
from dash.exceptions import PreventUpdate
import json
import os
from datetime import datetime
from pdf_exporter import export_survival_analysis_to_pdf, export_weibull_exponential_combined_pdf
from ollama_AI import generate_interpretation_for_pdf
from translations import get_translation
import pandas as pd
import traceback
import re
from weibull import build_weibull_analysis
from exponential import build_exponential_analysis
from rsf import build_rsf_analysis, build_rsf_profile_analysis


def clean_markdown_text(text):
    """Remove Markdown formatting for correct rendering in ReportLab."""
    if not text:
        return text
    
    # Remove Markdown headings (### ..., ## ..., # ...).
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    
    # Convert Markdown bold (**text**) to HTML <b></b>.
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    
    # Convert Markdown italic (*text*) to HTML <i></i>, excluding processed **.
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    
    # Normalize repeated spaces and line breaks.
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text


def _prepare_ai_text_for_pdf(text):
    """Normalize the existing explanation before inserting it into the PDF."""
    return clean_markdown_text((text or "").strip())


def _looks_like_incomplete_ai_text(text):
    """Detect responses that were probably cut off mid-sentence."""
    normalized_text = (text or "").strip()
    if not normalized_text:
        return False

    last_char = normalized_text[-1]
    complete_endings = ".!?)]}'\""
    if last_char in complete_endings:
        return False

    trailing_words = normalized_text.lower().split()[-3:]
    unfinished_connectors = {
        "de", "del", "la", "el", "los", "las", "y", "o", "pero", "aunque",
        "porque", "que", "con", "sin", "para", "por", "en", "of", "the",
        "and", "or", "but", "because", "with", "without", "for", "to", "in",
    }
    return bool(trailing_words and trailing_words[-1] in unfinished_connectors) or last_char not in complete_endings


def _validate_ai_explanation(ai_requested, ai_text, analysis_label, language='en'):
    """Require an explanation only if the user selected AI and no actual text exists."""
    if not ai_requested:
        return None

    normalized_text = (ai_text or "").strip()
    placeholders = {"La respuesta es...", "The answer is..."}
    if not normalized_text or normalized_text in placeholders:
        if language == 'en':
            return f"⚠️ Before exporting the {analysis_label} PDF, generate an AI explanation first."
        return f"⚠️ Antes de exportar el PDF de {analysis_label}, genera primero una explicación de IA."

    if _looks_like_incomplete_ai_text(normalized_text):
        if language == 'en':
            return f"⚠️ The {analysis_label} AI explanation seems incomplete. Click Explain again before exporting."
        return f"⚠️ La explicación IA de {analysis_label} parece incompleta. Pulsa de nuevo Explicar antes de exportar."

    return None


def _validate_ai_language(ai_requested, generated_language, current_language):
    """Check that the AI explanation was generated in the active language."""
    if not ai_requested:
        return None

    if not generated_language:
        return None

    if generated_language != current_language:
        if current_language == 'en':
            return "⚠️ The AI explanation was generated in another language. Please click Explain again before exporting."
        return "⚠️ La explicación IA se generó en otro idioma. Pulsa de nuevo Explicar antes de exportar."

    return None


def _get_report_name_from_filename(filename):
    """Get a clean report name from the PDF filename."""
    base_name = os.path.splitext(os.path.basename(filename or ""))[0].strip()
    return base_name or "informe"


def register_pdf_export_callbacks(app):
    """
    Register all callbacks for PDF export.
    Include modals for Kaplan-Meier, Cox regression, Log-Rank, Weibull, and RSF.
    """
    
    # ===== KAPLAN-MEIER PDF CALLBACKS =====
    
    @app.callback(
        [Output('km-pdf-modal-overlay', 'style'),
         Output('km-pdf-modal-container', 'style')],
        [Input('export-km-btn', 'n_clicks'),
         Input('km-pdf-modal-close-btn', 'n_clicks'),
         Input('km-pdf-modal-cancel-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def toggle_km_pdf_modal(export_clicks, close_clicks, cancel_clicks):
        """Open/close the Kaplan-Meier export modal."""
        if not callback_context.triggered:
            return {'display': 'none'}, {'display': 'none'}
        
        triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]
        
        if triggered_id == 'export-km-btn' and export_clicks:
            return _get_modal_styles(True)
        
        return {'display': 'none'}, {'display': 'none'}
    
    
    @app.callback(
        [Output('km-pdf-modal-download', 'data'),
         Output('km-pdf-modal-error', 'children'),
         Output('km-pdf-modal-error', 'style')],
        Input('km-pdf-modal-download-btn', 'n_clicks'),
        [State('km-pdf-modal-filename', 'value'),
         State('km-pdf-modal-checklist-content', 'value'),
         State('km-current-variable', 'data'),
         State('openai-answer-kaplan', 'value'),
         State('df-store', 'data'),
         State('language-store', 'data')],
        prevent_initial_call=True
    )
    def download_km_pdf(n_clicks, filename, options, current_variable, ai_text_from_page, df_json, language):
        """Generate a Kaplan-Meier PDF using ACTUAL data from the selected analysis."""
        try:
            # Generate the filename.
            if not filename or filename.strip() == '':
                var_name = current_variable if current_variable else 'general'
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"km_{var_name}_{timestamp}.pdf"
            else:
                if not filename.endswith('.pdf'):
                    filename += '.pdf'
            
            os.makedirs("downloads", exist_ok=True)
            pdf_path = f"downloads/{filename}"
            report_name = _get_report_name_from_filename(filename)
            
            options = options or ['summary', 'graph']
            ai_error = _validate_ai_explanation('ai_interpretation' in options, ai_text_from_page, 'Kaplan-Meier', language)
            if ai_error:
                return None, ai_error, {
                    'display': 'block',
                    'marginTop': '15px',
                    'padding': '10px 12px',
                    'borderRadius': '6px',
                    'backgroundColor': '#fff1f0',
                    'color': '#c0392b',
                    'border': '1px solid #f5c6cb',
                    'fontSize': '14px',
                    'fontWeight': 'bold'
                }
            
            # CAPTURE ACTUAL ANALYSIS DATA.
            df = None
            if df_json:
                try:
                    df = pd.read_json(df_json, orient='split')
                except:
                    df = None
            
            # CALCULATE ACTUAL STATISTICS FOR THE CURRENT ANALYSIS.
            n_patients = len(df) if df is not None else 0
            n_events = int(df['final_result'].sum()) if df is not None and 'final_result' in df.columns else 0
            follow_up_mean = float(df['date'].mean()) if df is not None and 'date' in df.columns else 0
            follow_up_median = float(df['date'].median()) if df is not None and 'date' in df.columns else 0
            
            summary_stats = {
                'n_patients': n_patients,
                'n_events': n_events,
                'follow_up_mean': follow_up_mean,
                'follow_up_median': follow_up_median,
                'variable_name': current_variable if current_variable else 'General'
            }
            
            # GENERATE THE ACTUAL SURVIVAL TABLE.
            km_table = None
            if df is not None and n_patients > 0:
                try:
                    from lifelines import KaplanMeierFitter
                    kmf = KaplanMeierFitter()
                    kmf.fit(df['date'], event_observed=df['final_result'])
                    km_table = kmf.survival_function_.reset_index()
                    km_table.columns = ['Time', 'Survival'] if language == 'en' else ['Tiempo', 'Supervivencia']
                except Exception as e:
                    print(f"""[KM PDF] Error generating KM table: {e}""")
                    km_table = None
            
            # GENERATE THE KM CHART IF REQUESTED.
            km_figure = None
            if 'graph' in options and df is not None and current_variable:
                try:
                    print(f"""[KM PDF] Rebuilding the KM chart for {current_variable}...""")
                    from kaplan_meier import _create_km_figure
                    
                    # Map the variable to the correct covariate name.
                    covariate_map = {
                        'gender': 'gender_F',
                        'disability': 'disability_N',
                        'age': 'age_band',
                        'education': 'highest_education',
                        'credit': 'studied_credits'
                    }
                    
                    # Find which covariate to use.
                    covariate_col = current_variable
                    for key, val in covariate_map.items():
                        if key.lower() in current_variable.lower():
                            covariate_col = val
                            break
                    
                    # Create a Plotly figure directly.
                    km_figure = _create_km_figure(df, covariate_col, language=language)
                    print(f"""[KM PDF] KM chart rebuilt for {covariate_col}""")
                    
                except Exception as e:
                    print(f"""[KM PDF] Error rebuilding chart: {e}""")
                    traceback.print_exc()
                    km_figure = None
            
            # REUSE THE AI INTERPRETATION ALREADY GENERATED ON SCREEN.
            ai_text = ""
            if 'ai_interpretation' in options:
                print(f"""[KM PDF] Reusing the AI interpretation generated on screen for {current_variable}...""")
                ai_text = _prepare_ai_text_for_pdf(ai_text_from_page)
            
            print(f"""[KM PDF] Generating PDF: {filename}""")
            print(f"""[KM PDF] Variable: {current_variable} | Patients: {n_patients} | Events: {n_events}""")
            print(f"""[KM PDF] Selected options: {options}""")
            
            # GENERATE A PDF WITH ACTUAL DATA, INCLUDING ONLY SELECTED CONTENT.
            report_title = f"KAPLAN-MEIER REPORT: {current_variable.upper() if current_variable else 'GENERAL'}"
            export_survival_analysis_to_pdf(
                filename=pdf_path,
                title=report_title,
                report_name=report_name,
                include_summary='summary' in options,  # Show the summary ONLY if selected.
                include_km='table' in options or 'graph' in options,  # Show KM if the table or chart was selected.
                km_figure=km_figure if 'graph' in options else None,
                km_table=km_table if 'table' in options else None,
                include_cox=False,
                include_logrank=False,
                include_ai_interpretation='ai_interpretation' in options,  # Show AI ONLY if selected.
                ai_text=ai_text,
                summary_stats=summary_stats if 'summary' in options else None,
                language=language
            )
            
            print(f"""[KM PDF] PDF generated at: {pdf_path}""")
            return dcc.send_file(pdf_path), "", {'display': 'none'}
        
        except Exception as e:
            print(f"""ERROR in download_km_pdf: {str(e)}""")
            print(traceback.format_exc())
            error_label = 'Kaplan-Meier' if language == 'es' else 'Kaplan-Meier'
            return None, (f"❌ Error al generar el PDF de Kaplan-Meier: {str(e)}" if language == 'es' else f"❌ Error generating the Kaplan-Meier PDF: {str(e)}"), {
                'display': 'block',
                'marginTop': '15px',
                'padding': '10px 12px',
                'borderRadius': '6px',
                'backgroundColor': '#fff1f0',
                'color': '#c0392b',
                'border': '1px solid #f5c6cb',
                'fontSize': '14px',
                'fontWeight': 'bold'
            }
    
    
    # ===== COX REGRESSION PDF CALLBACKS =====
    
    @app.callback(
        [Output('cox-pdf-modal-overlay', 'style'),
         Output('cox-pdf-modal-container', 'style')],
        [Input('export-cox-btn', 'n_clicks'),
         Input('cox-pdf-modal-close-btn', 'n_clicks'),
         Input('cox-pdf-modal-cancel-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def toggle_cox_pdf_modal(export_clicks, close_clicks, cancel_clicks):
        """Open/close the Cox regression export modal."""
        if not callback_context.triggered:
            return {'display': 'none'}, {'display': 'none'}
        
        triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]
        
        if triggered_id == 'export-cox-btn' and export_clicks:
            return _get_modal_styles(True)
        
        return {'display': 'none'}, {'display': 'none'}
    
    
    @app.callback(
        [Output('cox-pdf-modal-download', 'data'),
         Output('cox-pdf-modal-error', 'children'),
         Output('cox-pdf-modal-error', 'style')],
        Input('cox-pdf-modal-download-btn', 'n_clicks'),
        [State('cox-pdf-modal-filename', 'value'),
         State('cox-pdf-modal-checklist-content', 'value'),
         State('cox-current-variables', 'data'),
         State('cox-selected-covariables', 'data'),
         State('cox-regression-output-store', 'data'),
         State('openai-answer-cox', 'value'),
         State('df-store', 'data'),
         State('dataset-signature-store', 'data'),
         State('language-store', 'data')],
        prevent_initial_call=True
    )
    def download_cox_pdf(n_clicks, filename, options, cox_variables_text, cox_covariables, cox_store_data, ai_text_from_page, df_json, dataset_signature, language):
        """Generate a Cox regression PDF using ACTUAL data from the updated analysis."""
        try:
            # Generate the filename.
            if not filename or filename.strip() == '':
                vars_str = ','.join(cox_covariables[:2]) if cox_covariables else 'general'
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"cox_{vars_str}_{timestamp}.pdf"
            else:
                if not filename.endswith('.pdf'):
                    filename += '.pdf'
            
            os.makedirs("downloads", exist_ok=True)
            pdf_path = f"downloads/{filename}"
            report_name = _get_report_name_from_filename(filename)
            
            options = options or ['summary', 'graph']

            if not df_json:
                return None, (
                    "⚠️ Primero carga y preprocesa un dataset antes de exportar." if language == 'es'
                    else "⚠️ Load and preprocess a dataset before exporting."
                ), {
                    'display': 'block',
                    'marginTop': '15px',
                    'padding': '10px 12px',
                    'borderRadius': '6px',
                    'backgroundColor': '#fff1f0',
                    'color': '#c0392b',
                    'border': '1px solid #f5c6cb',
                    'fontSize': '14px',
                    'fontWeight': 'bold'
                }

            if not cox_covariables:
                return None, (
                    "⚠️ Selecciona al menos una covariable y ejecuta Cox antes de exportar." if language == 'es'
                    else "⚠️ Select at least one covariate and run Cox before exporting."
                ), {
                    'display': 'block',
                    'marginTop': '15px',
                    'padding': '10px 12px',
                    'borderRadius': '6px',
                    'backgroundColor': '#fff1f0',
                    'color': '#c0392b',
                    'border': '1px solid #f5c6cb',
                    'fontSize': '14px',
                    'fontWeight': 'bold'
                }

            if not cox_store_data or not isinstance(cox_store_data, dict) or not cox_store_data.get('summary_json'):
                return None, (
                    "⚠️ Ejecuta primero la regresión de Cox para generar resultados exportables." if language == 'es'
                    else "⚠️ Run Cox regression first to generate exportable results."
                ), {
                    'display': 'block',
                    'marginTop': '15px',
                    'padding': '10px 12px',
                    'borderRadius': '6px',
                    'backgroundColor': '#fff1f0',
                    'color': '#c0392b',
                    'border': '1px solid #f5c6cb',
                    'fontSize': '14px',
                    'fontWeight': 'bold'
                }

            if cox_store_data.get('dataset_signature', '') != (dataset_signature or ''):
                return None, (
                    "⚠️ El dataset ha cambiado. Recalcula Cox antes de exportar." if language == 'es'
                    else "⚠️ The dataset changed. Recalculate Cox before exporting."
                ), {
                    'display': 'block',
                    'marginTop': '15px',
                    'padding': '10px 12px',
                    'borderRadius': '6px',
                    'backgroundColor': '#fff1f0',
                    'color': '#c0392b',
                    'border': '1px solid #f5c6cb',
                    'fontSize': '14px',
                    'fontWeight': 'bold'
                }

            ai_error = _validate_ai_explanation('ai_interpretation' in options, ai_text_from_page, 'Cox Regression', language)
            if ai_error:
                return None, ai_error, {
                    'display': 'block',
                    'marginTop': '15px',
                    'padding': '10px 12px',
                    'borderRadius': '6px',
                    'backgroundColor': '#fff1f0',
                    'color': '#c0392b',
                    'border': '1px solid #f5c6cb',
                    'fontSize': '14px',
                    'fontWeight': 'bold'
                }
            
            print(f"""\n[COX PDF] ========== STARTING EXPORT ==========""")
            print(f"""[COX PDF] Selected options: {options}""")
            print(f"""[COX PDF] Selected variables: {cox_covariables}""")
            
            # CAPTURE ACTUAL DATA FROM THE CURRENT ANALYSIS.
            df = None
            cox_table = None
            
            if df_json and cox_covariables:
                try:
                    df = pd.read_json(df_json, orient='split')
                    print(f"""[COX PDF] DataFrame loaded: {len(df)} rows, {len(df.columns)} cols""")
                    
                    # RECALCULATE COX REGRESSION WITH THE SELECTED VARIABLES.
                    from cox_regression import run_cox_regression
                    print(f"""[COX PDF] Running run_cox_regression({cox_covariables})...""")
                    
                    summary, cox_table_html = run_cox_regression(df, cox_covariables)
                    print(f"""[COX PDF] Summary returned: empty={summary.empty}, shape={summary.shape}""")
                    print(f"[COX PDF] Summary columns: {list(summary.columns) if not summary.empty else 'N/A'}")
                    
                    # Convert the summary into a clean table.
                    if not summary.empty:
                        print(f"""[COX PDF] Creating coefficient table...""")
                        try:
                            # Check whether it contains ERROR.
                            if 'ERROR' in summary.get('Covariable', summary.get(summary.columns[0], [])).values:
                                print(f"""[COX PDF] Error detected in Cox: {summary}""")
                                cox_table = None
                            else:
                                # Use the formatted column names returned by run_cox_regression.
                                # Names: Covariable, Coef., exp(Coef.), SE(Coef.), Coef. lower 95%, etc.
                                cols_disponibles = [c for c in ['Covariable', 'Coef.', 'exp(Coef.)', 'Coef. lower 95%', 'Coef. upper 95%', 'p'] 
                                                   if c in summary.columns]
                                
                                if not cols_disponibles or len(cols_disponibles) < 2:
                                    print(f"""[COX PDF] Insufficient available columns: {list(summary.columns)}""")
                                    cox_table = None
                                else:
                                    cox_table = summary[cols_disponibles].copy()
                                    if 'Covariable' in cox_table.columns:
                                        cox_table.columns = ['Variable'] + [c for c in summary.columns[1:] if c in cols_disponibles[1:]]
                                    print(f"""[COX PDF] Table created: {cox_table.shape[0]} variables, {len(cols_disponibles)} columns""")
                                    print(f"""[COX PDF] First rows:\n{cox_table.head()}""")
                        except Exception as e:
                            print(f"""[COX PDF] Error creating table: {e}""")
                            traceback.print_exc()
                            cox_table = None
                    
                    print(f"""[COX PDF] Analysis recalculated for variables: {cox_covariables}""")
                
                except Exception as e:
                    print(f"""[COX PDF] Error recalculating Cox: {e}""")
                    traceback.print_exc()
                    cox_table = None
            
            # CALCULATE ACTUAL SUMMARY STATISTICS.
            n_patients = len(df) if df is not None else 0
            n_events = int(df['final_result'].sum()) if df is not None and 'final_result' in df.columns else 0
            follow_up_mean = float(df['date'].mean()) if df is not None and 'date' in df.columns else 0
            follow_up_median = float(df['date'].median()) if df is not None and 'date' in df.columns else 0
            
            summary_stats = {
                'n_patients': n_patients,
                'n_events': n_events,
                'follow_up_mean': follow_up_mean,
                'follow_up_median': follow_up_median,
                'variable_name': ', '.join(cox_covariables) if cox_covariables else 'N/A'
            }
            
            # REUSE THE AI INTERPRETATION ALREADY GENERATED ON SCREEN.
            ai_text = ""
            if 'ai_interpretation' in options:
                print(f"""[COX PDF] Reusing the AI interpretation generated on screen...""")
                ai_text = _prepare_ai_text_for_pdf(ai_text_from_page)
                print(f"""[COX PDF] Interpretation reused ({len(ai_text)} chars)""")
            
            # GENERATE THE FOREST PLOT IF REQUESTED.
            forest_figure = None
            if 'graph' in options and not summary.empty:
                print(f"""[COX PDF] Generating the same forest plot as the dashboard...""")
                try:
                    from survival_plots import plot_cox_hazard_ratios
                    forest_component = plot_cox_hazard_ratios(
                        summary,
                        'Selected covariates' if language == 'en' else 'Covariables seleccionadas',
                        language=language
                    )
                    forest_figure = getattr(forest_component, 'figure', None)
                    if forest_figure is not None:
                        print(f"""[COX PDF] Forest plot generated""")
                except Exception as e:
                    print(f"""[COX PDF] Error generating forest plot: {e}""")
                    forest_figure = None
            
            print(f"""[COX PDF] Generating PDF: {filename}""")
            print(f"""[COX PDF] Variables: {cox_covariables} | Patients: {n_patients} | Events: {n_events}""")
            print(f"""[COX PDF] Include Cox table: {'table' in options and cox_table is not None}""")
            print(f"""[COX PDF] Include forest plot: {'graph' in options and forest_figure is not None}""")
            # GENERATE A PDF WITH ACTUAL DATA, INCLUDING ONLY SELECTED CONTENT.
            report_title = (
                f"COX REGRESSION REPORT: {', '.join(cox_covariables) if cox_covariables else 'GENERAL'}"
            )
            export_survival_analysis_to_pdf(
                filename=pdf_path,
                title=report_title,
                report_name=report_name,
                include_summary='summary' in options,  # Show the summary ONLY if selected.
                include_km=False,
                include_cox='table' in options or 'graph' in options,  # Show Cox if the table OR chart was selected.
                cox_table=cox_table if 'table' in options else None,
                forest_figure=forest_figure if 'graph' in options else None,  # Show the chart ONLY if selected.
                include_logrank=False,
                include_ai_interpretation='ai_interpretation' in options,  # Show AI ONLY if selected.
                ai_text=ai_text,
                summary_stats=summary_stats if 'summary' in options else None,
                language=language
            )
            
            print(f"""[COX PDF] PDF generated at: {pdf_path}""")
            print(f"""[COX PDF] ========== EXPORT COMPLETED ==========\n""")
            return dcc.send_file(pdf_path), "", {'display': 'none'}
        
        except Exception as e:
            print(f"""ERROR in download_cox_pdf: {str(e)}""")
            print(traceback.format_exc())
            return None, (f"❌ Error al generar el PDF de Cox: {str(e)}" if language == 'es' else f"❌ Error generating the Cox PDF: {str(e)}"), {
                'display': 'block',
                'marginTop': '15px',
                'padding': '10px 12px',
                'borderRadius': '6px',
                'backgroundColor': '#fff1f0',
                'color': '#c0392b',
                'border': '1px solid #f5c6cb',
                'fontSize': '14px',
                'fontWeight': 'bold'
            }
    
    
    # ===== LOG-RANK TEST PDF CALLBACKS =====
    
    @app.callback(
        [Output('logrank-pdf-modal-overlay', 'style'),
         Output('logrank-pdf-modal-container', 'style')],
        [Input('export-logrank-btn', 'n_clicks'),
         Input('logrank-pdf-modal-close-btn', 'n_clicks'),
         Input('logrank-pdf-modal-cancel-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def toggle_logrank_pdf_modal(export_clicks, close_clicks, cancel_clicks):
        """Open/close the Log-Rank export modal."""
        if not callback_context.triggered:
            return {'display': 'none'}, {'display': 'none'}
        
        triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]
        
        if triggered_id == 'export-logrank-btn' and export_clicks:
            return _get_modal_styles(True)
        
        return {'display': 'none'}, {'display': 'none'}
    
    
    @app.callback(
        [Output('logrank-pdf-modal-download', 'data'),
         Output('logrank-pdf-modal-error', 'children'),
         Output('logrank-pdf-modal-error', 'style')],
        Input('logrank-pdf-modal-download-btn', 'n_clicks'),
        [State('logrank-pdf-modal-filename', 'value'),
         State('logrank-pdf-modal-checklist-content', 'value'),
         State('logrank-current-variable', 'data'),
         State('logrank-selected-covariables', 'data'),
         State('logrank-test-output-store', 'data'),
         State('openai-answer-logrank', 'value'),
         State('df-store', 'data'),
         State('dataset-signature-store', 'data'),
         State('language-store', 'data')],
        prevent_initial_call=True
    )
    def download_logrank_pdf(n_clicks, filename, options, logrank_variable, logrank_covariables, logrank_store_data, ai_text_from_page, df_json, dataset_signature, language):
        """Generate a Log-Rank PDF using ACTUAL data from the updated analysis."""
        try:
            # Generate the filename.
            if not filename or filename.strip() == '':
                var_str = logrank_variable if logrank_variable else 'general'
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"logrank_{var_str}_{timestamp}.pdf"
            else:
                if not filename.endswith('.pdf'):
                    filename += '.pdf'
            
            os.makedirs("downloads", exist_ok=True)
            pdf_path = f"downloads/{filename}"
            report_name = _get_report_name_from_filename(filename)
            
            options = options or ['summary', 'table']

            if not df_json:
                return None, (
                    "⚠️ Primero carga y preprocesa un dataset antes de exportar." if language == 'es'
                    else "⚠️ Load and preprocess a dataset before exporting."
                ), {
                    'display': 'block',
                    'marginTop': '15px',
                    'padding': '10px 12px',
                    'borderRadius': '6px',
                    'backgroundColor': '#fff1f0',
                    'color': '#c0392b',
                    'border': '1px solid #f5c6cb',
                    'fontSize': '14px',
                    'fontWeight': 'bold'
                }

            if not logrank_covariables:
                return None, (
                    "⚠️ Selecciona covariables y ejecuta Log-Rank antes de exportar." if language == 'es'
                    else "⚠️ Select covariates and run Log-Rank before exporting."
                ), {
                    'display': 'block',
                    'marginTop': '15px',
                    'padding': '10px 12px',
                    'borderRadius': '6px',
                    'backgroundColor': '#fff1f0',
                    'color': '#c0392b',
                    'border': '1px solid #f5c6cb',
                    'fontSize': '14px',
                    'fontWeight': 'bold'
                }

            if not logrank_store_data or not isinstance(logrank_store_data, dict) or not logrank_store_data.get('results'):
                return None, (
                    "⚠️ Ejecuta primero el Test de Log-Rank para generar resultados exportables." if language == 'es'
                    else "⚠️ Run the Log-Rank test first to generate exportable results."
                ), {
                    'display': 'block',
                    'marginTop': '15px',
                    'padding': '10px 12px',
                    'borderRadius': '6px',
                    'backgroundColor': '#fff1f0',
                    'color': '#c0392b',
                    'border': '1px solid #f5c6cb',
                    'fontSize': '14px',
                    'fontWeight': 'bold'
                }

            if logrank_store_data.get('dataset_signature', '') != (dataset_signature or ''):
                return None, (
                    "⚠️ El dataset ha cambiado. Recalcula Log-Rank antes de exportar." if language == 'es'
                    else "⚠️ The dataset changed. Recalculate Log-Rank before exporting."
                ), {
                    'display': 'block',
                    'marginTop': '15px',
                    'padding': '10px 12px',
                    'borderRadius': '6px',
                    'backgroundColor': '#fff1f0',
                    'color': '#c0392b',
                    'border': '1px solid #f5c6cb',
                    'fontSize': '14px',
                    'fontWeight': 'bold'
                }

            ai_error = _validate_ai_explanation('ai_interpretation' in options, ai_text_from_page, 'Log-Rank', language)
            if ai_error:
                return None, ai_error, {
                    'display': 'block',
                    'marginTop': '15px',
                    'padding': '10px 12px',
                    'borderRadius': '6px',
                    'backgroundColor': '#fff1f0',
                    'color': '#c0392b',
                    'border': '1px solid #f5c6cb',
                    'fontSize': '14px',
                    'fontWeight': 'bold'
                }
            
            print(f"""\n[LOGRANK PDF] ========== STARTING EXPORT ==========""")
            print(f"""[LOGRANK PDF] Selected options: {options}""")
            print(f"""[LOGRANK PDF] Selected variables: {logrank_covariables}""")
            
            # CAPTURE ACTUAL DATA FROM THE CURRENT ANALYSIS.
            df = None
            logrank_results = None
            logrank_figure = None
            
            if df_json and logrank_covariables:
                df = pd.read_json(df_json, orient='split')
                
                # RECALCULATE THE LOG-RANK TEST WITH THE SELECTED VARIABLE.
                from log_rank_test import perform_log_rank_test, create_logrank_figure
                
                # Process each covariate and accumulate the results.
                all_results = []
                for covariable in logrank_covariables:
                    res_df = perform_log_rank_test(df, covariable)
                    if not res_df.empty:
                        all_results.append(res_df)
                
                if all_results:
                    logrank_results = pd.concat(all_results, ignore_index=True)
                
                # GENERATE THE CHART IF REQUESTED.
                if 'graph' in options and logrank_covariables:
                    print(f"""[LOGRANK PDF] Generating chart...""")
                    try:
                        logrank_figure = create_logrank_figure(df, logrank_covariables[0], language=language)
                        if logrank_figure:
                            print(f"""[LOGRANK PDF] Log-Rank chart generated""")
                    except Exception as e:
                        print(f"""[LOGRANK PDF] Error generating chart: {e}""")
                        logrank_figure = None
            
            # CALCULATE SUMMARY STATISTICS.
            n_patients = len(df) if df is not None else 0
            n_events = int(df['final_result'].sum()) if df is not None and 'final_result' in df.columns else 0
            follow_up_mean = float(df['date'].mean()) if df is not None and 'date' in df.columns else 0
            follow_up_median = float(df['date'].median()) if df is not None and 'date' in df.columns else 0
            
            summary_stats = {
                'n_patients': n_patients,
                'n_events': n_events,
                'follow_up_mean': follow_up_mean,
                'follow_up_median': follow_up_median,
                'variable_name': ', '.join(logrank_covariables) if logrank_covariables else 'N/A'
            }
            
            print(f"""[LOGRANK PDF] Include table: {'table' in options and logrank_results is not None}""")
            print(f"""[LOGRANK PDF] Include chart: {'graph' in options and logrank_figure is not None}""")
            print(f"""[LOGRANK PDF] Include summary: {'summary' in options}""")

            ai_text = ""
            if 'ai_interpretation' in options:
                print(f"""[LOGRANK PDF] Generating AI interpretation...""")
                ai_text = ai_text_from_page or generate_interpretation_for_pdf('log-rank', summary_stats, logrank_results, language=language)
                ai_text = clean_markdown_text(ai_text)
            
            # GENERATE A PDF WITH ACTUAL DATA, INCLUDING ONLY SELECTED CONTENT.
            report_title = (
                f"LOG-RANK TEST REPORT: {', '.join(logrank_covariables) if logrank_covariables else 'GENERAL'}"
            )
            export_survival_analysis_to_pdf(
                filename=pdf_path,
                title=report_title,
                report_name=report_name,
                include_summary='summary' in options,  # Show the summary ONLY if selected.
                include_km=False,
                include_cox=False,
                include_logrank='table' in options or 'graph' in options,  # Show Log-Rank if the table OR chart was selected.
                logrank_figure=logrank_figure if 'graph' in options else None,  # Show the chart ONLY if selected.
                logrank_results=logrank_results if 'table' in options else None,
                include_ai_interpretation='ai_interpretation' in options,
                ai_text=ai_text,
                summary_stats=summary_stats if 'summary' in options else None,
                language=language,
                landscape_tables=['logrank']  # Use landscape orientation for the Log-Rank table.
            )
            
            print(f"""[LOGRANK PDF] PDF generated at: {pdf_path}""")
            print(f"""[LOGRANK PDF] ========== EXPORT COMPLETED ==========\n""")
            return dcc.send_file(pdf_path), "", {'display': 'none'}
        
        except Exception as e:
            print(f"""ERROR in download_logrank_pdf: {str(e)}""")
            print(traceback.format_exc())
            return None, (f"❌ Error al generar el PDF de Log-Rank: {str(e)}" if language == 'es' else f"❌ Error generating the Log-Rank PDF: {str(e)}"), {
                'display': 'block',
                'marginTop': '15px',
                'padding': '10px 12px',
                'borderRadius': '6px',
                'backgroundColor': '#fff1f0',
                'color': '#c0392b',
                'border': '1px solid #f5c6cb',
                'fontSize': '14px',
                'fontWeight': 'bold'
            }


    # ===== COMBINED WEIBULL + EXPONENTIAL PDF CALLBACKS =====

    @app.callback(
        [Output('weibexp-pdf-modal-overlay', 'style'),
         Output('weibexp-pdf-modal-container', 'style')],
        [Input('export-weibexp-btn', 'n_clicks'),
         Input('weibexp-pdf-modal-close-btn', 'n_clicks'),
         Input('weibexp-pdf-modal-cancel-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def toggle_weibexp_pdf_modal(export_clicks, close_clicks, cancel_clicks):
        """Open/close the combined Weibull + Exponential export modal."""
        if not callback_context.triggered:
            return {'display': 'none'}, {'display': 'none'}

        triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]

        if triggered_id == 'export-weibexp-btn' and export_clicks:
            return _get_modal_styles(True)

        return {'display': 'none'}, {'display': 'none'}


    @app.callback(
        [Output('weibexp-pdf-modal-download', 'data'),
         Output('weibexp-pdf-modal-error', 'children'),
         Output('weibexp-pdf-modal-error', 'style')],
        Input('weibexp-pdf-modal-download-btn', 'n_clicks'),
        [State('weibexp-pdf-modal-filename', 'value'),
         State('weibexp-pdf-modal-techniques', 'value'),
         State('weibexp-pdf-modal-content', 'value'),
         State('weibull-ai-text-store', 'data'),
         State('exponential-ai-text-store', 'data'),
         State('weibull-ai-language-store', 'data'),
         State('exponential-ai-language-store', 'data'),
         State('df-store', 'data'),
         State('language-store', 'data')],
        prevent_initial_call=True
    )
    def download_weibexp_pdf(
        n_clicks,
        filename,
        techniques,
        options,
        weibull_ai_text,
        exponential_ai_text,
        weibull_ai_language,
        exponential_ai_language,
        df_json,
        language
    ):
        """Generate a combined Weibull + Exponential PDF."""
        try:
            if not filename or filename.strip() == '':
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"weibull_exponential_combined_{timestamp}.pdf"
            else:
                if not filename.endswith('.pdf'):
                    filename += '.pdf'

            report_name = _get_report_name_from_filename(filename)
            os.makedirs("downloads", exist_ok=True)
            pdf_path = f"downloads/{filename}"

            techniques = techniques or ['weibull', 'exponential']
            options = options or ['summary', 'table', 'graph']

            if 'ai_interpretation' in options:
                if 'weibull' in techniques:
                    ai_error = _validate_ai_explanation(True, weibull_ai_text, 'Weibull', language)
                    if ai_error:
                        return None, ai_error, {
                            'display': 'block',
                            'marginTop': '15px',
                            'padding': '10px 12px',
                            'borderRadius': '6px',
                            'backgroundColor': '#fff1f0',
                            'color': '#c0392b',
                            'border': '1px solid #f5c6cb',
                            'fontSize': '14px',
                            'fontWeight': 'bold'
                        }

                    lang_error = _validate_ai_language(True, weibull_ai_language, language)
                    if lang_error:
                        return None, lang_error, {
                            'display': 'block',
                            'marginTop': '15px',
                            'padding': '10px 12px',
                            'borderRadius': '6px',
                            'backgroundColor': '#fff1f0',
                            'color': '#c0392b',
                            'border': '1px solid #f5c6cb',
                            'fontSize': '14px',
                            'fontWeight': 'bold'
                        }

                if 'exponential' in techniques:
                    ai_error = _validate_ai_explanation(True, exponential_ai_text, 'Exponencial' if language == 'es' else 'Exponential', language)
                    if ai_error:
                        return None, ai_error, {
                            'display': 'block',
                            'marginTop': '15px',
                            'padding': '10px 12px',
                            'borderRadius': '6px',
                            'backgroundColor': '#fff1f0',
                            'color': '#c0392b',
                            'border': '1px solid #f5c6cb',
                            'fontSize': '14px',
                            'fontWeight': 'bold'
                        }

                    lang_error = _validate_ai_language(True, exponential_ai_language, language)
                    if lang_error:
                        return None, lang_error, {
                            'display': 'block',
                            'marginTop': '15px',
                            'padding': '10px 12px',
                            'borderRadius': '6px',
                            'backgroundColor': '#fff1f0',
                            'color': '#c0392b',
                            'border': '1px solid #f5c6cb',
                            'fontSize': '14px',
                            'fontWeight': 'bold'
                        }

            if not techniques:
                return None, (
                    "⚠️ Selecciona al menos una técnica (Weibull o Exponencial)."
                    if language == 'es' else
                    "⚠️ Select at least one technique (Weibull or Exponential)."
                ), {
                    'display': 'block',
                    'marginTop': '15px',
                    'padding': '10px 12px',
                    'borderRadius': '6px',
                    'backgroundColor': '#fff1f0',
                    'color': '#c0392b',
                    'border': '1px solid #f5c6cb',
                    'fontSize': '14px',
                    'fontWeight': 'bold'
                }

            if not df_json:
                return None, (
                    "❌ No hay datos cargados para exportar el informe combinado."
                    if language == 'es' else
                    "❌ No dataset loaded to export the combined report."
                ), {
                    'display': 'block',
                    'marginTop': '15px',
                    'padding': '10px 12px',
                    'borderRadius': '6px',
                    'backgroundColor': '#fff1f0',
                    'color': '#c0392b',
                    'border': '1px solid #f5c6cb',
                    'fontSize': '14px',
                    'fontWeight': 'bold'
                }

            df = pd.read_json(df_json, orient='split')
            weibull_analysis = build_weibull_analysis(df, language=language) if 'weibull' in techniques else None
            exponential_analysis = build_exponential_analysis(df, language=language) if 'exponential' in techniques else None

            if ('weibull' in techniques and not weibull_analysis) and ('exponential' in techniques and not exponential_analysis):
                return None, (
                    "❌ No hay datos suficientes para construir el informe combinado."
                    if language == 'es' else
                    "❌ Not enough data to build the combined report."
                ), {
                    'display': 'block',
                    'marginTop': '15px',
                    'padding': '10px 12px',
                    'borderRadius': '6px',
                    'backgroundColor': '#fff1f0',
                    'color': '#c0392b',
                    'border': '1px solid #f5c6cb',
                    'fontSize': '14px',
                    'fontWeight': 'bold'
                }

            summary_stats = {
                'n_patients': len(df),
                'n_events': int(df['final_result'].sum()) if 'final_result' in df.columns else 0,
                'follow_up_mean': float(df['date'].mean()) if 'date' in df.columns else 0,
                'follow_up_median': float(df['date'].median()) if 'date' in df.columns else 0,
            }

            combined_ai_text = ""
            if 'ai_interpretation' in options:
                ai_sections = []
                if weibull_analysis:
                    weibull_ai = weibull_ai_text
                    if weibull_ai:
                        ai_sections.append("Weibull:\n" + clean_markdown_text(weibull_ai))

                if exponential_analysis:
                    exponential_ai = exponential_ai_text
                    if exponential_ai:
                        ai_sections.append(("Exponencial" if language == 'es' else "Exponential") + ":\n" + clean_markdown_text(exponential_ai))

                combined_ai_text = "\n\n".join(ai_sections)

            export_weibull_exponential_combined_pdf(
                filename=pdf_path,
                title="WEIBULL + EXPONENTIAL REPORT: COMBINED",
                report_name=report_name,
                include_summary='summary' in options,
                include_table='table' in options,
                include_graph='graph' in options,
                include_ai_interpretation='ai_interpretation' in options,
                include_weibull='weibull' in techniques and weibull_analysis is not None,
                include_exponential='exponential' in techniques and exponential_analysis is not None,
                weibull_table=weibull_analysis['summary_df'] if weibull_analysis is not None else None,
                weibull_figure=weibull_analysis['figure'] if weibull_analysis is not None else None,
                exponential_table=exponential_analysis['summary_df'] if exponential_analysis is not None else None,
                exponential_figure=exponential_analysis['figure'] if exponential_analysis is not None else None,
                ai_text=combined_ai_text,
                summary_stats=summary_stats,
                language=language
            )

            return dcc.send_file(pdf_path), "", {'display': 'none'}

        except Exception as e:
            print(f"""ERROR in download_weibexp_pdf: {str(e)}""")
            print(traceback.format_exc())
            return None, (
                f"❌ Error al generar el PDF combinado: {str(e)}"
                if language == 'es' else
                f"❌ Error generating the combined PDF: {str(e)}"
            ), {
                'display': 'block',
                'marginTop': '15px',
                'padding': '10px 12px',
                'borderRadius': '6px',
                'backgroundColor': '#fff1f0',
                'color': '#c0392b',
                'border': '1px solid #f5c6cb',
                'fontSize': '14px',
                'fontWeight': 'bold'
            }


    # ===== WEIBULL PDF CALLBACKS =====

    @app.callback(
        [Output('weibull-pdf-modal-overlay', 'style'),
         Output('weibull-pdf-modal-container', 'style')],
        [Input('export-weibull-btn', 'n_clicks'),
         Input('weibull-pdf-modal-close-btn', 'n_clicks'),
         Input('weibull-pdf-modal-cancel-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def toggle_weibull_pdf_modal(export_clicks, close_clicks, cancel_clicks):
        """Open/close the Weibull export modal."""
        if not callback_context.triggered:
            return {'display': 'none'}, {'display': 'none'}

        triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]

        if triggered_id == 'export-weibull-btn' and export_clicks:
            return _get_modal_styles(True)

        return {'display': 'none'}, {'display': 'none'}


    @app.callback(
        [Output('weibull-pdf-modal-download', 'data'),
         Output('weibull-pdf-modal-error', 'children'),
         Output('weibull-pdf-modal-error', 'style')],
        Input('weibull-pdf-modal-download-btn', 'n_clicks'),
        [State('weibull-pdf-modal-filename', 'value'),
         State('weibull-pdf-modal-checklist-content', 'value'),
         State('openai-answer-weibull', 'value'),
         State('weibull-ai-language-store', 'data'),
         State('df-store', 'data'),
         State('language-store', 'data')],
        prevent_initial_call=True
    )
    def download_weibull_pdf(n_clicks, filename, options, ai_text_from_page, ai_text_language, df_json, language):
        """Generate a Weibull PDF using ACTUAL data from the clean dataset."""
        try:
            if not filename or filename.strip() == '':
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"weibull_{timestamp}.pdf"
            else:
                if not filename.endswith('.pdf'):
                    filename += '.pdf'

            os.makedirs("downloads", exist_ok=True)
            pdf_path = f"downloads/{filename}"
            report_name = _get_report_name_from_filename(filename)

            options = options or ['summary', 'table']
            ai_error = _validate_ai_explanation('ai_interpretation' in options, ai_text_from_page, 'Weibull', language)
            if ai_error:
                return None, ai_error, {
                    'display': 'block',
                    'marginTop': '15px',
                    'padding': '10px 12px',
                    'borderRadius': '6px',
                    'backgroundColor': '#fff1f0',
                    'color': '#c0392b',
                    'border': '1px solid #f5c6cb',
                    'fontSize': '14px',
                    'fontWeight': 'bold'
                }

            ai_lang_error = _validate_ai_language('ai_interpretation' in options, ai_text_language, language)
            if ai_lang_error:
                return None, ai_lang_error, {
                    'display': 'block',
                    'marginTop': '15px',
                    'padding': '10px 12px',
                    'borderRadius': '6px',
                    'backgroundColor': '#fff1f0',
                    'color': '#c0392b',
                    'border': '1px solid #f5c6cb',
                    'fontSize': '14px',
                    'fontWeight': 'bold'
                }

            df = None
            if df_json:
                try:
                    df = pd.read_json(df_json, orient='split')
                except Exception:
                    df = None

            analysis = build_weibull_analysis(df, language=language) if df is not None else None
            if not analysis:
                return None, ("❌ No hay datos suficientes para generar el PDF de Weibull." if language == 'es' else "❌ Not enough data to generate the Weibull PDF."), {
                    'display': 'block',
                    'marginTop': '15px',
                    'padding': '10px 12px',
                    'borderRadius': '6px',
                    'backgroundColor': '#fff1f0',
                    'color': '#c0392b',
                    'border': '1px solid #f5c6cb',
                    'fontSize': '14px',
                    'fontWeight': 'bold'
                }

            summary_stats = {
                'n_patients': analysis['n_observations'],
                'n_events': analysis['n_events'],
                'event_rate': analysis['event_rate'],
                'variable_name': 'Weibull'
            }

            ai_text = ""
            if 'ai_interpretation' in options:
                ai_text = ai_text_from_page or generate_interpretation_for_pdf('weibull', summary_stats, analysis['summary_df'], language=language)
                ai_text = clean_markdown_text(ai_text)

            report_title = "WEIBULL METHOD REPORT: GENERAL"
            export_survival_analysis_to_pdf(
                filename=pdf_path,
                title=report_title,
                report_name=report_name,
                include_summary='summary' in options,
                include_km=False,
                include_cox=False,
                include_logrank=False,
                include_weibull='table' in options or 'graph' in options,
                weibull_table=analysis['summary_df'] if 'table' in options else None,
                weibull_figure=analysis['figure'] if 'graph' in options else None,
                include_ai_interpretation='ai_interpretation' in options,
                ai_text=ai_text,
                summary_stats=summary_stats if 'summary' in options else None,
                language=language
            )

            return dcc.send_file(pdf_path), "", {'display': 'none'}

        except Exception as e:
            print(f"""ERROR in download_weibull_pdf: {str(e)}""")
            print(traceback.format_exc())
            return None, (f"❌ Error al generar el PDF de Weibull: {str(e)}" if language == 'es' else f"❌ Error generating the Weibull PDF: {str(e)}"), {
                'display': 'block',
                'marginTop': '15px',
                'padding': '10px 12px',
                'borderRadius': '6px',
                'backgroundColor': '#fff1f0',
                'color': '#c0392b',
                'border': '1px solid #f5c6cb',
                'fontSize': '14px',
                'fontWeight': 'bold'
            }


    # ===== EXPONENTIAL PDF CALLBACKS =====

    @app.callback(
        [Output('exponential-pdf-modal-overlay', 'style'),
         Output('exponential-pdf-modal-container', 'style')],
        [Input('export-exponential-btn', 'n_clicks'),
         Input('exponential-pdf-modal-close-btn', 'n_clicks'),
         Input('exponential-pdf-modal-cancel-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def toggle_exponential_pdf_modal(export_clicks, close_clicks, cancel_clicks):
        """Open or close the Exponential model PDF export modal."""
        if not callback_context.triggered:
            return {'display': 'none'}, {'display': 'none'}

        triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]

        if triggered_id == 'export-exponential-btn' and export_clicks:
            return _get_modal_styles(True)

        return {'display': 'none'}, {'display': 'none'}


    @app.callback(
        [Output('exponential-pdf-modal-download', 'data'),
         Output('exponential-pdf-modal-error', 'children'),
         Output('exponential-pdf-modal-error', 'style')],
        Input('exponential-pdf-modal-download-btn', 'n_clicks'),
        [State('exponential-pdf-modal-filename', 'value'),
         State('exponential-pdf-modal-checklist-content', 'value'),
         State('openai-answer-exponential', 'value'),
         State('exponential-ai-language-store', 'data'),
         State('df-store', 'data'),
         State('language-store', 'data')],
        prevent_initial_call=True
    )
    def download_exponential_pdf(n_clicks, filename, options, ai_text_from_page, ai_text_language, df_json, language):
        """Generate and download the Exponential analysis PDF with the selected options."""
        try:
            if not filename or filename.strip() == '':
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"exponential_{timestamp}.pdf"
            else:
                if not filename.endswith('.pdf'):
                    filename += '.pdf'

            os.makedirs("downloads", exist_ok=True)
            pdf_path = f"downloads/{filename}"
            report_name = _get_report_name_from_filename(filename)

            options = options or ['summary', 'table']
            ai_error = _validate_ai_explanation('ai_interpretation' in options, ai_text_from_page, 'Exponencial', language)
            if ai_error:
                return None, ai_error, {
                    'display': 'block',
                    'marginTop': '15px',
                    'padding': '10px 12px',
                    'borderRadius': '6px',
                    'backgroundColor': '#fff1f0',
                    'color': '#c0392b',
                    'border': '1px solid #f5c6cb',
                    'fontSize': '14px',
                    'fontWeight': 'bold'
                }

            ai_lang_error = _validate_ai_language('ai_interpretation' in options, ai_text_language, language)
            if ai_lang_error:
                return None, ai_lang_error, {
                    'display': 'block',
                    'marginTop': '15px',
                    'padding': '10px 12px',
                    'borderRadius': '6px',
                    'backgroundColor': '#fff1f0',
                    'color': '#c0392b',
                    'border': '1px solid #f5c6cb',
                    'fontSize': '14px',
                    'fontWeight': 'bold'
                }

            df = None
            if df_json:
                try:
                    df = pd.read_json(df_json, orient='split')
                except Exception:
                    df = None

            analysis = build_exponential_analysis(df, language=language) if df is not None else None
            if not analysis:
                return None, ("❌ No hay datos suficientes para generar el PDF de Exponencial." if language == 'es' else "❌ Not enough data to generate the Exponential PDF."), {
                    'display': 'block',
                    'marginTop': '15px',
                    'padding': '10px 12px',
                    'borderRadius': '6px',
                    'backgroundColor': '#fff1f0',
                    'color': '#c0392b',
                    'border': '1px solid #f5c6cb',
                    'fontSize': '14px',
                    'fontWeight': 'bold'
                }

            summary_stats = {
                'n_patients': analysis['n_observations'],
                'n_events': analysis['n_events'],
                'follow_up_mean': float(df['date'].mean()) if df is not None and 'date' in df.columns else 0,
                'follow_up_median': float(df['date'].median()) if df is not None and 'date' in df.columns else 0,
                'variable_name': 'Exponencial'
            }

            ai_text = ""
            if 'ai_interpretation' in options:
                ai_text = ai_text_from_page or analysis['interpretation']
                ai_text = clean_markdown_text(ai_text)

            report_title = "EXPONENTIAL METHOD REPORT: GENERAL"
            export_survival_analysis_to_pdf(
                filename=pdf_path,
                title=report_title,
                report_name=report_name,
                include_summary='summary' in options,
                include_km=False,
                include_cox=False,
                include_logrank=False,
                include_weibull=False,
                include_exponential=True,
                exponential_table=analysis['summary_df'],
                exponential_figure=analysis['figure'] if 'graph' in options else None,
                include_ai_interpretation='ai_interpretation' in options,
                ai_text=ai_text,
                summary_stats=summary_stats if 'summary' in options else None,
                language=language
            )

            return dcc.send_file(pdf_path), "", {'display': 'none'}

        except Exception as e:
            print(f"""ERROR in download_exponential_pdf: {str(e)}""")
            print(traceback.format_exc())
            return None, (f"❌ Error al generar el PDF de Exponencial: {str(e)}" if language == 'es' else f"❌ Error generating the Exponential PDF: {str(e)}"), {
                'display': 'block',
                'marginTop': '15px',
                'padding': '10px 12px',
                'borderRadius': '6px',
                'backgroundColor': '#fff1f0',
                'color': '#c0392b',
                'border': '1px solid #f5c6cb',
                'fontSize': '14px',
                'fontWeight': 'bold'
            }


    # ===== RSF PDF CALLBACKS =====

    @app.callback(
        [Output('rsf-pdf-modal-overlay', 'style'),
         Output('rsf-pdf-modal-container', 'style')],
        [Input('export-rsf-btn', 'n_clicks'),
         Input('rsf-pdf-modal-close-btn', 'n_clicks'),
         Input('rsf-pdf-modal-cancel-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def toggle_rsf_pdf_modal(export_clicks, close_clicks, cancel_clicks):
        """Open/close the RSF export modal."""
        if not callback_context.triggered:
            return {'display': 'none'}, {'display': 'none'}

        triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]

        if triggered_id == 'export-rsf-btn' and export_clicks:
            return _get_modal_styles(True)

        return {'display': 'none'}, {'display': 'none'}


    @app.callback(
        [Output('rsf-pdf-modal-download', 'data'),
         Output('rsf-pdf-modal-error', 'children'),
         Output('rsf-pdf-modal-error', 'style')],
        Input('rsf-pdf-modal-download-btn', 'n_clicks'),
        [State('rsf-pdf-modal-filename', 'value'),
         State('rsf-pdf-modal-checklist-content', 'value'),
         State('rsf-analysis-data', 'data'),
         State('openai-answer-rsf', 'value'),
         State('rsf-profile-gender', 'value'),
         State('rsf-profile-disability', 'value'),
         State('rsf-profile-age-band', 'value'),
         State('rsf-profile-education', 'value'),
         State('rsf-profile-credits', 'value'),
         State('df-store', 'data'),
         State('language-store', 'data')],
        prevent_initial_call=True
    )
    def download_rsf_pdf(
        n_clicks,
        filename,
        options,
        rsf_store_data,
        ai_text_from_page,
        profile_gender,
        profile_disability,
        profile_age_band,
        profile_education,
        profile_credits_level,
        df_json,
        language,
    ):
        """Generate a Random Survival Forest PDF using actual data from the current analysis."""
        try:
            if not filename or filename.strip() == '':
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"rsf_{timestamp}.pdf"
            else:
                if not filename.endswith('.pdf'):
                    filename += '.pdf'

            os.makedirs("downloads", exist_ok=True)
            pdf_path = f"downloads/{filename}"
            report_name = _get_report_name_from_filename(filename)

            options = options or ['general_summary', 'model_summary']
            ai_error = _validate_ai_explanation('ai_interpretation' in options, ai_text_from_page, 'Random Survival Forest', language)
            if ai_error:
                return None, ai_error, {
                    'display': 'block',
                    'marginTop': '15px',
                    'padding': '10px 12px',
                    'borderRadius': '6px',
                    'backgroundColor': '#fff1f0',
                    'color': '#c0392b',
                    'border': '1px solid #f5c6cb',
                    'fontSize': '14px',
                    'fontWeight': 'bold'
                }

            df = None
            if df_json:
                try:
                    df = pd.read_json(df_json, orient='split')
                except Exception:
                    df = None

            analysis = build_rsf_analysis(df, language=language) if df is not None else None
            if not analysis:
                return None, ("❌ No hay datos suficientes para generar el PDF de RSF." if language == 'es' else "❌ Not enough data to generate the RSF PDF."), {
                    'display': 'block',
                    'marginTop': '15px',
                    'padding': '10px 12px',
                    'borderRadius': '6px',
                    'backgroundColor': '#fff1f0',
                    'color': '#c0392b',
                    'border': '1px solid #f5c6cb',
                    'fontSize': '14px',
                    'fontWeight': 'bold'
                }

            # Use the same table displayed in the dashboard so the PDF matches
            # the "RSF model summary" block shown above exactly.
            rsf_table_for_pdf = analysis['summary_df']
            if rsf_store_data and isinstance(rsf_store_data, dict) and rsf_store_data.get('summary_json') and rsf_store_data.get('language') == language:
                try:
                    rsf_table_for_pdf = pd.read_json(rsf_store_data['summary_json'], orient='split')
                except Exception:
                    rsf_table_for_pdf = analysis['summary_df']

            summary_stats = {
                'n_patients': analysis['n_observations'],
                'n_events': analysis['n_events'],
                'follow_up_mean': float(df['date'].mean()) if df is not None and 'date' in df.columns else 0,
                'follow_up_median': float(df['date'].median()) if df is not None and 'date' in df.columns else 0,
                'variable_name': analysis['top_feature']
            }

            rsf_profile_analysis = None
            if 'profile' in options:
                credits_map = {'few': 30, 'medium': 60, 'many': 120}
                profile = {
                    'gender_F': profile_gender if profile_gender is not None else 1,
                    'disability_N': profile_disability if profile_disability is not None else 1,
                    'age_band': profile_age_band if profile_age_band else 'age_band_0-35',
                    'highest_education': profile_education if profile_education else 'highest_education_A Level or Equivalent',
                    'studied_credits': credits_map.get(profile_credits_level, 30),
                }
                rsf_profile_analysis = build_rsf_profile_analysis(df, profile, language=language)

            ai_text = ""
            if 'ai_interpretation' in options:
                ai_text = ai_text_from_page or analysis['interpretation']
                ai_text = clean_markdown_text(ai_text)

            report_title = "RANDOM SURVIVAL FOREST REPORT: MODEL"
            export_kwargs = dict(
                filename=pdf_path,
                title=report_title,
                report_name=report_name,
                include_summary='general_summary' in options,
                include_km=False,
                include_cox=False,
                include_logrank=False,
                include_weibull=False,
                include_rsf='model_summary' in options or 'graph' in options or 'importance' in options,
                rsf_table=rsf_table_for_pdf if 'model_summary' in options else None,
                rsf_figure=analysis['figure'] if 'graph' in options else None,
                rsf_importance_figure=analysis['importance_figure'] if 'importance' in options else None,
                include_rsf_profile='profile' in options and rsf_profile_analysis is not None,
                rsf_profile_figure=rsf_profile_analysis['figure'] if rsf_profile_analysis is not None else None,
                rsf_profile_text=clean_markdown_text(rsf_profile_analysis['interpretation']) if rsf_profile_analysis is not None else "",
                include_ai_interpretation='ai_interpretation' in options,
                ai_text=ai_text,
                summary_stats=summary_stats if 'general_summary' in options else None,
                language=language,
            )

            try:
                export_survival_analysis_to_pdf(**export_kwargs)
            except TypeError as type_error:
                # Support partial hot reloads with an older exporter signature in memory.
                if "include_rsf_profile" not in str(type_error):
                    raise

                fallback_kwargs = dict(export_kwargs)
                fallback_kwargs.pop('include_rsf_profile', None)
                fallback_kwargs.pop('rsf_profile_figure', None)
                fallback_kwargs.pop('rsf_profile_text', None)
                export_survival_analysis_to_pdf(**fallback_kwargs)

            return dcc.send_file(pdf_path), "", {'display': 'none'}

        except Exception as e:
            print(f"""ERROR in download_rsf_pdf: {str(e)}""")
            print(traceback.format_exc())
            return None, (f"❌ Error al generar el PDF de RSF: {str(e)}" if language == 'es' else f"❌ Error generating the RSF PDF: {str(e)}"), {
                'display': 'block',
                'marginTop': '15px',
                'padding': '10px 12px',
                'borderRadius': '6px',
                'backgroundColor': '#fff1f0',
                'color': '#c0392b',
                'border': '1px solid #f5c6cb',
                'fontSize': '14px',
                'fontWeight': 'bold'
            }


def _get_modal_styles(show=True):
    """Get modal styles."""
    if not show:
        return {'display': 'none'}, {'display': 'none'}
    
    overlay_style = {
        'display': 'block',
        'position': 'fixed',
        'top': 0,
        'left': 0,
        'width': '100%',
        'height': '100%',
        'backgroundColor': 'rgba(0,0,0,0.5)',
        'zIndex': 999
    }
    container_style = {
        'display': 'block',
        'position': 'fixed',
        'top': '50%',
        'left': '50%',
        'transform': 'translate(-50%, -50%)',
        'backgroundColor': 'white',
        'padding': '30px',
        'borderRadius': '12px',
        'boxShadow': '0 4px 20px rgba(0,0,0,0.3)',
        'zIndex': 1000,
        'width': '90%',
        'maxWidth': '500px',
        'maxHeight': '600px',
        'overflowY': 'auto'
    }
    return overlay_style, container_style
