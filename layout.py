"""Build the dashboard pages and visual components.

This module contains pure Dash layout functions: modals, analysis pages,
tables, and panels. Callbacks for these components are registered
in analysis_callbacks.py and pdf_callbacks.py.
"""

import pandas as pd
from pathlib import Path
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output  
import plotly.express as px
import base64
import io
from kaplan_meier import plot_kaplan_meier, plot_km_G, plot_km_disc 
import numpy as np
from translations import get_translation, get_display_text, get_display_records
from pdf_exporter import export_survival_analysis_to_pdf

# Local cache for the global Kaplan-Meier view. Populate only when a page
# needs it, avoiding CSV loading during module import.
_LAYOUT_DF = None

# Map age_band from one-hot encoding.
def map_age_band(row):
    """Combine one-hot age columns into a readable label."""
    if row.get('age_band_0-35', 0) == 1:
        return '0-35 años'
    elif row.get('age_band_35-55', 0) == 1:
        return '35-55 años'
    elif row.get('age_band_55<=', 0) == 1:
        return '55+ años'
    return 'Desconocido'

# Map highest_education from one-hot encoding.
def map_highest_education(row):
    """Combine one-hot education columns into a readable label."""
    education_map = {
        'highest_education_A Level or Equivalent': 'A Level o Equivalente',
        'highest_education_HE Qualification': 'Calificación HE',
        'highest_education_Lower Than A Level': 'Inferior a A Level',
        'highest_education_No Formal quals': 'Sin Cualificaciones',
        'highest_education_Post Graduate Qualification': 'Postgrado'
    }
    for col, label in education_map.items():
        if col in row.index and row[col] == 1:
            return label
    return 'Desconocido'

def _get_layout_df():
    """Load the dataset only when needed to avoid expensive work during import."""
    global _LAYOUT_DF
    if _LAYOUT_DF is not None:
        return _LAYOUT_DF

    base_dir = Path(__file__).resolve().parent
    candidate_paths = [
        base_dir / 'dataset_limpio.csv',
    ]

    loaded_df = None
    for csv_path in candidate_paths:
        if csv_path.exists():
            try:
                loaded_df = pd.read_csv(csv_path, sep=';')
                break
            except Exception:
                continue

    if loaded_df is None:
        _LAYOUT_DF = pd.DataFrame()
        return _LAYOUT_DF

    if 'final_result' in loaded_df.columns:
        loaded_df['abandono'] = loaded_df['final_result'].apply(lambda x: 1 if x == 'Withdrawn' else 0)

    if 'gender_F' in loaded_df.columns:
        loaded_df['gender'] = loaded_df['gender_F'].map({1: 'Femenino', 0: 'Masculino'})

    if 'disability_N' in loaded_df.columns:
        loaded_df['disability'] = loaded_df['disability_N'].map({1: 'Con Discapacidad', 0: 'Sin Discapacidad'})

    if all(col in loaded_df.columns for col in ['age_band_0-35', 'age_band_35-55', 'age_band_55<=']):
        loaded_df['age_band'] = loaded_df.apply(map_age_band, axis=1)

    if any(col in loaded_df.columns for col in ['highest_education_A Level or Equivalent', 'highest_education_HE Qualification',
                                                 'highest_education_Lower Than A Level', 'highest_education_No Formal quals',
                                                 'highest_education_Post Graduate Qualification']):
        loaded_df['highest_education'] = loaded_df.apply(map_highest_education, axis=1)

    _LAYOUT_DF = loaded_df
    return _LAYOUT_DF


def _create_ai_response_box(answer_id, loading_message, language='en'):
    """Create an AI response box with consistent formatting across all techniques."""
    return html.Div([
        dcc.Textarea(
            id=answer_id,
            placeholder=get_translation(language, 'respuesta'),
            className='ai-output-textarea',
            style={
                'width': '80%',
                'height': '400px',
                'resize': 'none',
                'whiteSpace': 'pre-wrap',
                'margin': '20px auto',
                'display': 'block',
                'border': '1px solid #ccc',
                'borderRadius': '8px',
                'fontSize': '14px',
                'padding': '10px',
                'overflowY': 'auto',
                'color': '#1a1a1a'
            },
            disabled=True
        ),
        html.Div(
            loading_message,
            className='textarea-loading-message'
        ),
    ], style={'textAlign': 'center'})


# ===== PDF EXPORT MODAL COMPONENT =====
def create_pdf_export_modal(modal_id, analysis_type="kaplan-meier", language='en'):
    """
    Create a modal for selecting PDF export content.

    Args:
        modal_id: Unique modal ID.
        analysis_type: Analysis type ('kaplan-meier', 'cox-regression', 'log-rank').
    """
    if analysis_type == 'weibull-exponential-combined':
        return html.Div([
            html.Div(
                id=f"{modal_id}-overlay",
                style={
                    'display': 'none',
                    'position': 'fixed',
                    'top': 0,
                    'left': 0,
                    'width': '100%',
                    'height': '100%',
                    'backgroundColor': 'rgba(0,0,0,0.5)',
                    'zIndex': 999,
                    'cursor': 'pointer'
                }
            ),
            html.Div(
                id=f"{modal_id}-container",
                style={
                    'display': 'none',
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
                    'maxWidth': '560px',
                    'maxHeight': '650px',
                    'overflowY': 'auto'
                },
                children=[
                    html.Div([
                        html.H2(
                            "Informe combinado: Weibull + Exponencial" if language == 'es' else "Combined report: Weibull + Exponential",
                            style={'margin': 0, 'color': '#2c3e50'}
                        ),
                        html.Button(
                            "X",
                            id=f"{modal_id}-close-btn",
                            style={
                                'position': 'absolute',
                                'top': '15px',
                                'right': '15px',
                                'background': 'none',
                                'border': 'none',
                                'fontSize': '24px',
                                'cursor': 'pointer',
                                'color': '#999'
                            }
                        )
                    ], style={'position': 'relative', 'marginBottom': '20px'}),

                    html.Div([
                        html.Label(
                            "Nombre del informe" if language == 'es' else "Report name",
                            style={'fontWeight': 'bold', 'color': '#34495e'}
                        ),
                        dcc.Input(
                            id=f"{modal_id}-filename",
                            type="text",
                            placeholder=("Ejemplo: informe_weibull_exponencial" if language == 'es' else "Example: weibull_exponential_report"),
                            style={
                                'width': '100%',
                                'padding': '8px',
                                'border': '1px solid #ddd',
                                'borderRadius': '6px',
                                'marginTop': '5px',
                                'boxSizing': 'border-box'
                            }
                        ),
                        html.Small(
                            "No incluir .pdf" if language == 'es' else "Do not include .pdf",
                            style={'display': 'block', 'color': '#999', 'marginTop': '5px'}
                        )
                    ], style={'marginBottom': '18px'}),

                    html.Div([
                        html.H4("Técnicas" if language == 'es' else "Techniques", style={'color': '#34495e', 'marginTop': 0}),
                        dcc.Checklist(
                            id=f"{modal_id}-techniques",
                            options=[
                                {'label': ' Weibull', 'value': 'weibull'},
                                {'label': ' Exponencial' if language == 'es' else ' Exponential', 'value': 'exponential'}
                            ],
                            value=['weibull', 'exponential'],
                            style={'marginBottom': '8px', 'lineHeight': '1.8'},
                            labelStyle={'display': 'block', 'marginBottom': '8px'}
                        )
                    ], style={'marginBottom': '14px', 'padding': '15px', 'backgroundColor': '#f8fbff', 'borderRadius': '8px'}),

                    html.Div([
                        html.H4("Contenido" if language == 'es' else "Content", style={'color': '#34495e', 'marginTop': 0}),
                        dcc.Checklist(
                            id=f"{modal_id}-content",
                            options=[
                                {'label': ' Resumen' if language == 'es' else ' Summary', 'value': 'summary'},
                                {'label': ' Tabla de resultados' if language == 'es' else ' Results table', 'value': 'table'},
                                {'label': ' Gráfica' if language == 'es' else ' Graph', 'value': 'graph'},
                                {'label': ' Interpretación IA (opcional)' if language == 'es' else ' AI interpretation (optional)', 'value': 'ai_interpretation'}
                            ],
                            value=['summary', 'table', 'graph'],
                            style={'marginBottom': '8px', 'lineHeight': '1.8'},
                            labelStyle={'display': 'block', 'marginBottom': '8px'}
                        )
                    ], style={'marginBottom': '14px', 'padding': '15px', 'backgroundColor': '#f8fbff', 'borderRadius': '8px'}),

                    html.Small(
                        "Nota: Exponencial se incluirá como caso particular del modelo Weibull."
                        if language == 'es' else
                        "Note: Exponential will be included as a special case of the Weibull model.",
                        style={'display': 'block', 'color': '#5d6d7e', 'fontStyle': 'italic', 'marginBottom': '18px'}
                    ),

                    html.Div([
                        html.Button(
                            "Cancelar" if language == 'es' else "Cancel",
                            id=f"{modal_id}-cancel-btn",
                            style={
                                'padding': '10px 20px',
                                'marginRight': '10px',
                                'backgroundColor': '#95a5a6',
                                'color': 'white',
                                'border': 'none',
                                'borderRadius': '6px',
                                'cursor': 'pointer',
                                'fontSize': '14px',
                                'fontWeight': 'bold'
                            }
                        ),
                        html.Button(
                            "Descargar PDF" if language == 'es' else "Download PDF",
                            id=f"{modal_id}-download-btn",
                            style={
                                'padding': '10px 20px',
                                'backgroundColor': '#3498db',
                                'color': 'white',
                                'border': 'none',
                                'borderRadius': '6px',
                                'cursor': 'pointer',
                                'fontSize': '14px',
                                'fontWeight': 'bold',
                                'boxShadow': '0 2px 8px rgba(52, 152, 219, 0.3)'
                            }
                        ),
                    ], style={'textAlign': 'right'}),

                    html.Div(
                        id=f"{modal_id}-error",
                        className='analysis-loading-target',
                        style={
                            'display': 'none',
                            'marginTop': '15px',
                            'padding': '10px 12px',
                            'borderRadius': '6px',
                            'backgroundColor': '#fff1f0',
                            'color': '#c0392b',
                            'border': '1px solid #f5c6cb',
                            'fontSize': '14px',
                            'fontWeight': 'bold'
                        },
                        **{'data-loading-message': 'Preparando informe PDF...' if language == 'es' else 'Preparing PDF report...'}
                    ),

                    dcc.Download(id=f"{modal_id}-download"),
                ]
            )
        ])

    # All options start UNCHECKED by default.
    default_values = []
    
    # For KM: create options without a table (hidden with CSS).
    if analysis_type == 'kaplan-meier':
        options = [
            {'label': f" {'Resumen general' if language == 'es' else 'Summary'}", 'value': 'summary'},
            {'label': f" {'Gráfica principal' if language == 'es' else 'Main graph'}", 'value': 'graph'},
            {'label': f" {'Interpretación automática IA' if language == 'es' else 'AI interpretation'}", 'value': 'ai_interpretation'},
        ]
    elif analysis_type == 'rsf':
        options = [
            {'label': f" {get_translation(language, 'rsf_pdf_option_general_summary')}", 'value': 'general_summary'},
            {'label': f" {get_translation(language, 'rsf_pdf_option_model_summary')}", 'value': 'model_summary'},
            {'label': f" {get_translation(language, 'rsf_pdf_option_graph')}", 'value': 'graph'},
            {'label': f" {get_translation(language, 'rsf_pdf_option_importance')}", 'value': 'importance'},
            {'label': f" {get_translation(language, 'rsf_pdf_option_profile')}", 'value': 'profile'},
            {'label': f" {get_translation(language, 'rsf_pdf_option_ai')}", 'value': 'ai_interpretation'},
        ]
    elif analysis_type == 'weibull':
        options = [
            {'label': f" {'Resumen general' if language == 'es' else 'Summary'}", 'value': 'summary'},
            {'label': f" {'Gráfica principal' if language == 'es' else 'Main graph'}", 'value': 'graph'},
            {'label': f" {'Tabla de resultados' if language == 'es' else 'Results table'}", 'value': 'table'},
            {'label': f" {'Interpretación automática IA' if language == 'es' else 'AI interpretation'}", 'value': 'ai_interpretation'},
        ]
    elif analysis_type == 'exponential':
        options = [
            {'label': f" {'Resumen del ajuste exponencial' if language == 'es' else 'Exponential fit summary'}", 'value': 'summary'},
            {'label': f" {'Gráfica principal' if language == 'es' else 'Main graph'}", 'value': 'graph'},
            {'label': f" {'Interpretación automática IA' if language == 'es' else 'AI interpretation'}", 'value': 'ai_interpretation'},
        ]
    else:  # cox-regression, log-rank
        options = [
            {'label': f" {'Resumen general' if language == 'es' else 'Summary'}", 'value': 'summary'},
            {'label': f" {'Gráfica principal' if language == 'es' else 'Main graph'}", 'value': 'graph'},
            {'label': f" {'Tabla de resultados' if language == 'es' else 'Results table'}", 'value': 'table'},
            {'label': f" {'Interpretación automática IA' if language == 'es' else 'AI interpretation'}", 'value': 'ai_interpretation'},
        ]
    
    return html.Div([
        # Dark modal overlay.
        html.Div(
            id=f"{modal_id}-overlay",
            style={
                'display': 'none',
                'position': 'fixed',
                'top': 0,
                'left': 0,
                'width': '100%',
                'height': '100%',
                'backgroundColor': 'rgba(0,0,0,0.5)',
                'zIndex': 999,
                'cursor': 'pointer'
            }
        ),
        
        # Modal box.
        html.Div(
            id=f"{modal_id}-container",
            style={
                'display': 'none',
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
            },
            children=[
                # Header
                html.Div([
                    html.H2(f"{get_translation(language, 'pdf_modal_title')}", style={'margin': 0, 'color': '#2c3e50'}),
                    html.Button(
                        "X",
                        id=f"{modal_id}-close-btn",
                        style={
                            'position': 'absolute',
                            'top': '15px',
                            'right': '15px',
                            'background': 'none',
                            'border': 'none',
                            'fontSize': '24px',
                            'cursor': 'pointer',
                            'color': '#999'
                        }
                    )
                ], style={'position': 'relative', 'marginBottom': '20px'}),
                
                # Export options.
                html.Div([
                    # Instructions.
                    html.P(
                        get_translation(language, 'pdf_modal_instruction'),
                        style={'color': '#555', 'fontSize': '14px', 'fontStyle': 'italic', 'marginBottom': '15px'}
                    ),
                    
                    # Content section.
                    html.Div([
                        html.H4(f"{get_translation(language, 'pdf_modal_content')}", style={'color': '#34495e', 'marginTop': 0}),
                        
                        dcc.Checklist(
                            id=f"{modal_id}-checklist-content",
                            options=options,  # Dynamic options based on the analysis.
                            value=default_values,  # All unchecked by default.
                            style={'marginBottom': '15px', 'lineHeight': '1.8'},
                            labelStyle={'display': 'block', 'marginBottom': '8px'}
                        )
                    ], style={'marginBottom': '20px', 'padding': '15px', 'backgroundColor': '#f8fbff', 'borderRadius': '8px'}),
                    
                    # File name.
                    html.Div([
                        html.Label(f"{get_translation(language, 'pdf_modal_filename_label')}", style={'fontWeight': 'bold', 'color': '#34495e'}),
                        dcc.Input(
                            id=f"{modal_id}-filename",
                            type="text",
                            placeholder=get_translation(language, 'pdf_modal_filename_placeholder'),
                            style={
                                'width': '100%',
                                'padding': '8px',
                                'border': '1px solid #ddd',
                                'borderRadius': '6px',
                                'marginTop': '5px',
                                'boxSizing': 'border-box'
                            }
                        ),
                        html.Small(
                            get_translation(language, 'pdf_modal_no_extension'),
                            style={'display': 'block', 'color': '#999', 'marginTop': '5px'}
                        )
                        ], style={'marginBottom': '20px'}),
                    ], style={'marginBottom': '20px'}),

                    # Action buttons.
                    html.Div([
                        html.Button(
                            get_translation(language, 'pdf_modal_cancel'),
                            id=f"{modal_id}-cancel-btn",
                            style={
                                'padding': '10px 20px',
                                'marginRight': '10px',
                                'backgroundColor': '#95a5a6',
                                'color': 'white',
                                'border': 'none',
                                'borderRadius': '6px',
                                'cursor': 'pointer',
                                'fontSize': '14px',
                                'fontWeight': 'bold'
                            }
                        ),
                        html.Button(
                            get_translation(language, 'pdf_modal_download'),
                            id=f"{modal_id}-download-btn",
                            style={
                                'padding': '10px 20px',
                                'backgroundColor': '#3498db',
                                'color': 'white',
                                'border': 'none',
                                'borderRadius': '6px',
                                'cursor': 'pointer',
                                'fontSize': '14px',
                                'fontWeight': 'bold',
                                'boxShadow': '0 2px 8px rgba(52, 152, 219, 0.3)'
                            }
                        ),
                    ], style={'textAlign': 'right'}),

                html.Div(
                    id=f"{modal_id}-error",
                    className='analysis-loading-target',
                    style={
                        'display': 'none',
                        'marginTop': '15px',
                        'padding': '10px 12px',
                        'borderRadius': '6px',
                        'backgroundColor': '#fff1f0',
                        'color': '#c0392b',
                        'border': '1px solid #f5c6cb',
                        'fontSize': '14px',
                        'fontWeight': 'bold'
                    },
                    **{'data-loading-message': 'Preparando informe PDF...' if language == 'es' else 'Preparing PDF report...'}
                ),
                
                # Download component.
                dcc.Download(id=f"{modal_id}-download"),
            ]
        ),
        
        # Store the current analysis data.
        dcc.Store(id=f"{modal_id}-analysis-data")
    ])


# Create pages dynamically with multilingual support.
def create_survival_analysis_page(language='en'):
    """Build the main survival analysis page."""
    techniques = [
        {
            'image': '/assets/weibull_en.svg' if language == 'en' else '/assets/weibull.png?v=2',
            'image_style': {'width': '78%', 'display': 'block', 'margin': '0 auto', 'marginTop': '30px', 'marginBottom': '20px'},
            'label_key': 'weibull_analysis',
            'href': '/survival-analysis/weibull'
        },
        {
            'image': '/assets/rsf.png?v=2',
            'image_style': {'width': '78%', 'display': 'block', 'margin': '0 auto', 'marginTop': '30px', 'marginBottom': '20px'},
            'label_key': 'random_survival_forest',
            'href': '/survival-analysis/rsf'
        },
        {
            'image': '/assets/exponential.png?v=2',
            'image_style': {'width': '78%', 'display': 'block', 'margin': '0 auto', 'marginTop': '30px', 'marginBottom': '20px'},
            'label_key': 'exponential_analysis',
            'href': '/survival-analysis/exponential'
        },
        {
            'image': '/assets/kaplan.png',
            'image_style': {'width': '60%', 'display': 'block', 'margin': '0 auto', 'marginTop': '30px', 'marginBottom': '15px'},
            'label_key': 'kaplan_meier',
            'href': '/survival-analysis/kaplan-meier'
        },
        {
            'image': '/assets/cox.png',
            'image_style': {'width': '60%', 'display': 'block', 'margin': '0 auto', 'marginTop': '30px', 'marginBottom': '30px'},
            'label_key': 'cox_regression',
            'href': '/survival-analysis/cox-regression'
        },
        {
            'image': '/assets/logrank.png',
            'image_style': {'width': '50%', 'display': 'block', 'margin': '0 auto', 'marginTop': '30px', 'marginBottom': '30px'},
            'label_key': 'log_rank_test',
            'href': '/survival-analysis/log-rank'
        },
    ]

    return html.Div([  
        html.H1(get_translation(language, 'survival_analysis'), style={'textAlign': 'center'}),
        # Internal navigation bar.
        html.Div([
            html.Div([
                html.Img(src=technique['image'], style=technique['image_style']),
                dcc.Link(get_translation(language, technique['label_key']), href=technique['href'], className='home-link'),
            ], className='survival-technique-card', style={'textAlign': 'center', 'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'marginTop': '20px', 'width': '30%', 'minWidth': '220px'})
            for technique in techniques
        ], className='survival-techniques-grid', style={'textAlign': 'center', 'display': 'flex', 'justifyContent': 'center', 'flexWrap': 'wrap', 'gap': '30px', 'marginTop': '20px', 'paddingBottom': '28px'}),  

        html.Div([
            dcc.Link(
                get_translation(language, 'navbar_techniques_comparison'),
                href='/survival-analysis/comparacion-tecnicas',
                className='home-link techniques-comparison-wide'
            )
        ], className='techniques-comparison-cta'),

        # Container for the Kaplan-Meier chart.
        html.Div(id='survival_analysis_page', children=[]),
    ])


def create_techniques_comparison_page(language='en'):
    """Build the page comparing the available techniques."""
    is_en = language == 'en'

    comparison_rows = [
        {
            'tipo_modelo': 'No paramétrico' if not is_en else 'Non-parametric',
            'tecnica': 'Kaplan-Meier',
            'hipotesis': (
                'Censura no informativa; estima supervivencia empírica sin forma funcional.'
                if not is_en else
                'Non-informative censoring; empirical survival estimate with no functional form.'
            ),
            'que_responde': (
                'Cómo evoluciona la permanencia y si existen diferencias visuales entre grupos.'
                if not is_en else
                'How retention evolves and whether visual differences appear between groups.'
            ),
            'ventajas': (
                'Muy interpretable; útil como línea base visual.'
                if not is_en else
                'Highly interpretable; useful as a visual baseline.'
            ),
            'limitaciones': (
                'No ajusta covariables simultáneamente.'
                if not is_en else
                'Does not model multiple covariates simultaneously.'
            ),
            'cuando_usarlo': (
                'Comparación descriptiva inicial entre grupos.'
                if not is_en else
                'Initial descriptive comparison across groups.'
            ),
        },
        {
            'tipo_modelo': 'Paramétrico' if not is_en else 'Parametric',
            'tecnica': 'Exponencial' if not is_en else 'Exponential',
            'hipotesis': (
                'Riesgo constante en el tiempo (hazard constante).'
                if not is_en else
                'Constant hazard over time.'
            ),
            'que_responde': (
                'Si una hipótesis simple de riesgo constante describe razonablemente los datos.'
                if not is_en else
                'Whether a simple constant-hazard assumption describes the data reasonably well.'
            ),
            'ventajas': (
                'Simple, estable y fácil de interpretar.'
                if not is_en else
                'Simple, stable, and easy to interpret.'
            ),
            'limitaciones': (
                'Puede infraajustar si el riesgo cambia con el tiempo.'
                if not is_en else
                'Can underfit when hazard changes over time.'
            ),
            'cuando_usarlo': (
                'Cuando se espera comportamiento aproximadamente constante del riesgo.'
                if not is_en else
                'When hazard is expected to be roughly constant.'
            ),
        },
        {
            'tipo_modelo': 'Paramétrico' if not is_en else 'Parametric',
            'tecnica': 'Weibull',
            'hipotesis': (
                'Riesgo monótono (creciente o decreciente) según parámetro de forma.'
                if not is_en else
                'Monotonic hazard (increasing/decreasing) driven by shape parameter.'
            ),
            'que_responde': (
                'Si el riesgo de abandono aumenta, disminuye o se mantiene aproximadamente estable.'
                if not is_en else
                'Whether dropout risk increases, decreases, or remains roughly stable.'
            ),
            'ventajas': (
                'Flexible y con parámetros interpretables.'
                if not is_en else
                'Flexible with interpretable parameters.'
            ),
            'limitaciones': (
                'Si el riesgo no es monótono puede no capturar bien el patrón.'
                if not is_en else
                'May fail when hazard is not monotonic.'
            ),
            'cuando_usarlo': (
                'Cuando se requiere modelo paramétrico más flexible que Exponencial.'
                if not is_en else
                'When a parametric model more flexible than Exponential is needed.'
            ),
        },
        {
            'tipo_modelo': 'Semiparamétrico' if not is_en else 'Semi-parametric',
            'tecnica': 'Regresión de Cox' if not is_en else 'Cox Regression',
            'hipotesis': (
                'Proporcionalidad de riesgos entre grupos/covariables.'
                if not is_en else
                'Proportional hazards across groups/covariates.'
            ),
            'que_responde': (
                'Qué covariables aumentan o reducen el riesgo relativo de abandono.'
                if not is_en else
                'Which covariates increase or reduce relative dropout risk.'
            ),
            'ventajas': (
                'Ajusta múltiples covariables sin asumir forma base del hazard.'
                if not is_en else
                'Handles multiple covariates without specifying baseline hazard shape.'
            ),
            'limitaciones': (
                'Sensibilidad a violaciones de proporcionalidad.'
                if not is_en else
                'Sensitive to proportional-hazards violations.'
            ),
            'cuando_usarlo': (
                'Para cuantificar efecto de covariables (hazard ratios).'
                if not is_en else
                'To quantify covariate effects (hazard ratios).'
            ),
        },
        {
            'tipo_modelo': 'Test no paramétrico' if not is_en else 'Non-parametric test',
            'tecnica': 'Log-Rank Test',
            'hipotesis': (
                'Compara igualdad global de curvas de supervivencia entre grupos.'
                if not is_en else
                'Tests global equality of survival curves between groups.'
            ),
            'que_responde': (
                'Si las diferencias entre curvas son estadísticamente significativas.'
                if not is_en else
                'Whether differences between survival curves are statistically significant.'
            ),
            'ventajas': (
                'Simple para contraste de hipótesis entre grupos.'
                if not is_en else
                'Simple hypothesis test for group differences.'
            ),
            'limitaciones': (
                'No estima tamaño de efecto multivariable por sí solo.'
                if not is_en else
                'Does not provide multivariable effect estimates by itself.'
            ),
            'cuando_usarlo': (
                'Para evaluar si las curvas difieren de forma estadísticamente significativa.'
                if not is_en else
                'To assess whether curves differ significantly.'
            ),
        },
        {
            'tipo_modelo': 'Ensamble ML' if not is_en else 'ML ensemble',
            'tecnica': 'Random Survival Forest',
            'hipotesis': (
                'No requiere proporcionalidad ni forma paramétrica explícita.'
                if not is_en else
                'No proportionality or explicit parametric form required.'
            ),
            'que_responde': (
                'Cómo se comporta el riesgo al combinar muchas variables e interacciones.'
                if not is_en else
                'How risk behaves when many variables and interactions are combined.'
            ),
            'ventajas': (
                'Captura relaciones no lineales e interacciones complejas.'
                if not is_en else
                'Captures non-linear effects and complex interactions.'
            ),
            'limitaciones': (
                'Menor interpretabilidad que Cox/KM.'
                if not is_en else
                'Lower interpretability than Cox/KM.'
            ),
            'cuando_usarlo': (
                'Cuando prima capacidad predictiva y complejidad de patrones.'
                if not is_en else
                'When predictive power and pattern complexity are priorities.'
            ),
        },
    ]

    table_df = pd.DataFrame(comparison_rows)

    workflow_steps = [
        {
            'title': '1. Lectura descriptiva' if not is_en else '1. Descriptive reading',
            'body': (
                'Kaplan-Meier resume la supervivencia observada y Log-Rank comprueba si las curvas se separan con significación estadística.'
                if not is_en else
                'Kaplan-Meier summarizes observed survival and Log-Rank tests whether curve separation is statistically significant.'
            ),
        },
        {
            'title': '2. Efecto de covariables' if not is_en else '2. Covariate effect',
            'body': (
                'Cox permite cuantificar el riesgo relativo asociado a varias variables mediante hazard ratios.'
                if not is_en else
                'Cox quantifies relative risk associated with several variables through hazard ratios.'
            ),
        },
        {
            'title': '3. Modelos de referencia' if not is_en else '3. Reference models',
            'body': (
                'Exponencial y Weibull aportan modelos paramétricos comparables, útiles para evaluar supuestos sobre la forma del riesgo.'
                if not is_en else
                'Exponential and Weibull provide comparable parametric models for evaluating assumptions about hazard shape.'
            ),
        },
        {
            'title': '4. Predicción flexible' if not is_en else '4. Flexible prediction',
            'body': (
                'Random Survival Forest se reserva para patrones complejos, perfiles individuales e importancia de variables.'
                if not is_en else
                'Random Survival Forest is reserved for complex patterns, individual profiles, and variable importance.'
            ),
        },
    ]

    metrics_rows = [
        {'tecnica': 'Kaplan-Meier', 'Predictivo': 2, 'Interpretabilidad': 5, 'Flexibilidad': 2},
        {'tecnica': ('Exponencial' if not is_en else 'Exponential'), 'Predictivo': 2, 'Interpretabilidad': 4, 'Flexibilidad': 2},
        {'tecnica': 'Weibull', 'Predictivo': 3, 'Interpretabilidad': 4, 'Flexibilidad': 3},
        {'tecnica': ('Regresión de Cox' if not is_en else 'Cox Regression'), 'Predictivo': 4, 'Interpretabilidad': 4, 'Flexibilidad': 3},
        {'tecnica': 'Log-Rank Test', 'Predictivo': 1, 'Interpretabilidad': 4, 'Flexibilidad': 1},
        {'tecnica': 'Random Survival Forest', 'Predictivo': 5, 'Interpretabilidad': 2, 'Flexibilidad': 5},
    ]
    metrics_df = pd.DataFrame(metrics_rows)
    required_metric_cols = ['Predictivo', 'Interpretabilidad', 'Flexibilidad']
    available_metric_cols = [col for col in required_metric_cols if col in metrics_df.columns]

    # Ensure the chart uses valid columns and numeric values on a comparable scale.
    if available_metric_cols:
        for col in available_metric_cols:
            metrics_df[col] = pd.to_numeric(metrics_df[col], errors='coerce').fillna(0).clip(lower=0, upper=5)

    metrics_df['score_global'] = (
        0.5 * metrics_df['Predictivo']
        + 0.3 * metrics_df['Interpretabilidad']
        + 0.2 * metrics_df['Flexibilidad']
    )
    metrics_df = metrics_df.sort_values('score_global', ascending=False)

    metrics_long = metrics_df.melt(
        id_vars='tecnica',
        value_vars=available_metric_cols,
        var_name='Metrica' if not is_en else 'Metric',
        value_name='Puntuacion' if not is_en else 'Score'
    )

    metric_translation = {
        'Predictivo': ('Capacidad predictiva' if not is_en else 'Predictive power'),
        'Interpretabilidad': ('Interpretabilidad' if not is_en else 'Interpretability'),
        'Flexibilidad': ('Flexibilidad' if not is_en else 'Flexibility'),
    }
    metric_col_name = 'Metrica' if not is_en else 'Metric'
    score_col_name = 'Puntuacion' if not is_en else 'Score'
    metrics_long[metric_col_name] = metrics_long[metric_col_name].map(metric_translation)

    fig = px.bar(
        metrics_long,
        x='tecnica',
        y=score_col_name,
        color=metric_col_name,
        barmode='group',
        title=(
            'Comparativa orientativa por técnica (escala normalizada 0-5)'
            if not is_en else
            'Indicative technique comparison (normalized 0-5 scale)'
        ),
        labels={
            'tecnica': ('Técnica' if not is_en else 'Technique'),
            'Puntuacion': ('Puntuación (0-5)' if not is_en else 'Score (0-5)'),
            'Score': 'Score (0-5)',
        },
        color_discrete_sequence=['#1f77b4', '#2ca02c', '#ff7f0e']
    )
    fig.update_layout(
        template='plotly_white',
        xaxis_tickangle=-20,
        margin=dict(l=30, r=20, t=65, b=40),
        legend_title_text=('Métrica evaluada' if not is_en else 'Evaluated metric')
    )
    fig.update_yaxes(range=[0, 5], dtick=1)

    best_predictive = metrics_df.loc[metrics_df['Predictivo'].idxmax(), 'tecnica']
    most_interpretable = metrics_df.loc[metrics_df['Interpretabilidad'].idxmax(), 'tecnica']
    most_flexible = metrics_df.loc[metrics_df['Flexibilidad'].idxmax(), 'tecnica']
    explanation_text = (
        f"En esta comparación orientativa, {best_predictive} destaca cuando el objetivo principal es maximizar rendimiento predictivo, "
        f"mientras que {most_interpretable} resulta más adecuado en contextos donde la trazabilidad del resultado y la explicación clínica o académica son prioritarias. "
        f"Por su parte, {most_flexible} es especialmente útil cuando se espera heterogeneidad entre perfiles y relaciones no lineales entre covariables y tiempo al evento. "
        "A nivel metodológico, una estrategia sólida suele comenzar con Kaplan-Meier y Log-Rank para describir diferencias entre grupos y comprobar si existen señales de separación en las curvas. "
        "Después, la Regresión de Cox permite cuantificar el efecto de múltiples covariables bajo la hipótesis de riesgos proporcionales. "
        "Si esa hipótesis no es suficientemente estable o se busca un ajuste paramétrico con supuestos explícitos, Exponencial y Weibull aportan modelos más estructurados, siendo Weibull generalmente más versátil al permitir cambios monótonos del riesgo en el tiempo. "
        "Finalmente, Random Survival Forest es una alternativa potente cuando prima la capacidad predictiva en escenarios complejos, aunque conviene acompañarlo de análisis de importancia de variables y de una lectura crítica de su menor interpretabilidad."
        if not is_en else
        f"In this indicative comparison, {best_predictive} stands out when the main goal is predictive performance, "
        f"whereas {most_interpretable} is preferable when result transparency and explainability are priorities. "
        f"Additionally, {most_flexible} is particularly useful when heterogeneous profiles and non-linear relationships between covariates and time-to-event are expected. "
        "Methodologically, a robust workflow often starts with Kaplan-Meier and Log-Rank to describe between-group differences and verify whether curves show meaningful separation. "
        "Next, Cox Regression is typically used to quantify multivariable effects under the proportional-hazards assumption. "
        "If that assumption is not sufficiently stable, or if a parametric formulation is preferred, Exponential and Weibull provide more structured alternatives, with Weibull usually being more versatile due to monotonic hazard variation. "
        "Finally, Random Survival Forest is a strong option for complex predictive settings, ideally complemented by variable-importance analysis and careful interpretation given its lower transparency."
    )

    return html.Div([
        html.H1(
            'Comparación de técnicas de supervivencia' if not is_en else 'Comparison of survival techniques',
            style={'textAlign': 'center', 'fontSize': '35px', 'marginBottom': '20px', 'color': '#1a1a1a', 'fontWeight': 'bold'}
        ),
        html.Div([
            html.P(
                'Esta página resume los principales métodos de supervivencia para facilitar la elección metodológica según objetivo analítico, supuestos y nivel de interpretabilidad.'
                if not is_en else
                'This page summarizes key survival methods to help select the right approach according to analytical goals, assumptions, and interpretability.',
                style={'margin': 0, 'fontSize': '1.02em', 'lineHeight': '1.6', 'color': '#34495e'}
            ),
        ], style={
            'backgroundColor': '#f8fbff',
            'padding': '22px',
            'borderRadius': '10px',
            'border': '1px solid #dfe9f3',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.06)',
            'margin': '0 20px 25px 20px'
        }),
        html.Div([
            html.Div([
                html.H3(step['title'], style={
                    'margin': '0 0 8px 0',
                    'fontSize': '1.05rem',
                    'color': '#1b3147',
                    'fontWeight': 'bold'
                }),
                html.P(step['body'], style={
                    'margin': 0,
                    'lineHeight': '1.55',
                    'color': '#526477',
                    'fontSize': '0.95rem'
                })
            ], style={
                'backgroundColor': 'white',
                'border': '1px solid #d8e0e8',
                'borderRadius': '8px',
                'boxShadow': '0 2px 8px rgba(15, 23, 42, 0.06)',
                'padding': '16px'
            })
            for step in workflow_steps
        ], style={
            'display': 'grid',
            'gridTemplateColumns': 'repeat(auto-fit, minmax(230px, 1fr))',
            'gap': '14px',
            'margin': '0 20px 25px 20px'
        }),
        html.Div([
            dash_table.DataTable(
                id='techniques-comparison-table',
                columns=[
                    {'name': ('Técnica' if not is_en else 'Technique'), 'id': 'tecnica'},
                    {'name': ('Tipo de modelo' if not is_en else 'Model type'), 'id': 'tipo_modelo'},
                    {'name': ('Qué responde' if not is_en else 'Question answered'), 'id': 'que_responde'},
                    {'name': ('Hipótesis' if not is_en else 'Hypothesis'), 'id': 'hipotesis'},
                    {'name': ('Ventaja' if not is_en else 'Advantage'), 'id': 'ventajas'},
                    {'name': ('Limitación' if not is_en else 'Limitation'), 'id': 'limitaciones'},
                    {'name': ('Uso recomendado' if not is_en else 'Recommended use'), 'id': 'cuando_usarlo'},
                ],
                data=table_df.to_dict('records'),
                style_table={'overflowX': 'auto', 'marginTop': '8px'},
                style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto', 'lineHeight': '16px', 'padding': '10px'},
                style_header={'fontWeight': 'bold', 'backgroundColor': '#f4f7f6'},
                style_data_conditional=[{'if': {'column_id': 'tecnica'}, 'fontWeight': 'bold'}]
            ),
        ], style={
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '10px',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
            'margin': '0 20px 25px 20px'
        }),
        html.Div([
            dcc.Graph(figure=fig, config={'responsive': True, 'displayModeBar': True})
        ], style={
            'backgroundColor': '#f8f9fa',
            'padding': '20px',
            'borderRadius': '10px',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
            'margin': '0 20px 25px 20px'
        }),
        html.Div([
            html.P(explanation_text, style={'marginBottom': 0, 'lineHeight': '1.7', 'color': '#2c3e50'})
        ], style={
            'backgroundColor': '#eef7ff',
            'padding': '20px',
            'borderRadius': '10px',
            'border': '1px solid #d4e8fb',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.05)',
            'margin': '0 20px 30px 20px'
        })
    ])

def create_weibull_analysis_page(language='en'):
    """Build the dedicated Weibull model view."""
    return html.Div([
        html.H1(
            get_translation(language, 'survival_analysis_prefix').format(name=get_translation(language, 'weibull_analysis')),
            style={'textAlign': 'center', 'fontSize': '35px', 'marginBottom': '20px', 'color': '#1a1a1a', 'fontWeight': 'bold'}
        ),

        html.Div([
            html.P(
                get_translation(language, 'weibull_intro'),
                style={'margin': 0, 'fontSize': '1.05em', 'lineHeight': '1.6', 'color': '#34495e'}
            ),
            html.P(
                get_translation(language, 'weibull_note'),
                style={'marginTop': '12px', 'marginBottom': 0, 'fontSize': '0.98em', 'lineHeight': '1.6', 'color': '#5d6d7e', 'fontStyle': 'italic'}
            ),
        ], style={
            'backgroundColor': '#f8fbff',
            'padding': '22px',
            'borderRadius': '10px',
            'border': '1px solid #dfe9f3',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.06)',
            'margin': '0 20px 25px 20px'
        }),

        html.Div(
            dcc.Loading(
                id='loading-weibull',
                type='circle',
                children=html.Div(
                    id='weibull-analysis-output',
                    className='analysis-loading-target',
                    **{'data-loading-message': 'Calculando modelo Weibull y preparando tabla y gráfica...' if language == 'es' else 'Fitting Weibull model and preparing table and chart...'}
                )
            ),
            style={'margin': '0 20px 30px 20px'}
        ),

        html.Div([
            html.Button(get_translation(language, 'explicar_weibull'), id='btn-weibull',
                       title=("Primero carga y preprocesa un dataset" if language == 'es' else "Load and preprocess a dataset first"),
                       style={'padding': '10px 20px', 'backgroundColor': '#1abc9c', 'color': 'white', 'border': 'none',
                             'borderRadius': '8px', 'cursor': 'pointer', 'fontSize': '14px', 'fontWeight': 'bold', 'marginRight': '10px'}),
            html.Button(f"{'Exportar a PDF' if language == 'es' else 'Export to PDF'}", id='export-weibull-btn',
                       title=("Necesitas resultados Weibull válidos para exportar" if language == 'es' else "You need valid Weibull results before exporting"),
                       style={'padding': '10px 20px', 'backgroundColor': '#e74c3c', 'color': 'white', 'border': 'none',
                             'borderRadius': '8px', 'cursor': 'pointer', 'fontSize': '14px', 'fontWeight': 'bold', 'marginRight': '10px'}),
            html.Button(
                "Exportar informe combinado" if language == 'es' else "Export combined report",
                id='export-weibexp-btn',
                title=("Requiere análisis Weibull y Exponencial calculados" if language == 'es' else "Requires computed Weibull and Exponential analyses"),
                style={'padding': '10px 20px', 'backgroundColor': '#2980b9', 'color': 'white', 'border': 'none',
                       'borderRadius': '8px', 'cursor': 'pointer', 'fontSize': '14px', 'fontWeight': 'bold'}
            )
        ], style={'textAlign': 'center', 'marginTop': '20px', 'marginBottom': '20px'}),

        _create_ai_response_box(
            'openai-answer-weibull',
            'Generando interpretación del modelo Weibull...' if language == 'es' else 'Generating Weibull model interpretation...',
            language
        ),

        create_pdf_export_modal('weibull-pdf-modal', 'weibull', language),
        create_pdf_export_modal('weibexp-pdf-modal', 'weibull-exponential-combined', language)
    ])

def create_exponential_analysis_page(language='en'):
    """Build the dedicated Exponential model view."""
    return html.Div([
        html.H1(
            get_translation(language, 'survival_analysis_prefix').format(name=get_translation(language, 'exponential_analysis')),
            style={'textAlign': 'center', 'fontSize': '35px', 'marginBottom': '20px', 'color': '#1a1a1a', 'fontWeight': 'bold'}
        ),

        html.Div([
            html.P(
                get_translation(language, 'exponential_intro'),
                style={'margin': 0, 'fontSize': '1.05em', 'lineHeight': '1.6', 'color': '#34495e'}
            ),
            html.P(
                get_translation(language, 'exponential_note'),
                style={'marginTop': '12px', 'marginBottom': 0, 'fontSize': '0.98em', 'lineHeight': '1.6', 'color': '#5d6d7e', 'fontStyle': 'italic'}
            ),
        ], style={
            'backgroundColor': '#f8fbff',
            'padding': '22px',
            'borderRadius': '10px',
            'border': '1px solid #dfe9f3',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.06)',
            'margin': '0 20px 25px 20px'
        }),

        html.Div(
            dcc.Loading(
                id='loading-exponential',
                type='circle',
                children=html.Div(
                    id='exponential-analysis-output',
                    className='analysis-loading-target',
                    **{'data-loading-message': 'Calculando modelo exponencial y preparando tabla y gráfica...' if language == 'es' else 'Fitting Exponential model and preparing table and chart...'}
                )
            ),
            style={'margin': '0 20px 30px 20px'}
        ),

          html.Div([
            html.Button(get_translation(language, 'explicar_exponential'), id='btn-exponential',
                                         title=("Primero carga y preprocesa un dataset" if language == 'es' else "Load and preprocess a dataset first"),
                     style={'padding': '10px 20px', 'backgroundColor': '#1abc9c', 'color': 'white', 'border': 'none',
                         'borderRadius': '8px', 'cursor': 'pointer', 'fontSize': '14px', 'fontWeight': 'bold', 'marginRight': '10px'}),
            html.Button(f"{'Exportar a PDF' if language == 'es' else 'Export to PDF'}", id='export-exponential-btn',
                                         title=("Necesitas resultados Exponencial válidos para exportar" if language == 'es' else "You need valid Exponential results before exporting"),
                     style={'padding': '10px 20px', 'backgroundColor': '#e74c3c', 'color': 'white', 'border': 'none',
                                                 'borderRadius': '8px', 'cursor': 'pointer', 'fontSize': '14px', 'fontWeight': 'bold', 'marginRight': '10px'}),
                        html.Button(
                                "Exportar informe combinado" if language == 'es' else "Export combined report",
                                id='export-weibexp-btn',
                                                                title=("Requiere análisis Weibull y Exponencial calculados" if language == 'es' else "Requires computed Weibull and Exponential analyses"),
                                style={'padding': '10px 20px', 'backgroundColor': '#2980b9', 'color': 'white', 'border': 'none',
                                             'borderRadius': '8px', 'cursor': 'pointer', 'fontSize': '14px', 'fontWeight': 'bold'}
                        ),
          ], style={'textAlign': 'center', 'marginTop': '20px', 'marginBottom': '20px'}),

        _create_ai_response_box(
            'openai-answer-exponential',
            'Generando interpretación del modelo exponencial...' if language == 'es' else 'Generating Exponential model interpretation...',
            language
        ),

        create_pdf_export_modal('exponential-pdf-modal', 'exponential', language),
        create_pdf_export_modal('weibexp-pdf-modal', 'weibull-exponential-combined', language),
    ])


def create_rsf_analysis_page(language='en'):
    """Build the dedicated Random Survival Forest view."""
    if language == 'es':
        profile_card_title = 'Simular perfil individual'
        profile_card_note = 'Selecciona un perfil concreto para obtener su curva de supervivencia estimada por RSF.'
        profile_button_label = 'Simular perfil'
        profile_gender_label = 'Género'
        profile_gender_options = [
            {'label': 'Femenino', 'value': 1},
            {'label': 'Masculino', 'value': 0},
        ]
        profile_disability_label = 'Discapacidad'
        profile_disability_options = [
            {'label': 'Con discapacidad', 'value': 0},
            {'label': 'Sin discapacidad', 'value': 1},
        ]
        profile_age_label = 'Edad'
        profile_age_options = [
            {'label': '0-35 años', 'value': 'age_band_0-35'},
            {'label': '35-55 años', 'value': 'age_band_35-55'},
            {'label': '55+ años', 'value': 'age_band_55<='},
        ]
        profile_education_label = 'Nivel educativo'
        profile_education_options = [
            {'label': 'A Level o equivalente', 'value': 'highest_education_A Level or Equivalent'},
            {'label': 'HE Qualification', 'value': 'highest_education_HE Qualification'},
            {'label': 'Inferior a A Level', 'value': 'highest_education_Lower Than A Level'},
            {'label': 'Postgrado', 'value': 'highest_education_Post Graduate Qualification'},
        ]
        profile_credits_label = 'Créditos estudiados'
        profile_credits_options = [
            {'label': 'Pocos créditos (~30)', 'value': 'few'},
            {'label': 'Créditos medios (~60)', 'value': 'medium'},
            {'label': 'Muchos créditos (~120)', 'value': 'many'},
        ]
    else:
        profile_card_title = 'Simulate individual profile'
        profile_card_note = 'Choose a concrete profile to obtain its RSF-estimated survival curve.'
        profile_button_label = 'Simulate profile'
        profile_gender_label = 'Gender'
        profile_gender_options = [
            {'label': 'Female', 'value': 1},
            {'label': 'Male', 'value': 0},
        ]
        profile_disability_label = 'Disability'
        profile_disability_options = [
            {'label': 'With disability', 'value': 0},
            {'label': 'Without disability', 'value': 1},
        ]
        profile_age_label = 'Age band'
        profile_age_options = [
            {'label': '0-35 years', 'value': 'age_band_0-35'},
            {'label': '35-55 years', 'value': 'age_band_35-55'},
            {'label': '55+ years', 'value': 'age_band_55<='},
        ]
        profile_education_label = 'Education level'
        profile_education_options = [
            {'label': 'A Level or Equivalent', 'value': 'highest_education_A Level or Equivalent'},
            {'label': 'HE Qualification', 'value': 'highest_education_HE Qualification'},
            {'label': 'Lower Than A Level', 'value': 'highest_education_Lower Than A Level'},
            {'label': 'Post Graduate Qualification', 'value': 'highest_education_Post Graduate Qualification'},
        ]
        profile_credits_label = 'Studied credits'
        profile_credits_options = [
            {'label': 'Few credits (~30)', 'value': 'few'},
            {'label': 'Medium credits (~60)', 'value': 'medium'},
            {'label': 'Many credits (~120)', 'value': 'many'},
        ]

    return html.Div([
        html.H1(
            get_translation(language, 'survival_analysis_prefix').format(name=get_translation(language, 'random_survival_forest')),
            style={'textAlign': 'center', 'fontSize': '35px', 'marginBottom': '20px', 'color': '#1a1a1a', 'fontWeight': 'bold'}
        ),

        html.Div([
            html.P(
                get_translation(language, 'rsf_intro'),
                style={'margin': 0, 'fontSize': '1.05em', 'lineHeight': '1.6', 'color': '#34495e'}
            ),
            html.P(
                get_translation(language, 'rsf_note'),
                style={'marginTop': '12px', 'marginBottom': 0, 'fontSize': '0.98em', 'lineHeight': '1.6', 'color': '#5d6d7e', 'fontStyle': 'italic'}
            ),
        ], style={
            'backgroundColor': '#f8fbff',
            'padding': '22px',
            'borderRadius': '10px',
            'border': '1px solid #dfe9f3',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.06)',
            'margin': '0 20px 25px 20px'
        }),

        html.Div(
            dcc.Loading(
                id='loading-rsf',
                type='circle',
                children=html.Div(
                    id='rsf-analysis-output',
                    className='analysis-loading-target',
                    **{'data-loading-message': 'Entrenando Random Survival Forest y preparando métricas...' if language == 'es' else 'Training Random Survival Forest and preparing metrics...'}
                )
            ),
            style={'margin': '0 20px 30px 20px'}
        ),

        html.Div([
            html.H3(profile_card_title, style={'textAlign': 'center', 'color': '#0d0d0d', 'fontWeight': 'bold', 'marginBottom': '10px'}),
            html.P(profile_card_note, style={'margin': '0 0 18px 0', 'fontSize': '0.98em', 'lineHeight': '1.6', 'color': '#5d6d7e', 'fontStyle': 'italic', 'textAlign': 'center'}),
            html.Div([
                html.Div([
                    html.Label(profile_gender_label, style={'fontWeight': 'bold', 'marginBottom': '6px', 'display': 'block'}),
                    dcc.Dropdown(id='rsf-profile-gender', options=profile_gender_options, value=1, clearable=False),
                ]),
                html.Div([
                    html.Label(profile_disability_label, style={'fontWeight': 'bold', 'marginBottom': '6px', 'display': 'block'}),
                    dcc.Dropdown(id='rsf-profile-disability', options=profile_disability_options, value=1, clearable=False),
                ]),
                html.Div([
                    html.Label(profile_age_label, style={'fontWeight': 'bold', 'marginBottom': '6px', 'display': 'block'}),
                    dcc.Dropdown(id='rsf-profile-age-band', options=profile_age_options, value='age_band_0-35', clearable=False),
                ]),
                html.Div([
                    html.Label(profile_education_label, style={'fontWeight': 'bold', 'marginBottom': '6px', 'display': 'block'}),
                    dcc.Dropdown(id='rsf-profile-education', options=profile_education_options, value='highest_education_A Level or Equivalent', clearable=False),
                ]),
                html.Div([
                    html.Label(profile_credits_label, style={'fontWeight': 'bold', 'marginBottom': '6px', 'display': 'block'}),
                    dcc.Dropdown(id='rsf-profile-credits', options=profile_credits_options, value='few', clearable=False),
                ]),
            ], style={
                'display': 'grid',
                'gridTemplateColumns': 'repeat(auto-fit, minmax(220px, 1fr))',
                'gap': '14px',
                'marginBottom': '18px'
            }),
            html.Div([
                html.Button(profile_button_label, id='rsf-profile-simulate-btn', style={'padding': '10px 20px', 'backgroundColor': '#8e44ad', 'color': 'white', 'border': 'none', 'borderRadius': '8px', 'cursor': 'pointer', 'fontSize': '14px', 'fontWeight': 'bold'}),
            ], style={'textAlign': 'center'}),
            html.Div(
                id='rsf-profile-output',
                className='analysis-loading-target',
                style={'marginTop': '18px'},
                **{'data-loading-message': 'Simulando perfil individual con el modelo RSF...' if language == 'es' else 'Simulating individual profile with the RSF model...'}
            ),
            html.Div([
                html.Button(get_translation(language, 'explicar_rsf'), id='btn-rsf',
                                                     title=("Primero carga y preprocesa un dataset" if language == 'es' else "Load and preprocess a dataset first"),
                           style={'padding': '10px 20px', 'backgroundColor': '#1abc9c', 'color': 'white', 'border': 'none',
                                 'borderRadius': '8px', 'cursor': 'pointer', 'fontSize': '14px', 'fontWeight': 'bold'}),
                html.Button(f"{'Exportar a PDF' if language == 'es' else 'Export to PDF'}", id='export-rsf-btn',
                                                 title=("Necesitas ejecutar RSF y obtener resultados antes de exportar" if language == 'es' else "Run RSF and get results before exporting"),
                         style={'padding': '10px 20px', 'backgroundColor': '#e74c3c', 'color': 'white', 'border': 'none',
                             'borderRadius': '8px', 'cursor': 'pointer', 'fontSize': '14px', 'fontWeight': 'bold', 'marginLeft': '10px'})
            ], style={'textAlign': 'center', 'marginTop': '18px', 'marginBottom': '16px'}),
            _create_ai_response_box(
                'openai-answer-rsf',
                'Generando interpretación RSF...' if language == 'es' else 'Generating RSF interpretation...',
                language
            ),
        ], style={
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '10px',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
            'margin': '0 20px 30px 20px'
        }),

            create_pdf_export_modal('rsf-pdf-modal', 'rsf', language),
        dcc.Store(id='rsf-analysis-data'),
        dcc.Store(id='rsf-profile-data')
    ])

def create_ver_dataset_page(language='en'):
    """Build the clean dataset inspection page."""
    layout_df = _get_layout_df()

    variable_info_catalog = [
        {
            'Variable': 'id_student',
            'Descripcion_es': 'Identificador único del estudiante.',
            'Descripcion_en': 'Unique student identifier.',
            'Valores': 'Entero (ID)'
        },
        {
            'Variable': 'date',
            'Descripcion_es': 'Tiempo de seguimiento usado en supervivencia (por ejemplo, días).',
            'Descripcion_en': 'Follow-up time used in survival analysis (for example, days).',
            'Valores': 'Numérico >= 0'
        },
        {
            'Variable': 'final_result',
            'Descripcion_es': 'Indicador de evento: 1 = abandono (evento), 0 = censurado/sin abandono.',
            'Descripcion_en': 'Event indicator: 1 = withdrawal (event), 0 = censored/no withdrawal.',
            'Valores': '0 o 1'
        },
        {
            'Variable': 'gender_F',
            'Descripcion_es': 'Género codificado: 1 = femenino, 0 = masculino.',
            'Descripcion_en': 'Encoded gender: 1 = female, 0 = male.',
            'Valores': '0 o 1'
        },
        {
            'Variable': 'disability_N',
            'Descripcion_es': 'Discapacidad codificada: 1 = con discapacidad, 0 = sin discapacidad.',
            'Descripcion_en': 'Encoded disability: 1 = with disability, 0 = without disability.',
            'Valores': '0 o 1'
        },
        {
            'Variable': 'age_band_0-35',
            'Descripcion_es': 'Indicador one-hot del grupo de edad 0-35.',
            'Descripcion_en': 'One-hot flag for age group 0-35.',
            'Valores': '0 o 1'
        },
        {
            'Variable': 'age_band_35-55',
            'Descripcion_es': 'Indicador one-hot del grupo de edad 35-55.',
            'Descripcion_en': 'One-hot flag for age group 35-55.',
            'Valores': '0 o 1'
        },
        {
            'Variable': 'age_band_55<=',
            'Descripcion_es': 'Indicador one-hot del grupo de edad 55+.',
            'Descripcion_en': 'One-hot flag for age group 55+.',
            'Valores': '0 o 1'
        },
        {
            'Variable': 'highest_education_A Level or Equivalent',
            'Descripcion_es': 'Indicador one-hot de nivel educativo A Level o equivalente.',
            'Descripcion_en': 'One-hot flag for A Level or equivalent education level.',
            'Valores': '0 o 1'
        },
        {
            'Variable': 'highest_education_HE Qualification',
            'Descripcion_es': 'Indicador one-hot de titulación HE Qualification.',
            'Descripcion_en': 'One-hot flag for HE Qualification.',
            'Valores': '0 o 1'
        },
        {
            'Variable': 'highest_education_Lower Than A Level',
            'Descripcion_es': 'Indicador one-hot de nivel educativo inferior a A Level.',
            'Descripcion_en': 'One-hot flag for education lower than A Level.',
            'Valores': '0 o 1'
        },
        {
            'Variable': 'highest_education_No Formal quals',
            'Descripcion_es': 'Indicador one-hot de ausencia de cualificaciones formales.',
            'Descripcion_en': 'One-hot flag for no formal qualifications.',
            'Valores': '0 o 1'
        },
        {
            'Variable': 'highest_education_Post Graduate Qualification',
            'Descripcion_es': 'Indicador one-hot de estudios de posgrado.',
            'Descripcion_en': 'One-hot flag for postgraduate qualification.',
            'Valores': '0 o 1'
        },
        {
            'Variable': 'studied_credits',
            'Descripcion_es': 'Créditos cursados por el estudiante (covariable numérica).',
            'Descripcion_en': 'Credits studied by the student (numeric covariate).',
            'Valores': 'Numérico'
        },
    ]

    variable_info_rows = [
        {
            'Variable': item['Variable'],
            'Significado': item['Descripcion_en'] if language == 'en' else item['Descripcion_es'],
            'Valores': item['Valores']
        }
        for item in variable_info_catalog
    ]

    event_row = {
        'Variable': 'Evento (final_result)' if language == 'es' else 'Event (final_result)',
        'Significado': (
            'Representa si ocurrió el evento en el análisis de supervivencia: 1 = abandono (evento), 0 = censura (no abandono durante el seguimiento).'
            if language == 'es' else
            'Represents whether the survival event occurred: 1 = withdrawal (event), 0 = censoring (no withdrawal during follow-up).'
        ),
        'Valores': '0 o 1'
    }
    variable_info_rows.insert(0, event_row)

    return html.Div([
        html.H1(get_translation(language, 'dataset_limpio'), style={'textAlign': 'center', 'fontSize': '2.5em'}),
        html.Div([
            dash_table.DataTable(
                id='clean-dataset-table',
                columns=[{"name": col, "id": col} for col in layout_df.columns],  
                data=get_display_records(layout_df, language),  # Convert the DataFrame into a format Dash can use.
                style_table={
                    'overflowX': 'auto',
                    'overflowY': 'hidden',
                    'height': '74vh',
                    'maxHeight': '74vh',
                    'width': '100%'
                },
                style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto', 'lineHeight': '15px'},  # Ensure text is aligned.
            ),
        ], style={'textAlign': 'center', 'marginTop': '20px', 'width': '100%'}),

        html.Div([
            html.Table([
                html.Thead(
                    html.Tr([
                        html.Th('Variable', style={'border': '1px solid #d0d7de', 'padding': '10px', 'textAlign': 'left', 'backgroundColor': '#f4f7f6'}),
                        html.Th('Valores' if language == 'es' else 'Values', style={'border': '1px solid #d0d7de', 'padding': '10px', 'textAlign': 'left', 'backgroundColor': '#f4f7f6'}),
                        html.Th('Significado' if language == 'es' else 'Meaning', style={'border': '1px solid #d0d7de', 'padding': '10px', 'textAlign': 'left', 'backgroundColor': '#f4f7f6'}),
                    ])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(row['Variable'], style={'border': '1px solid #d0d7de', 'padding': '9px', 'fontWeight': 'bold'}),
                        html.Td(row['Valores'], style={'border': '1px solid #d0d7de', 'padding': '9px'}),
                        html.Td(row['Significado'], style={'border': '1px solid #d0d7de', 'padding': '9px'}),
                    ])
                    for row in variable_info_rows
                ])
            ], style={'width': '100%', 'borderCollapse': 'collapse', 'backgroundColor': 'white'})
        ], style={
            'margin': '14px auto 24px auto',
            'width': '96%',
            'padding': '0'
        })
    ], style={'width': '99%', 'margin': '0 auto'})

# Create the covariate analysis page with multilingual support.
# Global CSS styles to improve the interface, now applied inline.
# These can be used in RadioItems labelStyle.

def create_covariate_analysis_page(language='en'):
    """Build the covariate selection and analysis page."""
    return html.Div([
        # Header
        html.Div([
            html.H1(
                get_translation(language, 'analisis_covariables_title'),
                style={
                    'textAlign': 'center',
                    'fontSize': '2.5em',
                    'color': '#2c3e50',
                    'marginBottom': '10px',
                    'fontWeight': 'bold'
                }
            ),
            html.P(
                get_translation(language, 'analisis_covariables_intro'),
                style={
                    'textAlign': 'center',
                    'fontSize': '1.1em',
                    'color': '#7f8c8d',
                    'marginBottom': '30px'
                }
            )
        ], className='section-header formal-card', style={'backgroundColor': '#f8fbff', 'padding': '30px', 'borderRadius': '10px', 'margin': '20px'}),
        
        # Selection buttons.
        html.Div([
            html.Div([
                html.Label(
                    get_translation(language, 'selecciona_el_analisis'),
                    style={'fontSize': '1.1em', 'fontWeight': 'bold', 'color': '#2c3e50', 'display': 'block', 'marginBottom': '15px'}
                ),
                dcc.RadioItems(
                    options=[
                        {'label': f"{get_translation(language, 'abandono_total')}", 'value': 'abandono'},
                        {'label': f"{get_translation(language, 'abandono_genero')}", 'value': 'gender'},
                        {'label': f"{get_translation(language, 'abandono_discapacidad')}", 'value': 'disability'},
                        {'label': f"{get_translation(language, 'abandono_age_band')}", 'value': 'age_band'},
                        {'label': f"{get_translation(language, 'abandono_highest_education')}", 'value': 'highest_education'},
                        {'label': f"{get_translation(language, 'abandono_studied_credits')}", 'value': 'studied_credits'}
                    ],
                    value='abandono',
                    id='covariables-dropdown',
                    labelStyle={
                        'display': 'inline-block',
                        'padding': '12px 20px',
                        'margin': '8px',
                        'border': '2px solid #ddd',
                        'borderRadius': '8px',
                        'cursor': 'pointer',
                        'backgroundColor': '#f8f9fa',
                        'transition': 'all 0.3s ease',
                        'fontWeight': '500',
                        'textAlign': 'center'
                    }
                )
            ], className='formal-card', style={
                'padding': '20px',
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
                'margin': '0 20px'
            })
        ]),
        
        # Main content.
        html.Div([
            html.Div([
                # Chart.
                html.Div([
                    dcc.Loading(
                        id="loading-covariables",
                        type="circle",
                        children=[
                            dcc.Graph(
                                id='covariables-graph',
                                className='analysis-loading-component',
                                style={'height': '500px'},
                                config={'responsive': True, 'displayModeBar': True}
                            ),
                            html.Div(
                                'Actualizando visualización de covariables...' if language == 'es' else 'Updating covariate visualization...',
                                className='inline-loading-message'
                            )
                        ]
                    )
                ], className='graph-container', style={
                    'width': 'calc(100% - 340px)',
                    'display': 'inline-block',
                    'verticalAlign': 'top',
                    'marginRight': '20px'
                }),
                
                # Explanation panel.
                html.Div([
                    html.H3(f"{get_translation(language, 'interpretacion')}", style={'color': '#1abc9c', 'marginTop': '0', 'fontWeight': 'bold'}),
                    html.Div(
                        id='graph-explanation',
                        className='explanation-panel',
                        style={'minHeight': '400px', 'overflow': 'auto'}
                    )
                ], className='interpretation-card formal-card', style={
                    'width': '320px',
                    'display': 'inline-block',
                    'verticalAlign': 'top',
                    'backgroundColor': '#f8fbff',
                    'padding': '20px',
                    'borderRadius': '10px',
                    'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'
                })
            ], className='covariate-main-layout', style={'padding': '20px', 'display': 'flex', 'gap': '0'})
        ], style={'margin': '20px'})
    ])

# Create the Cox regression page with multilingual support.
def create_cox_regression_page(language='en'):
    """Build the Cox regression view."""
    return html.Div([
        html.H1(get_translation(language, 'survival_analysis_prefix').format(name=get_translation(language, 'cox_regression')), style={'textAlign': 'center','fontSize': '35px'}),
    
        html.Div([
            dcc.Dropdown(
                id='covariables-dropdown-cox',
                options=[
                    {'label': get_translation(language, 'genero'), 'value': 'gender_F'},
                    {'label': get_translation(language, 'discapacidad'), 'value': 'disability_N'},
                    {'label': get_translation(language, 'edad_banda'), 'value': 'age_band'},
                    {'label': get_translation(language, 'educacion_mas_alta'), 'value': 'highest_education'},
                    {'label': get_translation(language, 'creditos_estudiados'), 'value': 'studied_credits'}
                ],
                placeholder=get_translation(language, 'elige_covariable_cox'),  
                multi=True,  # Allow selecting multiple covariates.
                style={'width': '60%', 'margin': 'auto'}
            ),
        ], style={'textAlign': 'center', 'marginTop': '30px'}),
        
        html.Div(
            dcc.Loading(
                id='loading-cox-regression',
                type='circle',
                children=html.Div(
                    id='cox-regression-output',
                    className='analysis-loading-target',
                    **{'data-loading-message': 'Ejecutando regresión de Cox y preparando tabla/forest plot...' if language == 'es' else 'Running Cox regression and preparing table/forest plot...'}
                )
            ),
            style={'textAlign': 'center', 'marginTop': '20px'}
        ),
        
        # Action buttons.
        html.Div([
            html.Button(get_translation(language, 'explicar_cox'), id='btn-cox', 
                     title=("Selecciona covariables y ejecuta Cox antes de explicar" if language == 'es' else "Select covariates and run Cox before generating an explanation"),
                       style={'padding': '10px 20px', 'backgroundColor': '#1abc9c', 'color': 'white', 'border': 'none',
                             'borderRadius': '8px', 'cursor': 'pointer', 'fontSize': '14px', 'fontWeight': 'bold', 'marginRight': '10px'}),
            html.Button(f"{'Exportar a PDF' if language == 'es' else 'Export to PDF'}", id='export-cox-btn',
                     title=("Requiere resultados Cox vigentes (si cambias dataset, recalcula)" if language == 'es' else "Requires up-to-date Cox results (recalculate after dataset changes)"),
                       style={'padding': '10px 20px', 'backgroundColor': '#e74c3c', 'color': 'white', 'border': 'none',
                             'borderRadius': '8px', 'cursor': 'pointer', 'fontSize': '14px', 'fontWeight': 'bold'}),
        ], style={'textAlign': 'center', 'marginTop': '20px', 'marginBottom': '20px'}),
        
        _create_ai_response_box(
            'openai-answer-cox',
            'Generando interpretación de Cox...' if language == 'es' else 'Generating Cox interpretation...',
            language
        ),
        
        # PDF export modal.
        create_pdf_export_modal('cox-pdf-modal', 'cox-regression', language),
        
        # Store the current analysis data.
        dcc.Store(id='cox-analysis-data')
    ])

# Create the Kaplan-Meier page with multilingual support.
def create_kaplan_meier_page(language='en'):
    """Build the Kaplan-Meier view."""
    layout_df = _get_layout_df()
    km_global_component = (
        plot_kaplan_meier(layout_df, language=language)
        if not layout_df.empty and all(col in layout_df.columns for col in ['date', 'final_result'])
        else html.Div(
            "No hay datos suficientes para mostrar Kaplan-Meier" if language == 'es' else "Not enough data to display Kaplan-Meier",
            style={'textAlign': 'center', 'color': '#b03a2e', 'fontWeight': 'bold', 'padding': '10px'}
        )
    )

    return html.Div([
    html.H1(get_translation(language, 'survival_analysis_prefix').format(name=get_translation(language, 'kaplan_meier')), 
                style={'textAlign': 'center', 'fontSize': '35px', 'marginBottom': '30px', 'color': '#1a1a1a', 'fontWeight': 'bold'}),
        
        # ===== GLOBAL CHART =====
        html.Div([
            html.H3(f"{'Curva de Kaplan-Meier General' if language == 'es' else 'General Kaplan-Meier curve'}", style={'textAlign': 'center', 'color': '#0d0d0d', 'fontWeight': 'bold'}),
            html.Div(id='km-global-div', children=km_global_component,
                    style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'})
        ], style={'marginBottom': '40px', 'boxShadow': '0 2px 8px rgba(0,0,0,0.1)', 'padding': '20px', 'borderRadius': '10px'}),
        
        # ===== SELECTION BUTTONS =====
        html.Div([
            html.Label(get_translation(language, 'selecciona_covariable_kaplan'), 
                      style={'fontWeight':'bold', 'fontSize': '1.1em', 'marginBottom': '15px', 'display': 'block', 'color': '#1a1a1a'}),
            html.Div([
                html.Button(get_translation(language, 'genero'), id='botonG',
                           style={'borderRadius': '8px', 'padding': '10px 20px', 'margin': '8px', 'cursor': 'pointer',
                                  'backgroundColor': '#f0f0f0', 'border': '2px solid #ddd', 'fontSize': '14px', 'fontWeight': '500', 'color': '#1a1a1a'}),
                html.Button(get_translation(language, 'discapacidad'), id='botonDisc',
                           style={'borderRadius': '8px', 'padding': '10px 20px', 'margin': '8px', 'cursor': 'pointer',
                                  'backgroundColor': '#f0f0f0', 'border': '2px solid #ddd', 'fontSize': '14px', 'fontWeight': '500', 'color': '#1a1a1a'}),
                html.Button(get_translation(language, 'edad_banda'), id='botonAge',
                           style={'borderRadius': '8px', 'padding': '10px 20px', 'margin': '8px', 'cursor': 'pointer',
                                  'backgroundColor': '#f0f0f0', 'border': '2px solid #ddd', 'fontSize': '14px', 'fontWeight': '500', 'color': '#1a1a1a'}),
                html.Button(get_translation(language, 'educacion_mas_alta'), id='botonEdu',
                           style={'borderRadius': '8px', 'padding': '10px 20px', 'margin': '8px', 'cursor': 'pointer',
                                  'backgroundColor': '#f0f0f0', 'border': '2px solid #ddd', 'fontSize': '14px', 'fontWeight': '500', 'color': '#1a1a1a'}),
                html.Button(get_translation(language, 'creditos_estudiados'), id='botonCredits',
                           style={'borderRadius': '8px', 'padding': '10px 20px', 'margin': '8px', 'cursor': 'pointer',
                                  'backgroundColor': '#f0f0f0', 'border': '2px solid #ddd', 'fontSize': '14px', 'fontWeight': '500', 'color': '#1a1a1a'}),
                html.Button(get_translation(language, 'ninguna'), id='botonNone',
                           style={'borderRadius': '8px', 'padding': '10px 20px', 'margin': '8px', 'cursor': 'pointer',
                                  'backgroundColor': '#d4edda', 'border': '2px solid #28a745', 'fontSize': '14px', 'fontWeight': '500', 'color': '#0d3d1a'}),
            ], style={'textAlign': 'center', 'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'}),
        ], style={'textAlign': 'center', 'marginTop': '30px', 'marginBottom': '30px', 'padding': '20px', 
                 'backgroundColor': '#f8fbff', 'borderRadius': '10px', 'border': '1px solid #ddd'}),
        
        # ===== COVARIATE CHART (shown only when a selection is made) =====
        html.Div(
            dcc.Loading(
                id='loading-km-covariate',
                type='circle',
                children=html.Div(
                    id='km-cov-div',
                    className='analysis-loading-target',
                    **{'data-loading-message': 'Calculando curva Kaplan-Meier para la covariable seleccionada...' if language == 'es' else 'Computing Kaplan-Meier curve for the selected covariate...'}
                )
            ),
            style={'marginTop': '40px', 'padding': '20px', 'backgroundColor': '#f8f9fa',
                   'borderRadius': '10px', 'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'}
        ),
        
        # ===== ACTION BUTTONS =====
        html.Div([
            html.Button(get_translation(language, 'explicar_kaplan'), id='explicar-btn-kaplan', 
                     title=("Selecciona una covariable para generar una explicación estratificada" if language == 'es' else "Select a covariate to generate a stratified explanation"),
                       style={'padding': '10px 20px', 'backgroundColor': '#1abc9c', 'color': 'white', 'border': 'none',
                             'borderRadius': '8px', 'cursor': 'pointer', 'fontSize': '14px', 'fontWeight': 'bold', 'marginRight': '10px'}),
            html.Button(f"{'Exportar a PDF' if language == 'es' else 'Export to PDF'}", id='export-km-btn',
                     title=("Para incluir gráfica por grupo, selecciona primero una covariable" if language == 'es' else "To include grouped plot, select a covariate first"),
                       style={'padding': '10px 20px', 'backgroundColor': '#e74c3c', 'color': 'white', 'border': 'none',
                             'borderRadius': '8px', 'cursor': 'pointer', 'fontSize': '14px', 'fontWeight': 'bold'}),
        ], style={'textAlign': 'center', 'marginTop': '20px'}),
        
        # ===== EXPLANATION TEXT AREA =====
        _create_ai_response_box(
            'openai-answer-kaplan',
            'Generando interpretación Kaplan-Meier...' if language == 'es' else 'Generating Kaplan-Meier interpretation...',
            language
        ),
        
        # Store the current variable.
        dcc.Store(id='km-current-variable', data=''),
        
        # PDF export modal.
        create_pdf_export_modal('km-pdf-modal', 'kaplan-meier', language)
    ])

# Create the Log-Rank analysis page with multilingual support.
def create_log_rank_page(language='en'):
    """Build the Log-Rank test view."""
    return html.Div([
        html.H1(get_translation(language, 'survival_analysis_prefix').format(name=get_translation(language, 'log_rank_test')), style={'textAlign': 'center', 'fontSize': '35px'}),
        
        html.Div([
            dcc.Dropdown(
                id='covariables-dropdown-logrank',
                options=[
                    {'label': get_translation(language, 'genero'), 'value': 'gender_F'},
                    {'label': get_translation(language, 'discapacidad'), 'value': 'disability_N'},
                    {'label': get_translation(language, 'edad_banda'), 'value': 'age_band'},
                    {'label': get_translation(language, 'educacion_mas_alta'), 'value': 'highest_education'},
                    {'label': get_translation(language, 'creditos_estudiados'), 'value': 'studied_credits'}
                ],
                placeholder=get_translation(language, 'selecciona_covariable_logrank'),  
                value=[], 
                multi=True,
                style={'width': '60%', 'margin': 'auto'}
            ),
        ], style={'textAlign': 'center', 'marginTop': '30px'}),

        # This div is updated with the Log-Rank test results.
        html.Div(
            dcc.Loading(
                id='loading-logrank-test',
                type='circle',
                children=html.Div(
                    id='logrank-test-output',
                    className='analysis-loading-target',
                    **{'data-loading-message': 'Ejecutando Test Log-Rank y preparando curvas comparativas...' if language == 'es' else 'Running Log-Rank test and preparing comparative curves...'}
                )
            ),
            style={'textAlign': 'center', 'marginTop': '20px'}
        ),
        
        # Action buttons.
        html.Div([
            html.Button(get_translation(language, 'explicar_logrank'), id='explicar-btn-logrank', 
                     title=("Selecciona covariables y ejecuta Log-Rank antes de explicar" if language == 'es' else "Select covariates and run Log-Rank before generating an explanation"),
                       style={'padding': '10px 20px', 'backgroundColor': '#1abc9c', 'color': 'white', 'border': 'none',
                             'borderRadius': '8px', 'cursor': 'pointer', 'fontSize': '14px', 'fontWeight': 'bold', 'marginRight': '10px'}),
            html.Button(f"{'Exportar a PDF' if language == 'es' else 'Export to PDF'}", id='export-logrank-btn',
                     title=("Requiere resultados Log-Rank vigentes (si cambias dataset, recalcula)" if language == 'es' else "Requires up-to-date Log-Rank results (recalculate after dataset changes)"),
                       style={'padding': '10px 20px', 'backgroundColor': '#e74c3c', 'color': 'white', 'border': 'none',
                             'borderRadius': '8px', 'cursor': 'pointer', 'fontSize': '14px', 'fontWeight': 'bold'}),
        ], style={'textAlign': 'center', 'marginTop': '20px', 'marginBottom': '20px'}),

        _create_ai_response_box(
            'openai-answer-logrank',
            'Generando interpretación Log-Rank...' if language == 'es' else 'Generating Log-Rank interpretation...',
            language
        ),
        
        # PDF export modal.
        create_pdf_export_modal('logrank-pdf-modal', 'log-rank', language),
        
        # Store the current analysis data.
        dcc.Store(id='logrank-analysis-data')
    ])

def display_logrank_summary_table(result, language='en'):
    """Convert Log-Rank results into a readable Dash table."""
    df_show = result[['Covariable','Grupo A','Grupo B',
                      'n A','n B','test_statistic','p_value','-log2(p)',
                      'Decisión','Conclusión']].copy()

    return dash_table.DataTable(
        id='logrank-summary-table',
        columns=[{"name": get_display_text(language, col), "id": col} for col in df_show.columns],
        data=get_display_records(df_show, language),
        style_table={'height': '150px', 'overflowY': 'auto', 'width': '58%', 'margin': 'auto'},
        style_cell={'textAlign': 'center', 'whiteSpace': 'normal', 'height': 'auto', 'lineHeight': '15px'},
        style_header={'fontWeight': 'bold', 'backgroundColor': '#f4f7f6'},
        style_data_conditional=[
            {'if': {'filter_query': '{decision} = "Rechazar H0"'}, 'backgroundColor': '#ffecec'},
            {'if': {'filter_query': '{decision} = "No rechazar H0"'}, 'backgroundColor': '#ecffec'},
        ]
    )


