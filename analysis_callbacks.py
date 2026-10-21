"""Main callbacks for the survival analysis dashboard.

This module contains the interactive Dash logic: CSV upload and validation,
preprocessing, navigation between views, running analyses, and generating
interpretations using local AI.
"""

import os
import platform as _platform
from collections import namedtuple

# Workaround for Python 3.13 on some Windows systems: platform.system() may
# hang on a WMI query when importing Werkzeug/Dash.
if os.name == 'nt':
    _Uname = namedtuple('Uname', ['system', 'node', 'release', 'version', 'machine', 'processor'])
    _fake_uname = _Uname('Windows', 'localhost', '10', '10.0', 'AMD64', 'AMD64')
    _platform.uname = lambda: _fake_uname
    _platform.system = lambda: _fake_uname.system
    _platform.machine = lambda: _fake_uname.machine
    _platform.processor = lambda: _fake_uname.processor

import dash
from dash import Dash, dcc, html, dash_table
import pandas as pd
from pathlib import Path
import dash_daq as daq
from dash.dependencies import Input, Output, State
import base64
import plotly.express as px
import io
from layout import (
    create_survival_analysis_page, create_covariate_analysis_page, 
    create_kaplan_meier_page, create_cox_regression_page, 
    create_log_rank_page, create_ver_dataset_page, 
    create_weibull_analysis_page, create_exponential_analysis_page,
    create_rsf_analysis_page,
    create_techniques_comparison_page,
    display_logrank_summary_table
)
from kaplan_meier import plot_kaplan_meier, plot_km_G, plot_km_disc
from cox_regression import run_cox_regression
from log_rank_test import perform_log_rank_test
from survival_plots import plot_logrank_curves, plot_cox_hazard_ratios
from preprocesamiento import preprocess_data, preprocess_csv_file_streaming
from weibull import build_weibull_analysis
from exponential import build_exponential_analysis
from rsf import build_rsf_analysis, build_rsf_profile_analysis
import matplotlib.pyplot as plt
import requests
from ollama_AI import generate_interpretation_for_pdf, responder_pregunta_con_llama3
import plotly.graph_objs as go
from dash import callback_context
from dash.exceptions import PreventUpdate
from translations import get_covariate_label, get_feature_label, get_plotly_config, get_result_column_label, get_translation, get_display_text, get_display_records
from pdf_callbacks import register_pdf_export_callbacks  # PDF export ENABLED

# Initialize the Dash application.

# Project paths used by callbacks to read temporary data
# and the clean dataset regardless of Dash's working directory.
BASE_DIR = Path(__file__).resolve().parent
TEMP_DATA_PATH = BASE_DIR / 'data' / 'temp_data.csv'
CLEAN_DATA_PATH = BASE_DIR / 'dataset_limpio.csv'
DATASET_PREVIEW_MAX_ROWS = 100

# Simple in-memory cache. Avoid reading CSV files in every callback;
# initialize on demand with load_dataframes().
df = None
df_limpio = None

def load_dataframes():
    """Load dataframes on demand if they do not exist."""
    global df, df_limpio
    if df is None:
        try:
            df = pd.read_csv(TEMP_DATA_PATH, sep=';')
        except FileNotFoundError:
            print("⚠️  Warning: temp_data.csv not found")
    if df_limpio is None:
        try:
            df_limpio = pd.read_csv(CLEAN_DATA_PATH, sep=';')
        except FileNotFoundError:
            print("⚠️  Warning: dataset_limpio.csv not found")


def _read_split_json(json_value):
    """Read JSON with orient='split' without relying on literal strings."""
    if json_value is None:
        return None
    if isinstance(json_value, str):
        return pd.read_json(io.StringIO(json_value), orient='split')
    return pd.read_json(json_value, orient='split')


def _humanize_label(value):
    """Convert technical labels into readable text for narrative interpretation."""
    if value is None:
        return "N/A"

    text = str(value)
    replacements = {
        'highest_education_': 'education_',
        'age_band_': 'age_',
        'gender_F': 'Female',
        'disability_N': 'No disability',
        '_': ' ',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.strip()


def _dataset_signature_from_json(df_json):
    """Generate a stable signature to detect dataset changes in the workflow."""
    if not df_json:
        return ""
    return str(hash(df_json))


def _build_km_interpretation_context(df, variable_actual):
    """Build an interpretable summary of the current Kaplan-Meier curve."""
    if df is None or 'date' not in df.columns or 'final_result' not in df.columns:
        return ""

    from lifelines import KaplanMeierFitter
    from kaplan_meier import COVARIABLE_MAPPING

    context_lines = []
    kmf = KaplanMeierFitter()

    columns = COVARIABLE_MAPPING.get(variable_actual, [variable_actual])
    available_columns = [col for col in columns if col in df.columns]
    if not available_columns:
        return ""

    max_time = float(df['date'].max()) if len(df) else 0.0
    mid_followup_time = float(df['date'].median()) if len(df) else 0.0

    def _format_group_stats(label, group_df):
        """Summarize a visible Kaplan-Meier curve for the AI prompt."""
        if len(group_df) == 0:
            return ""

        kmf.fit(group_df['date'], event_observed=group_df['final_result'])
        final_survival = float(kmf.survival_function_.iloc[-1, 0]) if len(kmf.survival_function_) > 0 else 0.0
        mid_followup_survival = float(kmf.predict(mid_followup_time)) if mid_followup_time else final_survival
        max_survival = float(kmf.predict(max_time)) if max_time else final_survival
        km_median = kmf.median_survival_time_
        median_note = "not reached" if pd.isna(km_median) or km_median == float("inf") else f"{float(km_median):.0f}"
        events = int(group_df['final_result'].sum())
        event_rate = (events / len(group_df)) * 100

        return (
            f"- {label}: n={len(group_df)}, events={events} ({event_rate:.1f}%), "
            f"S(observed midpoint time={mid_followup_time:.0f}) approx. {mid_followup_survival:.3f}, "
            f"S(maximum observed time={max_time:.0f}) approx. {max_survival:.3f}, "
            f"final survival approx. {final_survival:.3f}, "
            f"Kaplan-Meier median={median_note}"
        )

    if len(columns) == 1 and variable_actual != 'studied_credits':
        col = columns[0]
        label_map = {
            'gender_F': {1: 'Female', 0: 'Male'},
            'disability_N': {1: 'Without disability', 0: 'With disability'}
        }
        df_filtered = df[df[col].notnull()]
        for group_value in sorted(df_filtered[col].unique()):
            group_df = df_filtered[df_filtered[col] == group_value]
            label = label_map.get(variable_actual, {}).get(group_value, f"{col} = {group_value}")
            line = _format_group_stats(label, group_df)
            if line:
                context_lines.append(line)

    elif len(columns) > 1:
        for col in columns:
            if col not in df.columns:
                continue
            group_df = df[df[col] == 1]
            if len(group_df) == 0:
                continue
            if col.startswith('age_band_'):
                label = col.replace('age_band_', '', 1)
            elif col.startswith('highest_education_'):
                label = col.replace('highest_education_', '', 1)
            else:
                label = col
            line = _format_group_stats(label, group_df)
            if line:
                context_lines.append(line)

    elif variable_actual == 'studied_credits':
        df_temp = df[['studied_credits', 'date', 'final_result']].dropna()
        if len(df_temp) > 0:
            unique_values = sorted(df_temp['studied_credits'].unique())
            if len(unique_values) > 10:
                df_temp['credits_group'] = pd.cut(df_temp['studied_credits'], bins=5, duplicates='drop')
            else:
                df_temp['credits_group'] = df_temp['studied_credits']

            groups = list(df_temp['credits_group'].dropna().unique())
            for group_value in groups:
                group_df = df_temp[df_temp['credits_group'] == group_value]
                label = f"Credits: {group_value}"
                line = _format_group_stats(label, group_df)
                if line:
                    context_lines.append(line)

            if not context_lines and len(unique_values) <= 5:
                unique_vals = sorted(df_temp['studied_credits'].unique())
                context_lines.append(f"- Visible credit values: {', '.join(map(str, unique_vals))}")

    if context_lines:
        context_lines.append(
            "Interpretation note: S(t) is estimated survival at observed times, rather than median survival. "
            "If the Kaplan-Meier median is not reached, do not claim that a median survival time exists. "
            "Compare the visual proximity of the curves and be cautious about curves with small n or perfect final survival."
        )

    return "\n".join(context_lines)


def _build_cox_interpretation_context(summary_df, selected_covariables=None, df=None, language='en'):
    """Build an accurate summary of the active Cox table for the AI prompt."""
    if summary_df is None or summary_df.empty:
        return ""

    from math import exp, isfinite
    from cox_regression import COVARIABLE_MAPPING

    selected_covariables = selected_covariables or []
    if isinstance(selected_covariables, str):
        selected_covariables = [item.strip() for item in selected_covariables.split(',') if item.strip()]

    label_map = {
        'gender_F': 'Femenino frente a Masculino',
        'disability_N': 'Sin discapacidad frente a Con discapacidad',
        'studied_credits': 'Créditos estudiados (por unidad)',
        'age_band_0-35': 'Edad 0-35',
        'age_band_35-55': 'Edad 35-55',
        'age_band_55<=': 'Edad 55 o más',
        'highest_education_A Level or Equivalent': 'A Level or Equivalent',
        'highest_education_HE Qualification': 'HE Qualification',
        'highest_education_Lower Than A Level': 'Lower Than A Level',
        'highest_education_Post Graduate Qualification': 'Post Graduate Qualification',
    }

    def _as_float(value):
        """Convert a numeric table value to float when possible."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _clean_label(covariable):
        """Convert internal column names into readable labels."""
        covariable = str(covariable)
        if covariable in label_map:
            return label_map[covariable]
        if covariable.startswith('age_band_'):
            return covariable.replace('age_band_', 'Edad ', 1)
        if covariable.startswith('highest_education_'):
            return covariable.replace('highest_education_', '', 1)
        return _humanize_label(covariable)

    def _selected_label(covariable):
        """Return the display name of the selected covariate."""
        covariable = str(covariable)
        if covariable == 'gender_F':
            return 'Género'
        if covariable == 'disability_N':
            return 'Discapacidad'
        if covariable == 'age_band':
            return 'Banda de edad'
        if covariable == 'highest_education':
            return 'Nivel educativo'
        if covariable == 'studied_credits':
            return 'Créditos estudiados'
        return _humanize_label(covariable)

    lines = []
    if df is not None and 'date' in df.columns and 'final_result' in df.columns:
        lines.append(
            f"Datos del modelo activo: n={len(df)}, eventos={int(df['final_result'].sum())}, "
            f"tiempo máximo observado={float(df['date'].max()):.0f}."
        )

    if selected_covariables:
        selected_labels = ", ".join(_selected_label(item) for item in selected_covariables)
        lines.append(f"Covariables seleccionadas en pantalla: {selected_labels}.")

    for cov in selected_covariables:
        mapped_cols = COVARIABLE_MAPPING.get(cov, [cov])
        existing_cols = [col for col in mapped_cols if df is None or col in df.columns]
        if len(existing_cols) > 1:
            reference = existing_cols[-1]
            lines.append(
                f"Referencia categórica para {_selected_label(cov)}: {_clean_label(reference)}. "
                "Los HR de sus categorías se interpretan frente a esta referencia."
            )

    lines.append("Resultados reales de la tabla Cox activa:")
    for _, row in summary_df.iterrows():
        cov_name = row.get('Covariable', row.get('Variable', 'N/A'))
        if str(cov_name).upper() == 'ERROR':
            continue

        hr = _as_float(row.get('exp(Coef.)', row.get('exp(coef)', None)))
        ci_low = _as_float(row.get('exp(Coef.) lower 95%', None))
        ci_high = _as_float(row.get('exp(Coef.) upper 95%', None))
        p_val = _as_float(row.get('p', None))

        if ci_low is None:
            coef_low = _as_float(row.get('Coef. lower 95%', None))
            ci_low = None if coef_low is None else float(exp(coef_low))
        if ci_high is None:
            coef_high = _as_float(row.get('Coef. upper 95%', None))
            ci_high = None if coef_high is None else float(exp(coef_high))

        if hr is None:
            continue

        significance = "significativa" if p_val is not None and p_val < 0.05 else "no significativa"
        point_direction = "HR puntual >1" if hr > 1 else "HR puntual <1"
        direction = (
            "mayor hazard/riesgo de abandono"
            if hr > 1 and significance == "significativa"
            else "menor hazard/riesgo de abandono"
            if hr < 1 and significance == "significativa"
            else "estimación puntual no concluyente"
        )
        ci_text = f"IC95% HR=[{ci_low:.3f}, {ci_high:.3f}]" if ci_low is not None and ci_high is not None else "IC95% HR=no disponible"
        p_text = f"p={p_val:.4f}" if p_val is not None else "p=no disponible"
        unstable_ci = (
            ci_low is not None and ci_high is not None and
            (ci_low <= 0 or not isfinite(ci_high) or ci_high > 1000)
        )
        crosses_one = (
            " El IC95% cruza 1, por lo que el efecto debe tratarse como incierto."
            if ci_low is not None and ci_high is not None and ci_low <= 1 <= ci_high else ""
        )
        unstable_note = (
            " El intervalo es extremadamente amplio o infinito, señal de estimación inestable; no interpretes la magnitud del HR como efecto real."
            if unstable_ci else ""
        )

        lines.append(
            f"- {_clean_label(cov_name)}: HR={hr:.3f}, {ci_text}, {p_text}, "
            f"{significance}; {point_direction}; interpretación: {direction}.{crosses_one}{unstable_note}"
        )

    lines.append(
        "Nota para interpretar: en Cox, HR>1 indica mayor hazard del evento y HR<1 indica menor hazard. "
        "No hagas afirmaciones causales; habla de asociación ajustada. "
        "Si p>=0.05 o el IC95% incluye 1, evita presentar el efecto como concluyente. "
        "Si el IC95% es infinito o extremadamente amplio, destaca inestabilidad y no interpretes la magnitud del HR."
    )

    return "\n".join(lines)


def _build_logrank_interpretation_context(logrank_store_data):
    """Convert active Log-Rank tables and curves into interpretable context."""
    if not isinstance(logrank_store_data, dict):
        return ""

    from lifelines import KaplanMeierFitter
    from log_rank_test import COVARIABLE_MAPPING

    def _clean_group_label(value, covariable=None):
        """Normalize group labels for display in the interpretation."""
        text = str(value)
        if covariable == 'gender_F':
            return {'0': 'Masculino', '1': 'Femenino'}.get(text, text)
        if covariable == 'disability_N':
            return {'0': 'Con discapacidad', '1': 'Sin discapacidad'}.get(text, text)
        if text.startswith('age_band_'):
            return text.replace('age_band_', '', 1)
        if text.startswith('highest_education_'):
            return text.replace('highest_education_', '', 1)
        if text.startswith('band_'):
            return text.replace('band_', '', 1)
        if text.startswith('education_'):
            return text.replace('education_', '', 1)
        return _humanize_label(text)

    def _selected_label(covariable):
        """Get the descriptive name of the covariate used in Log-Rank."""
        labels = {
            'gender_F': 'Género',
            'disability_N': 'Discapacidad',
            'age_band': 'Banda de edad',
            'highest_education': 'Nivel educativo',
            'studied_credits': 'Créditos estudiados',
        }
        return labels.get(str(covariable), _humanize_label(covariable))

    def _format_group_stats(label, group_df):
        """Summarize a group's size, events, and final survival."""
        if group_df is None or len(group_df) == 0:
            return ""

        kmf = KaplanMeierFitter()
        kmf.fit(group_df['date'], event_observed=group_df['final_result'])
        final_survival = float(kmf.survival_function_.iloc[-1, 0]) if len(kmf.survival_function_) > 0 else 0.0
        events = int(group_df['final_result'].sum())
        event_rate = (events / len(group_df)) * 100
        return f"{label}: n={len(group_df)}, eventos={events} ({event_rate:.1f}%), supervivencia final aprox. {final_survival:.3f}"

    def _build_curve_context(df, covariable):
        """Build curve context lines for the AI explanation."""
        if df is None or df.empty or 'date' not in df.columns or 'final_result' not in df.columns:
            return []

        columns = COVARIABLE_MAPPING.get(covariable, [covariable])
        curve_lines = []

        if covariable == 'studied_credits' and 'studied_credits' in df.columns:
            df_temp = df[['studied_credits', 'date', 'final_result']].dropna().copy()
            if df_temp.empty:
                return []
            unique_values = sorted(df_temp['studied_credits'].unique())
            if len(unique_values) <= 5:
                value_to_group = {val: i for i, val in enumerate(unique_values)}
                df_temp['group'] = df_temp['studied_credits'].map(value_to_group)
                labels = {i: f"Valor {unique_values[i]:.0f} créditos" for i in range(len(unique_values))}
            else:
                df_temp['group'] = pd.qcut(df_temp['studied_credits'], q=5, labels=False, duplicates='drop')
                labels = {int(group): f"Quintil {int(group) + 1}" for group in sorted(df_temp['group'].dropna().unique())}

            for group_value in sorted(df_temp['group'].dropna().unique()):
                group_df = df_temp[df_temp['group'] == group_value]
                line = _format_group_stats(labels.get(int(group_value), f"Grupo {group_value}"), group_df)
                if line:
                    curve_lines.append(line)

        elif len(columns) > 1:
            for col in columns:
                if col not in df.columns:
                    continue
                group_df = df[df[col] == 1]
                line = _format_group_stats(_clean_group_label(col, covariable), group_df)
                if line:
                    curve_lines.append(line)

        elif columns and columns[0] in df.columns:
            col = columns[0]
            for group_value in sorted(df[col].dropna().unique()):
                group_df = df[df[col] == group_value]
                label = _clean_group_label(group_value, covariable)
                line = _format_group_stats(label, group_df)
                if line:
                    curve_lines.append(line)

        return curve_lines

    result_blocks = []

    # Legacy format: {'results_json': '...'}
    if logrank_store_data.get('results_json'):
        result_blocks.append({
            'covariable': logrank_store_data.get('covariable', ''),
            'results_json': logrank_store_data.get('results_json')
        })

    # Current format: {'results': [{'results_json': '...'}, ...]}
    for item in logrank_store_data.get('results', []) or []:
        if isinstance(item, dict) and item.get('results_json'):
            result_blocks.append(item)

    if not result_blocks:
        return ""

    lines = []
    df_data = None
    if logrank_store_data.get('df_json'):
        try:
            df_data = _read_split_json(logrank_store_data.get('df_json'))
        except Exception:
            df_data = None

    selected_covariables = logrank_store_data.get('covariables', [])
    if selected_covariables:
        selected_labels = ", ".join(_selected_label(cov) for cov in selected_covariables)
        lines.append(f"Parámetro(s) Log-Rank seleccionado(s) en pantalla: {selected_labels}.")

    for block in result_blocks:
        try:
            results_df = _read_split_json(block.get('results_json'))
        except Exception:
            continue

        if results_df is None or results_df.empty:
            continue

        covariable = block.get('covariable', '')
        if covariable:
            lines.append(f"Curvas visibles para {_selected_label(covariable)}:")
            curve_lines = _build_curve_context(df_data, covariable)
            lines.extend(f"- {line}" for line in curve_lines)

        sortable_df = results_df.copy()
        sortable_df['_p_sort'] = pd.to_numeric(sortable_df.get('p_value', sortable_df.get('p', None)), errors='coerce')
        sortable_df = sortable_df.sort_values('_p_sort', na_position='last')

        lines.append(f"Comparaciones reales de la tabla Log-Rank para {_selected_label(covariable)}:")
        for _, row in sortable_df.head(10).iterrows():
            p_value = row.get('p_value', row.get('p', None))
            stat = row.get('test_statistic', row.get('chi2', None))
            conclusion = row.get('Conclusión', row.get('ConclusiÃ³n', ''))
            group_a = _clean_group_label(row.get('Grupo A', 'N/A'), covariable)
            group_b = _clean_group_label(row.get('Grupo B', 'N/A'), covariable)
            decision = row.get('Decisión', row.get('DecisiÃ³n', ''))
            significance = "significativo" if pd.notna(p_value) and float(p_value) < 0.05 else "no significativo"
            lines.append(
                f"- {group_a} vs {group_b}: chi2={stat}, p={p_value}, resultado={significance}, decisión={decision}, conclusión={conclusion}"
            )

    if lines:
        lines.append(
            "Nota para interpretar: Log-Rank contrasta si las curvas de supervivencia difieren entre grupos. "
            "Usa la tabla para significación estadística y las curvas para describir dirección/separación visual. "
            "Si p>=0.05, no afirmes diferencias concluyentes; si hay muchos contrastes pairwise, resume los más relevantes."
        )

    return "\n".join(lines)


def _build_weibull_interpretation_context(analysis, df=None, language='en'):
    """Summarize the active Weibull table and chart for the AI prompt."""
    if not analysis or not isinstance(analysis, dict):
        return ""

    shape = analysis.get('shape')
    scale = analysis.get('scale')
    median_survival = analysis.get('median_survival')
    event_rate = analysis.get('event_rate')
    n_observations = analysis.get('n_observations')
    n_events = analysis.get('n_events')
    weibull_aic = analysis.get('aic')
    best_fit_model = analysis.get('best_fit_model')
    comparison_text = analysis.get('model_comparison_interpretation', '')

    risk_text = (
        "aumenta con el tiempo" if shape is not None and shape > 1.05 else
        "disminuye con el tiempo" if shape is not None and shape < 0.95 else
        "se mantiene aproximadamente constante"
    )

    lines = [
        "Analisis Weibull activo: no hay covariable seleccionable; la tabla y la grafica son las mostradas en pantalla.",
        f"Tabla activa: n={n_observations}, eventos={n_events}, tasa de eventos={event_rate:.1f}%.",
        f"Parametros Weibull: shape/rho={shape:.4f}, scale/lambda={scale:.4f}, mediana estimada por Weibull={median_survival:.4f}.",
        f"Lectura del shape: el riesgo {risk_text}.",
    ]

    if weibull_aic is not None:
        lines.append(f"Ajuste Weibull: AIC={weibull_aic:.4f}.")
    if best_fit_model:
        lines.append(f"Comparacion de modelos activa: mejor ajuste por AIC={best_fit_model}. {comparison_text}")

    if df is not None and 'date' in df.columns and 'final_result' in df.columns:
        try:
            durations = pd.to_numeric(df['date'], errors='coerce').dropna()
            events = pd.to_numeric(df['final_result'], errors='coerce').dropna()
            if not durations.empty:
                max_observed_time = float(durations.max())
                lines.append(
                    f"Datos de la grafica: tiempo maximo observado={max_observed_time:.0f}; "
                    f"eventos observados={int(events.sum())}."
                )
                if median_survival is not None and median_survival > max_observed_time:
                    lines.append(
                        "Cautela: la mediana estimada por Weibull queda fuera del rango observado; interpretala como extrapolacion parametrica, no como mediana empirica observada."
                    )
        except Exception:
            pass

    figure = analysis.get('figure')
    if figure is not None and getattr(figure, 'data', None):
        try:
            trace_summaries = []
            for trace in figure.data:
                name = getattr(trace, 'name', 'curva')
                y_values = list(getattr(trace, 'y', []) or [])
                if y_values:
                    trace_summaries.append(f"{name}: supervivencia al final de la curva aprox. {float(y_values[-1]):.3f}")
            if trace_summaries:
                lines.append("Grafica activa: " + "; ".join(trace_summaries) + ".")
        except Exception:
            pass

    lines.append(
        "Nota para interpretar: usa la tabla para parametros y AIC; usa la grafica para comparar visualmente Kaplan-Meier empirico, Weibull ajustado y Exponencial ajustado. "
        "No afirmes causalidad ni hables de covariables, porque Weibull aqui modela la supervivencia global. "
        "Si la mediana estimada supera el tiempo maximo observado, menciona que es una extrapolacion del modelo."
    )

    return "\n".join(lines)


def _build_exponential_interpretation_context(analysis, df=None, language='en'):
    """Summarize the active Exponential table and chart for the AI prompt."""
    if not analysis or not isinstance(analysis, dict):
        return ""

    lambda_value = analysis.get('lambda_value')
    event_rate = analysis.get('event_rate')
    n_observations = analysis.get('n_observations')
    n_events = analysis.get('n_events')
    exponential_aic = analysis.get('aic')
    best_fit_model = analysis.get('best_fit_model')
    comparison_text = analysis.get('model_comparison_interpretation', '')

    lines = [
        "Analisis Exponencial activo: no hay covariable seleccionable; la tabla y la grafica son las mostradas en pantalla.",
        f"Tabla activa: n={n_observations}, eventos={n_events}, tasa de eventos={event_rate:.1f}%.",
        f"Parametro exponencial: lambda/tasa={lambda_value:.6f}.",
        "Supuesto del modelo: hazard/riesgo constante a lo largo del tiempo.",
    ]

    if exponential_aic is not None:
        lines.append(f"Ajuste Exponencial: AIC={exponential_aic:.4f}.")
    if best_fit_model:
        lines.append(f"Comparacion de modelos activa: mejor ajuste por AIC={best_fit_model}. {comparison_text}")

    if df is not None and 'date' in df.columns and 'final_result' in df.columns:
        try:
            durations = pd.to_numeric(df['date'], errors='coerce').dropna()
            events = pd.to_numeric(df['final_result'], errors='coerce').dropna()
            if not durations.empty:
                lines.append(
                    f"Datos de la grafica: tiempo maximo observado={float(durations.max()):.0f}; "
                    f"eventos observados={int(events.sum())}."
                )
        except Exception:
            pass

    figure = analysis.get('figure')
    if figure is not None and getattr(figure, 'data', None):
        try:
            trace_summaries = []
            for trace in figure.data:
                name = getattr(trace, 'name', 'curva')
                y_values = list(getattr(trace, 'y', []) or [])
                if y_values:
                    trace_summaries.append(f"{name}: supervivencia al final de la curva aprox. {float(y_values[-1]):.3f}")
            if trace_summaries:
                lines.append("Grafica activa: " + "; ".join(trace_summaries) + ".")
        except Exception:
            pass

    lines.append(
        "Nota para interpretar: usa la tabla para lambda, eventos y AIC; usa la grafica para comparar visualmente Kaplan-Meier empirico y curva Exponencial. "
        "Explica que Exponencial impone riesgo constante; si Weibull tiene menor AIC, menciona que el supuesto exponencial puede ser demasiado simple. "
        "No afirmes causalidad ni hables de covariables, porque Exponencial aqui modela la supervivencia global."
    )

    return "\n".join(lines)


def _build_rsf_interpretation_context(rsf_store_data, profile_store_data=None, language='en'):
    """Summarize the active RSF charts, table, and profile indicator."""
    if not isinstance(rsf_store_data, dict):
        return ""

    lines = [
        "Analisis RSF activo: usa la tabla resumen, curvas de supervivencia bajo/medio/alto riesgo, importancia de variables y perfil simulado visible."
    ]

    lines.append(
        f"Modelo global: observaciones={rsf_store_data.get('n_observations', 'N/A')}, "
        f"eventos={rsf_store_data.get('n_events', 'N/A')}, predictores={rsf_store_data.get('n_features', 'N/A')}, "
        f"c-index train={rsf_store_data.get('train_c_index', 'N/A')}, c-index OOB={rsf_store_data.get('oob_score', 'N/A')}."
    )

    top_feature = rsf_store_data.get('top_feature', 'N/A')
    top_importance = rsf_store_data.get('top_feature_importance', 'N/A')
    lines.append(f"Variable mas influyente activa: {top_feature} con importancia aproximada={top_importance}.")

    top_features = rsf_store_data.get('top_features', []) or []
    if top_features:
        feature_lines = []
        for item in top_features[:5]:
            feature_lines.append(f"{item.get('name', 'N/A')}={float(item.get('importance', 0.0)):.3f}")
        lines.append("Importancia de variables visible: " + "; ".join(feature_lines) + ".")

    curve_summaries = rsf_store_data.get('survival_curve_summaries', []) or []
    if curve_summaries:
        curve_lines = []
        for curve in curve_summaries:
            final_survival = curve.get('final_survival')
            final_text = f"{float(final_survival):.3f}" if final_survival is not None else "N/A"
            curve_lines.append(
                f"{curve.get('label', 'Curva')}: score={float(curve.get('risk_score', 0.0)):.3f}, supervivencia final aprox. {final_text}"
            )
        lines.append("Curvas globales activas: " + "; ".join(curve_lines) + ".")

    if isinstance(profile_store_data, dict):
        profile_summary = ""
        if profile_store_data.get('summary_json'):
            try:
                profile_df = _read_split_json(profile_store_data.get('summary_json'))
                if profile_df is not None and not profile_df.empty:
                    profile_parts = []
                    for _, row in profile_df.iterrows():
                        metric = row.get('Metrica', row.get('Metric', 'Métrica'))
                        value = row.get('Valor', row.get('Value', 'N/A'))
                        profile_parts.append(f"{metric}={value}")
                    profile_summary = "; ".join(profile_parts)
            except Exception:
                profile_summary = ""

        risk_score = profile_store_data.get('risk_score', None)
        risk_percentile = profile_store_data.get('risk_percentile', None)
        final_survival = profile_store_data.get('final_survival', None)
        end_time = profile_store_data.get('end_time', None)
        profile_text = (
            f"Perfil simulado activo: {profile_summary}. "
            f"Score de riesgo={float(risk_score):.3f}" if risk_score is not None else f"Perfil simulado activo: {profile_summary}. Score de riesgo=N/A"
        )
        if risk_percentile is not None:
            profile_text += f", percentil de riesgo aprox. {float(risk_percentile):.1f}"
        if final_survival is not None:
            profile_text += f", supervivencia final aprox. {float(final_survival):.3f}"
        if end_time is not None:
            profile_text += f" en t aprox. {float(end_time):.0f}"
        profile_text += "."
        lines.append(profile_text)
    else:
        lines.append("No hay perfil simulado activo disponible para interpretar.")

    lines.append(
        "Nota para interpretar: RSF es un modelo predictivo no paramétrico; no afirmes causalidad. "
        "La lectura global debe basarse en discriminación, separación de curvas e importancia de variables. "
        "La lectura individual debe basarse en el perfil simulado actualmente visible y cambia al modificar sus parámetros."
    )

    return "\n".join(lines)


def _interpretation_style_guidance(language='en'):
    """Shared instruction for AI to interpret results instead of copying tables."""
    if language == 'en':
        return (
            "Do not rewrite the table row by row. Start with the main conclusion suggested by the data, "
            "use only a few key numbers as evidence, explain what the active graph shows visually, "
            "and include a clear conclusion about the curve trend when curves are present: decreasing survival curves "
            "usually reflect accumulated events/dropouts over time, flat segments indicate few or no events in that interval, "
            "and abrupt drops or jumps suggest concentrated events or small group sizes. If a displayed curve increases, "
            "explain whether the axis represents risk, cumulative events, or another non-survival quantity. "
            "Close with a cautious interpretation for student dropout."
        )
    return (
        "No redactes la tabla fila por fila. Empieza por la conclusion principal que sugieren los datos, "
        "usa solo algunas cifras clave como evidencia, explica que muestra visualmente la grafica activa "
        "e incluye una conclusion clara sobre la tendencia de las curvas cuando existan: las curvas de supervivencia "
        "decrecientes suelen reflejar acumulacion de eventos o abandonos con el paso del tiempo, los tramos planos "
        "indican pocos o ningun evento en ese intervalo, y las caidas o saltos bruscos sugieren concentracion de eventos "
        "o grupos con pocos casos. Si una curva mostrada crece, explica si el eje representa riesgo, eventos acumulados "
        "u otra magnitud distinta de la supervivencia. Cierra con una interpretacion prudente sobre el abandono estudiantil."
    )



# HOME page function supporting dynamic translations.
def create_home_page(language='en'):
    """Build the home page with a video, CSV upload, and preprocessing button."""
    return html.Div([
        html.Div([ 
            html.Video(
                src='/assets/banner.mp4', 
                id='banner-video', 
                autoPlay=True, 
                muted=True, 
                loop=True, 
                style={
                    'width': '100%', 
                    'maxWidth': '100vw',
                    'maxHeight': '350px', 
                    'display': 'block', 
                    'marginTop': '0px', 
                    'marginBottom': '0px',
                    'objectFit': 'cover'  
                }
            )
        ], id="banner-container", style={'width': '100%', 'maxWidth': '100vw', 'padding': '0', 'margin': '0', 'overflow': 'hidden'}),

        html.Div([
            html.Div([
                html.H1(get_translation(language, 'dashboard_title'), className='home-title-formal', style={'textAlign': 'center'}),
                html.P(
                    'Entorno de análisis estadístico para el estudio de abandono y supervivencia académica.' if language == 'es'
                    else 'Statistical analysis environment for dropout and academic survival studies.',
                    className='home-subtitle-formal'
                ),

                dcc.Loading(
                    id="loading-spinner",
                    type="circle",
                    children=html.Div([
                        html.H3(get_translation(language, 'cargar_dataset'), id='upload-text', className='home-upload-title'),
                        dcc.Upload(
                            id='upload-data',
                            children=html.Button(get_translation(language, 'sube_csv'), className='home-upload-btn'),
                            multiple=False
                        ),
                        html.Div(id='output-data-upload', className='home-upload-feedback')
                    ], className='home-upload-block')
                ),

                html.Div([
                    html.Button(get_translation(language, 'preprocesa_csv'), id='load-clean', n_clicks=0, className='home-preprocess-btn'),
                ], className='home-preprocess-block')
            ], className='home-main-panel')
        ], style={'padding': '10px 0 30px 0'}),
    ])

# Home page.
home_page = create_home_page('es')
# Hide the initial "Load Dataset..." text.

def register_analysis_callbacks(app):
    """Register all Dash callbacks connecting the interface to the analyses."""
    @app.callback(
        Output('upload-text', 'style'),  # Change the text style.
        [Input('upload-data', 'contents')],
        prevent_initial_call=True
    )
    def hide_upload_text(contents):
        # Hide the text once a file has been uploaded.
        """Hide the initial upload text once the user has uploaded a file."""
        if contents is not None:
            return {'display': 'none'}  # Hide the text.
        return {'display': 'block'} 
    
    # Process the uploaded file and display it in a table.
    def display_data(df, title, language='en'):
        """Build a Dash table to preview the uploaded dataframe."""
        preview_df = df.head(DATASET_PREVIEW_MAX_ROWS).copy() if df is not None else pd.DataFrame()
        return html.Div([
            html.H5(title),
            html.Div([
                dash_table.DataTable(
                    id='data-table',
                    columns=[{"name": col, "id": col} for col in preview_df.columns],
                    data=preview_df.to_dict('records'),
                    page_size=10,
                    style_table={
                        'overflowX': 'auto',
                        'overflowY': 'auto',
                        'maxHeight': '400px',
                        'width': '100%',
                        'minWidth': '100%',
                        'maxWidth': '100%'
                    },
                    style_cell={
                        'textAlign': 'left',
                        'whiteSpace': 'normal',
                        'height': 'auto',
                        'lineHeight': '15px',
                        'minWidth': '125px',
                        'width': '125px',
                        'maxWidth': '210px'
                    },
                ),
            ], className='dataset-preview-table'),
        ], className='dataset-preview')
    # Load the CSV file.
    def parse_contents(contents, filename=None, preview_only=False):
        """
        Decode and load a CSV file from a Dash upload.
        Try to detect the correct delimiter automatically.

        Args:
            contents: Base64-encoded file contents.

        Returns:
            DataFrame containing the uploaded data.

        Raises:
            ValueError: If the file cannot be processed.
        """
        if filename == "temp_data.csv" and TEMP_DATA_PATH.exists():
            try:
                nrows = DATASET_PREVIEW_MAX_ROWS if preview_only else None
                return pd.read_csv(TEMP_DATA_PATH, sep=';', nrows=nrows)
            except Exception as disk_error:
                print(f"""⚠️  Could not read temp_data.csv from disk: {disk_error}""")

        try:
            if not isinstance(contents, str) or ',' not in contents:
                raise ValueError("The contents received by Dash are empty or incomplete.")

            content_type, content_string = contents.split(',')
            try:
                decoded = base64.b64decode(content_string)
            except MemoryError:
                raise ValueError("The file is too large to decode in memory.")

            best_candidate = None
            read_errors = []
            for encoding in ('utf-8', 'latin-1'):
                for separator in (';', ','):
                    try:
                        nrows = DATASET_PREVIEW_MAX_ROWS if preview_only else None
                        df_candidate = pd.read_csv(io.BytesIO(decoded), sep=separator, encoding=encoding, nrows=nrows)
                        if best_candidate is None:
                            best_candidate = df_candidate
                        if len(df_candidate.columns) > 1:
                            if separator == ',':
                                print("⚠️  Warning: Detected ',' as the delimiter instead of ';'")
                            return df_candidate
                    except UnicodeDecodeError:
                        read_errors.append(f"{encoding}/{separator}: invalid encoding")
                        continue
                    except Exception as read_error:
                        read_errors.append(f"{encoding}/{separator}: {read_error}")
                        continue

            if best_candidate is not None:
                return best_candidate

            details = "; ".join(read_errors[:3])
            raise ValueError(f"""Could not read the CSV. Check its format, delimiter, and encoding. {details}""")
                    
        except Exception as e:
            raise ValueError(f"""Error processing the file: {str(e)}""")
    
    def verificar_archivo_correcto(contents, filename):
        # Compare the uploaded filename with the expected filename.
        """Check that the uploaded filename matches the application's expected name."""
        archivo_esperado = "temp_data.csv"
        
        # Check whether the uploaded filename matches the expected name.
        if filename != archivo_esperado:
            return False
        return True
    
    
    def validate_uploaded_csv(df, filename, language='en'):
        """Validate the minimum CSV structure before preprocessing."""
        errors = []
    
        if not filename or not str(filename).lower().endswith('.csv'):
            errors.append(
                "El archivo debe tener extensión .csv."
                if language == 'es' else
                "The file must have .csv extension."
            )
    
        if df is None or df.empty:
            errors.append(
                "El CSV está vacío o no contiene filas válidas."
                if language == 'es' else
                "The CSV is empty or has no valid rows."
            )
            return errors
    
        if len(df.columns) <= 1:
            errors.append(
                "Formato CSV incorrecto: se detectó una sola columna. Revisa separador (; o ,) y codificación."
                if language == 'es' else
                "Invalid CSV format: only one column detected. Check delimiter (; or ,) and encoding."
            )
    
        required_columns = [
            'id_student', 'date', 'final_result',
            'gender_F', 'disability_N',
            'age_band_0-35', 'age_band_35-55', 'age_band_55<=',
            'highest_education_A Level or Equivalent',
            'highest_education_HE Qualification',
            'highest_education_Lower Than A Level',
            'highest_education_No Formal quals',
            'highest_education_Post Graduate Qualification',
            'studied_credits'
        ]
    
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(
                (
                    "Faltan columnas necesarias: " + ", ".join(missing_columns)
                ) if language == 'es' else (
                    "Missing required columns: " + ", ".join(missing_columns)
                )
            )
    
        critical_columns = ['id_student', 'date', 'final_result']
        null_critical = [col for col in critical_columns if col in df.columns and df[col].isna().any()]
        if null_critical:
            errors.append(
                (
                    "Hay valores nulos en columnas críticas: " + ", ".join(null_critical)
                ) if language == 'es' else (
                    "There are null values in critical columns: " + ", ".join(null_critical)
                )
            )
    
        if 'date' in df.columns:
            date_numeric = pd.to_numeric(df['date'], errors='coerce')
            if date_numeric.isna().any():
                errors.append(
                    "Formato incorrecto en 'date': debe ser numérico (sin textos)."
                    if language == 'es' else
                    "Invalid format in 'date': it must be numeric (no text values)."
                )
    
        if 'final_result' in df.columns:
            final_result_series = df['final_result'].astype(str).str.strip()
            final_result_lower = final_result_series.str.lower()
            invalid_final_result = final_result_series.eq('') | final_result_lower.eq('nan')
            allowed_final_results = {'withdrawn', 'pass', 'fail', 'distinction', '0', '1'}
            invalid_values = sorted(set(final_result_lower[~final_result_lower.isin(allowed_final_results)]))
            if invalid_final_result.any() or invalid_values:
                errors.append(
                    (
                        "Formato incorrecto en 'final_result': usa Withdrawn, Pass, Fail, Distinction, 0 o 1."
                    )
                    if language == 'es' else
                    "Invalid format in 'final_result': use Withdrawn, Pass, Fail, Distinction, 0 or 1."
                )
    
        return errors
    
    
    # Update the page and display the uploaded file.
    @app.callback(
        [Output('upload-data', 'style'),
         Output('load-clean', 'style'), 
         Output('output-data-upload', 'children'),
        Output('df-store', 'data'),
        Output('dataset-signature-store', 'data')], 
        [Input('upload-data', 'contents'),
         Input('upload-data', 'filename'), 
         Input('load-clean', 'n_clicks')],
        [State('language-store', 'data'),
         State('df-store', 'data')]
    )
    
    def update_output(contents, filename, n_clicks, language, current_df_json):
        """Handle CSV upload/preprocessing and update the dashboard stores."""
        try:
            if language not in ['es', 'en']:
                language = 'es'
    
            safe_clicks = n_clicks or 0
    
            if contents is None:
                # Keep an already preprocessed dataset in memory without forcing a reload.
                if current_df_json is not None:
                    try:
                        df_cached = _read_split_json(current_df_json)
                        return (
                            {'display': 'none'},
                            {'display': 'none'},
                            display_data(df_cached, get_translation(language, 'archivo_preprocesado'), language),
                            current_df_json,
                            _dataset_signature_from_json(current_df_json),
                        )
                    except Exception:
                        # Safely clear corrupted state if reconstruction fails.
                        return (
                            {'display': 'block'},
                            {'display': 'none'},
                            html.Div([get_translation(language, 'no_archivo_cargado')], style={'marginTop': '20px', 'marginBottom': '0px'}),
                            None,
                            '',
                        )
    
                return {'display': 'block'}, {'display': 'none'}, html.Div([get_translation(language, 'no_archivo_cargado')], style={'marginTop': '20px', 'marginBottom': '0px'}), None, ''
    
            if not verificar_archivo_correcto(contents, filename):
                return {'display': 'none'}, {'display': 'none'}, html.Div(
                    [
                        get_translation(language, 'error_archivo'),
                        html.Br(),
                        get_translation(language, 'archivo_correcto')
                    ],
                    style={
                        'color': 'red',
                        'fontSize': '20px',
                        'textAlign': 'center',
                        'marginTop': '20px'
                    }
                ), None, ''

            if safe_clicks > 0 and filename == "temp_data.csv" and TEMP_DATA_PATH.exists():
                try:
                    df_procesado = preprocess_csv_file_streaming(TEMP_DATA_PATH)
                    processed_json = df_procesado.to_json(date_format='iso', orient='split')
                    return {'display': 'none'}, {'display': 'none'}, display_data(df_procesado, get_translation(language, 'archivo_preprocesado'), language), processed_json, _dataset_signature_from_json(processed_json)
                except Exception as e:
                    return {'display': 'none'}, {'display': 'none'}, html.Div(
                        [
                            html.H3(f"❌ {get_translation(language, 'error_preprocess_title')}"),
                            html.P(str(e)),
                            html.Hr(),
                            html.P(get_translation(language, 'error_preprocess_body'))
                        ],
                        style={
                            'color': 'red',
                            'fontSize': '14px',
                            'padding': '20px',
                            'border': '2px solid red',
                            'borderRadius': '5px',
                            'marginTop': '20px',
                            'backgroundColor': '#fff5f5'
                        }
                    ), None, ''
    
            # Load the CSV file with error handling.
            try:
                df = parse_contents(contents, filename, preview_only=safe_clicks <= 0)
            except Exception as e:
                return {'display': 'none'}, {'display': 'none'}, html.Div(
                    [
                        html.H3(f"❌ {get_translation(language, 'error_loading_csv_title')}"),
                        html.P(get_translation(language, 'error_loading_csv_body')),
                        html.Ul([
                            html.Li(get_translation(language, 'error_loading_csv_tip_1')),
                            html.Li(get_translation(language, 'error_loading_csv_tip_2')),
                            html.Li(get_translation(language, 'error_loading_csv_tip_3'))
                        ]),
                        html.P(f"Detalles: {str(e)}")
                    ],
                    style={
                        'color': 'red',
                        'fontSize': '16px',
                        'padding': '20px',
                        'border': '2px solid red',
                        'borderRadius': '5px',
                        'marginTop': '20px'
                    }
                ), None, ''
    
            validation_errors = validate_uploaded_csv(df, filename, language)
            if validation_errors:
                return {'display': 'none'}, {'display': 'none'}, html.Div(
                    [
                        html.H3("❌ " + ("Validación del CSV fallida" if language == 'es' else "CSV validation failed")),
                        html.P(
                            "Revisa los siguientes errores antes de continuar:" if language == 'es'
                            else "Review the following errors before continuing:"
                        ),
                        html.Ul([html.Li(err) for err in validation_errors])
                    ],
                    style={
                        'color': 'red',
                        'fontSize': '16px',
                        'padding': '20px',
                        'border': '2px solid red',
                        'borderRadius': '5px',
                        'marginTop': '20px',
                        'backgroundColor': '#fff5f5'
                    }
                ), None, ''
    
            if safe_clicks > 0:
                # Run preprocessing with error handling.
                try:
                    df_procesado = preprocess_data(df)
                    processed_json = df_procesado.to_json(date_format='iso', orient='split')
                    return {'display': 'none'}, {'display': 'none'}, display_data(df_procesado, get_translation(language, 'archivo_preprocesado'), language), processed_json, _dataset_signature_from_json(processed_json)
                except (ValueError, TypeError) as e:
                    return {'display': 'none'}, {'display': 'none'}, html.Div(
                        [
                            html.H3(f"❌ {get_translation(language, 'error_preprocess_title')}"),
                            html.P(str(e)),
                            html.Hr(),
                            html.P(get_translation(language, 'error_preprocess_body'))
                            , html.Ul([
                                html.Li("id_student"),
                                html.Li("date"),
                                html.Li("final_result"),
                                html.Li("gender_F, disability_N"),
                                html.Li("age_band_* (0-35, 35-55, 55<=)"),
                                html.Li("highest_education_* (5 tipos)"),
                                html.Li("studied_credits")
                            ])
                        ],
                        style={
                            'color': 'red',
                            'fontSize': '14px',
                            'padding': '20px',
                            'border': '2px solid red',
                            'borderRadius': '5px',
                            'marginTop': '20px',
                            'backgroundColor': '#fff5f5'
                        }
                    ), None, ''
                except Exception as e:
                    return {'display': 'none'}, {'display': 'none'}, html.Div(
                        [
                            html.H3(f"❌ {get_translation(language, 'error_unexpected_title')}"),
                            html.P(get_translation(language, 'error_unexpected_body')),
                            html.P(f"Detalles: {str(e)}"),
                            html.P(get_translation(language, 'error_unexpected_contact'))
                        ],
                        style={
                            'color': 'red',
                            'fontSize': '14px',
                            'padding': '20px',
                            'border': '2px solid red',
                            'borderRadius': '5px',
                            'marginTop': '20px',
                            'backgroundColor': '#fff5f5'
                        }
                    ), None, ''
    
            # Display the raw file if the preprocessing button has not been clicked.
            return {'display': 'none'}, {'display': 'inline-block'}, display_data(df, get_translation(language, 'archivo_bruto'), language), None, ''
    
        except Exception as e:
            print(f"""[UPDATE_OUTPUT] Unhandled error: {e}""")
            return (
                {'display': 'block'},
                {'display': 'none'},
                html.Div(
                    f"Error interno del servidor: {str(e)}" if language == 'es' else f"Internal server error: {str(e)}",
                    style={'color': '#b03a2e', 'fontWeight': 'bold', 'marginTop': '20px'}
                ),
                current_df_json,
                _dataset_signature_from_json(current_df_json)
            )
    
    
    # Hide the navigation bar until the dataset has been cleaned.
    @app.callback(
        Output('navbar', 'style'),
        [Input('df-store', 'data')]
    )
    def toggle_navbar(df_json):
        """Show navigation only when a dataset is available."""
        if df_json is not None:
            return {'position': 'fixed', 'top': '0', 'left': '0', 'width': '100%', 'background-color': '#f4f7f6', 'padding': '10px', 'z-index': '1000'}
        return {'display': 'none'}  # Keep the navigation bar hidden until the button has been clicked.
    
    # Handle Kaplan-Meier navigation.
    @app.callback(
        [Output('km-cov-div', 'children'),
         Output('km-current-variable', 'data')],
        [Input('botonG', 'n_clicks'), Input('botonDisc', 'n_clicks'), 
         Input('botonAge', 'n_clicks'), Input('botonEdu', 'n_clicks'), 
         Input('botonCredits', 'n_clicks'), Input('botonNone', 'n_clicks'),
         Input('language-store', 'data')],
        [State('df-store', 'data'),
         State('km-current-variable', 'data')],
        prevent_initial_call=True
    )
    def update_km_cov(gender_clicks, disability_clicks, age_clicks, edu_clicks, credits_clicks, none_clicks, language, df_json, current_variable):
        """Determine which Kaplan-Meier covariate was selected and return its chart."""
        try:
            ctx = callback_context
            button_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else ''
            button_variables = {
                'botonG': 'gender_F',
                'botonDisc': 'disability_N',
                'botonAge': 'age_band',
                'botonEdu': 'highest_education',
                'botonCredits': 'studied_credits',
                'botonNone': '',
            }
            button_clicks = dict(zip(button_variables, (
                gender_clicks, disability_clicks, age_clicks,
                edu_clicks, credits_clicks, none_clicks,
            )))
            # Keep the selection when changing language or rebuilding the page.
            if button_id not in button_variables or not button_clicks[button_id]:
                button_id = next((
                    button for button, variable in button_variables.items()
                    if variable == current_variable
                ), 'botonNone')
            if button_id == 'botonNone':
                return None, ''
            
            # If no data is loaded, use the local clean dataset only as a fallback.
            if df_json is None:
                load_dataframes()
                if df_limpio is None:
                    raise PreventUpdate
                df = df_limpio.copy()
            else:
                # Rebuild the dataframe from JSON.
                df = _read_split_json(df_json)
            
            if button_id == 'botonG':
                from kaplan_meier import plot_km_by_covariate_with_figure
                graph_component, _ = plot_km_by_covariate_with_figure(df, 'gender_F', language=language)
                return html.Div([
                    html.H3("Kaplan-Meier Curve by Gender"if language == 'en'else "Curva de Kaplan-Meier por Género", style={'textAlign': 'center', 'color': '#0d0d0d', 'fontWeight': 'bold', 'marginBottom': '20px'}),
                    graph_component
                ]), 'gender_F'
            elif button_id == 'botonDisc':
                from kaplan_meier import plot_km_by_covariate_with_figure
                graph_component, _ = plot_km_by_covariate_with_figure(df, 'disability_N', language=language)
                return html.Div([
                    html.H3("Kaplan-Meier Curve by Disability"if language == 'en'else "Curva de Kaplan-Meier por Discapacidad", style={'textAlign': 'center', 'color': '#0d0d0d', 'fontWeight': 'bold', 'marginBottom': '20px'}),
                    graph_component
                ]), 'disability_N'
            elif button_id == 'botonAge':
                from kaplan_meier import plot_km_by_covariate_with_figure
                graph_component, _ = plot_km_by_covariate_with_figure(df, 'age_band', language=language)
                return html.Div([
                    html.H3("Kaplan-Meier Curve by Age Band"if language == 'en'else "Curva de Kaplan-Meier por Banda de Edad", style={'textAlign': 'center', 'color': '#0d0d0d', 'fontWeight': 'bold', 'marginBottom': '20px'}),
                    graph_component
                ]), 'age_band'
            elif button_id == 'botonEdu':
                from kaplan_meier import plot_km_by_covariate_with_figure
                graph_component, _ = plot_km_by_covariate_with_figure(df, 'highest_education', language=language)
                return html.Div([
                    html.H3("Kaplan-Meier Curve by Highest Education"if language == 'en'else "Curva de Kaplan-Meier por Educación Más Alta", style={'textAlign': 'center', 'color': '#0d0d0d', 'fontWeight': 'bold', 'marginBottom': '20px'}),
                    graph_component
                ]), 'highest_education'
            elif button_id == 'botonCredits':
                from kaplan_meier import plot_km_by_covariate_with_figure
                graph_component, _ = plot_km_by_covariate_with_figure(df, 'studied_credits', language=language)
                return html.Div([
                    html.H3("Kaplan-Meier Curve by Studied Credits"if language == 'en'else "Curva de Kaplan-Meier por Créditos Estudiados", style={'textAlign': 'center', 'color': '#0d0d0d', 'fontWeight': 'bold', 'marginBottom': '20px'}),
                    graph_component
                ]), 'studied_credits'
            # Fallback if no button is detected.
            return None, ''
        
        except Exception as e:
            # Handle Kaplan-Meier errors.
            import traceback
            error_div = html.Div([
                html.H3("❌ Error in Kaplan-Meier", style={'color': '#d32f2f', 'fontWeight': 'bold'}),
                html.P(f"Could not generate the chart: {str(e)}", style={'color': '#1a1a1a'}),
                html.Div(f"Technical details: {traceback.format_exc()}", style={'fontSize': '12px', 'color': '#666', 'whiteSpace': 'pre-wrap', 'marginTop': '10px'})
            ])
            return error_div, '' 
    
    
    
    @app.callback(
        Output('openai-answer-kaplan', 'value'),
        [Input('explicar-btn-kaplan', 'n_clicks')],
        [State('km-current-variable', 'data'),
         State('language-store', 'data'),
         State('df-store', 'data')],
        prevent_initial_call=True 
    )
    def explicar_kaplan(n_clicks, variable_actual, language, df_json):
        """Generate an explanation of the Kaplan-Meier chart using actual data."""
        try:
            if n_clicks is None or n_clicks <= 0:
                return ""
            
            if not variable_actual:
                return f"⚠️  {get_translation(language, 'error_select_variable')}"
            
            if language not in ['es', 'en']:
                language = 'es'
            
            # Extract actual statistics for the KM variable.
            stats_info = ""
            if df_json:
                try:
                    df = _read_split_json(df_json)
                    stats_info = _build_km_interpretation_context(df, variable_actual)
                except Exception as e:
                    print(f"""Error extracting KM statistics: {e}""")

            if not stats_info.strip():
                return (
                    "⚠️  Could not extract numerical data from the current Kaplan-Meier graph."
                    if language == 'en'
                    else "⚠️  No se pudieron extraer datos numéricos de la gráfica Kaplan-Meier actual."
                )
            
            # Build a prompt using actual data.
            if language == 'en':
                prompt = f"""Write an academic interpretation of the Kaplan-Meier graph currently displayed for '{_humanize_label(variable_actual)}'. Use only the real group results below and do not discuss variables that are not visible in this graph:
    {stats_info}
    {_interpretation_style_guidance(language)}
    Produce exactly 2 short paragraphs: first paragraph for statistical interpretation of the visible curves, second paragraph for practical implication and one limitation. Do not call the intermediate observed time a survival median. If the Kaplan-Meier median is not reached, say so. Mention when curves are visually close, and be cautious with groups that have small n or perfect final survival. Do not use bullet points. Finish with a complete sentence."""
            else:
                prompt = f"""Redacta una interpretación académica de la gráfica Kaplan-Meier mostrada actualmente para '{_humanize_label(variable_actual)}'. Usa solo los resultados reales de grupos siguientes y no menciones variables que no aparezcan en esta gráfica:
    {stats_info}
    {_interpretation_style_guidance(language)}
    Devuelve exactamente 2 párrafos breves: el primero con lectura estadística de las curvas visibles y el segundo con implicación práctica y una limitación. No llames mediana de supervivencia al tiempo intermedio observado. Si la mediana Kaplan-Meier no se alcanza, indícalo. Menciona cuando las curvas estén visualmente próximas y sé prudente con grupos de n pequeño o supervivencia final perfecta. Sin viñetas ni listas numeradas. Termina con una frase completa."""
            
            respuesta = responder_pregunta_con_llama3(prompt, language)
            return respuesta if respuesta else ("⚠️  Error: Empty response" if language == 'en' else "⚠️  Error: Respuesta vacía")
            
        except requests.exceptions.Timeout:
            return f"⚠️  {get_translation(language, 'error_timeout')}"
        except Exception as e:
            print(f"""❌ Error in explicar_kaplan: {str(e)}""")
            return f"❌ Error: {str(e)}"
    
    # Update the chart according to the Dropdown selection.
    @app.callback(
        [Output('covariables-graph', 'figure'),
        Output('graph-explanation', 'children')],
        [Input('covariables-dropdown', 'value'),
         Input('language-store', 'data')],
        [State('df-store', 'data')],
        prevent_initial_call=False
    )
    def update_graph(col_chosen, language, df_json):
        # VALIDATION 1: Read the dataset from the store.
        """Update the Kaplan-Meier visualization for the selected covariate."""
        df_data = None
        if df_json:
            try:
                df_data = _read_split_json(df_json)
            except:
                df_data = None
        
        # VALIDATION 2: Check that the dataset is loaded and is not empty.
        if df_data is None or df_data.empty:
            error_fig = go.Figure()
            error_fig.add_annotation(
                text=get_translation(language, 'covariate_error_load_dataset'),
                showarrow=False,
                font=dict(size=14, color='red')
            )
            error_fig.update_layout(title=get_translation(language, 'covariate_error_dataset_title'), template="plotly_white")
            return error_fig, html.Div([
                html.H4(get_translation(language, 'covariate_error_data_header'), style={'color': 'red'}),
                html.P(get_translation(language, 'covariate_error_data_body'))
            ])
        
        # VALIDATION 2: Check that language is valid.
        if language not in ['es', 'en']:
            language = 'es'
    
        def _pick_existing(df_local, candidates):
            """Return the first candidate column present in the dataframe."""
            for col in candidates:
                if col in df_local.columns:
                    return col
            return None
        
        try:
            if col_chosen == 'abandono':
                conteo_abandono = df_data['final_result'].value_counts().sort_index()
                fig = go.Figure()
                count_no_abandono = conteo_abandono[0] if 0 in conteo_abandono else 0
                count_abandono = conteo_abandono[1] if 1 in conteo_abandono else 0
                label_no_abandono = get_translation(language, 'no_abandono')
                label_abandono = get_translation(language, 'abandono')
                
                fig.add_trace(go.Bar(x=[label_no_abandono], y=[count_no_abandono], name='', marker_color='#1abc9c'))
                fig.add_trace(go.Bar(x=[label_abandono], y=[count_abandono], name='', marker_color='#006400'))
                fig.update_layout(
                    title=get_translation(language, 'abandono_vs_no_abandono'),
                    xaxis_title=get_translation(language, 'resultado_final'),
                    yaxis_title=get_translation(language, 'num_estudiantes'),
                    barmode='group'
                )
                return fig, get_translation(language, 'exp_abandono')
            
            elif col_chosen == 'gender':
                # Support datasets with the original column ('gender') or one-hot column ('gender_F').
                if 'gender' in df_data.columns:
                    plot_df = df_data.copy()
                    plot_df['gender_plot'] = plot_df['gender']
                elif 'gender_F' in df_data.columns:
                    plot_df = df_data.copy()
                    plot_df['gender_plot'] = plot_df['gender_F'].apply(
                        lambda x: get_translation(language, 'femenino') if x == 1 else get_translation(language, 'masculino')
                    )
                else:
                    raise KeyError('gender / gender_F')
    
                fig = px.histogram(plot_df, x='gender_plot', color='final_result', barmode='group',
                                   title=get_translation(language, 'abandono_genero_title'),
                                   color_discrete_map={0: '#1abc9c', 1: '#006400'})
                fig.update_layout(
                    xaxis_title=get_translation(language, 'genero'),
                    yaxis_title=get_translation(language, 'num_estudiantes'),
                    legend_title=get_translation(language, 'abandono')
                )
                return fig, get_translation(language, 'exp_genero')
            
            elif col_chosen == 'disability':
                # Support datasets with the original column ('disability') or one-hot column ('disability_N').
                if 'disability' in df_data.columns:
                    plot_df = df_data.copy()
                    plot_df['disability_plot'] = plot_df['disability']
                elif 'disability_N' in df_data.columns:
                    plot_df = df_data.copy()
                    # disability_N=1 -> no disability; disability_N=0 -> disability.
                    plot_df['disability_plot'] = plot_df['disability_N'].apply(
                        lambda x: get_translation(language, 'sin_discapacidad') if x == 1 else get_translation(language, 'con_discapacidad')
                    )
                else:
                    raise KeyError('disability / disability_N')
    
                fig = px.histogram(plot_df, x='disability_plot', color='final_result', barmode='group',
                                   title=get_translation(language, 'abandono_discapacidad_title'),
                                   color_discrete_map={0: '#1abc9c', 1: '#006400'})
                fig.update_layout(
                    xaxis_title=get_translation(language, 'discapacidad'),
                    yaxis_title=get_translation(language, 'num_estudiantes'),
                    legend_title=get_translation(language, 'abandono')
                )
                return fig, get_translation(language, 'exp_discapacidad')
            
            elif col_chosen == 'age_band':
                data_age = []
                age_candidates = [
                    ('age_band_0-35', '0-35'),
                    ('age_band_35-55', '35-55'),
                    ('age_band_55<=', '55+'),
                    ('age_band_55+', '55+'),
                ]
                available_age = [(col, label) for col, label in age_candidates if col in df_data.columns]
                if not available_age:
                    raise KeyError('age_band_*')
    
                for col, label in available_age:
                    for result in [0, 1]:
                        count = len(df_data[(df_data[col] == 1) & (df_data['final_result'] == result)])
                        data_age.append({'age_group': label, 'final_result': result, 'count': count})
                df_plot = pd.DataFrame(data_age)
                fig = px.bar(df_plot, x='age_group', y='count', color='final_result', barmode='group',
                            title=get_translation(language, 'abandono_age_band_title'),
                            color_discrete_map={0: '#1abc9c', 1: '#006400'})
                fig.update_layout(
                    xaxis_title=get_translation(language, 'grupo_edad'),
                    yaxis_title=get_translation(language, 'num_estudiantes'),
                    legend=dict(title=get_translation(language, 'abandono'))
                )
                return fig, get_translation(language, 'exp_age_band')
            
            elif col_chosen == 'highest_education':
                education_candidates = [
                    ('highest_education_A Level or Equivalent', 'A Level'),
                    ('highest_education_HE Qualification', 'HE Qualification'),
                    ('highest_education_Lower Than A Level', 'Lower than A Level'),
                    ('highest_education_No Formal quals', 'No Formal'),
                    ('highest_education_Post Graduate Qualification', 'Post Graduate'),
                ]
                available_edu = [(col, label) for col, label in education_candidates if col in df_data.columns]
                if not available_edu:
                    # Fallback: try the original categorical column.
                    edu_col = _pick_existing(df_data, ['highest_education', 'education'])
                    if edu_col is None:
                        raise KeyError('highest_education_* / highest_education')
    
                    plot_df = df_data.copy()
                    fig = px.histogram(plot_df, x=edu_col, color='final_result', barmode='group',
                                       title=get_translation(language, 'abandono_highest_education_title'),
                                       color_discrete_map={0: '#1abc9c', 1: '#006400'})
                    fig.update_layout(
                        xaxis_title=get_translation(language, 'nivel_educativo'),
                        yaxis_title=get_translation(language, 'num_estudiantes'),
                        legend=dict(title=get_translation(language, 'abandono')),
                        xaxis_tickangle=-45
                    )
                    return fig, get_translation(language, 'exp_highest_education')
    
                data_edu = []
                for col, label in available_edu:
                    for result in [0, 1]:
                        count = len(df_data[(df_data[col] == 1) & (df_data['final_result'] == result)])
                        data_edu.append({'education': label, 'final_result': result, 'count': count})
                df_plot = pd.DataFrame(data_edu)
                fig = px.bar(df_plot, x='education', y='count', color='final_result', barmode='group',
                            title=get_translation(language, 'abandono_highest_education_title'),
                            color_discrete_map={0: '#1abc9c', 1: '#006400'})
                fig.update_layout(
                    xaxis_title=get_translation(language, 'nivel_educativo'),
                    yaxis_title=get_translation(language, 'num_estudiantes'),
                    legend=dict(title=get_translation(language, 'abandono')),
                    xaxis_tickangle=-45
                )
                return fig, get_translation(language, 'exp_highest_education')
            
            elif col_chosen == 'studied_credits':
                if 'studied_credits' not in df_data.columns:
                    raise KeyError('studied_credits')
                df_credits = df_data[['studied_credits', 'final_result']].copy()
                
                # ERROR #6: Validate quintiles by checking that there is enough data.
                unique_values = df_credits['studied_credits'].nunique()
                if unique_values < 5:
                    # Use fewer groups if there are fewer than 5 unique values.
                    num_groups = max(2, unique_values)
                    print(f"""⚠️  [STUDIED_CREDITS] Only {unique_values} unique values; using {num_groups} groups""")
                    df_credits['credits_quintile'] = pd.qcut(df_credits['studied_credits'], q=num_groups, duplicates='drop', labels=False)
                else:
                    df_credits['credits_quintile'] = pd.qcut(df_credits['studied_credits'], q=5, duplicates='drop', labels=False)
                
                data_credits = []
                for quin in sorted(df_credits['credits_quintile'].unique()):
                    for result in [0, 1]:
                        count = len(df_credits[(df_credits['credits_quintile'] == quin) & (df_credits['final_result'] == result)])
                        min_val = df_credits[df_credits['credits_quintile'] == quin]['studied_credits'].min()
                        max_val = df_credits[df_credits['credits_quintile'] == quin]['studied_credits'].max()
                        label = f'Q{int(quin)+1} ({int(min_val)}-{int(max_val)})'
                        data_credits.append({'quintile': label, 'final_result': result, 'count': count})
                df_plot = pd.DataFrame(data_credits)
                fig = px.bar(df_plot, x='quintile', y='count', color='final_result', barmode='group',
                            title=get_translation(language, 'abandono_studied_credits_title'),
                            color_discrete_map={0: '#1abc9c', 1: '#006400'})
                fig.update_layout(
                    xaxis_title=get_translation(language, 'creditos_estudiados'),
                    yaxis_title=get_translation(language, 'num_estudiantes'),
                    legend=dict(title=get_translation(language, 'abandono')),
                    height=500
                )
                return fig, get_translation(language, 'exp_studied_credits')
            
            else:
                return go.Figure(), ""
        
        except KeyError as ke:
            # ERROR 2: An expected column is missing.
            error_fig = go.Figure()
            error_fig.add_annotation(
                text=f"{get_translation(language, 'covariate_error_missing_prefix')} '{str(ke)}'",
                showarrow=False,
                font=dict(size=12, color='red')
            )
            error_fig.update_layout(title=get_translation(language, 'covariate_error_missing_column_title'), template="plotly_white")
            return error_fig, html.Div([
                html.H4(get_translation(language, 'covariate_error_structure_header'), style={'color': 'red'}),
                html.P(f"{get_translation(language, 'covariate_error_missing_prefix')} {str(ke)}")
            ])
        
        except Exception as e:
            # ERROR 2: Any other error.
            error_fig = go.Figure()
            error_fig.add_annotation(text=f"❌ Error: {str(e)[:40]}...", showarrow=False, font=dict(size=12, color='red'))
            error_fig.update_layout(title=get_translation(language, 'covariate_error_generic_title'), template="plotly_white")
            return error_fig, html.Div([
                html.H4(get_translation(language, 'covariate_error_unexpected_header'), style={'color': 'red'}),
                html.P(str(e))
            ])
    
    @app.callback(
        Output('cox-selected-covariables', 'data'),
        Input('covariables-dropdown-cox', 'value')
    )
    def update_cox_store(covariables):
        """Save the current covariate selection for Cox regression."""
        return covariables if covariables else []
    
    @app.callback(
        [Output('cox-regression-output-store', 'data'),
         Output('cox-current-variables', 'data')],
        [Input('cox-selected-covariables', 'data'),
         Input('language-store', 'data'),
         Input('df-store', 'data'),
         Input('dataset-signature-store', 'data')]
    )
    def update_cox_model(covariables, language, df_json, dataset_signature):
        """Run Cox regression with the selected covariates and prepare the results."""
        if covariables is None or len(covariables) == 0:
            return None, ''
        
        if df_json is None:
            return None, ''
    
        # Rebuild the dataframe from JSON.
        df_data = _read_split_json(df_json)
        
        print(f"""\n[UPDATE COX] Available columns in df_data: {list(df_data.columns)}""")
        print(f"""[UPDATE COX] Dataframe shape: {df_data.shape}""")
    
        # Ensure covariables is a list.
        if isinstance(covariables, str):  
            covariables = [covariables]
            
        # Call Cox regression with the selected covariates.
        summary, cox_table_html = run_cox_regression(df_data, covariables)
        
        print(f"""[UPDATE COX] Summary empty: {summary.empty}""")
        print(f"[UPDATE COX] cox_table_html: {type(cox_table_html)}")
        
        # Save data in a JSON-serializable format for the Store.
        store_data = {
            'summary_json': summary.to_json(orient='split') if not summary.empty else None,
            'cox_table_html': cox_table_html,
            'covariables': covariables,
            'dataset_signature': dataset_signature or ''
        }
        
        # Save the selected variables as a string for the explanation.
        variables_str = ', '.join(covariables) if isinstance(covariables, list) else covariables
        
        print(f"""[UPDATE COX] Saving to Store: {store_data.keys()}""")
        return store_data, variables_str
    
    
    @app.callback(
        Output('cox-regression-output', 'children'),
        [Input('cox-regression-output-store', 'data'),
         Input('language-store', 'data')]
    )
    def render_cox_output(store_data, language):
        """Render the Cox table and chart from the Store."""
        if store_data is None:
            return html.Div([
                html.Br(),
                html.P(get_translation(language, 'respuesta'), style={'textAlign': 'center', 'color': '#999', 'fontSize': '18px'})
            ])
        
        try:
            output_elements = []
            
            # Rebuild the table from the summary saved in the Store.
            if store_data.get('summary_json'):
                try:
                    summary = _read_split_json(store_data['summary_json'])
                    if not summary.empty:
                        display_summary = summary.copy()
                        if 'Covariable' in display_summary.columns:
                            display_summary['Covariable'] = display_summary['Covariable'].map(lambda value: get_feature_label(language, str(value)))
                        cox_table = dash_table.DataTable(
                            id='cox-summary-table',
                            columns=[{"name": get_result_column_label(language, col), "id": col} for col in display_summary.columns],
                            data=get_display_records(display_summary, language),
                            style_table={
                                'height': '300px',
                                'overflowY': 'auto',  
                                'overflowX': 'auto',  
                                'width': '100%'
                            },
                            style_cell={
                                'textAlign': 'center',
                                'whiteSpace': 'normal',
                                'height': 'auto',
                                'lineHeight': '15px',
                                'backgroundColor': 'transparent'
                            },
                            style_header={'fontWeight': 'bold', 'backgroundColor': '#f4f7f6'},
                        )
                        output_elements.append(cox_table)
                        print(f"""[RENDER COX] Table created successfully: {summary.shape}""")
                except Exception as e:
                    print(f"""[RENDER COX] Error rebuilding table: {e}""")
            
            # Rebuild and display the chart if a summary is available.
            if store_data.get('summary_json'):
                try:
                    summary = _read_split_json(store_data['summary_json'])
                    if not summary.empty:
                        forest_plot = plot_cox_hazard_ratios(
                            summary,
                            'Selected covariates' if language == 'en' else 'Covariables seleccionadas',
                            language=language
                        )
                        output_elements.append(forest_plot)
                        print(f"""[RENDER COX] Forest plot created successfully""")
                except Exception as e:
                    print(f"""[RENDER COX] Error rebuilding forest plot: {e}""")
                    import traceback
                    traceback.print_exc()
            
            if not output_elements:
                return html.Div("No results to display" if language == 'en' else "No hay resultados para mostrar")
            
            print(f"""[RENDER COX] Rendering {len(output_elements)} elements""")
            return html.Div(output_elements, style={'textAlign': 'center', 'marginTop': '20px'})
        
        except Exception as e:
            print(f"[RENDER COX] Error: {e}")
            import traceback
            traceback.print_exc()
            return html.Div(
                f"Error rendering results: {str(e)}" if language == 'en' else f"Error renderizando resultados: {str(e)}"
            )
    
    
    @app.callback(
        Output('weibull-analysis-output', 'children'),
        [Input('language-store', 'data'), Input('df-store', 'data'), Input('url', 'pathname')]
    )
    def render_weibull_output(language, df_json, pathname):
        """Compute and render Weibull analysis when data is loaded."""
        if pathname != '/survival-analysis/weibull':
            raise PreventUpdate
    
        def _no_data_message():
            """Build the message displayed when Weibull has insufficient data."""
            return html.Div(
                get_translation(language, 'weibull_no_data'),
                style={'textAlign': 'center', 'color': '#b03a2e', 'fontWeight': 'bold', 'padding': '20px'}
            )
    
        if df_json is None:
            return _no_data_message()
    
        try:
            df_data = _read_split_json(df_json)
        except Exception:
            return _no_data_message()
    
        analysis = build_weibull_analysis(df_data, language=language)
        if not analysis:
            return _no_data_message()
    
        summary_df = analysis['summary_df']
        table = dash_table.DataTable(
            id='weibull-summary-table',
            columns=[
                {"name": get_translation(language, 'weibull_metric'), "id": "Metrica"},
                {"name": get_translation(language, 'weibull_value'), "id": "Valor"},
            ],
            data=summary_df.to_dict('records'),
            style_table={'overflowX': 'auto', 'maxHeight': '360px', 'overflowY': 'auto', 'marginTop': '10px'},
            style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto', 'lineHeight': '16px', 'padding': '10px'},
            style_header={'fontWeight': 'bold', 'backgroundColor': '#f4f7f6'},
            style_data_conditional=[{'if': {'column_id': 'Metrica'}, 'fontWeight': 'bold'}]
        )
    
        comparison_df = analysis.get('comparison_df')
        comparison_table = dash_table.DataTable(
            id='weibull-comparison-table',
            columns=[
                {"name": "Modelo" if language == 'es' else "Model", "id": "Modelo"},
                {"name": "AIC", "id": "AIC"},
                {"name": "Log-likelihood", "id": "LogLikelihood"},
                {"name": "ΔAIC", "id": "DeltaAIC"},
            ],
            data=comparison_df.to_dict('records') if comparison_df is not None else [],
            style_table={'overflowX': 'auto', 'marginTop': '10px'},
            style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto', 'lineHeight': '16px', 'padding': '10px'},
            style_header={'fontWeight': 'bold', 'backgroundColor': '#eef4ff'},
            style_data_conditional=[
                {'if': {'column_id': 'Modelo'}, 'fontWeight': 'bold'},
                {'if': {'filter_query': '{DeltaAIC} = 0'}, 'backgroundColor': '#eafaf1', 'fontWeight': 'bold'}
            ]
        )
    
        comparison_interpretation = analysis.get('model_comparison_interpretation', '')
    
        return html.Div([
            html.Div([
                html.H3(get_translation(language, 'weibull_summary_title'), style={'textAlign': 'center', 'color': '#0d0d0d', 'fontWeight': 'bold', 'marginBottom': '10px'}),
                table,
            ], style={
                'backgroundColor': 'white',
                'padding': '20px',
                'borderRadius': '10px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                'marginBottom': '30px'
            }),
            html.Div([
                html.H3(get_translation(language, 'weibull_graph_title'), style={'textAlign': 'center', 'color': '#0d0d0d', 'fontWeight': 'bold', 'marginBottom': '20px'}),
                dcc.Graph(figure=analysis['figure'], config=get_plotly_config(language, responsive=True, displayModeBar=True))
            ], style={
                'backgroundColor': '#f8f9fa',
                'padding': '20px',
                'borderRadius': '10px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                'marginBottom': '30px'
            }),
            html.Div([
                html.H3(
                    "Comparación de ajuste entre modelos" if language == 'es' else "Model fit comparison",
                    style={'textAlign': 'center', 'color': '#0d0d0d', 'fontWeight': 'bold', 'marginBottom': '10px'}
                ),
                comparison_table,
                html.P(
                    comparison_interpretation,
                    style={'marginTop': '12px', 'marginBottom': 0, 'fontSize': '0.98em', 'lineHeight': '1.6', 'color': '#2c3e50', 'fontWeight': '600'}
                )
            ], style={
                'backgroundColor': '#ffffff',
                'padding': '20px',
                'borderRadius': '10px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                'marginBottom': '30px'
            }),
        ])
    
    
    @app.callback(
        Output('exponential-analysis-output', 'children'),
        [Input('language-store', 'data'), Input('df-store', 'data'), Input('url', 'pathname')]
    )
    def render_exponential_output(language, df_json, pathname):
        """Compute and render Exponential analysis when data is loaded."""
        if pathname != '/survival-analysis/exponential':
            raise PreventUpdate
    
        def _no_data_message():
            """Build the message displayed when Exponential has insufficient data."""
            return html.Div(
                get_translation(language, 'exponential_no_data'),
                style={'textAlign': 'center', 'color': '#b03a2e', 'fontWeight': 'bold', 'padding': '20px'}
            )
    
        if df_json is None:
            return _no_data_message()
    
        try:
            df_data = _read_split_json(df_json)
        except Exception:
            return _no_data_message()
    
        analysis = build_exponential_analysis(df_data, language=language)
        if not analysis:
            return _no_data_message()
    
        summary_df = analysis['summary_df']
        table = dash_table.DataTable(
            id='exponential-summary-table',
            columns=[
                {"name": get_translation(language, 'exponential_metric'), "id": "Metrica"},
                {"name": get_translation(language, 'exponential_value'), "id": "Valor"},
                {"name": get_translation(language, 'exponential_interpretation'), "id": "Interpretacion"},
            ],
            data=summary_df.to_dict('records'),
            style_table={'overflowX': 'auto', 'maxHeight': '360px', 'overflowY': 'auto', 'marginTop': '10px'},
            style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto', 'lineHeight': '16px', 'padding': '10px'},
            style_header={'fontWeight': 'bold', 'backgroundColor': '#f4f7f6'},
            style_data_conditional=[{'if': {'column_id': 'Metrica'}, 'fontWeight': 'bold'}]
        )
    
        comparison_df = analysis.get('comparison_df')
        comparison_table = dash_table.DataTable(
            id='exponential-comparison-table',
            columns=[
                {"name": "Modelo" if language == 'es' else "Model", "id": "Modelo"},
                {"name": "AIC", "id": "AIC"},
                {"name": "Log-likelihood", "id": "LogLikelihood"},
                {"name": "ΔAIC", "id": "DeltaAIC"},
            ],
            data=comparison_df.to_dict('records') if comparison_df is not None else [],
            style_table={'overflowX': 'auto', 'marginTop': '10px'},
            style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto', 'lineHeight': '16px', 'padding': '10px'},
            style_header={'fontWeight': 'bold', 'backgroundColor': '#eef4ff'},
            style_data_conditional=[
                {'if': {'column_id': 'Modelo'}, 'fontWeight': 'bold'},
                {'if': {'filter_query': '{DeltaAIC} = 0'}, 'backgroundColor': '#eafaf1', 'fontWeight': 'bold'}
            ]
        )
    
        comparison_interpretation = analysis.get('model_comparison_interpretation', '')
    
        return html.Div([
            html.Div([
                html.H3(get_translation(language, 'exponential_summary_title'), style={'textAlign': 'center', 'color': '#0d0d0d', 'fontWeight': 'bold', 'marginBottom': '10px'}),
                table,
            ], style={
                'backgroundColor': 'white',
                'padding': '20px',
                'borderRadius': '10px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                'marginBottom': '30px'
            }),
            html.Div([
                html.H3(get_translation(language, 'exponential_graph_title'), style={'textAlign': 'center', 'color': '#0d0d0d', 'fontWeight': 'bold', 'marginBottom': '20px'}),
                dcc.Graph(figure=analysis['figure'], config=get_plotly_config(language, responsive=True, displayModeBar=True))
            ], style={
                'backgroundColor': '#f8f9fa',
                'padding': '20px',
                'borderRadius': '10px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                'marginBottom': '30px'
            }),
            html.Div([
                html.H3(
                    "Comparación de ajuste entre modelos" if language == 'es' else "Model fit comparison",
                    style={'textAlign': 'center', 'color': '#0d0d0d', 'fontWeight': 'bold', 'marginBottom': '10px'}
                ),
                comparison_table,
                html.P(
                    comparison_interpretation,
                    style={'marginTop': '12px', 'marginBottom': 0, 'fontSize': '0.98em', 'lineHeight': '1.6', 'color': '#2c3e50', 'fontWeight': '600'}
                )
            ], style={
                'backgroundColor': '#ffffff',
                'padding': '20px',
                'borderRadius': '10px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                'marginBottom': '30px'
            }),
        ])
    
    
    @app.callback(
        [Output('rsf-analysis-output', 'children'),
         Output('rsf-analysis-data', 'data')],
        [Input('language-store', 'data'), Input('df-store', 'data'), Input('url', 'pathname')]
    )
    def render_rsf_output(language, df_json, pathname):
        """Compute and render the Random Survival Forest model in its panel."""
        if pathname != '/survival-analysis/rsf':
            raise PreventUpdate
    
        def _no_data_message():
            """Build the message displayed when RSF has insufficient data."""
            return html.Div(
                get_translation(language, 'rsf_no_data'),
                style={'textAlign': 'center', 'color': '#b03a2e', 'fontWeight': 'bold', 'padding': '20px'}
            ), None
    
        if df_json is None:
            return _no_data_message()
    
        try:
            df_data = _read_split_json(df_json)
        except Exception:
            return _no_data_message()
    
        analysis = build_rsf_analysis(df_data, language=language)
        if not analysis:
            return _no_data_message()
    
        summary_df = analysis['summary_df']
        table = dash_table.DataTable(
            id='rsf-summary-table',
            columns=[
                {"name": get_translation(language, 'rsf_metric'), "id": "Metrica"},
                {"name": get_translation(language, 'rsf_value'), "id": "Valor"},
                {"name": get_translation(language, 'rsf_interpretation'), "id": "Interpretacion"},
            ],
            data=summary_df.to_dict('records'),
            style_table={'overflowX': 'auto', 'maxHeight': '360px', 'overflowY': 'auto', 'marginTop': '10px'},
            style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto', 'lineHeight': '16px', 'padding': '10px'},
            style_header={'fontWeight': 'bold', 'backgroundColor': '#f4f7f6'},
            style_data_conditional=[{'if': {'column_id': 'Metrica'}, 'fontWeight': 'bold'}]
        )
    
        output_children = html.Div([
            html.Div([
                html.H3(get_translation(language, 'rsf_summary_title'), style={'textAlign': 'center', 'color': '#0d0d0d', 'fontWeight': 'bold', 'marginBottom': '10px'}),
                table,
            ], style={
                'backgroundColor': 'white',
                'padding': '20px',
                'borderRadius': '10px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                'marginBottom': '30px'
            }),
            html.Div([
                html.H3(get_translation(language, 'rsf_graph_title'), style={'textAlign': 'center', 'color': '#0d0d0d', 'fontWeight': 'bold', 'marginBottom': '20px'}),
                dcc.Graph(figure=analysis['figure'], config=get_plotly_config(language, responsive=True, displayModeBar=True, scrollZoom=False, doubleClick='reset'))
            ], style={
                'backgroundColor': '#f8f9fa',
                'padding': '20px',
                'borderRadius': '10px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                'marginBottom': '30px'
            }),
            html.Div([
                html.H3(get_translation(language, 'rsf_importance_title'), style={'textAlign': 'center', 'color': '#0d0d0d', 'fontWeight': 'bold', 'marginBottom': '20px'}),
                dcc.Graph(figure=analysis['importance_figure'], config=get_plotly_config(language, responsive=True, displayModeBar=True))
            ], style={
                'backgroundColor': '#f8f9fa',
                'padding': '20px',
                'borderRadius': '10px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                'marginBottom': '30px'
            }),
        ])
    
        store_data = {
            'summary_json': summary_df.to_json(orient='split'),
            'top_feature': analysis['top_feature'],
            'top_feature_importance': analysis['top_feature_importance'],
            'train_c_index': analysis['train_c_index'],
            'oob_score': analysis['oob_score'],
            'n_observations': analysis['n_observations'],
            'n_events': analysis['n_events'],
            'n_features': analysis['n_features'],
            'interpretation': analysis['interpretation'],
            'top_features': analysis.get('top_features', []),
            'survival_curve_summaries': analysis.get('survival_curve_summaries', []),
            'language': language,
        }
    
        return output_children, store_data
    
    
    @app.callback(
        [Output('rsf-profile-output', 'children'),
         Output('rsf-profile-data', 'data')],
        [Input('rsf-profile-simulate-btn', 'n_clicks'),
         Input('df-store', 'data'),
         Input('language-store', 'data')],
        [State('rsf-profile-gender', 'value'),
         State('rsf-profile-disability', 'value'),
         State('rsf-profile-age-band', 'value'),
         State('rsf-profile-education', 'value'),
         State('rsf-profile-credits', 'value')],
        prevent_initial_call=False
    )
    def simulate_rsf_profile(n_clicks, df_json, language, gender, disability, age_band, education, credits_level):
        """Simulate survival for a specific profile using the RSF model."""
        if not df_json:
            return html.Div(
                get_translation(language, 'rsf_no_data'),
                style={'textAlign': 'center', 'color': '#b03a2e', 'fontWeight': 'bold', 'padding': '20px'}
            ), None
    
        try:
            df_data = _read_split_json(df_json)
        except Exception:
            return html.Div(
                get_translation(language, 'rsf_no_data'),
                style={'textAlign': 'center', 'color': '#b03a2e', 'fontWeight': 'bold', 'padding': '20px'}
            ), None
    
        credits_map = {'few': 30, 'medium': 60, 'many': 120}
        profile = {
            'gender_F': gender,
            'disability_N': disability,
            'age_band': age_band,
            'highest_education': education,
            'studied_credits': credits_map.get(credits_level, 30),
        }
    
        analysis = build_rsf_profile_analysis(df_data, profile, language=language)
        if not analysis:
            return html.Div(
                get_translation(language, 'rsf_no_data'),
                style={'textAlign': 'center', 'color': '#b03a2e', 'fontWeight': 'bold', 'padding': '20px'}
            ), None
    
        show_table = bool(n_clicks and n_clicks > 0)
    
        profile_table = dash_table.DataTable(
            columns=[{"name": "Métrica" if language == 'es' else "Metric", "id": "Metrica"}, {"name": "Valor" if language == 'es' else "Value", "id": "Valor"}],
            data=analysis['summary_df'].to_dict('records'),
            style_table={'overflowX': 'auto', 'maxHeight': '320px', 'overflowY': 'auto', 'marginTop': '10px'},
            style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto', 'lineHeight': '16px', 'padding': '10px'},
            style_header={'fontWeight': 'bold', 'backgroundColor': '#f4f7f6'},
            style_data_conditional=[{'if': {'column_id': 'Metrica'}, 'fontWeight': 'bold'}]
        )
    
        profile_title = 'Curva del perfil simulado' if language == 'es' else 'Simulated profile curve'
        profile_summary_title = 'Perfil simulado' if language == 'es' else 'Simulated profile'
    
        graph_block = html.Div([
            html.H4(profile_title, style={'textAlign': 'center', 'color': '#0d0d0d', 'fontWeight': 'bold', 'marginBottom': '10px'}),
            dcc.Graph(figure=analysis['figure'], config=get_plotly_config(language, responsive=True, displayModeBar=True, scrollZoom=False, doubleClick='reset'))
        ], style={
            'backgroundColor': '#f8f9fa',
            'padding': '18px',
            'borderRadius': '10px',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.06)',
            'marginBottom': '22px'
        })

        profile_store_data = {
            'summary_json': analysis['summary_df'].to_json(orient='split'),
            'interpretation': analysis['interpretation'],
            'risk_score': analysis.get('risk_score'),
            'risk_percentile': analysis.get('risk_percentile'),
            'final_survival': analysis.get('final_survival'),
            'min_survival': analysis.get('min_survival'),
            'max_survival': analysis.get('max_survival'),
            'end_time': analysis.get('end_time'),
            'profile': profile,
            'language': language,
        }
    
        if not show_table:
            return html.Div([
                graph_block,
                html.Div(analysis['interpretation'], style={
                    'backgroundColor': '#eef7ff',
                    'padding': '14px 16px',
                    'borderRadius': '8px',
                    'border': '1px solid #d9e8ff',
                    'color': '#2c3e50',
                    'lineHeight': '1.6'
                })
            ]), profile_store_data
    
        return html.Div([
            html.Div([
                html.H4(profile_summary_title, style={'textAlign': 'center', 'color': '#0d0d0d', 'fontWeight': 'bold', 'marginBottom': '10px'}),
                profile_table,
            ], style={
                'backgroundColor': '#ffffff',
                'padding': '18px',
                'borderRadius': '10px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.06)',
                'marginBottom': '22px'
            }),
            graph_block,
            html.Div(analysis['interpretation'], style={
                'backgroundColor': '#eef7ff',
                'padding': '14px 16px',
                'borderRadius': '8px',
                'border': '1px solid #d9e8ff',
                'color': '#2c3e50',
                'lineHeight': '1.6'
            })
        ]), profile_store_data
    
    
    @app.callback(
        Output('openai-answer-rsf', 'value'),
        [Input('btn-rsf', 'n_clicks')],
        [State('rsf-analysis-data', 'data'),
         State('rsf-profile-data', 'data'),
         State('language-store', 'data')],
        prevent_initial_call=True
    )
    def explicar_rsf(n_clicks, rsf_store_data, rsf_profile_data, language):
        """Generate a textual interpretation of the Random Survival Forest results."""
        if n_clicks is None or n_clicks <= 0:
            return get_translation(language, 'respuesta')
    
        if not rsf_store_data:
            return get_translation(language, 'rsf_no_data')
    
        rsf_context = _build_rsf_interpretation_context(rsf_store_data, rsf_profile_data, language=language)
        if not rsf_context.strip():
            return get_translation(language, 'rsf_no_data')
    
        if language == 'en':
            prompt = (
                "Write an academic interpretation of the active Random Survival Forest dashboard. "
                "Use only the active model table, risk curves, variable-importance chart, and simulated profile below:\n"
                f"{rsf_context}\n"
                f"{_interpretation_style_guidance(language)} "
                "Return exactly 2 short paragraphs, maximum 160 words total. "
                "First paragraph: global model quality, low/medium/high-risk survival curves, and most important variables. "
                "Second paragraph: interpret the currently simulated profile, including its risk score/percentile and survival curve. "
                "Do not make causal claims. Finish with a complete sentence."
            )
        else:
            prompt = (
                "Redacta una interpretación académica del dashboard Random Survival Forest activo. "
                "Usa solo la tabla del modelo, las curvas de riesgo, la importancia de variables y el perfil simulado siguientes:\n"
                f"{rsf_context}\n"
                f"{_interpretation_style_guidance(language)} "
                "Devuelve exactamente 2 párrafos breves, máximo 160 palabras en total. "
                "Primer párrafo: calidad global del modelo, curvas de bajo/medio/alto riesgo e importancia de variables. "
                "Segundo párrafo: interpreta el perfil simulado actualmente, incluyendo su score/percentil de riesgo y su curva de supervivencia. "
                "No hagas afirmaciones causales. Termina con una frase completa."
            )
    
        respuesta = responder_pregunta_con_llama3(prompt, language)
        if respuesta:
            return respuesta
    
        if language == 'en':
            top_feature = rsf_store_data.get('top_feature', 'N/A')
            oob_score = rsf_store_data.get('oob_score', None)
            train_c_index = rsf_store_data.get('train_c_index', None)
            n_features = rsf_store_data.get('n_features', 0)
            return (
                f"RSF uses {n_features} predictors. Train c-index={train_c_index}, OOB c-index={oob_score}, and top feature={top_feature}. "
                "The dashboard curves and simulated profile should be interpreted as predictive estimates; confirm with external validation."
            )
        top_feature = rsf_store_data.get('top_feature', 'N/A')
        oob_score = rsf_store_data.get('oob_score', None)
        train_c_index = rsf_store_data.get('train_c_index', None)
        n_features = rsf_store_data.get('n_features', 0)
        return (
            f"RSF usa {n_features} predictores. c-index entrenamiento={train_c_index}, c-index OOB={oob_score}, y variable principal={top_feature}. "
            "Las curvas del dashboard y el perfil simulado deben interpretarse como estimaciones predictivas; conviene validar externamente."
        )
    
    
    @app.callback(
        [Output('openai-answer-weibull', 'value'),
         Output('weibull-ai-text-store', 'data'),
         Output('weibull-ai-language-store', 'data')],
        [Input('btn-weibull', 'n_clicks')],
        [State('df-store', 'data'), State('language-store', 'data')],
        prevent_initial_call=True
    )
    def explicar_weibull(n_clicks, df_json, language):
        """Generate a textual interpretation of the current Weibull fit."""
        if n_clicks is None or n_clicks <= 0:
            return "", "", ""
    
        if not df_json:
            no_data_msg = get_translation(language, 'weibull_no_data')
            return no_data_msg, "", ""
    
        try:
            df_data = _read_split_json(df_json)
            analysis = build_weibull_analysis(df_data, language=language)
            if not analysis:
                no_data_msg = get_translation(language, 'weibull_no_data')
                return no_data_msg, "", ""
    
            def _fallback_weibull_explanation():
                """Create a local Weibull explanation if AI returns no response."""
                shape = analysis['shape']
                scale = analysis['scale']
                median = analysis['median_survival']
                event_rate = analysis['event_rate']
                risk_text = (
                    'aumenta con el tiempo' if shape > 1.05 else
                    'disminuye con el tiempo' if shape < 0.95 else
                    'se mantiene aproximadamente constante'
                )
    
                if language == 'en':
                    return (
                        f"The Weibull model fitted to the cleaned dataset gives a shape parameter of {shape:.3f}, which indicates that the risk {('increases' if shape > 1.05 else 'decreases' if shape < 0.95 else 'remains approximately constant')} over time. "
                        f"The scale parameter is {scale:.3f} and the median survival time is {median:.3f}. The event rate is {event_rate:.1f}%, so the fitted curve summarizes the overall survival pattern of the cleaned data."
                    )
    
                return (
                    f"El modelo Weibull ajustado al dataset limpio da un parámetro de forma de {shape:.3f}, lo que significa que el riesgo {risk_text}. "
                    f"El parámetro de escala es {scale:.3f} y la mediana de supervivencia es {median:.3f}. La tasa de eventos es {event_rate:.1f}%, por lo que la curva ajustada resume el patrón global de supervivencia de los datos limpios."
                )
    
            weibull_context = _build_weibull_interpretation_context(analysis, df=df_data, language=language)
            if not weibull_context.strip():
                final_text = _fallback_weibull_explanation()
                return final_text, final_text, language

            if language == 'en':
                prompt = (
                    "Write an academic interpretation of the active Weibull analysis shown in the dashboard. "
                    "Use only the table, model-comparison, and curve information below:\n"
                    f"{weibull_context}\n"
                    f"{_interpretation_style_guidance(language)} "
                    "Return exactly 2 short paragraphs, maximum 130 words total. "
                    "First paragraph: interpret rho/shape, scale, event rate, and median estimated by Weibull. "
                    "Second paragraph: compare the fitted Weibull curve with empirical Kaplan-Meier and Exponential using AIC and visual fit. "
                    "If the median exceeds the observed time range, call it a parametric extrapolation. "
                    "Do not mention covariates or causal effects. Finish with a complete sentence."
                )
            else:
                prompt = (
                    "Redacta una interpretación académica del análisis Weibull activo mostrado en el dashboard. "
                    "Usa solo la tabla, la comparación de modelos y la información de curvas siguientes:\n"
                    f"{weibull_context}\n"
                    f"{_interpretation_style_guidance(language)} "
                    "Devuelve exactamente 2 párrafos breves, máximo 130 palabras en total. "
                    "Primer párrafo: interpreta rho/shape, escala, tasa de eventos y mediana estimada por Weibull. "
                    "Segundo párrafo: compara la curva Weibull ajustada con Kaplan-Meier empírico y Exponencial usando AIC y ajuste visual. "
                    "Si la mediana supera el rango observado, descríbela como extrapolación paramétrica. "
                    "No menciones covariables ni efectos causales. Termina con una frase completa."
                )

            respuesta = responder_pregunta_con_llama3(prompt, language)
            final_text = respuesta if respuesta else _fallback_weibull_explanation()
            return final_text, final_text, language
        except Exception as e:
            print(f"""❌ Error in explicar_weibull: {str(e)}""")
            error_msg = get_translation(language, 'weibull_no_data')
            return error_msg, "", ""
    
    
    @app.callback(
        [Output('openai-answer-exponential', 'value'),
         Output('exponential-ai-text-store', 'data'),
         Output('exponential-ai-language-store', 'data')],
        [Input('btn-exponential', 'n_clicks')],
        [State('df-store', 'data'), State('language-store', 'data')],
        prevent_initial_call=True
    )
    def explicar_exponential(n_clicks, df_json, language):
        """Generate a textual interpretation of the current Exponential model."""
        if n_clicks is None or n_clicks <= 0:
            return "", "", ""
    
        if not df_json:
            no_data_msg = get_translation(language, 'exponential_no_data')
            return no_data_msg, "", ""
    
        try:
            df_data = _read_split_json(df_json)
            analysis = build_exponential_analysis(df_data, language=language)
            if not analysis:
                no_data_msg = get_translation(language, 'exponential_no_data')
                return no_data_msg, "", ""
    
            exponential_context = _build_exponential_interpretation_context(analysis, df=df_data, language=language)
            if not exponential_context.strip():
                final_text = analysis['interpretation']
                return final_text, final_text, language

            if language == 'en':
                prompt = (
                    "Write an academic interpretation of the active Exponential survival analysis shown in the dashboard. "
                    "Use only the table, model-comparison, and curve information below:\n"
                    f"{exponential_context}\n"
                    f"{_interpretation_style_guidance(language)} "
                    "Return exactly 2 short paragraphs, maximum 130 words total. "
                    "First paragraph: interpret lambda, event rate, and the constant-hazard assumption. "
                    "Second paragraph: compare the Exponential curve with empirical Kaplan-Meier and Weibull using AIC and visual fit. "
                    "If Weibull has lower AIC, say the Exponential assumption may be too simple. "
                    "Do not mention covariates or causal effects. Finish with a complete sentence."
                )
            else:
                prompt = (
                    "Redacta una interpretación académica del análisis Exponencial activo mostrado en el dashboard. "
                    "Usa solo la tabla, la comparación de modelos y la información de curvas siguientes:\n"
                    f"{exponential_context}\n"
                    f"{_interpretation_style_guidance(language)} "
                    "Devuelve exactamente 2 párrafos breves, máximo 130 palabras en total. "
                    "Primer párrafo: interpreta lambda, tasa de eventos y el supuesto de riesgo constante. "
                    "Segundo párrafo: compara la curva Exponencial con Kaplan-Meier empírico y Weibull usando AIC y ajuste visual. "
                    "Si Weibull tiene menor AIC, indica que el supuesto Exponencial puede ser demasiado simple. "
                    "No menciones covariables ni efectos causales. Termina con una frase completa."
                )

            respuesta = responder_pregunta_con_llama3(prompt, language)
            final_text = respuesta if respuesta else analysis['interpretation']
            return final_text, final_text, language
        except Exception as e:
            print(f"""❌ Error in explicar_exponential: {str(e)}""")
            error_msg = get_translation(language, 'exponential_no_data')
            return error_msg, "", ""
    
    @app.callback(
        Output('openai-answer-cox', 'value'),
        [Input('btn-cox', 'n_clicks')],
        [State('cox-regression-output-store', 'data'),
         State('cox-current-variables', 'data'),
         State('df-store', 'data'),
         State('language-store', 'data'),
         State('dataset-signature-store', 'data')],
        prevent_initial_call=True
    )
    def explicar_cox(n_clicks, cox_store_data, variables_seleccionadas, df_json, language, dataset_signature):
        """Generate a Cox regression explanation using actual data."""
        try:
            if n_clicks is None or n_clicks <= 0:
                return ""
            
            if not variables_seleccionadas:
                return f"⚠️  {get_translation(language, 'error_select_covariate')}"
    
            if not cox_store_data or not isinstance(cox_store_data, dict) or not cox_store_data.get('summary_json'):
                return (
                    "⚠️  Primero ejecuta la regresión de Cox seleccionando covariables y esperando a que aparezca la tabla."
                    if language == 'es' else
                    "⚠️  Run Cox regression first: select covariates and wait until the summary table appears."
                )
    
            if cox_store_data.get('dataset_signature', '') != (dataset_signature or ''):
                return (
                    "⚠️  El dataset ha cambiado. Recalcula la regresión de Cox antes de generar la explicación."
                    if language == 'es' else
                    "⚠️  The dataset changed. Recalculate Cox regression before generating the explanation."
                )
            
            if language not in ['es', 'en']:
                language = 'es'
            
            df_data = _read_split_json(df_json) if df_json else None
            summary_df = _read_split_json(cox_store_data['summary_json'])
            table_summary = _build_cox_interpretation_context(
                summary_df,
                cox_store_data.get('covariables', variables_seleccionadas),
                df=df_data,
                language=language
            )

            if not table_summary.strip():
                return (
                    "⚠️  Could not extract numerical data from the current Cox model."
                    if language == 'en'
                    else "⚠️  No se pudieron extraer datos numéricos del modelo Cox actual."
                )
            
            # Build a prompt using actual data.
            if language == 'en':
                prompt = (f"Write an academic interpretation of the Cox regression model currently displayed for {_humanize_label(variables_seleccionadas)}. "
                         f"Use only this real table summary and do not discuss covariates that are not in the active model:\n{table_summary}\n"
                         f"{_interpretation_style_guidance(language)} "
                         f"Return exactly 2 short paragraphs, maximum 130 words total. "
                         f"First paragraph: HR, 95% CI, p-value, direction, and significance. "
                         f"Second paragraph: practical reading and one methodological limitation. "
                         f"Be cautious when CI includes 1 or p>=0.05, mention reference groups if present, and do not make causal claims. No bullet points. Finish with a complete sentence.")
            else:
                prompt = (f"Redacta una interpretación académica del modelo de Regresión de Cox mostrado actualmente para {_humanize_label(variables_seleccionadas)}. "
                         f"Usa solo este resumen real de tabla y no menciones covariables que no estén en el modelo activo:\n{table_summary}\n"
                         f"{_interpretation_style_guidance(language)} "
                         f"Devuelve exactamente 2 párrafos breves, máximo 130 palabras en total. "
                         f"Primer párrafo: HR, IC95%, p-valor, dirección y significación. "
                         f"Segundo párrafo: lectura práctica y una limitación metodológica. "
                         f"Sé prudente cuando el IC95% incluya 1 o p>=0.05, menciona referencias si existen y no hagas afirmaciones causales. Sin viñetas ni listas. Termina con una frase completa.")
            
            respuesta = responder_pregunta_con_llama3(prompt, language)
            return respuesta if respuesta else ("⚠️  Error: Empty response" if language == 'en' else "⚠️  Error: Respuesta vacía")
            
        except requests.exceptions.Timeout:
            return f"⚠️  {get_translation(language, 'error_timeout')}"
        except Exception as e:
            print(f"""❌ Error in explicar_cox: {str(e)}""")
            return f"❌ Error: {str(e)}"
            return f"❌ Error generating explanation: {str(e)}"
    
    # Callback controlling Log-Rank analysis.
    @app.callback(
        Output('logrank-selected-covariables', 'data'),
        Input('covariables-dropdown-logrank', 'value')
    )
    def update_logrank_store(covariables):
        """Save the selected covariates for the Log-Rank test."""
        return covariables if covariables else []
    
    @app.callback(
        [Output('logrank-test-output-store', 'data'),
         Output('logrank-current-variable', 'data')],
        [Input('logrank-selected-covariables', 'data'),
         Input('language-store', 'data'),
         Input('df-store', 'data'),
         Input('dataset-signature-store', 'data')]
    )
    
    def update_logrank_test(covariables, language, df_json, dataset_signature):
        # Check that at least one covariate has been selected.
        """Run the Log-Rank test for each covariate and save tables and charts."""
        if not covariables:
            return None, ''
        
        if df_json is None:
            return None, ''
    
        # Rebuild the dataframe from JSON.
        df_data = _read_split_json(df_json)
        
        print(f"""\n[UPDATE LOGRANK] Selected covariates: {covariables}""")
        print(f"""[UPDATE LOGRANK] Dataframe shape: {df_data.shape}""")
    
        panels = []
        results_payload = []
    
        # Run the Log-Rank test for each selected covariate.
        for covariable in covariables:
            print(f"""\n[UPDATE LOGRANK] Processing covariate: {covariable}""")
            try:
                res_df = perform_log_rank_test(df_data, covariable)
                print(f"""[UPDATE LOGRANK] Result for {covariable}: {len(res_df)} rows""")
                if res_df.empty:
                    print(f"""[WARNING LOGRANK] Empty res_df for {covariable}""")
                    continue
                
                # Create a panel with the test results table and stratified Kaplan-Meier chart.
                table = display_logrank_summary_table(res_df, language=language)
                graph_component = plot_logrank_curves(df_data, covariable, language=language)
    
                results_payload.append({
                    'covariable': covariable,
                    'results_json': res_df.to_json(orient='split')
                })
                
                panels.append(html.Div([
                    html.H3(f"{'Log-Rank Test results for' if language == 'en' else 'Resultado del Test de Log-Rank para'} {get_covariate_label(language, covariable)}", style={'textAlign': 'center'}),
                    graph_component,
                    table
                ]))
                
    
                    
            except Exception as e:
                print(f"""[ERROR LOGRANK] Error processing {covariable}: {str(e)}""")
                import traceback
                traceback.print_exc()
        
        # Save structured data for the explanation and view rendering.
        variables_str = ', '.join(covariables) if isinstance(covariables, list) else covariables
        store_data = {
            'df_json': df_json,
            'covariables': covariables,
            'results': results_payload,
            'dataset_signature': dataset_signature or ''
        }
    
        return store_data, variables_str
    
    @app.callback(
        Output('openai-answer-logrank', 'value'),
        [Input('explicar-btn-logrank', 'n_clicks')],
        [State('logrank-test-output-store', 'data'),
         State('logrank-current-variable', 'data'),
        State('language-store', 'data'),
        State('dataset-signature-store', 'data')]  
    )
    def explicar_logrank(n_clicks, logrank_content, variables_seleccionadas, language, dataset_signature):
        """Generate a Log-Rank test explanation using actual data."""
        try:
            if n_clicks is None or n_clicks <= 0:
                return ""
            
            if not logrank_content:
                return f"⚠️  {get_translation(language, 'error_select_logrank')}"
    
            if isinstance(logrank_content, dict) and logrank_content.get('dataset_signature', '') != (dataset_signature or ''):
                return (
                    "⚠️  El dataset ha cambiado. Recalcula el Test de Log-Rank antes de generar la explicación."
                    if language == 'es' else
                    "⚠️  The dataset changed. Recalculate the Log-Rank test before generating the explanation."
                )
            
            if language not in ['es', 'en']:
                language = 'es'
            
            # Format store data, limiting its size to keep the prompt concise.
            logrank_data_str = _build_logrank_interpretation_context(logrank_content)

            if not logrank_data_str.strip():
                return (
                    "⚠️  Could not extract numerical data from the current Log-Rank result."
                    if language == 'en'
                    else "⚠️  No se pudieron extraer datos numéricos del resultado Log-Rank actual."
                )
            
            # Build a prompt using actual data.
            if language == 'en':
                prompt = (f"Write an academic interpretation of the Log-Rank result currently displayed for {_humanize_label(variables_seleccionadas)}. "
                         f"Use only the active table and curve summaries below:\n"
                         f"{logrank_data_str}\n"
                         f"{_interpretation_style_guidance(language)} "
                         f"Return exactly 2 short paragraphs, maximum 130 words total. "
                         f"First paragraph: identify significant/non-significant comparisons and describe visible curve separation using group survival. "
                         f"Second paragraph: practical interpretation and one limitation. Do not claim differences when p>=0.05. No bullet points. Finish with a complete sentence.")
            else:
                prompt = (f"Redacta una interpretación académica del resultado Log-Rank mostrado actualmente para {_humanize_label(variables_seleccionadas)}. "
                         f"Usa solo la tabla activa y el resumen de curvas siguientes:\n"
                         f"{logrank_data_str}\n"
                         f"{_interpretation_style_guidance(language)} "
                         f"Devuelve exactamente 2 párrafos breves, máximo 130 palabras en total. "
                         f"Primer párrafo: identifica comparaciones significativas/no significativas y describe la separación visible de curvas usando la supervivencia por grupo. "
                         f"Segundo párrafo: interpretación práctica y una limitación. No afirmes diferencias si p>=0.05. Sin viñetas ni listas numeradas. Termina con una frase completa.")
            
            respuesta = responder_pregunta_con_llama3(prompt, language)
            if len(respuesta) > 3000:
                max_length = 1500
                chunks = [respuesta[i:i + max_length] for i in range(0, len(respuesta), max_length)]
                return '\n\n'.join(chunks)
            return respuesta if respuesta else ("⚠️  Error: Empty response" if language == 'en' else "⚠️  Error: Respuesta vacía")
            
        except requests.exceptions.Timeout:
            return f"⚠️  {get_translation(language, 'error_timeout')}"
        except Exception as e:
            print(f"""❌ Error in explicar_logrank: {str(e)}""")
            return f"❌ Error: {str(e)}"
    
    # Synchronize stores with divs on the pages.
    @app.callback(
        Output('logrank-test-output', 'children'),
        [Input('logrank-test-output-store', 'data'),
         Input('language-store', 'data')]
    )
    def sync_logrank_output(data, language):
        """Rebuild the Log-Rank visual output from the stored data."""
        if not data or not isinstance(data, dict):
            return data
    
        try:
            df_json = data.get('df_json')
            covariables = data.get('covariables') or []
            results = data.get('results') or []
    
            if df_json:
                df_data = _read_split_json(df_json)
            else:
                df_data = None
    
            panels = []
            for result_info in results:
                covariable = result_info.get('covariable')
                results_json = result_info.get('results_json')
                if not covariable or not results_json or df_data is None:
                    continue
                res_df = _read_split_json(results_json)
                if res_df.empty:
                    continue
                table = display_logrank_summary_table(res_df, language=language)
                graph_component = plot_logrank_curves(df_data, covariable, language=language)
                panels.append(html.Div([
                    html.H3(
                        f"Log-Rank Test results for {get_covariate_label(language, covariable)}" if language == 'en' else f"Resultado del Test de Log-Rank para {get_covariate_label(language, covariable)}",
                        style={'textAlign': 'center'}
                    ),
                    graph_component,
                    table
                ]))
    
            return html.Div(panels) if panels else html.Div("No results to display" if language == 'en' else "No hay resultados para mostrar")
        except Exception as e:
            print(f"""[SYNC LOGRANK] Error rebuilding output: {e}""")
            return html.Div(
                f"Error rendering results: {str(e)}" if language == 'en' else f"Error renderizando resultados: {str(e)}"
            )
    
    
