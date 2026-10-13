"""Integrate the local AI model for narrative explanations.

Build academic prompts for each survival technique, call the local
OpenAI-compatible server, and normalize responses for display
in the application or inclusion in PDF reports.
"""

import requests
import json
import time
import sys
import re
import pandas as pd
from config import LLAMA_SERVER_URL, MODEL_NAME

DEFAULT_TEMPERATURE = 0.15
DEFAULT_TOP_P = 0.85
PDF_MAX_TOKENS = 560
EXPLAIN_MAX_TOKENS = 320
QUESTION_MAX_TOKENS = 560
REQUEST_TIMEOUT_S = 240


def _looks_like_list_output(text):
    """Detect whether the model response looks like a list rather than prose."""
    if not text:
        return False
    stripped = text.strip()
    if stripped.startswith('-') or stripped.startswith('*'):
        return True
    return any(marker in stripped for marker in ('1)', '2)', '3)', '1.', '2.', '3.'))


def _trim_incomplete_final_sentence(text):
    """Avoid displaying responses cut off mid-sentence by the token limit."""
    if not text:
        return text

    stripped = text.strip()
    if stripped.endswith(('.', '!', '?', ')', ']', '"', "'")):
        return stripped

    sentence_endings = list(re.finditer(r'[.!?](?=\s|$)', stripped))
    if not sentence_endings:
        return stripped

    trimmed = stripped[:sentence_endings[-1].end()].strip()
    return trimmed or stripped


def _academic_system_prompt(language='en'):
    """Create a system message requiring a concise academic tone."""
    if language == 'en':
        return (
            "You are a survival analysis expert writing for an academic thesis on university student dropout. "
            "Write in formal, evidence-based academic prose using only the numerical results provided. "
            "Cite only the most relevant values, such as survival probabilities, p-values, hazard ratios or model parameters, as evidence. "
            "Do not transcribe tables row by row: synthesize the main pattern, compare groups or curves, and explain what it means in the student dropout context. "
            "Return 2-3 short paragraphs with no bullet points, no numbered lists, and no invented data. "
            "Always finish with a complete final sentence."
        )
    return (
        "Eres un experto en análisis de supervivencia redactando para un TFG sobre abandono académico universitario. "
        "Redacta en prosa académica formal y basada en evidencias, usando solo los resultados numéricos proporcionados. "
        "Cita solo los valores mas relevantes, como probabilidades de supervivencia, p-valores, hazard ratios o parametros del modelo, como evidencia. "
        "No transcribas tablas fila por fila: sintetiza el patron principal, compara grupos o curvas y explica que significa en el contexto del abandono estudiantil. "
        "Devuelve 2-3 párrafos breves, sin viñetas ni listas numeradas, sin inventar datos. "
        "Termina siempre con una frase final completa."
    )


def _rewrite_to_prose_if_needed(content, max_tokens, language='en', timeout=REQUEST_TIMEOUT_S):
    """Rewrite list responses as paragraphs."""
    if not _looks_like_list_output(content):
        return content

    rewrite_payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": _academic_system_prompt(language)},
            {
                "role": "user",
                "content": (
                    "Reescribe este contenido en prosa académica manteniendo exactamente los hechos, "
                    "sin viñetas ni numeración:\n\n" + content
                ),
            },
        ],
        "temperature": 0.1,
        "top_p": DEFAULT_TOP_P,
        "max_tokens": max_tokens,
        "stream": False,
    }
    response = requests.post(LLAMA_SERVER_URL, json=rewrite_payload, timeout=timeout)
    response.raise_for_status()
    result = response.json()
    rewritten = result['choices'][0]['message']['content'].strip()
    return rewritten or content


def _call_llm(prompt, max_tokens, temperature=DEFAULT_TEMPERATURE, timeout=REQUEST_TIMEOUT_S, language='en'):
    """Invoke llama-server with consistent parameters and academic output."""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": _academic_system_prompt(language)},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "top_p": DEFAULT_TOP_P,
        "max_tokens": max_tokens,
        "stream": False,
    }
    response = requests.post(LLAMA_SERVER_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    result = response.json()
    content = result['choices'][0]['message']['content'].strip()
    content = _rewrite_to_prose_if_needed(content, max_tokens=max_tokens, language=language, timeout=timeout)
    return _trim_incomplete_final_sentence(content)


def responder_pregunta_con_llama3(pregunta: str, language: str = 'en') -> str:
    """Shared interface for AI explanations requested from the dashboard."""
    try:
        return _call_llm(
            pregunta,
            max_tokens=QUESTION_MAX_TOKENS,
            timeout=REQUEST_TIMEOUT_S,
            language=language,
        )
    except requests.exceptions.ConnectionError:
        return ""
    except requests.exceptions.Timeout:
        return ""
    except Exception as e:
        print(f"""Error querying the AI model: {e}""")
        return ""


def generate_interpretation_for_pdf(analysis_type, data_summary=None, table_data=None, language='en'):
    """
    Generate automatic PDF interpretations using an LLM.

    Args:
        analysis_type: 'kaplan-meier', 'cox', or 'log-rank'.
        data_summary: General statistics {n_patients, n_events, variable_name}.
        table_data: DataFrame containing analysis results.

    Returns:
        str: Interpretation text formatted for PDF.
    """
    
    # Build a prompt specific to the analysis type.
    if analysis_type == 'kaplan-meier':
        prompt = _build_km_prompt(data_summary, table_data, language)
    elif analysis_type == 'cox':
        prompt = _build_cox_prompt(data_summary, table_data, language)
    elif analysis_type == 'log-rank':
        prompt = _build_logrank_prompt(data_summary, table_data, language)
    elif analysis_type == 'weibull':
        prompt = _build_weibull_prompt(data_summary, table_data, language)
    elif analysis_type == 'exponential':
        prompt = _build_exponential_prompt(data_summary, table_data, language)
    else:
        return "Could not generate an interpretation for this type of analysis." if language == 'en' else "No se pudo generar interpretación para este tipo de análisis."
    
    try:
        print(f"""⏳ Generating an AI interpretation for {analysis_type}...""")

        inicio = time.time()
        interpretation = _call_llm(prompt, max_tokens=PDF_MAX_TOKENS, language=language)
        tiempo_respuesta = time.time() - inicio
        print(f"""Interpretation generated in {tiempo_respuesta:.2f}s""")
        
        return interpretation
    
    except requests.exceptions.ConnectionError:
        return ""
    except Exception as e:
        print(f"""❌ Error generating interpretation: {str(e)}""")
        return ""


def _build_km_prompt(data_summary, table_data, language='en'):
    """Build a Kaplan-Meier prompt."""
    data_summary = data_summary or {}
    km_lines = []
    if table_data is not None and isinstance(table_data, pd.DataFrame) and len(table_data) > 0:
        try:
            sample = table_data.head(8)
            col_t = sample.columns[0]
            col_s = sample.columns[1] if len(sample.columns) > 1 else None
            for _, row in sample.iterrows():
                t_val = row.get(col_t, '')
                s_val = row.get(col_s, '') if col_s else ''
                if col_s:
                    km_lines.append(f"- t={t_val}, S(t)={s_val}")
                else:
                    km_lines.append(f"- {t_val}")
        except Exception:
            km_lines.append("- KM table available but could not be parsed")

    km_text = "\n".join(km_lines) if km_lines else "- No KM table available"

    if language == 'en':
        prompt = f"""
You are an expert in biostatistics and survival analysis.
Provide a concise interpretation (max 140 words) of this Kaplan-Meier analysis.
Use only the information provided and do not invent values.

STUDY DATA:
- Number of students/observations: {data_summary.get('n_patients', 0)}
- Number of events: {data_summary.get('n_events', 0)}
- Analyzed variable: {data_summary.get('variable_name', 'General')}
- Mean follow-up: {data_summary.get('follow_up_mean', 0):.1f} months
- Median follow-up: {data_summary.get('follow_up_median', 0):.1f} months

KM TABLE (sample):
{km_text}

Return exactly 3 short paragraphs:
1) curve behavior and risk trend
2) practical meaning
3) one limitation
"""
    else:
        prompt = f"""
Eres experto en bioestadística y supervivencia.
Da una interpretación concisa (máximo 140 palabras) del análisis Kaplan-Meier.
Usa solo los datos disponibles y no inventes valores.

DATOS DEL ESTUDIO:
- Número de estudiantes/observaciones: {data_summary.get('n_patients', 0)}
- Número de eventos: {data_summary.get('n_events', 0)}
- Variable analizada: {data_summary.get('variable_name', 'General')}
- Follow-up medio: {data_summary.get('follow_up_mean', 0):.1f} meses
- Follow-up mediano: {data_summary.get('follow_up_median', 0):.1f} meses

TABLA KM (muestra):
{km_text}

Devuelve exactamente 3 párrafos cortos:
1) comportamiento de la curva y tendencia de riesgo
2) interpretación práctica
3) una limitación
"""
    return prompt


def _build_cox_prompt(data_summary, table_data, language='en'):
    """Build a Cox regression prompt."""
    data_summary = data_summary or {}
    
    # Create a text table for the prompt.
    table_text = ""
    if table_data is not None and isinstance(table_data, pd.DataFrame) and len(table_data) > 0:
        try:
            # Use the correct Cox model column names.
            # Expected columns: Variable/Covariable, exp(Coef.), Coef. lower 95%, Coef. upper 95%, p.
            lines = []
            for idx, row in table_data.iterrows():
                # Get the variable name from the first element or 'Variable'/'Covariable' column.
                var_name = row.get('Variable') or row.get('Covariable') or row.iloc[0] if isinstance(row.iloc[0], str) else f"Variable_{idx}"
                
                # Get HR (exp(Coef.)).
                hr = row.get('exp(Coef.)') or row.get('HR') or 1.0
                
                # Get the 95% CI.
                ic_lower = row.get('Coef. lower 95%') or 0
                ic_upper = row.get('Coef. upper 95%') or 0
                
                # Get the p-value.
                p_val = row.get('p') or row.get('p-valor') or 1.0
                
                lines.append(f"  - {var_name}: HR={float(hr):.2f} (IC 95%: {float(ic_lower):.2f}-{float(ic_upper):.2f}), p={float(p_val):.4f}")
            
            table_text = "\n".join(lines)
        except Exception as e:
            print(f"""Error processing Cox table for the prompt: {e}""")
            table_text = "Results table [unsupported format]"
    
    if language == 'en':
        prompt = f"""
You are an expert statistician in multivariate survival analysis.
Provide a professional, direct, and concise interpretation of the Cox regression analysis (2-3 paragraphs):

GENERAL DATA:
- Total number of students/observations: {data_summary.get('n_patients', 0)}
- Documented events: {data_summary.get('n_events', 0)}
- Included variables: {data_summary.get('variable_name', 'multiple')}

MODEL RESULTS:
{table_text}

Please:
1. Explain what each Hazard Ratio means
2. Identify the significant variables (p < 0.05)
3. Interpret relative risk in academic dropout terms
4. Mention the Cox model assumptions and limitations concisely
"""
    else:
        prompt = f"""
Eres un experto estadístico en análisis de supervivencia multivariado.
Proporciona una interpretación profesional, directa y breve del análisis de regresión de Cox (2-3 párrafos):

DATOS GENERALES:
- Número total de estudiantes/observaciones: {data_summary.get('n_patients', 0)}
- Eventos documentados: {data_summary.get('n_events', 0)}
- Variables incluidas: {data_summary.get('variable_name', 'múltiples')}

RESULTADOS DEL MODELO:
{table_text}

Por favor:
1. Explica qué significa cada Hazard Ratio
2. Identifica las variables significativas (p < 0.05)
3. Interpreta el riesgo relativo en términos de abandono académico
4. Menciona supuestos y limitaciones del modelo Cox de forma concisa
"""
    return prompt


def _build_logrank_prompt(data_summary, table_data, language='en'):
    """Build a Log-Rank test prompt."""
    data_summary = data_summary or {}
    logrank_text = "- No Log-Rank table available"
    if table_data is not None and isinstance(table_data, pd.DataFrame) and len(table_data) > 0:
        try:
            rows = []
            for _, row in table_data.head(8).iterrows():
                cov = row.get('Covariable', row.get('Variable', 'N/A'))
                chi2 = row.get('test_statistic', row.get('chi2', 'N/A'))
                pval = row.get('p_value', row.get('p', 'N/A'))
                rows.append(f"- {cov}: chi2={chi2}, p={pval}")
            logrank_text = "\n".join(rows)
        except Exception:
            logrank_text = "- Log-Rank table available but could not be parsed"

    if language == 'en':
        prompt = f"""
You are an expert in biostatistics.
Provide a concise interpretation of this log-rank test (max 130 words).
Do not invent any values.

CONTEXT:
- Number of students/observations: {data_summary.get('n_patients', 0)}
- Total events: {data_summary.get('n_events', 0)}
- Compared variable: {data_summary.get('variable_name', 'unknown')}

RESULTS:
{logrank_text}

Return exactly 2 paragraphs:
1) significance and p-value interpretation
2) practical implication and one caveat
"""
    else:
        prompt = f"""
Eres experto en bioestadística.
Da una interpretación breve del test Log-Rank (máximo 130 palabras).
No inventes datos.

CONTEXTO:
- Número de estudiantes/observaciones: {data_summary.get('n_patients', 0)}
- Eventos totales: {data_summary.get('n_events', 0)}
- Variable comparada: {data_summary.get('variable_name', 'desconocida')}

RESULTADOS:
{logrank_text}

Devuelve exactamente 2 párrafos:
1) significación e interpretación del p-valor
2) implicación práctica y una limitación
"""
    return prompt


def _build_weibull_prompt(data_summary, table_data, language='en'):
    """Build a Weibull prompt."""
    data_summary = data_summary or {}
    table_text = ""
    if table_data is not None and isinstance(table_data, pd.DataFrame) and len(table_data) > 0:
        try:
            lines = []
            for _, row in table_data.iterrows():
                metric = row.get('Metrica') or row.get('Metric') or row.iloc[0]
                value = row.get('Valor') or row.get('Value') or row.iloc[1]
                lines.append(f"  - {metric}: {value}")
            table_text = "\n".join(lines)
        except Exception as e:
            print(f"""Error processing Weibull table for the prompt: {e}""")
            table_text = "Results table [unsupported format]"

    if language == 'en':
        prompt = f"""
You are an expert in survival analysis and parametric modeling.
Provide a concise interpretation of a Weibull survival analysis (2-3 paragraphs):

GENERAL DATA:
- Total number of observations: {data_summary.get('n_patients', 0)}
- Events: {data_summary.get('n_events', 0)}
- Event rate: {data_summary.get('event_rate', 0):.1f}%

MODEL RESULTS:
{table_text}

Please:
1. Explain the meaning of the shape and scale parameters
2. State whether the risk increases, decreases, or remains roughly constant over time
3. Comment on the fitted curve and the main practical interpretation
"""
    else:
        prompt = f"""
Eres un experto en análisis de supervivencia y modelado paramétrico.
Proporciona una interpretación clara, directa y concisa de un análisis Weibull de supervivencia en 2-3 párrafos.
No devuelvas listas ni texto vacío. Redacta una interpretación completa y natural.

DATOS GENERALES:
- Número total de observaciones: {data_summary.get('n_patients', 0)}
- Eventos: {data_summary.get('n_events', 0)}
- Tasa de eventos: {data_summary.get('event_rate', 0):.1f}%

RESULTADOS DEL MODELO:
{table_text}

Por favor:
1. Explica el significado de los parámetros de forma y escala
2. Indica si el riesgo aumenta, disminuye o se mantiene aproximadamente constante con el tiempo
3. Comenta la curva ajustada y la interpretación práctica principal
"""
    return prompt


def _build_exponential_prompt(data_summary, table_data, language='en'):
    """Build an Exponential survival prompt."""
    data_summary = data_summary or {}
    table_text = ""
    if table_data is not None and isinstance(table_data, pd.DataFrame) and len(table_data) > 0:
        try:
            lines = []
            for _, row in table_data.iterrows():
                metric = row.get('Metrica') or row.get('Metric') or row.iloc[0]
                value = row.get('Valor') or row.get('Value') or row.iloc[1]
                lines.append(f"- {metric}: {value}")
            table_text = "\n".join(lines)
        except Exception:
            table_text = "- Results table could not be parsed"
    else:
        table_text = "- No results table available"

    if language == 'en':
        return f"""
You are an expert in survival analysis.
Interpret the Exponential model shown in the dashboard using only the provided table/curve information.

DATA:
- Observations: {data_summary.get('n_patients', 0)}
- Events: {data_summary.get('n_events', 0)}
- Event rate: {data_summary.get('event_rate', 0):.1f}%

TABLE SUMMARY:
{table_text}

Return exactly 3 short paragraphs:
1) what lambda implies (constant hazard over time)
2) how the exponential fit compares conceptually to empirical KM curve
3) one practical implication and one limitation
"""

    return f"""
Eres experto en análisis de supervivencia.
Interpreta el modelo Exponencial mostrado en el dashboard usando solo la información de tabla y curva.

DATOS:
- Observaciones: {data_summary.get('n_patients', 0)}
- Eventos: {data_summary.get('n_events', 0)}
- Tasa de eventos: {data_summary.get('event_rate', 0):.1f}%

TABLA RESUMEN:
{table_text}

Devuelve exactamente 3 párrafos cortos:
1) qué implica lambda (riesgo constante en el tiempo)
2) cómo encaja el modelo exponencial frente a la curva KM empírica
3) una implicación práctica y una limitación
"""

def _format_context_for_prompt(context_data, max_chars=3500):
    """Convert chart/table data into compact text for the prompt."""
    if context_data is None:
        return "No additional numerical data was provided."
    if isinstance(context_data, str) and not context_data.strip():
        return "No additional numerical data was provided."

    try:
        if isinstance(context_data, pd.DataFrame):
            text = context_data.head(20).to_string(index=False)
        elif isinstance(context_data, dict):
            text = json.dumps(context_data, ensure_ascii=False, default=str, indent=2)
        elif isinstance(context_data, (list, tuple)):
            if context_data and all(isinstance(item, dict) for item in context_data):
                text = pd.DataFrame(context_data).head(20).to_string(index=False)
            else:
                text = json.dumps(context_data, ensure_ascii=False, default=str, indent=2)
        else:
            text = str(context_data)
    except Exception:
        text = str(context_data)

    return text[:max_chars]


def generate_explanation(graph_data, model_type, language='en'):
    """
    Generate explanations using llama.cpp (llama-server) with Qwen2.5-1.5B-Instruct.
    Use a local endpoint without external dependencies.
    Apply the configured timeout to avoid blocking the interface.
    """
    context_text = _format_context_for_prompt(graph_data)

    if language == 'en':
        prompt_map = {
            'kaplan-meier': "Interpret the Kaplan-Meier results using only the data below.",
            'log-rank': "Interpret the Log-Rank test using only the data below.",
            'cox-regression': "Interpret the Cox regression using only the data below.",
            'weibull': "Interpret the Weibull survival model using only the data below.",
            'exponential': "Interpret the Exponential survival model using only the data below.",
            'rsf': "Interpret the Random Survival Forest results using only the data below.",
        }
        task = prompt_map.get(model_type, "Interpret these survival-analysis results using only the data below.")
        prompt = f"""{task}

DATA:
{context_text}

Return 2 short academic paragraphs. Mention concrete values when present, do not invent values, and include one limitation."""
    else:
        prompt_map = {
            'kaplan-meier': "Interpreta los resultados de Kaplan-Meier usando solo los datos siguientes.",
            'log-rank': "Interpreta el test Log-Rank usando solo los datos siguientes.",
            'cox-regression': "Interpreta la regresión de Cox usando solo los datos siguientes.",
            'weibull': "Interpreta el modelo Weibull de supervivencia usando solo los datos siguientes.",
            'exponential': "Interpreta el modelo Exponencial de supervivencia usando solo los datos siguientes.",
            'rsf': "Interpreta los resultados del Random Survival Forest usando solo los datos siguientes.",
        }
        task = prompt_map.get(model_type, "Interpreta estos resultados de supervivencia usando solo los datos siguientes.")
        prompt = f"""{task}

DATOS:
{context_text}

Devuelve 2 párrafos académicos breves. Cita valores concretos cuando existan, no inventes valores e incluye una limitación."""

    try:
        # Log the start.
        inicio = time.time()
        print(f"""\n⏳ Sending request to the LLM...""")
        print(f"""Waiting for a response...\n""")
        sys.stdout.flush()
        explanation = _call_llm(prompt, max_tokens=EXPLAIN_MAX_TOKENS, timeout=REQUEST_TIMEOUT_S, language=language)
        
        # Calculate response time.
        tiempo_respuesta = time.time() - inicio
        
        # Display processing time.
        minutos = int(tiempo_respuesta // 60)
        segundos = int(tiempo_respuesta % 60)
        tiempo_str = f"{minutos}m {segundos}s" if minutos > 0 else f"{segundos}s"
        print(f"""\nResponse generated in: {tiempo_str}""")
        sys.stdout.flush()
        
        # Limit the length if necessary.
        max_length = 3000
        if len(explanation) > max_length:
            explanation_parts = [explanation[i:i + 1000] for i in range(0, len(explanation), 1000)]
            return '\n\n'.join(explanation_parts)
        
        return explanation
        
    except requests.exceptions.ConnectionError:
        return "❌ Error: Could not connect to llama-server. Make sure it is running at http://127.0.0.1:8000"
    except requests.exceptions.Timeout:
        return "❌ Error: Timeout (the model took more than 10 minutes to respond). Try shorter prompts."
    except Exception as e:
        return f"❌ Error generating explanation: {str(e)}"
