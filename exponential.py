"""
Exponential survival analysis helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter, ExponentialFitter, WeibullFitter


def build_exponential_analysis(df: pd.DataFrame, language: str = 'en'):
    """Fit an Exponential model and build the figure and summary table."""
    if df is None or df.empty:
        return None

    required_columns = ["date", "final_result"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        return None

    working_df = df[required_columns].copy()
    working_df["date"] = pd.to_numeric(working_df["date"], errors="coerce")
    working_df["final_result"] = pd.to_numeric(working_df["final_result"], errors="coerce")
    working_df = working_df.dropna(subset=["date", "final_result"])

    if working_df.empty:
        return None

    durations = working_df["date"].astype(float).where(working_df["date"] > 0, 1e-6)
    events = working_df["final_result"].astype(int)

    if durations.nunique() < 2:
        return None

    expf = ExponentialFitter()
    expf.fit(durations, event_observed=events, label="Exponential")

    wbf = WeibullFitter()
    wbf.fit(durations, event_observed=events, label="Weibull")

    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=events, label="Kaplan-Meier")

    max_time = float(durations.max())
    time_grid = np.linspace(0, max_time, 200)
    fitted_survival = expf.survival_function_at_times(time_grid).values

    fitted_times = [float(value) for value in time_grid.tolist()]
    fitted_values = [float(value) for value in fitted_survival.tolist()]

    is_en = language == 'en'

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=kmf.timeline.tolist(),
            y=kmf.survival_function_.iloc[:, 0].tolist(),
            mode="lines",
            name="Empirical Kaplan-Meier" if is_en else "Kaplan-Meier empírico",
            line=dict(color="#2c3e50", width=2, dash="dash"),
            hovertemplate=(
                "<b>Kaplan-Meier</b><br>Time: %{x:.0f}<br>Survival: %{y:.3f}<extra></extra>"
                if is_en else
                "<b>Kaplan-Meier</b><br>Tiempo: %{x:.0f}<br>Supervivencia: %{y:.3f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fitted_times,
            y=fitted_values,
            mode="lines",
            name="Exponential survival curve" if is_en else "Curva de supervivencia exponencial",
            line=dict(color="#e67e22", width=3),
            hovertemplate=(
                "<b>Exponential</b><br>Time: %{x:.0f}<br>Survival: %{y:.3f}<extra></extra>"
                if is_en else
                "<b>Exponencial</b><br>Tiempo: %{x:.0f}<br>Supervivencia: %{y:.3f}<extra></extra>"
            ),
        )
    )

    lambda_value = float(expf.lambda_)
    log_likelihood = float(expf.log_likelihood_)
    aic = float(expf.AIC_)
    weibull_log_likelihood = float(wbf.log_likelihood_)
    weibull_aic = float(wbf.AIC_)

    better_model = "Weibull" if weibull_aic <= aic else ("Exponential" if is_en else "Exponencial")
    aic_difference = abs(weibull_aic - aic)

    if aic_difference < 2:
        evidence_note = (
            "Difference in AIC is small; both models show similar support."
            if is_en else
            "La diferencia de AIC es pequeña; ambos modelos tienen soporte similar."
        )
    elif aic_difference < 6:
        evidence_note = (
            "Difference in AIC suggests moderate evidence in favor of the best model."
            if is_en else
            "La diferencia de AIC sugiere evidencia moderada a favor del mejor modelo."
        )
    else:
        evidence_note = (
            "Difference in AIC suggests strong evidence in favor of the best model."
            if is_en else
            "La diferencia de AIC sugiere evidencia fuerte a favor del mejor modelo."
        )

    fig.update_layout(
        title="Exponential survival curve" if is_en else "Curva de supervivencia exponencial",
        xaxis_title="Time" if is_en else "Tiempo",
        yaxis_title="Survival probability" if is_en else "Probabilidad de supervivencia",
        yaxis=dict(range=[0, 1]),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=30, t=70, b=40),
        annotations=[
            dict(
                x=0.99,
                y=0.03,
                xref="paper",
                yref="paper",
                xanchor="right",
                yanchor="bottom",
                align="left",
                showarrow=False,
                bordercolor="#d0d7de",
                borderwidth=1,
                borderpad=6,
                bgcolor="rgba(255,255,255,0.9)",
                font=dict(size=12, color="#2c3e50"),
                text=(
                    f"<b>{'Lambda value (rate)' if is_en else 'Valor de λ (tasa)'}:</b> {lambda_value:.6f}<br>"
                    f"<b>{'Metrics' if is_en else 'Métricas'}</b><br>"
                    f"• Log-likelihood: {log_likelihood:.4f}<br>"
                    f"• AIC: {aic:.4f}"
                ),
            )
        ],
    )

    event_rate = (events.sum() / len(events) * 100) if len(events) else 0

    summary_df = pd.DataFrame([
        {"Metrica": "Number of observations" if is_en else "Numero de observaciones", "Valor": f"{len(events)}", "Interpretacion": "Sample size used for model fitting." if is_en else "Tamanio muestral usado en el ajuste."},
        {"Metrica": "Number of events" if is_en else "Numero de eventos", "Valor": f"{int(events.sum())}", "Interpretacion": "Cases where the event occurs." if is_en else "Casos en los que ocurre el evento de interes."},
        {"Metrica": "Event rate" if is_en else "Tasa de eventos", "Valor": f"{event_rate:.1f}%", "Interpretacion": "Dropout proportion over the total." if is_en else "Proporcion de abandonos sobre el total."},
        {"Metrica": "Lambda (rate)" if is_en else "Lambda (tasa)", "Valor": f"{lambda_value:.6f}", "Interpretacion": "Rate parameter of the Exponential model." if is_en else "Parametro de tasa del modelo exponencial."},
        {"Metrica": "Log-likelihood", "Valor": f"{log_likelihood:.4f}", "Interpretacion": "Higher values indicate relatively better fit." if is_en else "Cuanto mayor, mejor ajuste relativo."},
        {"Metrica": "AIC", "Valor": f"{aic:.4f}", "Interpretacion": "Lower values indicate better fit-complexity tradeoff." if is_en else "Menor valor indica mejor equilibrio entre ajuste y complejidad."},
        {"Metrica": "Weibull log-likelihood" if is_en else "Log-likelihood Weibull", "Valor": f"{weibull_log_likelihood:.4f}", "Interpretacion": "Reference value to compare with Exponential." if is_en else "Sirve como referencia para comparar con Exponencial."},
        {"Metrica": "Weibull AIC", "Valor": f"{weibull_aic:.4f}", "Interpretacion": "Reference value to compare with Exponential." if is_en else "Sirve como referencia para comparar con Exponencial."},
        {"Metrica": "Best AIC fit" if is_en else "Mejor ajuste AIC", "Valor": better_model, "Interpretacion": "The model with lower AIC is preferred." if is_en else "El modelo con menor AIC se considera preferible."},
        {"Metrica": "Equivalent Weibull case" if is_en else "Caso Weibull equivalente", "Valor": "k = 1", "Interpretacion": "The Exponential model is a special case of Weibull with constant shape." if is_en else "El modelo exponencial es un caso particular del Weibull con forma constante."},
    ])

    comparison_df = pd.DataFrame([
        {
            "Modelo": "Exponential" if is_en else "Exponencial",
            "AIC": round(aic, 4),
            "LogLikelihood": round(log_likelihood, 4),
            "DeltaAIC": round(aic - min(aic, weibull_aic), 4),
        },
        {
            "Modelo": "Weibull",
            "AIC": round(weibull_aic, 4),
            "LogLikelihood": round(weibull_log_likelihood, 4),
            "DeltaAIC": round(weibull_aic - min(aic, weibull_aic), 4),
        },
    ])

    model_comparison_interpretation = (
        f"Best fit by AIC: {better_model}. ΔAIC = {aic_difference:.3f}. {evidence_note}"
        if is_en else
        f"Mejor ajuste según AIC: {better_model}. ΔAIC = {aic_difference:.3f}. {evidence_note}"
    )

    interpretation = (
        (
            f"Lambda = {lambda_value:.6f}. "
            f"This implies a constant dropout rate over time, so risk does not change with time. "
            f"The Exponential model is a special case of Weibull when k = 1. "
            f"The Exponential curve is compared against empirical Kaplan-Meier to assess overall fit."
        ) if is_en else (
            f"Lambda = {lambda_value:.6f}. "
            f"Esto implica una tasa de abandono constante en el tiempo, por lo que el riesgo no cambia con el tiempo. "
            f"El modelo Exponencial es un caso particular del Weibull cuando k = 1. "
            f"La curva exponencial se compara con Kaplan-Meier empírico para ver el ajuste global."
        )
    )

    return {
        "figure": fig,
        "summary_df": summary_df,
        "interpretation": interpretation,
        "n_observations": len(events),
        "n_events": int(events.sum()),
        "event_rate": event_rate,
        "lambda_value": lambda_value,
        "log_likelihood": log_likelihood,
        "aic": aic,
        "comparison_df": comparison_df,
        "model_comparison_interpretation": model_comparison_interpretation,
        "best_fit_model": better_model,
    }
