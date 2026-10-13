"""Combined Weibull and Exponential parametric analyses.

Prepare survival data, fit parametric models with lifelines,
and return the figure, summary, and interpretation for the Weibull page.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter, WeibullFitter, ExponentialFitter


def build_weibull_analysis(df, language='en'):
    """Fit a Weibull model and build the figure and summary table."""
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

    wbf = WeibullFitter()
    wbf.fit(durations, event_observed=events, label="Weibull")

    expf = ExponentialFitter()
    expf.fit(durations, event_observed=events, label="Exponential")

    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=events, label="Kaplan-Meier")

    max_time = float(durations.max())
    time_grid = np.linspace(0, max_time, 200)
    fitted_survival = wbf.survival_function_at_times(time_grid).values

    fitted_times = [float(value) for value in time_grid.tolist()]
    fitted_values = [float(value) for value in fitted_survival.tolist()]

    exponential_survival = expf.survival_function_at_times(time_grid).values
    exponential_values = [float(value) for value in exponential_survival.tolist()]

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
            name="Fitted Weibull" if is_en else "Weibull ajustado",
            line=dict(color="#e74c3c", width=3),
            hovertemplate=(
                "<b>Weibull</b><br>Time: %{x:.0f}<br>Survival: %{y:.3f}<extra></extra>"
                if is_en else
                "<b>Weibull</b><br>Tiempo: %{x:.0f}<br>Supervivencia: %{y:.3f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fitted_times,
            y=exponential_values,
            mode="lines",
            name="Fitted Exponential" if is_en else "Exponencial ajustado",
            line=dict(color="#f39c12", width=2.5, dash="dot"),
            hovertemplate=(
                "<b>Exponential</b><br>Time: %{x:.0f}<br>Survival: %{y:.3f}<extra></extra>"
                if is_en else
                "<b>Exponencial</b><br>Tiempo: %{x:.0f}<br>Supervivencia: %{y:.3f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=("Weibull vs Kaplan-Meier vs Exponential" if is_en else "Método Weibull vs Kaplan-Meier vs Exponencial"),
        xaxis_title=("Time" if is_en else "Tiempo"),
        yaxis_title=("Survival probability" if is_en else "Probabilidad de supervivencia"),
        yaxis=dict(range=[0, 1]),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=30, t=70, b=40),
    )

    event_rate = (events.sum() / len(events) * 100) if len(events) else 0
    shape = float(wbf.rho_)
    scale = float(wbf.lambda_)
    median_survival = float(wbf.median_survival_time_)
    exponential_scale = float(expf.lambda_)
    hazard_note = (
        "Increases over time" if shape > 1.05 else
        "Decreases over time" if shape < 0.95 else
        "Remains approximately constant"
    ) if is_en else (
        "Aumenta con el tiempo" if shape > 1.05 else
        "Disminuye con el tiempo" if shape < 0.95 else
        "Se mantiene aproximadamente constante"
    )
    weibull_aic = float(wbf.AIC_)
    exponential_aic = float(expf.AIC_)
    weibull_log_likelihood = float(wbf.log_likelihood_)
    exponential_log_likelihood = float(expf.log_likelihood_)
    better_model = "Weibull" if weibull_aic <= exponential_aic else ("Exponential" if is_en else "Exponencial")
    aic_difference = abs(weibull_aic - exponential_aic)

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

    summary_df = pd.DataFrame([
        {"Metrica": "Number of observations" if is_en else "Numero de observaciones", "Valor": f"{len(events)}", "Interpretacion": "Sample size used for model fitting" if is_en else "Tamanio muestral usado en el ajuste"},
        {"Metrica": "Number of events" if is_en else "Numero de eventos", "Valor": f"{int(events.sum())}", "Interpretacion": "Cases where the event occurs" if is_en else "Casos en los que ocurre el evento"},
        {"Metrica": "Event rate" if is_en else "Tasa de eventos", "Valor": f"{event_rate:.1f}%", "Interpretacion": "Dropout proportion over the total" if is_en else "Proporcion de abandonos sobre el total"},
        {"Metrica": "Shape (rho)", "Valor": f"{shape:.4f}", "Interpretacion": (f"Risk {hazard_note.lower()}" if is_en else f"El riesgo {hazard_note.lower()}")},
        {"Metrica": "Scale (lambda)", "Valor": f"{scale:.4f}", "Interpretacion": "Time scale of the Weibull model" if is_en else "Escala temporal del modelo Weibull"},
        {"Metrica": "Exponential scale (lambda)" if is_en else "Scale exponencial (lambda)", "Valor": f"{exponential_scale:.4f}", "Interpretacion": "Time scale of the Exponential model" if is_en else "Escala temporal del modelo exponencial"},
        {"Metrica": "Median survival" if is_en else "Mediana de supervivencia", "Valor": f"{median_survival:.4f}", "Interpretacion": "Time when survival reaches 50%" if is_en else "Tiempo en el que la supervivencia cae al 50%"},
        {"Metrica": "Log-likelihood", "Valor": f"{weibull_log_likelihood:.4f}", "Interpretacion": "Higher values indicate relatively better fit" if is_en else "Cuanto mayor, mejor ajuste relativo"},
        {"Metrica": "AIC", "Valor": f"{weibull_aic:.4f}", "Interpretacion": "Lower values indicate better fit-complexity tradeoff" if is_en else "Menor valor indica mejor equilibrio entre ajuste y complejidad"},
        {"Metrica": "Exponential AIC" if is_en else "AIC Exponencial", "Valor": f"{exponential_aic:.4f}", "Interpretacion": "Reference value to compare with Weibull" if is_en else "Sirve como referencia para comparar con Weibull"},
        {"Metrica": "Best AIC fit" if is_en else "Mejor ajuste AIC", "Valor": better_model, "Interpretacion": "The model with lower AIC is preferred" if is_en else "El modelo con menor AIC se considera preferible"},
    ])

    comparison_df = pd.DataFrame([
        {
            "Modelo": "Weibull",
            "AIC": round(weibull_aic, 4),
            "LogLikelihood": round(weibull_log_likelihood, 4),
            "DeltaAIC": round(weibull_aic - min(weibull_aic, exponential_aic), 4),
        },
        {
            "Modelo": "Exponential" if is_en else "Exponencial",
            "AIC": round(exponential_aic, 4),
            "LogLikelihood": round(exponential_log_likelihood, 4),
            "DeltaAIC": round(exponential_aic - min(weibull_aic, exponential_aic), 4),
        },
    ])

    model_comparison_interpretation = (
        f"Best fit by AIC: {better_model}. ΔAIC = {aic_difference:.3f}. {evidence_note}"
        if is_en else
        f"Mejor ajuste según AIC: {better_model}. ΔAIC = {aic_difference:.3f}. {evidence_note}"
    )

    interpretation = (
        (
            f"Shape = {shape:.3f}. "
            f"If it is greater than 1, risk increases over time; if lower than 1, it decreases. "
            f"In this fit, risk behavior is: {hazard_note.lower()}. "
            f"The Weibull curve is shown together with empirical Kaplan-Meier and Exponential to visually compare fit quality. "
            f"According to AIC, the model with better relative fit is: {better_model}."
        ) if is_en else (
            f"Shape = {shape:.3f}. "
            f"Si es mayor que 1, el riesgo aumenta con el tiempo; si es menor que 1, disminuye. "
            f"En este ajuste, el comportamiento del riesgo es: {hazard_note.lower()}. "
            f"La curva Weibull se muestra junto a la Kaplan-Meier empírica y la exponencial para comprobar visualmente el ajuste. "
            f"Según el AIC, el modelo con mejor ajuste relativo es: {better_model}."
        )
    )

    return {
        "figure": fig,
        "summary_df": summary_df,
        "interpretation": interpretation,
        "n_observations": len(events),
        "n_events": int(events.sum()),
        "event_rate": event_rate,
        "shape": shape,
        "scale": scale,
        "median_survival": median_survival,
        "log_likelihood": weibull_log_likelihood,
        "aic": weibull_aic,
        "comparison_df": comparison_df,
        "model_comparison_interpretation": model_comparison_interpretation,
        "best_fit_model": better_model,
    }
