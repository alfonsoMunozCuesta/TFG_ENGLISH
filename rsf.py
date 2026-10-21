"""
Random Survival Forest helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from translations import get_feature_label


PROFILE_CATEGORIES = {
    "gender_F": {
        1: "Femenino",
        0: "Masculino",
    },
    "disability_N": {
        1: "Sin discapacidad",
        0: "Con discapacidad",
    },
    "age_band": {
        "age_band_0-35": "0-35 años",
        "age_band_35-55": "35-55 años",
        "age_band_55<=": "55+ años",
    },
    "highest_education": {
        "highest_education_A Level or Equivalent": "A Level o equivalente",
        "highest_education_HE Qualification": "HE Qualification",
        "highest_education_Lower Than A Level": "Inferior a A Level",
        "highest_education_Post Graduate Qualification": "Postgrado",
    },
}

PROFILE_CREDIT_LEVELS = {
    "few": 30,
    "medium": 60,
    "many": 120,
}


def _coerce_event_column(series: pd.Series) -> pd.Series:
    """Convert the event column into binary values compatible with survival analysis."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).astype(int)

    normalized = series.astype(str).str.strip().str.lower()
    positive_values = {"1", "true", "yes", "y", "withdrawn", "dropout", "abandono", "event"}
    return normalized.isin(positive_values).astype(int)


def _build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Prepare predictors, time, and events for RSF model training."""
    if df is None or df.empty:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=int)

    if "date" not in df.columns or "final_result" not in df.columns:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=int)

    working_df = df.copy()
    working_df["date"] = pd.to_numeric(working_df["date"], errors="coerce")
    working_df["final_result"] = _coerce_event_column(working_df["final_result"])
    working_df = working_df.dropna(subset=["date", "final_result"])

    if working_df.empty:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=int)

    durations = working_df["date"].astype(float).clip(lower=1e-6)
    events = working_df["final_result"].astype(int)

    feature_df = working_df.drop(columns=["date", "final_result"], errors="ignore")
    feature_df = feature_df.select_dtypes(include=["number", "bool", "object", "category"])
    feature_df = pd.get_dummies(feature_df, drop_first=False)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

    if feature_df.empty:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=int)

    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
    feature_df = feature_df.fillna(feature_df.median(numeric_only=True)).fillna(0.0)

    valid_rows = durations.notna() & events.notna()
    feature_df = feature_df.loc[valid_rows].reset_index(drop=True)
    durations = durations.loc[valid_rows].reset_index(drop=True)
    events = events.loc[valid_rows].reset_index(drop=True)

    return feature_df, durations, events


def _build_importance_proxy(feature_df: pd.DataFrame, risk_scores: np.ndarray) -> np.ndarray:
    """Approximate importance by correlating variables with risk."""
    correlations = []
    for column in feature_df.columns:
        series = pd.to_numeric(feature_df[column], errors="coerce").fillna(feature_df[column].median())
        if series.nunique(dropna=True) <= 1:
            correlations.append(0.0)
            continue
        try:
            value = np.corrcoef(series.to_numpy(dtype=float), risk_scores)[0, 1]
            correlations.append(float(abs(value)) if np.isfinite(value) else 0.0)
        except Exception:
            correlations.append(0.0)
    return np.asarray(correlations, dtype=float)


def _fit_rsf_model(df: pd.DataFrame):
    """Train the Random Survival Forest and return its supporting data."""
    feature_df, durations, events = _build_feature_matrix(df)
    if feature_df.empty or durations.nunique() < 2:
        return None

    if len(feature_df) < 5 or feature_df.shape[1] < 1:
        return None

    y = Surv.from_arrays(event=events.astype(bool).to_numpy(), time=durations.to_numpy(dtype=float))

    rsf = RandomSurvivalForest(
        n_estimators=200,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
        oob_score=True,
    )
    rsf.fit(feature_df, y)

    risk_scores = rsf.predict(feature_df)
    train_c_index = concordance_index_censored(events.astype(bool).to_numpy(), durations.to_numpy(dtype=float), risk_scores)[0]
    oob_score = getattr(rsf, "oob_score_", None)

    return {
        "model": rsf,
        "feature_df": feature_df,
        "durations": durations,
        "events": events,
        "risk_scores": risk_scores,
        "train_c_index": float(train_c_index),
        "oob_score": None if oob_score is None else float(oob_score),
    }


def _build_profile_defaults(df: pd.DataFrame) -> dict:
    """Get reasonable defaults for simulating individual profiles."""
    defaults = {}
    if df is None or df.empty:
        return defaults

    feature_columns = [column for column in df.columns if column not in {"date", "final_result"}]
    for column in feature_columns:
        series = df[column]
        numeric_series = pd.to_numeric(series, errors="coerce")
        if numeric_series.notna().any():
            defaults[column] = float(numeric_series.median())
        else:
            mode = series.mode(dropna=True)
            if not mode.empty:
                defaults[column] = mode.iloc[0]
            else:
                defaults[column] = 0

    return defaults


def _apply_profile_overrides(defaults: dict, profile: dict, df: pd.DataFrame) -> dict:
    """Apply the user's selected options to the base profile."""
    profile_values = dict(defaults)

    if "gender_F" in df.columns:
        profile_values["gender_F"] = 1 if int(profile.get("gender_F", 1)) == 1 else 0

    if "disability_N" in df.columns:
        profile_values["disability_N"] = 1 if int(profile.get("disability_N", 1)) == 1 else 0

    age_choice = profile.get("age_band", "age_band_0-35")
    for column in PROFILE_CATEGORIES["age_band"]:
        if column in df.columns:
            profile_values[column] = 1 if column == age_choice else 0

    education_choice = profile.get("highest_education", "highest_education_A Level or Equivalent")
    for column in PROFILE_CATEGORIES["highest_education"]:
        if column in df.columns:
            profile_values[column] = 1 if column == education_choice else 0

    if "studied_credits" in df.columns:
        credits_value = profile.get("studied_credits", PROFILE_CREDIT_LEVELS["few"])
        try:
            profile_values["studied_credits"] = float(credits_value)
        except (TypeError, ValueError):
            profile_values["studied_credits"] = float(PROFILE_CREDIT_LEVELS["few"])

    return profile_values


def _build_profile_feature_frame(df: pd.DataFrame, profile: dict, feature_columns: pd.Index) -> pd.DataFrame:
    """Build a predictor row compatible with the model for the simulated profile."""
    defaults = _build_profile_defaults(df)
    profile_values = _apply_profile_overrides(defaults, profile, df)
    profile_df = pd.DataFrame([profile_values])
    profile_df = pd.get_dummies(profile_df, drop_first=False)
    profile_df = profile_df.reindex(columns=feature_columns, fill_value=0)
    profile_df = profile_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return profile_df


def build_rsf_analysis(df: pd.DataFrame, language: str = 'en'):
    """Fit a Random Survival Forest and build reusable outputs for the dashboard."""
    fitted = _fit_rsf_model(df)
    if not fitted:
        return None

    rsf = fitted["model"]
    feature_df = fitted["feature_df"]
    durations = fitted["durations"]
    events = fitted["events"]
    risk_scores = fitted["risk_scores"]
    train_c_index = fitted["train_c_index"]
    oob_score = fitted["oob_score"]

    sorted_indices = np.argsort(risk_scores)
    if len(sorted_indices) == 0:
        return None

    percentile_positions = [0.2, 0.5, 0.8]
    representative_indices = []
    for percentile in percentile_positions:
        position = int(round((len(sorted_indices) - 1) * percentile))
        candidate = int(sorted_indices[position])
        while candidate in representative_indices and position + 1 < len(sorted_indices):
            position += 1
            candidate = int(sorted_indices[position])
        representative_indices.append(candidate)

    survival_curves = rsf.predict_survival_function(feature_df.iloc[representative_indices], return_array=True)
    curve_times = np.asarray(rsf.unique_times_, dtype=float)
    curve_times_plot = [float(value) for value in curve_times.tolist()]

    if len(feature_df.columns) > 0:
        try:
            importances = rsf.feature_importances_
        except (AttributeError, NotImplementedError):
            importances = None

        if importances is None or not np.isfinite(importances).any() or np.allclose(importances, 0):
            importances = _build_importance_proxy(feature_df, risk_scores)
    else:
        importances = np.asarray([])

    importances = np.asarray(importances, dtype=float)
    if importances.size:
        normalized_importances = importances / importances.sum() if importances.sum() > 0 else importances
        top_indices = np.argsort(normalized_importances)[::-1][:10]
        top_features = [
            {
                "name": feature_df.columns[index],
                "importance": float(normalized_importances[index]),
            }
            for index in top_indices
            if normalized_importances[index] > 0
        ]
    else:
        top_features = []

    top_feature_name = top_features[0]["name"] if top_features else "N/A"
    top_feature_value = top_features[0]["importance"] if top_features else 0.0

    is_en = language == 'en'
    top_feature_label = get_feature_label(language, top_feature_name)

    survival_labels = ["Low risk", "Medium risk", "High risk"] if is_en else ["Bajo riesgo", "Riesgo medio", "Alto riesgo"]
    if len(representative_indices) < 3:
        survival_labels = [f"Group {index + 1}" if is_en else f"Grupo {index + 1}" for index in range(len(representative_indices))]

    fig_survival = go.Figure()
    colors = ["#1abc9c", "#2980b9", "#e74c3c"]
    y_min = 1.0
    survival_curve_summaries = []

    def _hex_to_rgba(hex_color: str, alpha: float) -> str:
        """Convert hexadecimal colors to RGBA to shade intervals in Plotly."""
        hex_color = hex_color.lstrip("#")
        red = int(hex_color[0:2], 16)
        green = int(hex_color[2:4], 16)
        blue = int(hex_color[4:6], 16)
        return f"rgba({red}, {green}, {blue}, {alpha})"

    for index, row_index in enumerate(representative_indices):
        curve = survival_curves[index]
        curve_plot = [float(value) for value in np.asarray(curve, dtype=float).tolist()]
        y_min = min(y_min, float(np.min(curve)))
        survival_curve_summaries.append({
            "label": survival_labels[index],
            "risk_score": float(risk_scores[row_index]),
            "final_survival": float(curve_plot[-1]) if curve_plot else None,
            "min_survival": float(np.min(curve)) if len(curve) else None,
        })
        fig_survival.add_trace(
            go.Scatter(
                x=curve_times_plot,
                y=curve_plot,
                mode="lines+markers",
                name=f"{survival_labels[index]} ({'score' if is_en else 'puntuación'}={risk_scores[row_index]:.3f})",
                line=dict(color=colors[index % len(colors)], width=4),
                line_shape="linear",
                marker=dict(size=7, color=colors[index % len(colors)]),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>Time: %{x:.0f}<br>Survival: %{y:.3f}<extra></extra>"
                    if is_en else
                    "<b>%{fullData.name}</b><br>Tiempo: %{x:.0f}<br>Supervivencia: %{y:.3f}<extra></extra>"
                ),
                showlegend=True,
            )
        )

    x_axis_max = float(np.max(curve_times)) if len(curve_times) else 1.0
    if x_axis_max <= 0:
        x_axis_max = 1.0

    y_axis_min = max(0.70, y_min - 0.08)
    if y_axis_min > 0.92:
        y_axis_min = 0.92

    fig_survival.update_layout(
        title="Random Survival Forest: estimated survival curves" if is_en else "Random Survival Forest: curvas de supervivencia estimadas",
        xaxis_title="Time" if is_en else "Tiempo",
        yaxis_title="Survival probability" if is_en else "Probabilidad de supervivencia",
        xaxis=dict(range=[0, x_axis_max], fixedrange=True),
        yaxis=dict(range=[y_axis_min, 1], fixedrange=True),
        height=720,
        hovermode="x unified",
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=90, b=60),
    )

    importance_df = pd.DataFrame(top_features)
    if not importance_df.empty:
        importance_fig = go.Figure(
            go.Bar(
                x=importance_df["importance"].iloc[::-1],
                y=importance_df["name"].iloc[::-1].map(lambda name: get_feature_label(language, name)),
                orientation="h",
                marker=dict(color="#2980b9"),
                hovertemplate=(
                    "<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>"
                    if is_en else
                    "<b>%{y}</b><br>Importancia: %{x:.3f}<extra></extra>"
                ),
            )
        )
    else:
        importance_fig = go.Figure()

    importance_fig.update_layout(
        title="Variable importance in RSF" if is_en else "Importancia de variables en RSF",
        xaxis_title="Normalized importance" if is_en else "Importancia normalizada",
        yaxis_title="Variable",
        template="plotly_white",
        margin=dict(l=180, r=30, t=70, b=40),
        height=max(460, 60 * max(len(top_features), 3)),
    )

    summary_df = pd.DataFrame([
        {
            "Metrica": "Observations" if is_en else "Observaciones",
            "Valor": f"{len(feature_df)}",
            "Interpretacion": "Rows used to train the forest." if is_en else "Número de filas usadas para entrenar el bosque.",
        },
        {
            "Metrica": "Events" if is_en else "Eventos",
            "Valor": f"{int(events.sum())}",
            "Interpretacion": "Cases where the event occurred." if is_en else "Casos en los que se produjo el evento de interés.",
        },
        {
            "Metrica": "Features" if is_en else "Variables",
            "Valor": f"{feature_df.shape[1]}",
            "Interpretacion": "Predictors used after preprocessing." if is_en else "Número de predictores usados tras el preprocesado.",
        },
        {
            "Metrica": "Train concordance" if is_en else "Concordancia train",
            "Valor": f"{train_c_index:.3f}",
            "Interpretacion": "The closer to 1, the better risk ranking." if is_en else "Cuanto más cerca de 1, mejor ordena los riesgos.",
        },
        {
            "Metrica": "OOB concordance" if is_en else "Concordancia OOB",
            "Valor": f"{oob_score:.3f}" if oob_score is not None else "N/A",
            "Interpretacion": "Out-of-bag estimate of overall performance." if is_en else "Estimación fuera de bolsa del rendimiento general.",
        },
        {
            "Metrica": "Most influential variable" if is_en else "Variable más influyente",
            "Valor": top_feature_label,
            "Interpretacion": (f"Approximate relative weight: {top_feature_value:.3f}" if is_en else f"Peso relativo aproximado: {top_feature_value:.3f}"),
        },
    ])

    if oob_score is not None:
        interpretation = (
            (
                f"Random Survival Forest combines {feature_df.shape[1]} variables through trees trained on bootstrap samples. "
                f"Training concordance is {train_c_index:.3f} and OOB estimate is {oob_score:.3f}. "
                f"The most influential variable is {top_feature_label}. Curves show how survival changes across low-, medium- and high-risk profiles."
            ) if is_en else (
                f"El Random Survival Forest ha combinado {feature_df.shape[1]} variables mediante árboles entrenados sobre muestras bootstrap. "
                f"La concordancia en entrenamiento es {train_c_index:.3f} y la estimación OOB es {oob_score:.3f}. "
                f"La variable más influyente es {top_feature_label}. Las curvas muestran cómo cambia la supervivencia para perfiles de bajo, medio y alto riesgo."
            )
        )
    else:
        interpretation = (
            (
                f"Random Survival Forest combines {feature_df.shape[1]} variables through trees trained on bootstrap samples. "
                f"Training concordance is {train_c_index:.3f}. The most influential variable is {top_feature_label}. "
                f"Curves show how survival changes across low-, medium- and high-risk profiles."
            ) if is_en else (
                f"El Random Survival Forest ha combinado {feature_df.shape[1]} variables mediante árboles entrenados sobre muestras bootstrap. "
                f"La concordancia en entrenamiento es {train_c_index:.3f}. La variable más influyente es {top_feature_label}. "
                f"Las curvas muestran cómo cambia la supervivencia para perfiles de bajo, medio y alto riesgo."
            )
        )

    return {
        "figure": fig_survival,
        "importance_figure": importance_fig,
        "summary_df": summary_df,
        "interpretation": interpretation,
        "n_observations": len(feature_df),
        "n_events": int(events.sum()),
        "n_features": int(feature_df.shape[1]),
        "train_c_index": float(train_c_index),
        "oob_score": None if oob_score is None else float(oob_score),
        "top_feature": top_feature_name,
        "top_feature_importance": float(top_feature_value),
        "top_features": top_features,
        "survival_labels": survival_labels,
        "survival_curve_summaries": survival_curve_summaries,
    }


def build_rsf_profile_analysis(df: pd.DataFrame, profile: dict, language: str = 'en'):
    """Fit RSF and build a survival curve for a single simulated profile."""

    fitted = _fit_rsf_model(df)
    if not fitted:
        return None

    rsf = fitted["model"]
    feature_df = fitted["feature_df"]
    risk_scores = fitted["risk_scores"]
    train_c_index = fitted["train_c_index"]
    oob_score = fitted["oob_score"]

    profile_features = _build_profile_feature_frame(df, profile, feature_df.columns)
    survival_curve = rsf.predict_survival_function(profile_features, return_array=True)[0]
    curve_times = np.asarray(rsf.unique_times_, dtype=float)
    curve_times_plot = [float(value) for value in curve_times.tolist()]
    survival_curve_plot = [float(value) for value in np.asarray(survival_curve, dtype=float).tolist()]
    profile_risk_score = float(rsf.predict(profile_features)[0])

    fig_profile = go.Figure()
    curve_min = float(np.min(survival_curve)) if len(survival_curve) else 0.0
    curve_max = float(np.max(survival_curve)) if len(survival_curve) else 1.0
    is_en = language == 'en'

    fig_profile.add_trace(
        go.Scatter(
            x=curve_times_plot,
            y=survival_curve_plot,
            mode="lines+markers",
            name="Simulated profile" if is_en else "Perfil simulado",
            line=dict(color="#8e44ad", width=5),
            marker=dict(color="#8e44ad", size=8),
            line_shape="linear",
            fill="tozeroy",
            fillcolor="rgba(142, 68, 173, 0.12)",
            hovertemplate=(
                "<b>Simulated profile</b><br>Time: %{x:.0f}<br>Survival: %{y:.3f}<extra></extra>"
                if is_en else
                "<b>Perfil simulado</b><br>Tiempo: %{x:.0f}<br>Supervivencia: %{y:.3f}<extra></extra>"
            ),
        )
    )
    y_min_visible = max(0.70, min(0.95, curve_min - 0.06))
    y_max_visible = min(1.02, max(1.0, curve_max + 0.02))
    if y_max_visible - y_min_visible < 0.08:
        y_min_visible = max(0.75, curve_min - 0.05)
        y_max_visible = 1.02

    fig_profile.update_layout(
        title="RSF: curve for an individual profile" if is_en else "RSF: curva para un perfil individual",
        xaxis_title="Time" if is_en else "Tiempo",
        yaxis_title="Survival probability" if is_en else "Probabilidad de supervivencia",
        xaxis=dict(range=[0, float(np.max(curve_times)) if len(curve_times) else 1.0], fixedrange=True),
        yaxis=dict(range=[y_min_visible, y_max_visible], fixedrange=True),
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=60, r=30, t=80, b=60),
        showlegend=False,
    )
    fig_profile.add_hline(y=1.0, line_dash="dot", line_color="#95a5a6", opacity=0.55)

    profile_labels = {
        "gender_F": ("Female" if int(profile.get("gender_F", 1)) == 1 else "Male") if is_en else PROFILE_CATEGORIES["gender_F"].get(int(profile.get("gender_F", 1)), "Femenino"),
        "disability_N": ("Without disability" if int(profile.get("disability_N", 1)) == 1 else "With disability") if is_en else PROFILE_CATEGORIES["disability_N"].get(int(profile.get("disability_N", 1)), "Sin discapacidad"),
        "age_band": (
            {
                "age_band_0-35": "0-35 years",
                "age_band_35-55": "35-55 years",
                "age_band_55<=": "55+ years",
            }.get(profile.get("age_band", "age_band_0-35"), "0-35 years")
            if is_en else PROFILE_CATEGORIES["age_band"].get(profile.get("age_band", "age_band_0-35"), "0-35 años")
        ),
        "highest_education": (
            {
                "highest_education_A Level or Equivalent": "A Level or equivalent",
                "highest_education_HE Qualification": "HE Qualification",
                "highest_education_Lower Than A Level": "Lower than A Level",
                "highest_education_Post Graduate Qualification": "Postgraduate",
            }.get(profile.get("highest_education", "highest_education_A Level or Equivalent"), "A Level or equivalent")
            if is_en else PROFILE_CATEGORIES["highest_education"].get(profile.get("highest_education", "highest_education_A Level or Equivalent"), "A Level o equivalente")
        ),
        "studied_credits": f"{float(profile.get('studied_credits', PROFILE_CREDIT_LEVELS['few'])):.0f}",
    }

    profile_summary = pd.DataFrame([
        {"Metrica": "Gender" if is_en else "Género", "Valor": profile_labels["gender_F"]},
        {"Metrica": "Disability" if is_en else "Discapacidad", "Valor": profile_labels["disability_N"]},
        {"Metrica": "Age" if is_en else "Edad", "Valor": profile_labels["age_band"]},
        {"Metrica": "Education level" if is_en else "Nivel educativo", "Valor": profile_labels["highest_education"]},
        {"Metrica": "Studied credits" if is_en else "Créditos estudiados", "Valor": profile_labels["studied_credits"]},
        {"Metrica": "Risk score" if is_en else "Score de riesgo", "Valor": f"{profile_risk_score:.3f}"},
        {"Metrica": "Train concordance" if is_en else "Concordancia train", "Valor": f"{train_c_index:.3f}"},
        {"Metrica": "OOB concordance" if is_en else "Concordancia OOB", "Valor": f"{oob_score:.3f}" if oob_score is not None else "N/A"},
    ])

    interpretation = (
        (
            f"This individual profile is simulated with RSF using {profile_labels['gender_F']} gender, "
            f"{profile_labels['disability_N'].lower()}, age {profile_labels['age_band']} and {profile_labels['studied_credits']} credits. "
            f"The curve shows estimated survival for this specific case, which is the most faithful way to represent RSF outputs."
        ) if is_en else (
            f"Este perfil individual se simula con RSF usando género {profile_labels['gender_F']}, "
            f"{profile_labels['disability_N'].lower()}, edad {profile_labels['age_band']} y {profile_labels['studied_credits']} créditos. "
            f"La curva muestra la supervivencia estimada para ese caso concreto, que es la forma más fiel de representar RSF."
        )
    )

    risk_percentile = float((np.mean(risk_scores <= profile_risk_score) * 100)) if len(risk_scores) else 0.0
    final_survival = float(survival_curve_plot[-1]) if survival_curve_plot else None

    return {
        "figure": fig_profile,
        "summary_df": profile_summary,
        "interpretation": interpretation,
        "risk_score": profile_risk_score,
        "risk_percentile": risk_percentile,
        "final_survival": final_survival,
        "min_survival": float(curve_min),
        "max_survival": float(curve_max),
        "end_time": float(curve_times_plot[-1]) if curve_times_plot else None,
    }
