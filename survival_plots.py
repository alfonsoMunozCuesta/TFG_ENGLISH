"""
Functions for creating survival analysis charts.
Include charts for the Log-Rank test and Cox regression.
"""

import plotly.graph_objs as go
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from translations import get_display_text
from dash import dcc, html
from translations import get_covariate_label, get_feature_label, get_group_label, get_plotly_config

# Map parameters to actual columns.
COVARIABLE_MAPPING = {
    'gender_F': ['gender_F'],
    'disability_N': ['disability_N'],
    'age_band': ['age_band_0-35', 'age_band_35-55', 'age_band_55<='],
    'highest_education': ['highest_education_A Level or Equivalent', 
                         'highest_education_HE Qualification',
                         'highest_education_Lower Than A Level', 
                         'highest_education_Post Graduate Qualification'],
    'studied_credits': ['studied_credits']
}

def plot_logrank_curves(df, covariable, language='en'):
    """
    Create Kaplan-Meier survival curves stratified by a covariate.

    Args:
        df: DataFrame containing the data.
        covariable: Covariate name used for stratification.

    Returns:
        dcc.Graph containing the stratified Kaplan-Meier chart.
    """
    
    is_en = language == 'en'
    columns = COVARIABLE_MAPPING.get(covariable, [covariable])
    
    fig = go.Figure()
    kmf = KaplanMeierFitter()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Continuous variable (studied_credits).
    if covariable == 'studied_credits':
        df_temp = df[['studied_credits', 'date', 'final_result']].dropna()
        
        # Create quintiles.
        n_unique = df_temp['studied_credits'].nunique()
        if n_unique <= 5:
            unique_vals = sorted(df_temp['studied_credits'].unique())
            value_to_group = {val: i for i, val in enumerate(unique_vals)}
            df_temp['group'] = df_temp['studied_credits'].map(value_to_group)
            groups = sorted(df_temp['group'].unique())
            labels = [f"Quintile {i+1}\n({unique_vals[i]:.0f} credits)" if is_en else f"Quintil {i+1}\n({unique_vals[i]:.0f} créditos)" for i in groups]
        else:
            df_temp['group'] = pd.qcut(df_temp['studied_credits'], q=5, labels=False, duplicates='drop')
            groups = sorted(df_temp['group'].unique())
            labels = [f"Quintile {i+1}" if is_en else f"Quintil {i+1}" for i in groups]
        
        for i, group_val in enumerate(groups):
            df_group = df_temp[df_temp['group'] == group_val]
            if len(df_group) > 0:
                kmf.fit(df_group['date'], event_observed=df_group['final_result'], label=labels[i])
                
                fig.add_trace(go.Scatter(
                    x=kmf.timeline.tolist(),
                    y=kmf.survival_function_.iloc[:, 0].tolist(),
                    mode='lines',
                    name=labels[i],
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>Time: %{x:.0f}<br>Survival: %{y:.3f}<extra></extra>"
                        if is_en else
                        "<b>%{fullData.name}</b><br>Tiempo: %{x:.0f}<br>Supervivencia: %{y:.3f}<extra></extra>"
                    )
                ))
                
                # Add confidence intervals.
                ci = kmf.confidence_interval_survival_function_
                fig.add_trace(go.Scatter(
                    x=kmf.timeline.tolist(),
                    y=ci.iloc[:, 1].tolist(),
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False,
                    name=f"{labels[i]} ({'Upper CI' if is_en else 'IC superior'})"
                ))
                fig.add_trace(go.Scatter(
                    x=kmf.timeline.tolist(),
                    y=ci.iloc[:, 0].tolist(),
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False,
                    name=f"{labels[i]} ({'Lower CI' if is_en else 'IC inferior'})",
                    fillcolor=f"rgba({int(colors[i % len(colors)][1:3], 16)}, {int(colors[i % len(colors)][3:5], 16)}, {int(colors[i % len(colors)][5:7], 16)}, 0.2)"
                ))
    
    # Categorical variables with multiple classes (age_band, highest_education).
    elif len(columns) > 1:
        # Iterate over ALL mapped columns, even if absent from the current dataframe.
        groups_info = []
        for col in columns:
            # Resolve the group label in the selected language.
            label = get_covariate_label(language, col)
            
            # Get data if the column exists; otherwise use an empty dataframe.
            if col in df.columns:
                df_group = df[df[col] == 1]
            else:
                df_group = pd.DataFrame()  # The column does not exist in the data.
            
            groups_info.append({
                'col': col, 
                'label': label, 
                'data': df_group, 
                'n': len(df_group)
            })
        
        print(f"""[LOGRANK CURVES] Mapped groups: {[info["label"] + f" (n={info["n"]})" for info in groups_info]}""")
        
        for i, info in enumerate(groups_info):
            if len(info['data']) > 0:
                kmf.fit(info['data']['date'], event_observed=info['data']['final_result'], 
                        label=f"{info['label']} (n={info['n']})")
                
                fig.add_trace(go.Scatter(
                    x=kmf.timeline.tolist(),
                    y=kmf.survival_function_.iloc[:, 0].tolist(),
                    mode='lines',
                    name=f"{info['label']} (n={info['n']})",
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>Time: %{x:.0f}<br>Survival: %{y:.3f}<extra></extra>"
                        if is_en else
                        "<b>%{fullData.name}</b><br>Tiempo: %{x:.0f}<br>Supervivencia: %{y:.3f}<extra></extra>"
                    )
                ))
                
                # Confidence intervals.
                ci = kmf.confidence_interval_survival_function_
                fig.add_trace(go.Scatter(
                    x=kmf.timeline.tolist(),
                    y=ci.iloc[:, 1].tolist(),
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False,
                    name=f"{info['label']} ({'Upper CI' if is_en else 'IC superior'})"
                ))
                fig.add_trace(go.Scatter(
                    x=kmf.timeline.tolist(),
                    y=ci.iloc[:, 0].tolist(),
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False,
                    name=f"{info['label']} ({'Lower CI' if is_en else 'IC inferior'})",
                    fillcolor=f"rgba({int(colors[i % len(colors)][1:3], 16)}, {int(colors[i % len(colors)][3:5], 16)}, {int(colors[i % len(colors)][5:7], 16)}, 0.2)"
                ))
            else:
                # Show groups without data as dashed gray lines with n=0.
                fig.add_trace(go.Scatter(
                    x=[],
                    y=[],
                    mode='lines+markers',
                    name=f"{info['label']} (n=0 - {'No data' if is_en else 'Sin datos'})",
                    line=dict(color='lightgray', width=2, dash='dash'),
                    marker=dict(symbol='x'),
                    hovertemplate=f"<b>{info['label']}</b><br>{'No observations in this group' if is_en else 'Sin observaciones en este grupo'}<extra></extra>"
                ))
    
    # Simple binary variables (gender_F, disability_N).
    else:
        col = columns[0]
        groups = sorted(df[col].dropna().unique())
        labels = [get_group_label(language, covariable, value) for value in groups]
        
        for i, group_val in enumerate(groups):
            df_group = df[df[col] == group_val]
            if len(df_group) > 0:
                kmf.fit(df_group['date'], event_observed=df_group['final_result'], label=labels[i])
                
                fig.add_trace(go.Scatter(
                    x=kmf.timeline.tolist(),
                    y=kmf.survival_function_.iloc[:, 0].tolist(),
                    mode='lines',
                    name=labels[i],
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>Time: %{x:.0f}<br>Survival: %{y:.3f}<extra></extra>"
                        if is_en else
                        "<b>%{fullData.name}</b><br>Tiempo: %{x:.0f}<br>Supervivencia: %{y:.3f}<extra></extra>"
                    )
                ))
                
                # Confidence intervals.
                ci = kmf.confidence_interval_survival_function_
                fig.add_trace(go.Scatter(
                    x=kmf.timeline.tolist(),
                    y=ci.iloc[:, 1].tolist(),
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False,
                    name=f"{labels[i]} ({'Upper CI' if is_en else 'IC superior'})"
                ))
                fig.add_trace(go.Scatter(
                    x=kmf.timeline.tolist(),
                    y=ci.iloc[:, 0].tolist(),
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False,
                    name=f"{labels[i]} ({'Lower CI' if is_en else 'IC inferior'})",
                    fillcolor=f"rgba({int(colors[i % len(colors)][1:3], 16)}, {int(colors[i % len(colors)][3:5], 16)}, {int(colors[i % len(colors)][5:7], 16)}, 0.2)"
                ))
    
    fig.update_layout(
        title=(
            f"Survival Curves (Kaplan-Meier) - {get_covariate_label(language, covariable)}"
            if is_en else
            f"Curvas de Supervivencia (Kaplan-Meier) - {get_covariate_label(language, covariable)}"
        ),
        xaxis_title="Time" if is_en else "Tiempo",
        yaxis_title="Survival Probability" if is_en else "Probabilidad de Supervivencia",
        yaxis=dict(range=[0, 1]),
        hovermode='x unified',
        template='plotly_white'
    )
    
    return dcc.Graph(figure=fig, config=get_plotly_config(language))


def plot_cox_hazard_ratios(summary_df, covariable_name, language='en'):
    """
    Create a forest plot with hazard ratios and confidence intervals.

    Args:
        summary_df: Cox regression results from cph.summary.
        covariable_name: Covariate name.

    Returns:
        dcc.Graph containing the forest plot.
    """
    
    # Ensure the correct data is available.
    summary_df = summary_df.copy()
    
    # Detect column names, which may vary across lifelines versions.
    coef_col = 'coef' if 'coef' in summary_df.columns else 'Coef.'
    se_col = 'se(coef)' if 'se(coef)' in summary_df.columns else 'SE(Coef.)'
    
    # Calculate HR (hazard ratio) and its confidence intervals.
    summary_df['HR'] = np.exp(pd.to_numeric(summary_df[coef_col], errors='coerce'))
    summary_df['se_coef'] = pd.to_numeric(summary_df[se_col], errors='coerce')
    summary_df['HR_lower'] = np.exp(pd.to_numeric(summary_df[coef_col], errors='coerce') - 1.96 * summary_df['se_coef'])
    summary_df['HR_upper'] = np.exp(pd.to_numeric(summary_df[coef_col], errors='coerce') + 1.96 * summary_df['se_coef'])
    
    # Remove rows with NaN.
    summary_df = summary_df.dropna(subset=['HR', 'HR_lower', 'HR_upper'])
    
    is_en = language == 'en'

    if summary_df.empty:
        return dcc.Graph(figure=go.Figure().add_annotation(text="Could not compute Hazard Ratios" if is_en else "No se pudieron calcular Hazard Ratios"), config=get_plotly_config(language))
    
    fig = go.Figure()
    
    # Detect the variable column name.
    var_col = 'Covariable' if 'Covariable' in summary_df.columns else summary_df.columns[0]
    
    colors = ['green' if hr < 1 else 'red' for hr in summary_df['HR']]
    y_labels = []
    
    # Add a vertical reference line at HR=1 (no effect).
    fig.add_vline(x=1, line_dash="dash", line_color="black", 
                  annotation_text="HR = 1 (No effect)" if is_en else "HR = 1 (Sin efecto)", annotation_position="top",
                  annotation_textangle=0)
    
    # Add points and error bars.
    for idx, (i, row) in enumerate(summary_df.iterrows()):
        var_name = get_feature_label(language, str(row[var_col])) if var_col in row and pd.notna(row[var_col]) else f"Var {idx}"
        y_labels.append(var_name)
        
        fig.add_trace(go.Scatter(
            x=[row['HR']],
            y=[idx],
            mode='markers',
            marker=dict(size=12, color=colors[idx]),
            name=var_name,
            hovertemplate=f"<b>{var_name}</b><br>HR: %{{x:.3f}}<br>{'95% CI' if is_en else 'IC 95%'}: ({row['HR_lower']:.3f} - {row['HR_upper']:.3f})<extra></extra>"
        ))
        
        # Error bars (horizontal lines).
        fig.add_trace(go.Scatter(
            x=[row['HR_lower'], row['HR_upper']],
            y=[idx, idx],
            mode='lines',
            line=dict(color=colors[idx], width=2),
            showlegend=False,
            hoverinfo='none'
        ))
    
    fig.update_layout(
        title="Forest Plot - Hazard Ratios (Cox Regression)" if is_en else "Forest Plot - Hazard Ratios (Regresión de Cox)",
        xaxis_title="Hazard Ratio (log scale)" if is_en else "Hazard Ratio (escala logarítmica)",
        yaxis_title="Variables" if is_en else "Variables",
        yaxis=dict(
            autorange="reversed",
            tickmode='array',
            tickvals=list(range(len(y_labels))),
            ticktext=y_labels
        ),
        xaxis=dict(type="log"),
        height=max(400, len(summary_df) * 50),
        template='plotly_white',
        hovermode='closest',
        showlegend=False
    )
    
    return dcc.Graph(figure=fig, config=get_plotly_config(language))
