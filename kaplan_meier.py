"""Functions for building Kaplan-Meier curves.

Group global and covariate-stratified empirical survival logic,
including confidence intervals and reusable Plotly figures.
"""

import pandas as pd
from dash import dcc
import plotly.graph_objs as go
from lifelines import KaplanMeierFitter
from translations import localize_figure

# Map the selected interface covariate to the actual columns.
# age_band and highest_education are one-hot encoded variables.
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


def _rgba_with_alpha(color, alpha):
    """Convert a hexadecimal color into RGBA with transparency."""
    if color.startswith('rgba(') and color.endswith(')'):
        parts = [part.strip() for part in color[5:-1].split(',')]
        if len(parts) >= 3:
            return f"rgba({parts[0]}, {parts[1]}, {parts[2]}, {alpha})"
    return color


def _add_km_trace_with_ci(fig, kmf, name, color, show_legend=True):
    """Add the Kaplan-Meier curve and its confidence interval to the figure."""
    survival_col = kmf.survival_function_.columns[0]
    timeline = kmf.timeline.tolist()
    survival_values = kmf.survival_function_[survival_col].tolist()
    ci = kmf.confidence_interval_survival_function_
    upper_values = ci.iloc[:, 1].tolist()
    lower_values = ci.iloc[:, 0].tolist()

    fig.add_trace(go.Scatter(
        x=timeline,
        y=upper_values,
        mode='lines',
        line=dict(color='rgba(0,0,0,0)', width=0),
        hoverinfo='skip',
        showlegend=False,
        name=f'{name} IC superior'
    ))

    fig.add_trace(go.Scatter(
        x=timeline,
        y=lower_values,
        mode='lines',
        line=dict(color='rgba(0,0,0,0)', width=0),
        fill='tonexty',
        fillcolor=_rgba_with_alpha(color, 0.18),
        hoverinfo='skip',
        showlegend=False,
        name=f'{name} IC inferior'
    ))

    fig.add_trace(go.Scatter(
        x=timeline,
        y=survival_values,
        mode='lines',
        name=name,
        line=dict(color=_rgba_with_alpha(color, 1.0), width=2),
        hovertemplate='(%{x:.0f}, %{y:.3f})<extra></extra>',
        showlegend=show_legend
    ))

def plot_kaplan_meier(df, language='en'):
    # ERROR #3: Validate the data structure before analysis.
    """Calculate and plot the dataset's global Kaplan-Meier curve."""
    if df is None or len(df) == 0:
        raise ValueError("❌ Empty DataFrame - No data to plot")
    
    if 'date' not in df.columns or 'final_result' not in df.columns:
        raise ValueError("❌ Missing critical columns: 'date' or 'final_result'")
    
    kmf = KaplanMeierFitter()

    kmf.fit(df['date'], event_observed=df['final_result'])

    # Create the figure with Plotly.
    fig = go.Figure()

    # Kaplan-Meier curve.
    _add_km_trace_with_ci(fig, kmf, '', 'rgba(0, 0, 255, 1.0)', show_legend=False)

    fig.update_layout(
        title="Curva de Supervivencia Kaplan-Meier",
        xaxis_title="Tiempo",
        yaxis_title="Probabilidad de Supervivencia",
        yaxis=dict(range=[0, 1]),

    )

    return dcc.Graph(figure=localize_figure(fig, language))


def plot_km_G(df, group_by='gender_F', language='en'):
    """Plot Kaplan-Meier for any covariate."""
    return plot_km_by_covariate(df, group_by, language=language)

def plot_km_disc(df, group_by='disability_N', language='en'):
    """Plot Kaplan-Meier by disability."""
    return plot_km_by_covariate(df, group_by, language=language)

def plot_km_by_covariate(df, covariable_name='gender_F', language='en'):
    """
    Plot Kaplan-Meier grouped by any covariate.
    Handle binary variables and categorical variables with multiple groups.

    Raises:
        ValueError: If critical data is missing from the dataframe.
    """
    # VALIDATION 1: Check that the dataframe is not empty.
    if df is None or len(df) == 0:
        raise ValueError("❌ Empty DataFrame - No data to plot")
    
    # VALIDATION 2: Check that critical columns exist.
    columnas_criticas = ['date', 'final_result']
    faltantes = [col for col in columnas_criticas if col not in df.columns]
    if faltantes:
        raise ValueError(f"""❌ Missing critical columns: {', '.join(faltantes)}""")
    
    # VALIDATION 3: Check that the requested covariate exists.
    columns = COVARIABLE_MAPPING.get(covariable_name, [covariable_name])
    cols_disponibles = [col for col in columns if col in df.columns]
    if not cols_disponibles:
        raise ValueError(f"""❌ Covariate '{covariable_name}' does not exist in the dataset""")
    
    kmf = KaplanMeierFitter()
    fig = go.Figure()
    
    # Get the columns associated with the covariate.
    columns = COVARIABLE_MAPPING.get(covariable_name, [covariable_name])
    
    # Map labels according to the variable type.
    label_map = {
        'gender_F': {1: "Femenino", 0: "Masculino"},
        'disability_N': {1: "Sin discapacidad", 0: "Con discapacidad"},
        'studied_credits': {}  # No mapping for the continuous variable.
    }
    label_mapping = label_map.get(covariable_name, {})
    
    colors = ['rgba(0, 123, 255, 0.3)', 'rgba(255, 165, 0, 0.3)', 'rgba(0, 200, 0, 0.3)', 
             'rgba(200, 0, 0, 0.3)', 'rgba(128, 0, 128, 0.3)', 'rgba(255, 192, 203, 0.3)']
    
    # Simple binary variables: one column, except studied_credits.
    if len(columns) == 1 and covariable_name != 'studied_credits':
        col = columns[0]
        df_filtered = df[df[col].notnull()]
        
        # Plot each group.
        groups = sorted(df_filtered[col].unique())
        for idx, group in enumerate(groups):
            df_group = df_filtered[df_filtered[col] == group]
            
            if len(df_group) > 0:
                # Get the label.
                if label_mapping:
                    group_label = label_mapping.get(group, str(group))
                else:
                    group_label = f"{col} = {group}"
                
                # Fit Kaplan-Meier.
                kmf.fit(df_group['date'], 
                       event_observed=df_group['final_result'],
                       label=group_label)

                _add_km_trace_with_ci(
                    fig,
                    kmf,
                    group_label,
                    colors[idx % len(colors)]
                )
    
    # Categorical variables with multiple values (age_band, highest_education).
    elif len(columns) > 1:
        idx = 0
        # Iterate over ALL mapped columns, even if they do not exist.
        for col in columns:
            if col.startswith('age_band_'):
                label = col.replace('age_band_', '', 1)
            elif col.startswith('highest_education_'):
                label = col.replace('highest_education_', '', 1)
            else:
                label = col
            
            # Get data if available.
            if col in df.columns:
                df_group = df[df[col] == 1].copy()
            else:
                df_group = pd.DataFrame()  # The column does not exist.
            
            if len(df_group) > 0:
                # Fit Kaplan-Meier.
                kmf.fit(df_group['date'],
                       event_observed=df_group['final_result'],
                       label=f"{label} (n={len(df_group)})")

                _add_km_trace_with_ci(
                    fig,
                    kmf,
                    f"{label} (n={len(df_group)})",
                    colors[idx % len(colors)]
                )
            else:
                # Show groups without observations as dashed gray lines in the legend.
                fig.add_trace(go.Scatter(
                    x=[],
                    y=[],
                    mode='lines+markers',
                    name=f"{label} (n=0 - Sin datos)",
                    line=dict(width=2, dash='dash', color='lightgray'),
                    marker=dict(symbol='x'),
                    hovertemplate=f"<b>{label}</b><br>Sin observaciones en este grupo<extra></extra>"
                ))
            idx += 1
    
    # studied_credits: continuous/numeric variable.
    elif covariable_name == 'studied_credits':
        col = columns[0]
        df_filtered = df[df[col].notnull()].copy()
        
        if len(df_filtered) > 0:
            # Get unique values and sort them.
            unique_values = sorted(df_filtered[col].unique())
            
            # Group into ranges if there are too many values.
            if len(unique_values) > 10:
                # Divide into quartiles or ranges.
                df_filtered['credits_group'] = pd.cut(df_filtered[col], bins=5, duplicates='drop')
                groups_col = 'credits_group'
            else:
                df_filtered['credits_group'] = df_filtered[col]
                groups_col = 'credits_group'
            
            for idx, group_val in enumerate(df_filtered[groups_col].unique()):
                df_group = df_filtered[df_filtered[groups_col] == group_val]
                
                if len(df_group) > 0:
                    group_label = f"Créditos: {group_val}"
                    
                    kmf.fit(df_group['date'],
                           event_observed=df_group['final_result'],
                           label=group_label)

                    _add_km_trace_with_ci(
                        fig,
                        kmf,
                        group_label,
                        colors[idx % len(colors)]
                    )
    
    # Configure the chart.
    fig.update_layout(
        title=f'Curva de Kaplan-Meier para {covariable_name}',
        xaxis_title='Tiempo',
        yaxis_title='Probabilidad de Supervivencia',
        yaxis=dict(range=[0, 1]),
        legend_title=covariable_name,
        template='plotly_white',
    )
    
    return dcc.Graph(figure=localize_figure(fig, language))


def plot_km_by_covariate_with_figure(df, covariable_name='gender_F', language='en'):
    """
    Return both the dcc.Graph component and the JSON figure for the Store.

    Returns:
        Tuple[dcc.Graph, dict]: (graph component, JSON figure).
    """
    # Use the existing logic to generate the figure.
    fig = _create_km_figure(df, covariable_name, language=language)
    
    # Return the graph component AND the JSON figure.
    return dcc.Graph(figure=localize_figure(fig, language)), fig.to_dict()


def _create_km_figure(df, covariable_name='gender_F', language='en'):
    """
    Create and return a Plotly go.Figure for use in multiple contexts.
    """
    # VALIDATION 1: Check that the dataframe is not empty.
    if df is None or len(df) == 0:
        raise ValueError("❌ Empty DataFrame - No data to plot")
    
    # VALIDATION 2: Check that critical columns exist.
    columnas_criticas = ['date', 'final_result']
    faltantes = [col for col in columnas_criticas if col not in df.columns]
    if faltantes:
        raise ValueError(f"""❌ Missing critical columns: {', '.join(faltantes)}""")
    
    # VALIDATION 3: Check that the requested covariate exists.
    columns = COVARIABLE_MAPPING.get(covariable_name, [covariable_name])
    cols_disponibles = [col for col in columns if col in df.columns]
    if not cols_disponibles:
        raise ValueError(f"""❌ Covariate '{covariable_name}' does not exist in the dataset""")
    
    kmf = KaplanMeierFitter()
    fig = go.Figure()
    
    # Get the columns associated with the covariate.
    columns = COVARIABLE_MAPPING.get(covariable_name, [covariable_name])
    
    # Map labels according to the variable type.
    label_map = {
        'gender_F': {1: "Femenino", 0: "Masculino"},
        'disability_N': {1: "Sin discapacidad", 0: "Con discapacidad"},
        'studied_credits': {}  # No mapping for the continuous variable.
    }
    label_mapping = label_map.get(covariable_name, {})
    
    colors = ['rgba(0, 123, 255, 0.3)', 'rgba(255, 165, 0, 0.3)', 'rgba(0, 200, 0, 0.3)', 
             'rgba(200, 0, 0, 0.3)', 'rgba(128, 0, 128, 0.3)', 'rgba(255, 192, 203, 0.3)']
    
    # Simple binary variables: one column, except studied_credits.
    if len(columns) == 1 and covariable_name != 'studied_credits':
        col = columns[0]
        df_filtered = df[df[col].notnull()]
        
        # Plot each group.
        groups = sorted(df_filtered[col].unique())
        for idx, group in enumerate(groups):
            df_group = df_filtered[df_filtered[col] == group]
            
            if len(df_group) > 0:
                # Get the label.
                if label_mapping:
                    group_label = label_mapping.get(group, str(group))
                else:
                    group_label = f"{col} = {group}"
                
                # Fit Kaplan-Meier.
                kmf.fit(df_group['date'], 
                       event_observed=df_group['final_result'],
                       label=group_label)

                _add_km_trace_with_ci(
                    fig,
                    kmf,
                    group_label,
                    colors[idx % len(colors)]
                )
    
    # Categorical variables with multiple values (age_band, highest_education).
    elif len(columns) > 1:
        idx = 0
        # Process only columns present in the dataframe.
        for col in columns:
            # Process only if the column exists.
            if col not in df.columns:
                continue
            
            if col.startswith('age_band_'):
                label = col.replace('age_band_', '', 1)
            elif col.startswith('highest_education_'):
                label = col.replace('highest_education_', '', 1)
            else:
                label = col
            
            # Get records where this column equals 1.
            df_group = df[df[col] == 1].copy()
            
            if len(df_group) > 0:
                kmf.fit(df_group['date'],
                       event_observed=df_group['final_result'],
                       label=label)

                _add_km_trace_with_ci(
                    fig,
                    kmf,
                    label,
                    colors[idx % len(colors)]
                )
                idx += 1
    
    # Continuous variables (studied_credits).
    else:
        for col in columns:
            if col in df.columns:
                df_filtered = df[[col, 'date', 'final_result']].dropna()
                
                if len(df_filtered) > 0:
                    # Divide into quartiles or ranges.
                    df_filtered = df_filtered.copy()
                    df_filtered['credits_group'] = pd.cut(df_filtered[col], bins=5, duplicates='drop')
                    groups_col = 'credits_group'
                    
                    for idx, group_val in enumerate(df_filtered[groups_col].unique()):
                        df_group = df_filtered[df_filtered[groups_col] == group_val]
                        
                        if len(df_group) > 0:
                            group_label = f"Créditos: {group_val}"
                            
                            kmf.fit(df_group['date'],
                                   event_observed=df_group['final_result'],
                                   label=group_label)

                            _add_km_trace_with_ci(
                                fig,
                                kmf,
                                group_label,
                                colors[idx % len(colors)]
                            )
    
    # Configure the chart.
    fig.update_layout(
        title=f'Curva de Kaplan-Meier para {covariable_name}',
        xaxis_title='Tiempo',
        yaxis_title='Probabilidad de Supervivencia',
        yaxis=dict(range=[0, 1]),
        legend_title=covariable_name,
        template='plotly_white',
    )
    
    return localize_figure(fig, language)
