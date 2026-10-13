"""Log-Rank test and comparative survival curves.

Compare groups for binary variables, one-hot categorical variables, and
continuous variables divided into groups, returning tables and figures.
"""

from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter
from translations import localize_figure
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Map interface covariate names to actual columns.
# Variables with multiple categories are one-hot encoded in the CSV.
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

def perform_log_rank_test(df, group_by, alpha: float = 0.05):
    """
    Run the Log-Rank test to compare survival curves.
    Handle binary and categorical variables.
    Group continuous variables (studied_credits) into quintiles.

    Raises:
        ValueError: If the data is invalid.
    """
    
    # VALIDATION 0: Check basic inputs.
    if df is None or len(df) == 0:
        raise ValueError("❌ Empty DataFrame - No data for the Log-Rank test")
    
    if not group_by:
        raise ValueError("❌ No covariate selected")
    
    # VALIDATION 1: Check critical columns.
    columnas_criticas = ['date', 'final_result']
    faltantes = [col for col in columnas_criticas if col not in df.columns]
    if faltantes:
        raise ValueError(f"""❌ Missing critical columns: {', '.join(faltantes)}""")
    
    print(f"""[LOGRANK] Starting perform_log_rank_test for: {group_by}""")
    print(f"""[LOGRANK] Available columns: {list(df.columns)}""")
    
    # Get the columns associated with the covariate.
    columns = COVARIABLE_MAPPING.get(group_by, [group_by])
    print(f"""[LOGRANK] Mapped columns for '{group_by}': {columns}""")
    
    # Variables with multiple one-hot columns (age_band, highest_education).
    if len(columns) > 1:
        print(f"""[LOGRANK] Multiple-group categorical variable""")
        return _perform_log_rank_multicategory(df, group_by, columns, alpha)
    
    col = columns[0]
    print(f"""[LOGRANK] Column to process: {col}""")
    
    # Group studied_credits, a continuous variable, into quintiles.
    if group_by == 'studied_credits':
        print(f"""[LOGRANK] Continuous variable detected: studied_credits - create quintiles""")
        return _perform_log_rank_quintiles(df, group_by, col, alpha)
    
    # Binary or numeric variables.
    vals = sorted(pd.unique(df[col].dropna()))
    
    if len(vals) != 2:
        # Continuous variables or variables with more than 2 values.
        if len(vals) > 2:
            return _perform_log_rank_multicategory(df, group_by, columns, alpha)
        else:
            return pd.DataFrame({
                'Covariable': [group_by],
                'Grupo A': [vals[0] if len(vals) > 0 else None],
                'Grupo B': [vals[1] if len(vals) > 1 else None],
                'n A': [np.nan], 'n B': [np.nan],
                'test_statistic': [np.nan], 'p_value': [np.nan], '-log2(p)': [np.nan],
                'Decisión': ['No evaluable'],
                'Conclusión': ['La covariable no tiene exactamente dos grupos.']
            })
    
    # Log-Rank test for binary variables.
    a, b = vals[0], vals[1]
    gA = df[df[col] == a]
    gB = df[df[col] == b]

    res = logrank_test(
        gA['date'], gB['date'],
        event_observed_A=gA['final_result'],
        event_observed_B=gB['final_result']
    )

    p = float(res.p_value)
    stat = float(res.test_statistic)
    decision = 'Rechazar H0' if p < alpha else 'No rechazar H0'
    interpretacion = ('Hay diferencias significativas entre las curvas'
                      if p < alpha else
                      'No se observan diferencias significativas entre las curvas')

    return pd.DataFrame({
        'Covariable': [group_by],
        'Grupo A': [a],
        'Grupo B': [b],
        'n A': [int(len(gA))], 'n B': [int(len(gB))],
        'test_statistic': [stat], 'p_value': [p],
        '-log2(p)': [(-np.log2(p) if p > 0 else np.inf)],
        'Decisión': [decision],
        'Conclusión': [interpretacion],
    })


def _perform_log_rank_quintiles(df, group_by, col, alpha: float = 0.05):
    """
    Group a continuous variable into quintiles, or by unique values when few exist.
    """
    print(f"""\n[LOGRANK QUINTILES] Starting for {group_by}, column: {col}""")
    
    results = []
    
    # Create quintiles.
    df_temp = df[[col, 'date', 'final_result']].copy()
    df_temp = df_temp.dropna()
    
    print(f"""[LOGRANK QUINTILES] Rows after dropna: {len(df_temp)}""")
    
    if len(df_temp) == 0:
        print(f"""[LOGRANK QUINTILES] ERROR: Empty DataFrame after dropna""")
        return pd.DataFrame()
    
    # Check variability in the data.
    n_unique = df_temp[col].nunique()
    print(f"""[LOGRANK QUINTILES] Unique values in {col}: {n_unique}""")
    
    if n_unique < 2:
        print(f"""[LOGRANK QUINTILES] ERROR: The variable has insufficient variability""")
        return pd.DataFrame()
    
    # STRATEGY: Use unique values directly as groups when there are few (<=5).
    # Use qcut() with percentiles when there are many values.
    if n_unique <= 5:
        print(f"""[LOGRANK QUINTILES] Strategy: Groups by unique values ({n_unique} groups)""")
        # Assign each unique value to a group.
        unique_vals = sorted(df_temp[col].unique())
        value_to_group = {val: i for i, val in enumerate(unique_vals)}
        df_temp['quintil'] = df_temp[col].map(value_to_group)
        actual_bins = len(df_temp['quintil'].unique())
        print(f"""[LOGRANK QUINTILES] Created {actual_bins} groups by unique values""")
    else:
        print(f"""[LOGRANK QUINTILES] Strategy: Percentiles (qcut)""")
        # Try creating quintiles, reducing their number if necessary.
        for n_bins in [5, 4, 3, 2]:
            try:
                df_temp['quintil'] = pd.qcut(df_temp[col], q=n_bins, labels=False, duplicates='drop')
                actual_bins = len(df_temp['quintil'].unique())
                print(f"""[LOGRANK QUINTILES] Created {actual_bins} groups (requested {n_bins})""")
                break
            except Exception as e:
                if n_bins == 2:
                    print(f"""[LOGRANK QUINTILES] ERROR: Cannot create even 2 groups""")
                    return pd.DataFrame()
                continue
    
    # Get unique quintiles and sort them correctly.
    quintiles = sorted(df_temp['quintil'].unique())
    
    print(f"""[LOGRANK QUINTILES] Quintiles found: {quintiles}""")
    print(f"""[LOGRANK QUINTILES] Number of bins: {len(quintiles)}""")
    
    # No comparisons are possible if only 1 bin was generated.
    if len(quintiles) <= 1:
        print(f"""[LOGRANK QUINTILES] ERROR: Only 1 group was created""")
        return pd.DataFrame()
    
    # Run pairwise comparisons between quintiles.
    for i in range(len(quintiles)):
        for j in range(i + 1, len(quintiles)):
            q_a = quintiles[i]
            q_b = quintiles[j]
            
            gA = df_temp[df_temp['quintil'] == q_a]
            gB = df_temp[df_temp['quintil'] == q_b]
            
            print(f"""[LOGRANK QUINTILES] Comparing Quintile {q_a+1} (n={len(gA)}) vs Quintile {q_b+1} (n={len(gB)})""")
            
            if len(gA) > 0 and len(gB) > 0:
                try:
                    res = logrank_test(
                        gA['date'], gB['date'],
                        event_observed_A=gA['final_result'],
                        event_observed_B=gB['final_result']
                    )
                    
                    p = float(res.p_value)
                    stat = float(res.test_statistic)
                    decision = 'Rechazar H0' if p < alpha else 'No rechazar H0'
                    interpretacion = ('Hay diferencias significativas' if p < alpha 
                                    else 'No hay diferencias significativas')
                    
                    results.append({
                        'Covariable': f"{group_by}: Quintil {q_a+1} vs Quintil {q_b+1}",
                        'Grupo A': f"Quintil {q_a+1}",
                        'Grupo B': f"Quintil {q_b+1}",
                        'n A': int(len(gA)),
                        'n B': int(len(gB)),
                        'test_statistic': stat,
                        'p_value': p,
                        '-log2(p)': (-np.log2(p) if p > 0 else np.inf),
                        'Decisión': decision,
                        'Conclusión': interpretacion,
                    })
                    print(f"""[LOGRANK QUINTILES] Test completed for Quintile {q_a+1} vs Quintile {q_b+1}""")
                except Exception as e:
                    print(f"""[LOGRANK QUINTILES] ERROR in test for Quintile {q_a+1} vs Quintile {q_b+1}: {str(e)}""")
    
    result_df = pd.DataFrame(results)
    print(f"""[LOGRANK QUINTILES] Total results: {len(result_df)}""")
    return result_df


def _perform_log_rank_multicategory(df, group_by, columns, alpha: float = 0.05):
    """
    Run pairwise Log-Rank comparisons for categorical variables with >2 groups.
    Show ALL groups, even if they are empty.
    """
    results = []
    
    # Find ALL mapped groups, even if absent from the current dataframe.
    groups_info = []
    for col in columns:
        # Extract the label from the column name.
        label = col.split('_', 1)[1] if '_' in col else col
        
        # Get data if the column exists.
        if col in df.columns:
            df_group = df[df[col] == 1]
        else:
            # Create an empty dataframe if the column does not exist.
            df_group = pd.DataFrame()
        
        groups_info.append({
            'column': col,
            'label': label,
            'data': df_group,
            'n': len(df_group)
        })
    
    print(f"""[LOGRANK MULTICATEGORY] Mapped groups for {group_by}: {len(groups_info)}""")
    for info in groups_info:
        print(f"""  - {info['label']}: {info['n']} observations""")
    
    # Run pairwise comparisons between groups.
    for i in range(len(groups_info)):
        for j in range(i + 1, len(groups_info)):
            gA = groups_info[i]['data']
            gB = groups_info[j]['data']
            label_a = groups_info[i]['label']
            label_b = groups_info[j]['label']
            n_a = len(gA)
            n_b = len(gB)
            
            if n_a > 0 and n_b > 0:
                try:
                    res = logrank_test(
                        gA['date'], gB['date'],
                        event_observed_A=gA['final_result'],
                        event_observed_B=gB['final_result']
                    )
                    
                    p = float(res.p_value)
                    stat = float(res.test_statistic)
                    decision = 'Rechazar H0' if p < alpha else 'No rechazar H0'
                    interpretacion = ('Hay diferencias significativas' if p < alpha 
                                    else 'No hay diferencias significativas')
                    
                    results.append({
                        'Covariable': f"{group_by}: {label_a} vs {label_b}",
                        'Grupo A': label_a,
                        'Grupo B': label_b,
                        'n A': int(n_a),
                        'n B': int(n_b),
                        'test_statistic': stat,
                        'p_value': p,
                        '-log2(p)': (-np.log2(p) if p > 0 else np.inf),
                        'Decisión': decision,
                        'Conclusión': interpretacion,
                    })
                except Exception as e:
                    print(f"""[LOGRANK MULTICATEGORY] Error in comparison: {str(e)}""")
            else:
                # Record the comparison as not evaluable if at least one group is empty.
                grupo_vacio = label_a if n_a == 0 else label_b
                results.append({
                    'Covariable': f"{group_by}: {label_a} vs {label_b}",
                    'Grupo A': label_a,
                    'Grupo B': label_b,
                    'n A': int(n_a),
                    'n B': int(n_b),
                    'test_statistic': np.nan,
                    'p_value': np.nan,
                    '-log2(p)': np.nan,
                    'Decisión': 'No evaluable',
                    'Conclusión': f'Grupo vacío: {grupo_vacio} (n=0)',
                })
    
    return pd.DataFrame(results)


def create_logrank_figure(df, group_by, language='en'):
    """
    Create a Plotly chart with Kaplan-Meier curves for Log-Rank comparison.

    Args:
        df: DataFrame with 'date', 'final_result', and covariate columns.
        group_by: Covariate name used for grouping.

    Returns:
        go.Figure: Plotly figure containing survival curves.
    """
    try:
        if df is None or df.empty:
            print("[LOGRANK FIGURE] Error: Empty DataFrame")
            return None
        
        # Get the columns associated with the covariate.
        columns = COVARIABLE_MAPPING.get(group_by, [group_by])
        columns_presentes = [col for col in columns if col in df.columns]

        # Prepare data.
        df_clean = df[['date', 'final_result'] + columns_presentes].dropna(subset=['date', 'final_result']) if columns_presentes else df[['date', 'final_result']].dropna(subset=['date', 'final_result'])

        # Create the figure.
        fig = go.Figure()
        kmf = KaplanMeierFitter()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        # Continuous variable: studied_credits.
        if group_by == 'studied_credits' and 'studied_credits' in df.columns:
            df_temp = df[['studied_credits', 'date', 'final_result']].dropna()
            n_unique = df_temp['studied_credits'].nunique()

            if n_unique <= 5:
                unique_vals = sorted(df_temp['studied_credits'].unique())
                value_to_group = {val: i for i, val in enumerate(unique_vals)}
                df_temp['group'] = df_temp['studied_credits'].map(value_to_group)
                groups = sorted(df_temp['group'].unique())
                labels = [f"Grupo {i+1} ({unique_vals[i]:.0f} créditos)" for i in groups]
            else:
                df_temp['group'] = pd.qcut(df_temp['studied_credits'], q=5, labels=False, duplicates='drop')
                groups = sorted(df_temp['group'].unique())
                labels = [f"Quintil {i+1}" for i in groups]

            for i, group_val in enumerate(groups):
                group_df = df_temp[df_temp['group'] == group_val]
                if len(group_df) > 0:
                    kmf.fit(group_df['date'], event_observed=group_df['final_result'], label=labels[i])
                    fig.add_trace(go.Scatter(
                        x=kmf.timeline.tolist(),
                        y=kmf.survival_function_.iloc[:, 0].tolist(),
                        mode='lines',
                        name=labels[i],
                        line=dict(color=colors[i % len(colors)], width=2),
                        hovertemplate='<b>%{fullData.name}</b><br>Tiempo: %{x:.0f}<br>Supervivencia: %{y:.3f}<extra></extra>'
                    ))

        # One-hot variables with multiple categories (age_band, highest_education).
        elif len(columns_presentes) > 1:
            for i, col in enumerate(columns_presentes):
                label = col.split('_', 1)[1] if '_' in col else col
                group_df = df[df[col] == 1]
                if len(group_df) > 0:
                    kmf.fit(group_df['date'], event_observed=group_df['final_result'], label=f"{label} (n={len(group_df)})")
                    fig.add_trace(go.Scatter(
                        x=kmf.timeline.tolist(),
                        y=kmf.survival_function_.iloc[:, 0].tolist(),
                        mode='lines',
                        name=f"{label} (n={len(group_df)})",
                        line=dict(color=colors[i % len(colors)], width=2),
                        hovertemplate='<b>%{fullData.name}</b><br>Tiempo: %{x:.0f}<br>Supervivencia: %{y:.3f}<extra></extra>'
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=[],
                        y=[],
                        mode='lines',
                        name=f"{label} (n=0)",
                        line=dict(color='lightgray', width=2, dash='dash'),
                        showlegend=True
                    ))

        # Simple binary variables.
        elif len(columns_presentes) == 1:
            col = columns_presentes[0]
            groups = sorted(df_clean[col].dropna().unique()) if col in df_clean.columns else []

            for i, group_val in enumerate(groups):
                group_df = df_clean[df_clean[col] == group_val]
                if len(group_df) > 0:
                    kmf.fit(group_df['date'], event_observed=group_df['final_result'], label=f"{group_by}={group_val}")
                    fig.add_trace(go.Scatter(
                        x=kmf.timeline.tolist(),
                        y=kmf.survival_function_.iloc[:, 0].tolist(),
                        mode='lines',
                        name=f"{group_val} (n={len(group_df)})",
                        line=dict(color=colors[i % len(colors)], width=2),
                        hovertemplate='<b>%{fullData.name}</b><br>Tiempo: %{x:.0f}<br>Supervivencia: %{y:.3f}<extra></extra>'
                    ))

        # Configure the layout.
        fig.update_layout(
            title=f"Curvas Kaplan-Meier - {group_by}",
            xaxis_title="Tiempo (meses)",
            yaxis_title="Función de Supervivencia",
            hovermode='x unified',
            plot_bgcolor='rgba(240,240,240,0.5)',
            width=1200,
            height=600,
            font=dict(size=12),
            yaxis=dict(range=[0, 1])
        )
        
        print(f"""[LOGRANK FIGURE] Chart created for {group_by}""")
        return localize_figure(fig, language)
        
    except Exception as e:
        print(f"""[LOGRANK FIGURE] Error creating chart: {e}""")
        import traceback
        traceback.print_exc()
        return None
