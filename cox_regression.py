"""Cox regression model and associated visualization.

Prepare covariates, fit the model with lifelines, and create
the tables and forest plots displayed in the dashboard.
"""

from lifelines import CoxPHFitter
import numpy as np
import pandas as pd
from dash import dash_table
from translations import get_display_text
import plotly.graph_objects as go

# Map interface names to actual dataset columns.
# Expand one-hot categorical variables into multiple columns.
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

def run_cox_regression(df_limpio, covariables):
    """
    Run Cox regression with the selected covariates.
    Handle both simple and composite parameters (age_band, highest_education).
    Remove a reference column for each one-hot encoded categorical variable.

    Args:
        df_limpio: DataFrame containing the data.
        covariables: Covariate names, e.g. ['gender_F', 'age_band', 'studied_credits'].

    Returns:
        summary: DataFrame containing the model summary.
        cox_table_html: HTML table for display.

    Raises:
        ValueError: If the data is invalid.
    """
    
    try:
        # VALIDATION 0: Check basic inputs.
        if df_limpio is None or len(df_limpio) == 0:
            raise ValueError("❌ Empty DataFrame - No data for Cox regression")
        
        if not covariables or len(covariables) == 0:
            raise ValueError("❌ No covariates selected")
        
        # VALIDATION 1: Check critical columns.
        columnas_criticas = ['date', 'final_result']
        faltantes = [col for col in columnas_criticas if col not in df_limpio.columns]
        if faltantes:
            raise ValueError(f"""❌ Missing critical columns: {', '.join(faltantes)}""")
        
        # Suppress lifelines convergence warnings.
        import warnings
        warnings.filterwarnings('ignore', category=Warning)
        
        # Expand composite parameters into their actual columns.
        columnas_finales = ['date', 'final_result']
        for cov in covariables:
            if cov in COVARIABLE_MAPPING:
                columnas_finales.extend(COVARIABLE_MAPPING[cov])
            else:
                columnas_finales.append(cov)
        
        # Remove duplicates while preserving order.
        columnas_finales = list(dict.fromkeys(columnas_finales))
        
        # Check that all columns exist in the DataFrame.
        columnas_validas = [col for col in columnas_finales if col in df_limpio.columns]
        
        print(f"""[DEBUG Cox] Requested covariates: {covariables}""")
        print(f"""[DEBUG Cox] Valid columns found: {columnas_validas}""")
        
        # Prepare the DataFrame for Cox regression.
        df_cox = df_limpio[columnas_validas].copy()
        
        # Remove rows with missing values.
        df_cox = df_cox.dropna()
        
        if df_cox.empty:
            print("[DEBUG Cox] Empty DataFrame after dropna()")
            return pd.DataFrame(), None
        
        # Remove the last one-hot categorical column to avoid perfect collinearity.
        cov_cols = [col for col in columnas_validas if col not in ['date', 'final_result']]
        
        # Identify one-hot groups. Cox cannot include all
        # categories at once because this would cause perfect collinearity.
        columnas_por_grupo = {}
        for col in cov_cols:
            if '_' in col:
                partes = col.rsplit('_', 1)
                if len(partes) == 2:
                    prefijo = partes[0]
                    if prefijo not in columnas_por_grupo:
                        columnas_por_grupo[prefijo] = []
                    columnas_por_grupo[prefijo].append(col)
                else:
                    if 'simple' not in columnas_por_grupo:
                        columnas_por_grupo['simple'] = []
                    columnas_por_grupo['simple'].append(col)
            else:
                if 'simple' not in columnas_por_grupo:
                    columnas_por_grupo['simple'] = []
                columnas_por_grupo['simple'].append(col)
        
        # Remove the last column of each categorical group as the reference.
        cov_cols_ajustadas = []
        for grupo, cols in columnas_por_grupo.items():
            if grupo == 'simple' or len(cols) == 1:
                cov_cols_ajustadas.extend(cols)
            else:
                cols_sin_referencia = cols[:-1]
                cov_cols_ajustadas.extend(cols_sin_referencia)
                print(f"""[DEBUG Cox] Categorical variable '{grupo}': reference: {cols[-1]}""")
        
        print(f"""[DEBUG Cox] Final covariates: {cov_cols_ajustadas}""")
        
        # Prepare the final DataFrame.
        df_cox = df_cox[['date', 'final_result'] + cov_cols_ajustadas].copy()
        df_cox = df_cox.dropna()
        
        if df_cox.empty:
            print("[DEBUG Cox] Empty DataFrame")
            return pd.DataFrame(), None
        
        # Create the CoxPHFitter object and fit the model.
        cph = CoxPHFitter()
        
        # Try fitting with all variables.
        try:
            cph.fit(df_cox, duration_col='date', event_col='final_result', show_progress=False)
            print("[DEBUG Cox] Model fitted successfully")
        except Exception as e:
            print(f"""[DEBUG Cox] Initial error: {str(e)}""")
            print(f"""[DEBUG Cox] Trying regularization...""")
            
            fitted_with_fallback = False
            try:
                # Try regularization (penalization).
                cph = CoxPHFitter(penalizer=0.1)
                cph.fit(df_cox, duration_col='date', event_col='final_result', show_progress=False)
                fitted_with_fallback = True
                print("[DEBUG Cox] Model fitted with regularization")
            except Exception as e2:
                print(f"""[DEBUG Cox] Error with regularization: {str(e2)}""")
                print(f"""[DEBUG Cox] Removing problematic variables...""")
            
            # Remove known problematic variables one at a time.
            variables_problematicas = [
                'age_band_55<=',
                'disability_N',
                'highest_education_Lower Than A Level',
                'studied_credits'
            ]
            
            exito = False
            for var_problematica in ([] if fitted_with_fallback else variables_problematicas):
                if var_problematica in cov_cols_ajustadas:
                    cov_cols_temp = [c for c in cov_cols_ajustadas if c != var_problematica]
                    if not cov_cols_temp:
                        continue
                    
                    df_cox_temp = df_cox[['date', 'final_result'] + cov_cols_temp].copy()
                    df_cox_temp = df_cox_temp.dropna()
                    
                    try:
                        cph.fit(df_cox_temp, duration_col='date', event_col='final_result', show_progress=False)
                        print(f"""[DEBUG Cox] Model fitted without: {var_problematica}""")
                        df_cox = df_cox_temp
                        cov_cols_ajustadas = cov_cols_temp
                        exito = True
                        break
                    except:
                        continue
            
            if not fitted_with_fallback and not exito:
                print("[ERROR Cox] Could not fit the model with any combination of variables")
                # Return an error with additional information.
                error_summary = pd.DataFrame({
                    'Covariable': ['ERROR'],
                    'Estado': ['The Cox model did not converge. Try more variables or more data.']
                })
                return error_summary, None

        # Cox model summary.
        summary = cph.summary.copy()
        
        # ERROR #4: Check that there are enough events for the analysis.
        total_events = int(df_cox['final_result'].sum())
        min_events = 5
        if total_events < min_events:
            print(f"""[WARN Cox] Only {total_events} observed events (recommended minimum: {min_events})""")
            # Continue with a warning.
        
        # ERROR #4: Check for division by zero when calculating p-values.
        if summary.empty:
            return pd.DataFrame(), None
        
        # Safely handle very small or undefined p-values.
        summary['p'] = summary['p'].fillna(1.0)  # Replace NaN with 1.0 (not significant).
        summary['-log2(p)'] = -np.log2(summary['p'].clip(lower=1e-10))  # Avoid taking the logarithm of 0.

        # Build an HTML table from the Cox model summary.
        cols_mostrar = ['coef', 'exp(coef)', 'se(coef)', 'coef lower 95%', 'coef upper 95%',
                        'exp(coef) lower 95%', 'exp(coef) upper 95%', 'cmp to', 'z', 'p', '-log2(p)']
        summary = summary[[c for c in cols_mostrar if c in summary.columns]]
        
        summary.reset_index(inplace=True)
        summary.columns = ['Covariable', 'Coef.', 'exp(Coef.)', 'SE(Coef.)', 'Coef. lower 95%', 'Coef. upper 95%',
                           'exp(Coef.) lower 95%', 'exp(Coef.) upper 95%', 'cmp to', 'z', 'p', '-log2(p)']

        # Create the DataTable.
        cox_table_html = dash_table.DataTable(
            id='cox-summary-table',
            columns=[{"name": get_display_text("en", col), "id": col} for col in summary.columns],
            data=summary.to_dict('records'),
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
            style_cell_conditional=[
                {'if': {'column_id': 'Covariable'}, 'textAlign': 'center'}
            ]
        )

        # Return both the summary and the HTML table.
        return summary, cox_table_html
    
    except Exception as e:
        print(f"[ERROR Cox] {str(e)}")
        return pd.DataFrame(), None


def create_forest_plot(summary_data):
    """
    Create a forest plot to visualize Cox regression coefficients.

    Args:
        summary_data: DataFrame containing the Cox model summary.
            Expected columns: 'Covariable', 'exp(Coef.)',
            'Coef. lower 95%', and 'Coef. upper 95%'.

    Returns:
        go.Figure: Plotly forest plot, or None on error.
    """
    try:
        if summary_data is None or summary_data.empty:
            print("[FOREST] No data to create a forest plot")
            return None
        
        # Check the required columns.
        cols_requeridas = ['Covariable', 'exp(Coef.)', 'Coef. lower 95%', 'Coef. upper 95%']
        if not all(col in summary_data.columns for col in cols_requeridas):
            print(f"""[FOREST] Missing columns. Available: {list(summary_data.columns)}""")
            return None
        
        # Prepare data and use HR with consistent intervals.
        summary_data = summary_data.copy()
        summary_data['HR'] = pd.to_numeric(summary_data['exp(Coef.)'], errors='coerce')
        summary_data['CI_lower'] = pd.to_numeric(summary_data['exp(Coef.) lower 95%'], errors='coerce') if 'exp(Coef.) lower 95%' in summary_data.columns else np.exp(pd.to_numeric(summary_data['Coef. lower 95%'], errors='coerce'))
        summary_data['CI_upper'] = pd.to_numeric(summary_data['exp(Coef.) upper 95%'], errors='coerce') if 'exp(Coef.) upper 95%' in summary_data.columns else np.exp(pd.to_numeric(summary_data['Coef. upper 95%'], errors='coerce'))
        summary_data = summary_data.dropna(subset=['HR', 'CI_lower', 'CI_upper'])

        if summary_data.empty:
            print("[FOREST] No valid rows remain after cleaning the data")
            return None

        summary_data = summary_data.iloc[::-1].reset_index(drop=True)
        summary_data['label'] = summary_data['Covariable'].astype(str)
        summary_data['y_pos'] = list(range(len(summary_data)))
        summary_data['log_hr'] = np.log(summary_data['HR'].clip(lower=1e-10))
        summary_data['err_minus'] = summary_data['HR'] - summary_data['CI_lower']
        summary_data['err_plus'] = summary_data['CI_upper'] - summary_data['HR']

        fig = go.Figure()
        fig.add_vline(x=1, line_dash="dash", line_color="red", annotation_text="No Effect", annotation_position="top")

        fig.add_trace(go.Scatter(
            x=summary_data['HR'],
            y=summary_data['y_pos'],
            mode='markers',
            marker=dict(size=10, color='darkblue', symbol='circle'),
            error_x=dict(
                type='data',
                symmetric=False,
                array=summary_data['err_plus'],
                arrayminus=summary_data['err_minus'],
                color='steelblue',
                thickness=2,
                width=0
            ),
            text=[
                f"HR: {hr:.2f} (IC 95%: {ci_l:.2f}-{ci_u:.2f})"
                for hr, ci_l, ci_u in zip(summary_data['HR'], summary_data['CI_lower'], summary_data['CI_upper'])
            ],
            hovertemplate='<b>%{text}</b><extra></extra>',
            showlegend=False
        ))

        fig.update_layout(
            title="Forest Plot - Cox Model Coefficients",
            xaxis_title="Hazard Ratio",
            yaxis_title="Variables",
            height=max(420, 120 + len(summary_data) * 55),
            hovermode='closest',
            plot_bgcolor='rgba(240,240,240,0.5)',
            margin=dict(l=220, r=80, t=80, b=60),
            yaxis=dict(
                tickmode='array',
                tickvals=summary_data['y_pos'],
                ticktext=summary_data['label'],
                autorange='reversed'
            ),
            xaxis=dict(type='log')
        )
        
        print(f"""[FOREST] Forest plot created with {len(summary_data)} variables""")
        return fig
        
    except Exception as e:
        print(f"""[FOREST] Error creating forest plot: {e}""")
        import traceback
        traceback.print_exc()
        return None
