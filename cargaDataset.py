"""Entry point for the Dash application.

Define the Dash instance, navigation bar, shared stores, and basic navigation
callbacks. The main analysis logic is registered from analysis_callbacks.py
and pdf_callbacks.py.
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
from dash import Dash, dcc, html
import dash_daq as daq
from dash.dependencies import Input, Output, State
from layout import (
    create_survival_analysis_page, create_covariate_analysis_page, 
    create_kaplan_meier_page, create_cox_regression_page, 
    create_log_rank_page, create_ver_dataset_page, 
    create_weibull_analysis_page, create_exponential_analysis_page,
    create_rsf_analysis_page,
    create_techniques_comparison_page
)
from dash import callback_context
from translations import get_translation
from pdf_callbacks import register_pdf_export_callbacks  # PDF export ENABLED
from analysis_callbacks import create_home_page, register_analysis_callbacks

# Initialize Dash. suppress_callback_exceptions allows declaring
# callbacks for components that appear only on certain pages.
app = Dash(__name__, suppress_callback_exceptions=True)
app.config['suppress_callback_exceptions'] = True


# Fixed navigation bar at the top. Labels update
# dynamically when the language changes.
navbar = html.Div([
    html.Div([ 
        html.Div([
            html.Button(get_translation('en', 'navbar_home'), id='inicio-btn', n_clicks=0, className='navbar-link', style={'border': 'none', 'background': 'none', 'cursor': 'pointer', 'fontWeight': 'bold', 'fontSize': '14px'}),
            dcc.Link(get_translation('en', 'navbar_view_dataset'), href='/ver-dataset', className='navbar-link', id='navbar-view-dataset'),
            dcc.Link(get_translation('en', 'navbar_covariate_analysis'), href='/covariate-analysis', className='navbar-link', id='navbar-covariate-analysis'),
            dcc.Link(get_translation('en', 'navbar_survival_analysis'), href='/survival-analysis', className='navbar-link', id='navbar-survival-analysis'),
        ], className='navbar-links', style={'flex': '1'}),
        # Language selector with flags and a switch in the right corner.
        html.Div([
            html.Img(
                src='/assets/spain.jpg',
                alt='Spanish',
                style={
                    'width': '28px',
                    'height': '20px',
                    'objectFit': 'cover',
                    'borderRadius': '3px',
                    'boxShadow': '0 1px 4px rgba(0,0,0,0.2)'
                }
            ),
            daq.ToggleSwitch(
                id='language-toggle',
                value=True,
                size=36,
                color='#4a4a4a',
                style={'margin': '0 10px'}
            ),
            html.Img(
                src='/assets/ingles.png',
                alt='English',
                style={
                    'width': '28px',
                    'height': '20px',
                    'objectFit': 'cover',
                    'borderRadius': '3px',
                    'boxShadow': '0 1px 4px rgba(0,0,0,0.2)'
                }
            )
        ], className='language-switch-wrap', style={'marginRight': '42px', 'display': 'flex', 'alignItems': 'center', 'gap': '2px'})
    ], className='navbar-inner', style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between', 'width': '100%'})
], id='navbar', style={'position': 'fixed', 'top': '0', 'left': '0', 'width': '100%', 'background-color': '#f4f7f6', 'padding': '10px', 'z-index': '1000'})

app.layout = html.Div([ 
    navbar, 
    dcc.Location(id='url', refresh=False),  
    # Stores hold shared callback state without writing to
    # disk: processed dataset, language, current results, and AI text.
    dcc.Store(id='language-store', data='en'),  # Store the current language.
    dcc.Store(id='df-store', data=None),  # Store the processed dataframe.
    dcc.Store(id='dataset-signature-store', data=''),  # Dataset signature used to invalidate outdated analyses.
    dcc.Store(id='km-current-variable', data=''),  # Track the current Kaplan-Meier variable.
    dcc.Store(id='cox-current-variables', data=''),  # Track Cox regression variables.
    dcc.Store(id='logrank-current-variable', data=''),  # Track the Log-Rank variable.
    dcc.Store(id='logrank-selected-covariables', data=[]),  # Store selected Log-Rank covariates.
    dcc.Store(id='cox-selected-covariables', data=[]),  # Store selected Cox covariates.
    dcc.Store(id='logrank-test-output-store', data=None),  # Store Log-Rank results.
    dcc.Store(id='cox-regression-output-store', data=None),  # Store Cox results.
    dcc.Store(id='cox-figure-store', data=None),  # Store the Cox forest plot.
    dcc.Store(id='logrank-figure-store', data=None),  # Store the Log-Rank chart.
    dcc.Store(id='weibull-ai-text-store', data=''),  # Store the Weibull AI explanation.
    dcc.Store(id='exponential-ai-text-store', data=''),  # Store the Exponential AI explanation.
    dcc.Store(id='weibull-ai-language-store', data=''),  # Language used to generate the Weibull AI explanation.
    dcc.Store(id='exponential-ai-language-store', data=''),  # Language used to generate the Exponential AI explanation.
    html.Div(id='page-content'),  
    # Confirmation dialog.
    dcc.ConfirmDialog(
        id='confirm-dialog',
        message='Are you sure you want to return to the home page? You will lose the loaded dataset and all analysis performed.',
        displayed=False,  # Initially hidden.
        submit_n_clicks=0,  # Keep the click counter.
        cancel_n_clicks=0  
    )
])
# Store the selected language.
@app.callback(
    Output('language-store', 'data'),
    Input('language-toggle', 'value')
)
def update_language(toggle_value):
    """Convert the language switch value into the code used by the application."""
    return 'en' if toggle_value else 'es'


@app.callback(
    [Output('inicio-btn', 'children'),
     Output('navbar-view-dataset', 'children'),
     Output('navbar-covariate-analysis', 'children'),
     Output('navbar-survival-analysis', 'children')],
    Input('language-store', 'data')
)
def update_navbar_labels(language):
    """Update navigation menu labels for the active language."""
    return (
        get_translation(language, 'navbar_home'),
        get_translation(language, 'navbar_view_dataset'),
        get_translation(language, 'navbar_covariate_analysis'),
        get_translation(language, 'navbar_survival_analysis'),
    )


@app.callback(
    Output('confirm-dialog', 'displayed'),
    Input('inicio-btn', 'n_clicks'),
    prevent_initial_call=True  # Prevent activation when the page loads.
)
def mostrar_confirmacion(n_clicks):
    """Open the confirmation dialog when attempting to load another dataset."""
    if n_clicks:
        return True 
    return False

def _create_dataset_locked_message(language):
    """Build the notice that blocks loading a second dataset."""
    return html.Div(
        [
            html.Div(
                [
                    html.H2("❌ " + get_translation(language, 'error_preprocess_title'), style={'marginBottom': '12px', 'color': '#b03a2e'}),
                    html.P(get_translation(language, 'error_dataset_not_loaded'), style={'fontSize': '18px', 'fontWeight': 'bold', 'marginBottom': '10px'}),
                    html.P(
                        (
                            'First click "Preprocess CSV" on the upload page to enable this section.'
                            if language == 'en' else
                            'Primero pulsa "Preprocesar CSV" desde la página de carga para habilitar esta sección.'
                        ),
                        style={'fontSize': '15px', 'color': '#555', 'marginBottom': '0'}
                    )
                ],
                style={
                    'maxWidth': '900px',
                    'margin': '90px auto 40px auto',
                    'padding': '28px',
                    'backgroundColor': '#fff5f5',
                    'border': '2px solid #f5c6cb',
                    'borderRadius': '14px',
                    'boxShadow': '0 10px 30px rgba(176, 58, 46, 0.08)',
                    'textAlign': 'center'
                }
            )
        ]
    )

@app.callback(
    Output('confirm-dialog', 'message'),
    Input('language-store', 'data')
)
def update_confirm_message(language):
    """Refresh the confirmation dialog message when the language changes."""
    return get_translation(language, 'confirmar_inicio')


# Handle navigation between pages.
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname'),
     Input('language-store', 'data')],
    [State('df-store', 'data'),
     State('km-current-variable', 'data')]
)
def display_page(pathname, language, df_json, km_current_variable):
    """Select the dashboard page to display for the current route."""
    dataset_loaded = df_json is not None

    if pathname == '/':
        return create_home_page(language)

    if pathname in ['/ver-dataset', '/covariate-analysis', '/survival-analysis', '/survival-analysis/kaplan-meier', '/survival-analysis/cox-regression', '/survival-analysis/log-rank', '/survival-analysis/weibull', '/survival-analysis/exponential', '/survival-analysis/rsf', '/survival-analysis/comparacion-tecnicas'] and not dataset_loaded:
        return _create_dataset_locked_message(language)

    if pathname == '/covariate-analysis':
        return create_covariate_analysis_page(language)
    elif pathname == '/survival-analysis':
        return create_survival_analysis_page(language)
    elif pathname == '/survival-analysis/kaplan-meier':
        return create_kaplan_meier_page(language, current_variable=km_current_variable, df_json=df_json)
    elif pathname == '/survival-analysis/cox-regression':
        return create_cox_regression_page(language)
    elif pathname == '/survival-analysis/log-rank':
        return create_log_rank_page(language)
    elif pathname == '/survival-analysis/weibull':
        return create_weibull_analysis_page(language)
    elif pathname == '/survival-analysis/exponential':
        return create_exponential_analysis_page(language)
    elif pathname == '/survival-analysis/rsf':
        return create_rsf_analysis_page(language)
    elif pathname == '/survival-analysis/comparacion-tecnicas':
        return create_techniques_comparison_page(language)
    elif pathname == '/ver-dataset':
        return create_ver_dataset_page(language)
    else:
        return create_home_page(language)


@app.callback(
    Output('url', 'pathname'),
    [Input('confirm-dialog', 'submit_n_clicks'),
     Input('confirm-dialog', 'cancel_n_clicks')],
    prevent_initial_call=True
)
def navegar_a_inicio(submit_n_clicks, cancel_n_clicks):

    """Decide whether to return home to load data or keep the current page."""
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update
    
    trigger_id = ctx.triggered[0]['prop_id']
    if 'submit_n_clicks' in trigger_id:
        print("Confirm clicked")
        return '/'
    
    if 'cancel_n_clicks' in trigger_id:
        print("Cancel clicked")
        return dash.no_update
    
    return dash.no_update


if __name__ == '__main__':
    print("[BOOT] Registering main callbacks...", flush=True)
    register_analysis_callbacks(app)

    print("[BOOT] Registering PDF export callbacks...", flush=True)
    register_pdf_export_callbacks(app)  # PDF export ENABLED

    host = '127.0.0.1'
    port = 8050
    debug_mode = os.environ.get('DASH_DEBUG', '0').lower() in ('1', 'true', 'yes', 'on')

    print(f"""[BOOT] Starting Dash at http://{host}:{port} (debug={debug_mode}, reloader=False)""", flush=True)
    if hasattr(app, 'run'):
        app.run(host=host, port=port, debug=debug_mode, use_reloader=False)
    else:
        app.run_server(host=host, port=port, debug=debug_mode, use_reloader=False)
