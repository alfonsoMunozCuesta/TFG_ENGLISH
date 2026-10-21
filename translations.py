"""Bilingual interface text dictionary.

Centralize Spanish and English labels so pages can render
dynamically according to the active language.
"""

# Spanish-English translation dictionary. Keys must match
# those used in layout.py, cargaDataset.py, and callbacks.
translations = {
    'es': {
        # Navbar
        'navbar_home': 'INICIO',
        'navbar_view_dataset': 'VER DATASET',
        'navbar_covariate_analysis': 'ANÁLISIS DE COVARIABLES',
        'navbar_survival_analysis': 'ANÁLISIS DE SUPERVIVENCIA',
        'navbar_techniques_comparison': 'COMPARACIÓN DE TÉCNICAS',
        'inicio': 'INICIO',
        'ver_dataset': 'VER DATASET',
        'analisis_covariables': 'ANÁLISIS DE COVARIABLES',
        'analisis_supervivencia': 'ANÁLISIS DE SUPERVIVENCIA',
        
        # Home page
        'dashboard_title': "DASHBOARD 'SURVIVAL ANALYSIS'",
        'cargar_dataset': 'Cargar Dataset para preprocesarlo y analizarlo',
        'sube_csv': 'Sube tu CSV',
        'preprocesa_csv': 'Preprocesa CSV',
        'no_archivo_cargado': 'No se ha cargado ningún archivo aún.',
        'error_archivo': 'ERROR: El archivo cargado no es el adecuado.',
        'archivo_correcto': 'Este sistema es compatible únicamente con el dataset temp_data.csv',
        'archivo_bruto': 'Archivo Bruto',
        'archivo_preprocesado': 'Archivo Preprocesado',
        'home_banner_subtitle': 'Analiza, compara y exporta resultados de supervivencia con soporte en español e inglés.',
        'home_upload_hint': 'Carga un CSV para comenzar.',
        'pdf_modal_title': 'Generar Informe PDF',
        'pdf_modal_instruction': 'Haz clic y selecciona qué quieres incluir en el PDF:',
        'pdf_modal_content': 'Contenido a incluir:',
        'pdf_modal_filename_label': 'Nombre del archivo:',
        'pdf_modal_filename_placeholder': 'informe_supervivencia',
        'pdf_modal_no_extension': 'No incluir extensión .pdf',
        'pdf_modal_cancel': 'Cancelar',
        'pdf_modal_download': 'Descargar PDF',
        'pdf_modal_error_prefix': 'Error',
        
        # Confirmation dialog.
        'confirmar_inicio': '¿Estás seguro de que deseas volver a la página inicial? Perderás el dataset cargado y todo el análisis realizado.',
        
        # Survival Analysis
        'survival_analysis': 'Análisis de Supervivencia',
        'survival_analysis_prefix': 'Análisis de Supervivencia: {name}',
        'kaplan_meier': 'Kaplan-Meier',
        'cox_regression': 'Regresión de Cox',
        'log_rank_test': 'Test de Log-Rank',
        'weibull_analysis': 'Análisis Weibull',
        'random_survival_forest': 'Random Survival Forest',
        'exponential_analysis': 'Análisis Exponencial',
        'exponential_intro': 'Este análisis ajusta un modelo Exponencial a los tiempos de supervivencia para modelar un riesgo constante a lo largo del tiempo.',
        'exponential_note': 'La curva ajustada resume el comportamiento del modelo exponencial y la tabla recoge sus parámetros clave.',
        'exponential_summary_title': 'Resumen del ajuste Exponencial',
        'exponential_graph_title': 'Método Exponencial',
        'explicar_exponential': 'Explicar',
        'exponential_metric': 'Métrica',
        'exponential_value': 'Valor',
        'exponential_interpretation': 'Interpretación',
        'exponential_no_data': 'No hay datos suficientes para aplicar Exponencial.',
        'weibull_intro': 'Este análisis ajusta una distribución Weibull a los tiempos de supervivencia para modelar cómo cambia el riesgo a lo largo del tiempo.',
        'weibull_note': 'La curva ajustada resume el comportamiento paramétrico estimado del modelo y la tabla recoge los parámetros clave del ajuste.',
        'rsf_intro': 'Random Survival Forest construye muchos árboles a partir de muestras bootstrap y combina sus predicciones para estimar la supervivencia.',
        'rsf_note': 'La gráfica principal muestra curvas de supervivencia para perfiles de bajo, medio y alto riesgo. La barra inferior resume qué variables influyen más en el modelo.',
        'rsf_summary_title': 'Resumen del modelo RSF',
        'rsf_graph_title': 'Curvas de supervivencia estimadas',
        'rsf_importance_title': 'Importancia de variables',
        'rsf_pdf_option_general_summary': 'Resumen general',
        'rsf_pdf_option_model_summary': 'Resumen del modelo RSF',
        'rsf_pdf_option_summary': 'Resumen del modelo',
        'rsf_pdf_option_graph': 'Curvas de supervivencia estimadas',
        'rsf_pdf_option_importance': 'Importancia de variables',
        'rsf_pdf_option_profile': 'Simular perfil individual',
        'rsf_pdf_option_ai': 'Interpretación IA',
        'rsf_metric': 'Métrica',
        'rsf_value': 'Valor',
        'rsf_interpretation': 'Interpretación',
        'explicar_rsf': 'Explicar',
        'pdf_modal_title': 'Generar Informe PDF',
        'pdf_modal_instruction': 'Haz clic y selecciona qué quieres incluir en el PDF:',
        'pdf_modal_content': 'Contenido a incluir:',
        'pdf_modal_filename_label': 'Nombre del archivo:',
        'pdf_modal_filename_placeholder': 'informe_supervivencia',
        'pdf_modal_no_extension': 'No incluir extensión .pdf',
        'pdf_modal_cancel': 'Cancelar',
        'pdf_modal_download': 'Descargar PDF',
        'rsf_no_data': 'No hay datos suficientes para aplicar Random Survival Forest.',
        'weibull_graph_title': 'Método Weibull',
        'weibull_summary_title': 'Resumen del ajuste Weibull',
        'explicar_weibull': 'Explicar',
        'weibull_metric': 'Métrica',
        'weibull_value': 'Valor',
        'weibull_interpretation': 'Interpretación',
        'weibull_no_data': 'No hay datos suficientes para aplicar Weibull.',
        
        # Clean dataset.
        'dataset_limpio': 'Dataset Limpio',
        
        # Covariate analysis.
        'analisis_covariables_title': 'Análisis de Covariables',
        'analisis_covariables_intro': 'Analiza cómo diferentes variables influyen en el abandono estudiantil',
        'selecciona_el_analisis': 'Selecciona el análisis:',
        'interpretacion': 'Interpretación',
        'abandono_total': 'Abandono Total',
        'abandono_genero': 'Abandono por Género',
        'abandono_discapacidad': 'Abandono por Discapacidad',
        'abandono_age_band': 'Abandono por Banda de Edad',
        'abandono_highest_education': 'Abandono por Educación Más Alta',
        'femenino': 'Femenino',
        'masculino': 'Masculino',
        'con_discapacidad': 'Con Discapacidad',
        'sin_discapacidad': 'Sin Discapacidad',
        'abandono': 'Abandono',
        'no_abandono': 'No abandono',
        'withdrawn': 'Withdrawn',
        'edad_banda': 'Banda de Edad',
        'educacion_mas_alta': 'Educación Más Alta',
        'creditos_estudiados': 'Créditos Estudiados',
        'edad_0_35': '0-35 años',
        'edad_35_55': '35-55 años',
        'edad_55_mas': '55+ años',
        'educacion_a_level': 'A Level o equivalente',
        'educacion_superior': 'Titulación de educación superior',
        'educacion_inferior_a_level': 'Inferior a A Level',
        'educacion_posgrado': 'Posgrado',
        'educacion_sin_titulacion': 'Sin titulación formal',
        'identificador_estudiante': 'Identificador del estudiante',
        
        # Kaplan-Meier
        'curva_supervivencia': 'Curva de Supervivencia Kaplan-Meier',
        'tiempo': 'Tiempo',
        'probabilidad_supervivencia': 'Probabilidad de Supervivencia',
        'explicar_kaplan': 'Explicar',
        'explicar_cox': 'Explicar',
        'explicar_logrank': 'Explicar',
        'selecciona_covariable': 'Selecciona 1 covariable para ver su curva de Kaplan:',
        'selecciona_covariable_kaplan': 'Selecciona 1 covariable para ver su curva de Kaplan:',
        'genero': 'Género',
        'discapacidad': 'Discapacidad',
        'grupo_edad': 'Grupo de Edad',
        'nivel_educativo': 'Nivel Educativo',
        'ninguna': 'Ninguna',
        
        # Cox Regression page
        'elige_covariable_cox': 'Elige 1 o más covariables para la regresión de Cox',
        'selecciona_covariable_logrank': 'Selecciona 1 o más covariables para el Test de Log-Rank:',
        'selecciona_covariable_minimo': 'Selecciona al menos una covariable.',
        'selecciona_covariable_comparar': 'Selecciona al menos una covariable para comparar.',
        'covariate_error_load_dataset': '❌ Por favor, carga primero un dataset válido',
        'covariate_error_dataset_title': 'Error: Dataset no cargado',
        'covariate_error_data_header': '❌ Error de datos',
        'covariate_error_data_body': 'No hay datos disponibles.',
        'covariate_error_missing_column_title': 'Error: Columna faltante',
        'covariate_error_structure_header': '❌ Error de estructura',
        'covariate_error_missing_prefix': 'Falta:',
        'covariate_error_generic_title': 'Error',
        'covariate_error_unexpected_header': '❌ Error inesperado',
        
        # Charts: titles.
        'abandono_vs_no_abandono': 'Abandono vs No Abandono',
        'resultado_final': 'Resultado Final',
        'num_estudiantes': 'Número de Estudiantes',
        'abandono_genero_title': 'Abandono según Género',
        'abandono_discapacidad_title': 'Abandono según Discapacidad',
        'abandono_age_band_title': 'Abandono según Banda de Edad',
        'abandono_highest_education_title': 'Abandono según Educación Más Alta',
        'evento': 'Evento',
        
        # Explanations.
        'exp_abandono': '''Este gráfico muestra la distribución de los estudiantes en función de si han abandonado o no el curso. 
            La columna verde representa a los estudiantes que no han abandonado, mientras que la morada a los que sí.
            Como se puede observar, la mayoría de los estudiantes no han llevado a cabo el evento de abandono 
            lo que sugiere que la tasa de abandono es relativamente baja en este conjunto de datos.''',
        'exp_genero': '''Este gráfico muestra la distribución del abandono influido por el género. Podemos concluir en que la mayoría de los estudiantes
            no han abandonado el curso y se observa una diferencia destacable en el abandono por género. Las estudiantes femeninas
            presentan una mayor proporción de abandono en comparación con los estudiantes masculinos, por ello, en este conjunto las mujeres tienen una
            tasa de abandono superior. A pesar de ello, en ambos sexos, el número de estudiantes 
            que no abandonan el curso es significativamente mayor que el número de los que si abandonan.''',
        'exp_discapacidad': '''Este gráfico muestra la distribución del abandono influido por la discapacidad. A pesar de que el número de estudiantes sin discapacidad
            es considerablemente menor, la tasa de abandono en este conjunto es más alta en comparación con los estudiantes que presentan discapacidad.
            Aún así, la mayoría de estudiantes, tanto con como sin, no abandonan el curso, y es la conclusión que podemos destacar del grafo presente. Esto
            nos lleva a entender que aunque la discapacidad puede estar asociada a un riesgo superior de que suceda este evento, la diferencia no es 
            considerable para este conjunto de datos.''',
        'exp_age_band': '''Este gráfico muestra la distribución del abandono según la banda de edad de los estudiantes. Podemos observar cómo la tasa de abandono
            varía según los diferentes grupos de edad. Este análisis es importante para comprender si existe una relación entre la edad del estudiante y la probabilidad
            de abandonar el curso. La mayoría de estudiantes no abandonan, pero es notable ver si hay diferencias significativas entre grupos de edad.''',
        'exp_highest_education': '''Este gráfico muestra la distribución del abandono según el nivel de educación más alto de los estudiantes. Podemos analizar si existe
            una correlación entre el nivel educativo previo y el abandono del curso. Este análisis ayuda a entender cómo el trasfondo educativo influye en la persistencia
            del estudiante. Es importante observar si ciertos niveles educativos tienen tasas de abandono más altas o más bajas.''',
        'abandono_studied_credits': 'Abandono por Créditos Estudiados',
        'abandono_studied_credits_title': 'Abandono según Créditos Estudiados',
        'exp_studied_credits': '''Este gráfico muestra la distribución del abandono según los créditos estudiados por los estudiantes. Podemos observar si existe una relación
            entre la cantidad de créditos cursados y la probabilidad de abandonar. Este análisis es importante para entender cómo el progreso académico
            influye en la retención del estudiante.''',
        'error_loading_csv_title': 'ERROR al cargar archivo',
        'error_loading_csv_body': 'No se pudo leer el archivo CSV. Verifica que:',
        'error_loading_csv_tip_1': 'El archivo sea un CSV válido',
        'error_loading_csv_tip_2': 'Utilice ";" como separador',
        'error_loading_csv_tip_3': 'La codificación sea UTF-8',
        'error_preprocess_title': 'ERROR en preprocesamiento',
        'error_preprocess_body': 'Verifica que el archivo contiene todas las columnas requeridas:',
        'error_unexpected_title': 'ERROR inesperado',
        'error_unexpected_body': 'Algo salió mal durante el preprocesamiento.',
        'error_unexpected_contact': 'Contacta al administrador si el problema persiste.',
        'error_dataset_not_loaded': 'Para acceder a esta parte del dashboard es necesario cargar el dataset académico primero.',
        'error_no_data': 'No hay datos disponibles.',
        'error_select_variable': 'Por favor, selecciona una covariable primero',
        'error_select_covariate': 'Por favor, selecciona al menos una covariable primero',
        'error_select_logrank': 'Por favor, realiza un Test de Log-Rank primero',
        'error_timeout': 'Timeout: El servidor tardó más de 10 minutos. Intenta más tarde.',
        'error_connection': 'Servidor llama-server no disponible. Ejecuta: START_LLAMA_SERVER.bat',
        
        # General
        'cargando': 'Cargando...',
        'respuesta': 'La respuesta es...',
    },
    'en': {
        # Navbar
        'navbar_home': 'HOME',
        'navbar_view_dataset': 'VIEW DATASET',
        'navbar_covariate_analysis': 'COVARIATE ANALYSIS',
        'navbar_survival_analysis': 'SURVIVAL ANALYSIS',
        'navbar_techniques_comparison': 'TECHNIQUES COMPARISON',
        'inicio': 'HOME',
        'ver_dataset': 'VIEW DATASET',
        'analisis_covariables': 'COVARIATE ANALYSIS',
        'analisis_supervivencia': 'SURVIVAL ANALYSIS',
        
        # Home page
        'dashboard_title': "SURVIVAL ANALYSIS DASHBOARD",
        'cargar_dataset': 'Load Dataset to preprocess and analyze',
        'sube_csv': 'Upload your CSV',
        'preprocesa_csv': 'Preprocess CSV',
        'no_archivo_cargado': 'No file has been loaded yet.',
        'error_archivo': 'ERROR: The uploaded file is not suitable.',
        'archivo_correcto': 'This system is only compatible with the temp_data.csv dataset',
        'archivo_bruto': 'Raw File',
        'archivo_preprocesado': 'Preprocessed File',
        'home_banner_subtitle': 'Analyze, compare, and export survival results with Spanish and English support.',
        'home_upload_hint': 'Upload a CSV to get started.',
        'pdf_modal_title': 'Generate PDF Report',
        'pdf_modal_instruction': 'Click and select what you want to include in the PDF:',
        'pdf_modal_content': 'Content to include:',
        'pdf_modal_filename_label': 'File name:',
        'pdf_modal_filename_placeholder': 'survival_report',
        'pdf_modal_no_extension': 'Do not include the .pdf extension',
        'pdf_modal_cancel': 'Cancel',
        'pdf_modal_download': 'Download PDF',
        'pdf_modal_error_prefix': 'Error',
        
        # Confirmation dialog.
        'confirmar_inicio': 'Are you sure you want to return to the home page? You will lose the loaded dataset and all analysis performed.',
        
        # Survival Analysis
        'survival_analysis': 'Survival Analysis',
        'kaplan_meier': 'Kaplan-Meier',
        'cox_regression': 'Cox Regression',
        'log_rank_test': 'Log-Rank Test',
        'weibull_analysis': 'Weibull Analysis',
        'random_survival_forest': 'Random Survival Forest',
        'exponential_analysis': 'Exponential Analysis',
        'exponential_intro': 'This analysis fits an Exponential model to survival times to model a constant risk over time.',
        'exponential_note': 'The fitted curve summarizes the exponential model behavior, and the table collects the key parameters of the fit.',
        'exponential_summary_title': 'Exponential fit summary',
        'exponential_graph_title': 'Exponential method',
        'explicar_exponential': 'Explain',
        'exponential_metric': 'Metric',
        'exponential_value': 'Value',
        'exponential_interpretation': 'Interpretation',
        'exponential_no_data': 'Not enough data is available to apply Exponential.',
        'weibull_intro': 'This analysis fits a Weibull distribution to the survival times to model how the risk changes over time.',
        'weibull_note': 'The fitted curve summarizes the estimated parametric behavior of the model, and the table collects the key parameters of the fit.',
        'rsf_intro': 'Random Survival Forest builds many trees from bootstrap samples and combines their predictions to estimate survival.',
        'rsf_note': 'The main chart shows survival curves for low-, medium-, and high-risk profiles. The lower bar chart summarizes which variables matter most in the model.',
        'rsf_summary_title': 'RSF model summary',
        'rsf_graph_title': 'Estimated survival curves',
        'rsf_importance_title': 'Variable importance',
        'rsf_pdf_option_general_summary': 'General summary',
        'rsf_pdf_option_model_summary': 'RSF model summary',
        'rsf_pdf_option_summary': 'Model summary',
        'rsf_pdf_option_graph': 'Estimated survival curves',
        'rsf_pdf_option_importance': 'Variable importance',
        'rsf_pdf_option_profile': 'Simulate individual profile',
        'rsf_pdf_option_ai': 'AI interpretation',
        'rsf_metric': 'Metric',
        'rsf_value': 'Value',
        'rsf_interpretation': 'Interpretation',
        'explicar_rsf': 'Explain',
        'pdf_modal_title': 'Generate PDF Report',
        'pdf_modal_instruction': 'Click and select what you want to include in the PDF:',
        'pdf_modal_content': 'Content to include:',
        'pdf_modal_filename_label': 'File name:',
        'pdf_modal_filename_placeholder': 'survival_report',
        'pdf_modal_no_extension': 'Do not include the .pdf extension',
        'pdf_modal_cancel': 'Cancel',
        'pdf_modal_download': 'Download PDF',
        'rsf_no_data': 'Not enough data is available to apply Random Survival Forest.',
        'weibull_graph_title': 'Weibull method',
        'weibull_summary_title': 'Weibull fit summary',
        'explicar_weibull': 'Explain',
        'weibull_metric': 'Metric',
        'weibull_value': 'Value',
        'weibull_interpretation': 'Interpretation',
        'weibull_no_data': 'Not enough data is available to apply Weibull.',

        # Clean dataset.
        'dataset_limpio': 'Clean Dataset',

        # Covariate Analysis
        'analisis_covariables_title': 'Covariate Analysis',
        'analisis_covariables_intro': 'Analyze how different variables influence student dropout.',
        'selecciona_el_analisis': 'Select the analysis:',
        'interpretacion': 'Interpretation',
        'abandono_total': 'Total Dropout',
        'abandono_genero': 'Dropout by Gender',
        'abandono_discapacidad': 'Dropout by Disability',
        
        'abandono_age_band': 'Dropout by Age Band',
        'abandono_highest_education': 'Dropout by Highest Education',
        'femenino': 'Female',
        'masculino': 'Male',
        'con_discapacidad': 'With Disability',
        'sin_discapacidad': 'Without Disability',
        'abandono': 'Dropout',
        'no_abandono': 'No Dropout',
        'withdrawn': 'Withdrawn',
        'edad_banda': 'Age Band',
        'educacion_mas_alta': 'Highest Education',
        'creditos_estudiados': 'Studied Credits',
        'edad_0_35': '0-35 years',
        'edad_35_55': '35-55 years',
        'edad_55_mas': '55+ years',
        'educacion_a_level': 'A Level or Equivalent',
        'educacion_superior': 'HE Qualification',
        'educacion_inferior_a_level': 'Lower Than A Level',
        'educacion_posgrado': 'Post Graduate Qualification',
        'educacion_sin_titulacion': 'No Formal Qualifications',
        'identificador_estudiante': 'Student ID',
        'abandono_discapacidad': 'Dropout by Disability',
        'survival_analysis_prefix': 'Survival Analysis: {name}',
        
        # Kaplan-Meier
        'curva_supervivencia': 'Kaplan-Meier Survival Curve',
        'tiempo': 'Time',
        'probabilidad_supervivencia': 'Survival Probability',
        'explicar_kaplan': 'Explain',
        'explicar_cox': 'Explain',
        'explicar_logrank': 'Explain',
        'selecciona_covariable': 'Select 1 covariate to see its Kaplan curve:',
        'selecciona_covariable_kaplan': 'Select 1 covariate to see its Kaplan curve:',
        'genero': 'Gender',
        'discapacidad': 'Disability',
        'grupo_edad': 'Age Group',
        'nivel_educativo': 'Education Level',
        'ninguna': 'None',
        
        # Cox Regression page
        'abandono_age_band_title': 'Dropout by Age Band',
        'abandono_highest_education_title': 'Dropout by Highest Education',
        'elige_covariable_cox': 'Choose 1 or more covariates for Cox regression',
        'selecciona_covariable_logrank': 'Select 1 or more covariates for the Log-Rank Test:',
        'selecciona_covariable_minimo': 'Please select at least one covariate.',
        'selecciona_covariable_comparar': 'Please select at least one covariate to compare.',
        'covariate_error_load_dataset': '❌ Please load a valid dataset first',
        'covariate_error_dataset_title': 'Error: Dataset not loaded',
        'covariate_error_data_header': '❌ Data error',
        'covariate_error_data_body': 'No data available.',
        'covariate_error_missing_column_title': 'Error: Missing column',
        'covariate_error_structure_header': '❌ Structure error',
        'covariate_error_missing_prefix': 'Missing:',
        'covariate_error_generic_title': 'Error',
        'covariate_error_unexpected_header': '❌ Unexpected error',
        
        # Charts: titles.
        'abandono_vs_no_abandono': 'Dropout vs No Dropout',
        'resultado_final': 'Final Result',
        'num_estudiantes': 'Number of Students',
        'abandono_genero_title': 'Dropout by Gender',
        'abandono_discapacidad_title': 'Dropout by Disability',
        'evento': 'Event',
        
        # Explanations.
        'exp_age_band': '''This chart shows the distribution of dropout by the age band of students. We can observe how the dropout rate
            varies among different age groups. This analysis is important to understand if there is a relationship between student age and the probability
            of dropping out of the course. Most students do not drop out, but it is notable to see if there are significant differences between age groups.''',
        'exp_highest_education': '''This chart shows the distribution of dropout by the highest education level of students. We can analyze if there is
            a correlation between prior educational level and course dropout. This analysis helps understand how educational background influences student
            persistence. It is important to observe if certain education levels have higher or lower dropout rates.''',
        'abandono_studied_credits': 'Dropout by Studied Credits',
        'abandono_studied_credits_title': 'Dropout by Studied Credits',
        'exp_studied_credits': '''This chart shows the distribution of dropout by the studied credits of students. We can observe if there is a relationship
            between the number of credits taken and the probability of dropping out. This analysis is important to understand how academic progress
            influences student retention.''',
        'error_loading_csv_title': 'ERROR loading file',
        'error_loading_csv_body': 'Could not read the CSV file. Check that:',
        'error_loading_csv_tip_1': 'The file is a valid CSV',
        'error_loading_csv_tip_2': 'It uses ";" as separator',
        'error_loading_csv_tip_3': 'The encoding is UTF-8',
        'error_preprocess_title': 'ERROR during preprocessing',
        'error_preprocess_body': 'Make sure the file contains all required columns:',
        'error_unexpected_title': 'Unexpected ERROR',
        'error_unexpected_body': 'Something went wrong during preprocessing.',
        'error_unexpected_contact': 'Contact the administrator if the problem persists.',
        'error_dataset_not_loaded': 'To access this part of the dashboard, you must load the academic dataset first.',
        'error_no_data': 'No data available.',
        'error_select_variable': 'Please select a covariate first',
        'error_select_covariate': 'Please select at least one covariate first',
        'error_select_logrank': 'Please run a Log-Rank Test first',
        'error_timeout': 'Timeout: The server took more than 10 minutes. Try again later.',
        'error_connection': 'llama-server is not available. Run: START_LLAMA_SERVER.bat',
        'exp_abandono': '''This chart shows the distribution of students based on whether they have dropped out or not. 
            The green column represents students who have not dropped out, while the purple represents those who have.
            As can be seen, most students have not experienced the dropout event, suggesting that the dropout rate is relatively low in this dataset.''',
        'exp_genero': '''This chart shows the distribution of dropout influenced by gender. We can conclude that most students
            have not dropped out of the course and there is a notable difference in dropout by gender. Female students
            have a higher proportion of dropout compared to male students, therefore, in this dataset women have a
            higher dropout rate. However, in both sexes, the number of students 
            who do not drop out is significantly greater than the number who do.''',
        'exp_discapacidad': '''This chart shows the distribution of dropout influenced by disability. Although the number of students without disability
            is considerably lower, the dropout rate in this dataset is higher compared to students with disability.
            Still, most students, both with and without disability, do not drop out, and this is the conclusion we can highlight from this chart. This
            leads us to understand that although disability can be associated with a higher risk of this event occurring, the difference is not 
            significant for this dataset.''',
        
        # General
        'cargando': 'Loading...',
        'respuesta': 'The answer is...',
    }
}

def get_display_text(language, text):
    """Translate display text while preserving internal column and data keys."""
    if language != 'en' or not isinstance(text, str):
        return text

    labels = {
        'Covariable': 'Covariate',
        'Grupo A': 'Group A',
        'Grupo B': 'Group B',
        'Decisión': 'Decision',
        'Conclusión': 'Conclusion',
        'Tiempo': 'Time',
        'Supervivencia': 'Survival',
        'Metrica': 'Metric',
        'Métrica': 'Metric',
        'Valor': 'Value',
        'Interpretacion': 'Interpretation',
        'Interpretación': 'Interpretation',
        'Modelo': 'Model',
        'Género': 'Gender',
        'Discapacidad': 'Disability',
        'Créditos Estudiados': 'Studied Credits',
        'Postgrado': 'Postgraduate',
        'Femenino': 'Female',
        'Masculino': 'Male',
        'Con discapacidad': 'With disability',
        'Sin discapacidad': 'Without disability',
        'Con Discapacidad': 'With Disability',
        'Sin Discapacidad': 'Without Disability',
        'Desconocido': 'Unknown',
        'A Level o Equivalente': 'A Level or Equivalent',
        'Calificación HE': 'HE Qualification',
        'Inferior a A Level': 'Lower Than A Level',
        'Sin Cualificaciones': 'No Formal Qualifications',
        'Rechazar H0': 'Reject H0',
        'No rechazar H0': 'Do not reject H0',
        'No evaluable': 'Not evaluable',
        'La covariable no tiene exactamente dos grupos.': 'The covariate does not have exactly two groups.',
        'Hay diferencias significativas entre las curvas': 'There are significant differences between the curves',
        'No se observan diferencias significativas entre las curvas': 'No significant differences between the curves were observed',
        'Hay diferencias significativas': 'There are significant differences',
        'No hay diferencias significativas': 'There are no significant differences',
        'Curva de Supervivencia Kaplan-Meier': 'Kaplan-Meier Survival Curve',
        'Probabilidad de Supervivencia': 'Survival Probability',
        'Tiempo (meses)': 'Time (months)',
        'Función de Supervivencia': 'Survival Function',
        'Forest Plot - Coeficientes del Modelo Cox': 'Forest Plot - Cox Model Coefficients',
        'Sin observaciones en este grupo': 'No observations in this group',
        'Sin datos': 'No data',
    }
    if text in labels:
        return labels[text]

    fragments = {
        'Curva de Kaplan-Meier para ': 'Kaplan-Meier Curve for ',
        'Curvas Kaplan-Meier - ': 'Kaplan-Meier Curves - ',
        'Grupo vacío: ': 'Empty group: ',
        'Quintil ': 'Quintile ',
        'Grupo ': 'Group ',
        'Créditos: ': 'Credits: ',
        ' créditos': ' credits',
        ' años': ' years',
        ' IC superior': ' upper CI',
        ' IC inferior': ' lower CI',
        'Sin observaciones en este grupo': 'No observations in this group',
        'Sin datos': 'No data',
        'Tiempo: ': 'Time: ',
        'Supervivencia: ': 'Survival: ',
    }
    for original, translated in fragments.items():
        text = text.replace(original, translated)
    return text


def get_display_records(dataframe, language):
    """Translate table cells for display without modifying the source dataframe."""
    return [
        {key: get_display_text(language, value) for key, value in row.items()}
        for row in dataframe.to_dict('records')
    ]


def localize_figure(figure, language):
    """Translate figure labels without changing traces or numerical results."""
    if language != 'en':
        return figure
    for title in (figure.layout.title, figure.layout.xaxis.title, figure.layout.yaxis.title, figure.layout.legend.title):
        if title.text:
            title.text = get_display_text(language, title.text)
    for trace in figure.data:
        if trace.name:
            trace.name = get_display_text(language, trace.name)
        if getattr(trace, 'hovertemplate', None):
            trace.hovertemplate = get_display_text(language, trace.hovertemplate)
    return figure


def get_covariate_label(language, covariable):
    """Translate chart labels without changing column names."""
    label_keys = {
        'gender_F': 'genero',
        'disability_N': 'discapacidad',
        'age_band': 'edad_banda',
        'highest_education': 'educacion_mas_alta',
        'studied_credits': 'creditos_estudiados',
        'id_student': 'identificador_estudiante',
        'age_band_0-35': 'edad_0_35',
        'age_band_35-55': 'edad_35_55',
        'age_band_55<=': 'edad_55_mas',
        'highest_education_A Level or Equivalent': 'educacion_a_level',
        'highest_education_HE Qualification': 'educacion_superior',
        'highest_education_Lower Than A Level': 'educacion_inferior_a_level',
        'highest_education_Post Graduate Qualification': 'educacion_posgrado',
        'highest_education_No Formal quals': 'educacion_sin_titulacion',
    }
    key = label_keys.get(covariable)
    return get_translation(language, key) if key else covariable


def get_feature_label(language, feature):
    """Label a predictor while preserving the meaning of its binary indicators."""
    if feature in ('gender_F', 'disability_N'):
        category = 'femenino' if feature == 'gender_F' else 'sin_discapacidad'
        return f"{get_covariate_label(language, feature)}: {get_translation(language, category)}"
    for prefix, covariable in (('age_band_', 'age_band'), ('highest_education_', 'highest_education')):
        if feature.startswith(prefix):
            return f"{get_covariate_label(language, covariable)}: {get_covariate_label(language, feature)}"
    return get_covariate_label(language, feature)


def get_group_label(language, covariable, group):
    """Translate group labels while preserving the codes stored in the data."""
    binary_keys = {
        'gender_F': {0: 'masculino', 1: 'femenino'},
        'disability_N': {0: 'con_discapacidad', 1: 'sin_discapacidad'},
    }
    if covariable in binary_keys:
        key = binary_keys[covariable].get(group)
        if key:
            return get_translation(language, key)
    label = str(group)
    if covariable == 'age_band' and not label.startswith('age_band_'):
        label = 'age_' + label if label.startswith('band_') else 'age_band_' + label
    elif covariable == 'highest_education' and not label.startswith('highest_education_'):
        label = 'highest_' + label if label.startswith('education_') else 'highest_education_' + label
    elif covariable == 'studied_credits' and language == 'en':
        label = label.replace('Quintil ', 'Quintile ')
    return get_covariate_label(language, label)


def get_result_column_label(language, column):
    """Translate result headers without changing column identifiers."""
    labels = {
        'en': {'Covariable': 'Covariate', 'Grupo A': 'Group A', 'Grupo B': 'Group B',
               'Decisión': 'Decision', 'Conclusión': 'Conclusion', 'Estado': 'Status',
               'test_statistic': 'Test statistic', 'p_value': 'p-value'},
        'es': {'test_statistic': 'Estadístico', 'p_value': 'p-valor',
               'Coef. lower 95%': 'Coef. inferior 95%', 'Coef. upper 95%': 'Coef. superior 95%',
               'exp(Coef.) lower 95%': 'HR inferior 95%', 'exp(Coef.) upper 95%': 'HR superior 95%',
               'cmp to': 'Comparación'},
    }
    return labels.get(language, labels['es']).get(column, column)


def get_plotly_config(language, **options):
    """Apply the selected language to Plotly controls and formats."""
    config = {'locale': 'en' if language == 'en' else 'es'}
    if language != 'en':
        config['locales'] = {'es': {
            'dictionary': {
                'Autoscale': 'Ajustar escala', 'Box Select': 'Seleccionar con un rectángulo',
                'Compare data on hover': 'Comparar datos al pasar el cursor',
                'Double-click on legend to isolate one trace': 'Haz doble clic en la leyenda para aislar una serie',
                'Double-click to zoom back out': 'Haz doble clic para restaurar la escala',
                'Download plot as a png': 'Descargar gráfica como PNG', 'Download plot': 'Descargar gráfica',
                'Lasso Select': 'Seleccionar con un lazo', 'Pan': 'Desplazar',
                'Produced with Plotly.js': 'Creado con Plotly.js', 'Reset': 'Restablecer',
                'Reset axes': 'Restablecer ejes', 'Reset view': 'Restablecer vista',
                'Reset views': 'Restablecer vistas',
                'Show closest data on hover': 'Mostrar el dato más cercano al pasar el cursor',
                'Snapshot succeeded': 'Imagen guardada',
                'Sorry, there was a problem downloading your snapshot!': 'No se ha podido descargar la imagen',
                'Taking snapshot - this may take a few seconds': 'Preparando la imagen...',
                'Toggle Spike Lines': 'Mostrar u ocultar guías',
                'Toggle show closest data on hover': 'Mostrar u ocultar el dato más cercano',
                'Zoom': 'Ampliar o reducir', 'Zoom in': 'Ampliar', 'Zoom out': 'Reducir',
            },
            'format': {
                'days': ['Domingo', 'Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado'],
                'shortDays': ['Dom', 'Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb'],
                'months': ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'],
                'shortMonths': ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'],
                'date': '%d/%m/%Y', 'decimal': ',', 'thousands': '.',
            },
        }}
    config.update(options)
    return config


def get_translation(lang, key):
    """
    Get a translation for the specified language and key.

    ERROR #5: Safely handle missing translations.
    - Validate that the language exists.
    - Return the key as a fallback if the translation is missing.
    - Log warnings for debugging.
    """
    # Validate the language.
    if lang not in translations:
        print(f"""⚠️  [TRANSLATION] Language '{lang}' is unsupported; using 'es'""")
        lang = 'es'
    
    # Get the translation.
    translation = translations.get(lang, {}).get(key, None)
    
    # Return the key as a fallback if the translation does not exist.
    if translation is None:
        print(f"""⚠️  [TRANSLATION] Key '{key}' not found for language '{lang}'""")
        return key
    
    return translation
