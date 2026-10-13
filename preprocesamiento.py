"""Clean and reduce the academic dataset.

Convert the original CSV into a table with one row per student and the
columns required for survival analysis, validating structure and key values.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
import os
import csv

def load_dataset(file_path):
    """Read the original CSV using the delimiter expected by the project."""
    df = pd.read_csv(file_path, sep=';')
    df.columns = df.columns.str.strip()  # Trim whitespace from column names.
    return df

# Clean activity columns.
def clean_columns(df):
    """Normalize column names for consistency in the analysis."""
    columnas_actividad = [
        "ouelluminate", "sharedsubpage", "ouwiki", "folder", "page", "externalquiz",
        "quiz", "dualpane", "questionnaire", "htmlactivity", "oucollaborate",
        "dataplus", "glossary", "repeatactivity", "resource", "forumng", "url",
        "subpage", "oucontent", "homepage"
    ]
    return columnas_actividad

# Preprocess the dataset.
def preprocess_csv_file_streaming(file_path):
    """Preprocess temp_data.csv row by row to avoid loading the entire CSV into memory."""
    columnas_actividad = clean_columns(None)
    columnas_a_conservar = [
        'id_student', 'date', 'final_result',
        'gender_F', 'disability_N',
        'age_band_0-35', 'age_band_35-55', 'age_band_55<=',
        'highest_education_A Level or Equivalent', 'highest_education_HE Qualification',
        'highest_education_Lower Than A Level', 'highest_education_No Formal quals',
        'highest_education_Post Graduate Qualification',
        'studied_credits'
    ]
    columnas_necesarias = set(columnas_a_conservar + columnas_actividad)

    def _to_float(value, default=0.0):
        """Convert CSV values to float, using a default value if conversion fails."""
        try:
            if value is None or str(value).strip() == "":
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _to_int(value, default=0):
        """Convert CSV values to integers, accepting numbers written as decimals."""
        try:
            if value is None or str(value).strip() == "":
                return default
            return int(float(value))
        except (TypeError, ValueError):
            return default

    def _normalizar_fila(raw_row):
        """Trim column names in a row read from the CSV."""
        return {
            (key.strip() if key else key): value
            for key, value in raw_row.items()
            if key is not None
        }

    def _preparar_registro(row):
        """Create a helper record with normalized time and event values."""
        final_result_text = str(row.get('final_result', '')).strip().lower()
        date_value = _to_float(row.get('date'))
        record = {col: row.get(col, '') for col in columnas_necesarias if col in row}
        record['_date_value'] = date_value
        record['_event_value'] = 1 if final_result_text == 'withdrawn' else 0
        return record

    def _tiene_actividad(record):
        """Check whether the record contains any positive student activity."""
        return any(_to_float(record.get(col), 0.0) > 0.0 for col in columnas_actividad)

    seleccion_por_estudiante = {}

    with open(file_path, newline='', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=';')
        fieldnames = {name.strip() for name in (reader.fieldnames or []) if name}
        columnas_faltantes = [col for col in ['id_student', 'date', 'final_result'] if col not in fieldnames]
        if columnas_faltantes:
            raise ValueError(f"""[ERROR] Missing critical columns: {', '.join(columnas_faltantes)}""")

        for raw_row in reader:
            row = _normalizar_fila(raw_row)
            student_id = str(row.get('id_student', '')).strip()
            if not student_id:
                continue

            record = _preparar_registro(row)
            date_value = record['_date_value']
            student_state = seleccion_por_estudiante.setdefault(
                student_id,
                {
                    'latest': record,
                    'oldest': record,
                    'date_269': None,
                    'date_0': None,
                    'activity_latest': None,
                }
            )

            if date_value > student_state['latest']['_date_value']:
                student_state['latest'] = record
            if date_value < student_state['oldest']['_date_value']:
                student_state['oldest'] = record
            if int(date_value) == 269 and student_state['date_269'] is None:
                student_state['date_269'] = record
            if int(date_value) == 0 and student_state['date_0'] is None:
                student_state['date_0'] = record
            if _tiene_actividad(record) and (
                student_state['activity_latest'] is None
                or date_value > student_state['activity_latest']['_date_value']
            ):
                student_state['activity_latest'] = record

    filas_finales = []
    for student_id in sorted(seleccion_por_estudiante, key=lambda value: _to_int(value)):
        state = seleccion_por_estudiante[student_id]
        latest = state['latest']
        if latest['_event_value'] == 0:
            selected = state['date_269'] or latest
        else:
            selected = state['activity_latest'] or state['date_0'] or state['oldest']

        output_row = {}
        for col in columnas_a_conservar:
            if col == 'final_result':
                output_row[col] = int(selected['_event_value'])
            else:
                output_row[col] = _to_int(selected.get(col), 0)
        filas_finales.append(output_row)

    df_final = pd.DataFrame(filas_finales, columns=columnas_a_conservar)
    if df_final.empty:
        raise ValueError("[ERROR] The dataset is empty. There are no rows to process.")

    print("\n[OK] Streaming preprocessing completed:")
    print(f"""  - Rows processed: {df_final.shape[0]}""")
    print(f"""  - Columns retained: {df_final.shape[1]}""")
    print(f"""  - final_result distribution: {df_final['final_result'].value_counts().to_dict()}\n""")
    return df_final


def preprocess_data(df_tmp):
    """
    Preprocess the dataset by removing duplicates, keeping essential columns,
    and validating data integrity.

    Raises:
        ValueError: If critical columns are missing or the dataframe is empty.
        TypeError: If data types are invalid.
    """
    print("Starting data preprocessing...")
    
    # VALIDATION 1: Empty DataFrame.
    if df_tmp is None or len(df_tmp) == 0:
        raise ValueError("[ERROR] The dataset is empty. There are no rows to process.")
    
    # VALIDATION 2: Critical columns.
    columnas_criticas = ['id_student', 'date', 'final_result']
    columnas_faltantes = [col for col in columnas_criticas if col not in df_tmp.columns]
    if columnas_faltantes:
        raise ValueError(f"""[ERROR] Missing critical columns: {', '.join(columnas_faltantes)}""")
    
    print(f"""[OK] Initial validations passed""")
    print(f"""  - File rows: {len(df_tmp)}""")
    print(f"""  - Columns found: {len(df_tmp.columns)}""")

    df_tmp.columns = df_tmp.columns.str.strip() 
    
    # Convert "final_result" to binary: 1 = dropout, 0 = no dropout.
    try:
        df_tmp['final_result'] = df_tmp['final_result'].astype(str).str.strip().str.lower()
        df_tmp['final_result'] = df_tmp['final_result'].apply(lambda x: 1 if x == 'withdrawn' else 0)
    except Exception as e:
        raise TypeError(f"""[ERROR] Could not process the 'final_result' column: {str(e)}""")

    # Sort by id_student and date in descending order, newest to oldest.
    df_tmp = df_tmp.sort_values(by=['id_student', 'date'], ascending=[True, False])

    # Remove 'Unnamed: 0' if unnecessary.
    if 'Unnamed: 0' in df_tmp.columns:
        df_tmp = df_tmp.drop(columns=['Unnamed: 0'])

    # Get activity columns before defining the function.
    # IMPORTANT: Keep only columns present in the dataframe.
    todas_columnas_actividad = clean_columns(df_tmp)
    columnas_actividad_validas = [col for col in todas_columnas_actividad if col in df_tmp.columns]

    # Select the appropriate row for each student.
    def seleccionar_fila(grupo):
        """Select the final representative row for each student."""
        if grupo['final_result'].iloc[0] == 0:
            fila_269 = grupo[grupo['date'] == 269]
            if not fila_269.empty:
                return fila_269.iloc[0]
            else:
                return grupo.iloc[0]  # Use the latest row if date=269 is absent.
        else:
            # Look for a row with activity if activity columns exist.
            if len(columnas_actividad_validas) > 0:
                grupo_con_actividad = grupo[(grupo[columnas_actividad_validas] > 0.0).any(axis=1)]
                if not grupo_con_actividad.empty:
                    return grupo_con_actividad.iloc[0]
            # Use the row with date=0 if there is no activity or no activity columns.
            fila_0 = grupo[grupo['date'] == 0]
            if not fila_0.empty:
                return fila_0.iloc[0]
            else:
                return grupo.iloc[-1]

    # Remove duplicates, keeping the latest row for each id_student.
    df_final = df_tmp.groupby('id_student', group_keys=False).apply(seleccionar_fila)

    # Convert columns to int, except activity columns that may be floats.
    # Use 'Int64' with a capital I to allow NaN values.
    for col in df_final.columns:
        if col not in columnas_actividad_validas:
            try:
                # Try converting to Int64 (nullable integer).
                df_final[col] = pd.to_numeric(df_final[col], errors='coerce').astype('Int64')
            except Exception as e:
                print(f"""Warning: Could not convert {col} to int: {e}""")
                # Keep the column unchanged if conversion fails.

    # Keep id_student, date, final_result, binary attributes, and new nonbinary attributes.
    columnas_a_conservar = [
        'id_student', 'date', 'final_result', 
        'gender_F', 'disability_N',
        # New nonbinary attributes.
        'age_band_0-35', 'age_band_35-55', 'age_band_55<=',
        'highest_education_A Level or Equivalent', 'highest_education_HE Qualification',
        'highest_education_Lower Than A Level', 'highest_education_No Formal quals',
        'highest_education_Post Graduate Qualification',
        'studied_credits'
    ]

    # VALIDATION 3: Check that all expected columns exist.
    columnas_faltantes = [col for col in columnas_a_conservar if col not in df_final.columns]
    if columnas_faltantes:
        print(f"""[WARNING] Missing expected columns: {', '.join(columnas_faltantes)}""")
        print(f"""   Only available columns will be retained""")
        columnas_a_conservar = [col for col in columnas_a_conservar if col in df_final.columns]
    
    # Check that columns exist before dropping them.
    columnas_a_eliminar = [col for col in df_final.columns if col not in columnas_a_conservar]

    # Drop the columns.
    df_final = df_final.drop(columns=columnas_a_eliminar)

    # ERROR #7: Validate NaN values in critical columns.
    critical_cols = ['id_student', 'date', 'final_result']
    nan_en_criticas = []
    for col in critical_cols:
        if col in df_final.columns:
            nan_count = df_final[col].isna().sum()
            if nan_count > 0:
                nan_en_criticas.append((col, nan_count))
    
    if nan_en_criticas:
        print(f"""[WARNING] NaN values in critical columns:""")
        for col, count in nan_en_criticas:
            print(f"""   - {col}: {count} NaN values""")
        # Remove rows with NaN in critical columns.
        df_final = df_final.dropna(subset=critical_cols)
        print(f"""   → Rows with NaN removed. Remaining rows: {df_final.shape[0]}""")
    
    # [OK] ERROR #7: Report NaN values in all columns.
    nan_summary = df_final.isna().sum()
    if nan_summary.sum() > 0:
        print(f"""\n[WARNING] NaN values by column:""")
        for col, count in nan_summary[nan_summary > 0].items():
            print(f"""   - {col}: {count} NaN values ({count/len(df_final)*100:.1f}%)""")

    # LOG: Display a summary of the cleaning performed.
    print(f"""\n[OK] Preprocessing completed:""")
    print(f"""  - Rows processed: {df_final.shape[0]}""")
    print(f"""  - Columns retained: {df_final.shape[1]}""")
    print(f"""  - Columns removed: {len(columnas_a_eliminar)}""")
    print(f"""  - final_result distribution: {df_final['final_result'].value_counts().to_dict()}""")
    print(f"""  - highest_education columns present: {[c for c in df_final.columns if 'highest_education' in c]}""")
    print(f"""  - All retained columns: {list(df_final.columns)}\n""")

    return df_final
