"""Shared project configuration.

Centralize paths, local AI server parameters, and lazy dataset loading
so other modules do not duplicate constants or disk reads.
"""

from pathlib import Path

import pandas as pd


# Resolve all paths relative to the project directory so the app
# works even when launched from a different working directory.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TEMP_DATA_PATH = DATA_DIR / "temp_data.csv"
CLEAN_DATA_PATH = BASE_DIR / "dataset_limpio.csv"

# OpenAI-compatible API endpoint exposed by the local server.
LLAMA_SERVER_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL_NAME = "qwen2.5-1.5b-instruct"

# In-memory caches avoid reloading CSV files on every call.
df = None
df_limpio = None


def load_temp_data():
    """Load the original temporary dataset from disk."""
    global df
    if df is None:
        df = pd.read_csv(TEMP_DATA_PATH, sep=";")
    return df


def load_clean_data():
    """Load the clean dataset used by the analyses."""
    global df_limpio
    if df_limpio is None:
        df_limpio = pd.read_csv(CLEAN_DATA_PATH, sep=";")
    return df_limpio
