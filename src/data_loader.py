"""
data_loader.py

Responsible for:
- Loading raw data from file (CSV)
- Returning a DataFrame/Matrix
- Handling file-related errors
"""

import pandas as pd
from pathlib import Path

def load_data(filepath: str):
    """
    Loads raw data from a CSV file.

    Parameters:
        filepath: Path to the CSV file

    Returns:
        pd.DataFrame: Loaded data

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file is empty or invalid
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            raise ValueError(f"The data file is empty: {filepath}")
        return df
    except Exception as e:
        raise ValueError(f"Error loading data from {filepath}: {str(e)}")

def validate_data(df: pd.DataFrame, required_columns: list = None):
    """
    Performs basic validation on the loaded DataFrame.
    """
    if df is None or df.empty:
        return False
    if required_columns:
        return all(col in df.columns for col in required_columns)
    return True
