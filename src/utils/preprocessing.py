"""
Utilidades para el preprocesamiento de datos.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Any

def encode_categorical_columns(df: pd.DataFrame,
                             columns: List[str]) -> pd.DataFrame:
    """
    Codifica variables categóricas usando LabelEncoder.
    
    Args:
        df (pd.DataFrame): Dataset
        columns (List[str]): Lista de columnas a codificar
        
    Returns:
        pd.DataFrame: Dataset con columnas codificadas
    """
    df_encoded = df.copy()
    encoders = {}
    
    for col in columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = le
        
    return df_encoded, encoders

def handle_missing_values(df: pd.DataFrame,
                         numeric_strategy: str = 'mean',
                         categorical_strategy: str = 'mode') -> pd.DataFrame:
    """
    Maneja valores faltantes en el dataset.
    
    Args:
        df (pd.DataFrame): Dataset
        numeric_strategy (str): Estrategia para variables numéricas ('mean', 'median', 'zero')
        categorical_strategy (str): Estrategia para variables categóricas ('mode', 'unknown')
        
    Returns:
        pd.DataFrame: Dataset sin valores faltantes
    """
    df_clean = df.copy()
    
    # Manejo de variables numéricas
    numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            if numeric_strategy == 'mean':
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            elif numeric_strategy == 'median':
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            elif numeric_strategy == 'zero':
                df_clean[col].fillna(0, inplace=True)
    
    # Manejo de variables categóricas
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().any():
            if categorical_strategy == 'mode':
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
            elif categorical_strategy == 'unknown':
                df_clean[col].fillna('unknown', inplace=True)
    
    return df_clean

def create_date_features(df: pd.DataFrame,
                        date_column: str) -> pd.DataFrame:
    """
    Crea features a partir de una columna de fecha.
    
    Args:
        df (pd.DataFrame): Dataset
        date_column (str): Nombre de la columna de fecha
        
    Returns:
        pd.DataFrame: Dataset con nuevas features de fecha
    """
    df_dates = df.copy()
    df_dates[date_column] = pd.to_datetime(df_dates[date_column])
    
    # Extraer componentes de fecha
    df_dates[f'{date_column}_year'] = df_dates[date_column].dt.year
    df_dates[f'{date_column}_month'] = df_dates[date_column].dt.month
    df_dates[f'{date_column}_day'] = df_dates[date_column].dt.day
    df_dates[f'{date_column}_dayofweek'] = df_dates[date_column].dt.dayofweek
    
    return df_dates 