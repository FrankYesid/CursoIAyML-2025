"""
Funciones para la carga y preprocesamiento de datos del curso.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional

def load_dataset(filename: str) -> pd.DataFrame:
    """
    Carga un dataset desde la carpeta data.
    
    Args:
        filename (str): Nombre del archivo a cargar
        
    Returns:
        pd.DataFrame: Dataset cargado
    """
    return pd.read_csv(f"../data/{filename}")

def preprocess_churn_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesa el dataset de churn según lo visto en el Módulo 4.
    
    Args:
        df (pd.DataFrame): Dataset original
        
    Returns:
        pd.DataFrame: Dataset procesado
    """
    df_processed = df.copy()
    
    # Convertir fechas
    df_processed['Fecha'] = pd.to_datetime(df_processed['Fecha'])
    df_processed['Año'] = df_processed['Fecha'].dt.year
    df_processed['Mes'] = df_processed['Fecha'].dt.month
    df_processed['Día'] = df_processed['Fecha'].dt.day
    
    return df_processed

def split_features_target(df: pd.DataFrame, 
                         target_col: str,
                         test_size: float = 0.2,
                         random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide el dataset en features y target, y en conjuntos de entrenamiento y prueba.
    
    Args:
        df (pd.DataFrame): Dataset completo
        target_col (str): Nombre de la columna objetivo
        test_size (float): Proporción del conjunto de prueba
        random_state (int): Semilla aleatoria
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split
    
    X = df.drop(columns=target_col)
    y = df[target_col]
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state) 