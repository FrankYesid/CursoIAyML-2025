"""
Funciones de visualización para el análisis de datos.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional

def plot_target_distribution(df: pd.DataFrame, 
                           target_col: str,
                           title: Optional[str] = None,
                           figsize: tuple = (10, 6)) -> None:
    """
    Visualiza la distribución de la variable objetivo.
    
    Args:
        df (pd.DataFrame): Dataset
        target_col (str): Nombre de la columna objetivo
        title (str, optional): Título del gráfico
        figsize (tuple): Tamaño de la figura
    """
    plt.figure(figsize=figsize)
    sns.countplot(data=df, x=target_col)
    plt.title(title or f"Distribución de {target_col}")
    plt.show()

def plot_feature_importance(feature_names: list,
                          importance_values: list,
                          title: str = "Importancia de Features",
                          figsize: tuple = (12, 6)) -> None:
    """
    Visualiza la importancia de las features en un modelo.
    
    Args:
        feature_names (list): Nombres de las features
        importance_values (list): Valores de importancia
        title (str): Título del gráfico
        figsize (tuple): Tamaño de la figura
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_values
    }).sort_values('importance', ascending=True)
    
    plt.figure(figsize=figsize)
    sns.barplot(data=importance_df, y='feature', x='importance')
    plt.title(title)
    plt.show()

def plot_confusion_matrix(cm: pd.DataFrame,
                         labels: list,
                         title: str = "Matriz de Confusión",
                         figsize: tuple = (8, 6)) -> None:
    """
    Visualiza una matriz de confusión.
    
    Args:
        cm (pd.DataFrame): Matriz de confusión
        labels (list): Etiquetas de las clases
        title (str): Título del gráfico
        figsize (tuple): Tamaño de la figura
    """
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('Real')
    plt.xlabel('Predicción')
    plt.show() 