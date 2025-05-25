"""
Implementación de modelos supervisados utilizados en el curso.
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

class ChurnPredictor:
    """
    Modelo de predicción de churn basado en árboles de decisión.
    """
    
    def __init__(self, max_depth: int = 5, random_state: int = 42):
        """
        Inicializa el modelo.
        
        Args:
            max_depth (int): Profundidad máxima del árbol
            random_state (int): Semilla aleatoria
        """
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=random_state
        )
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Entrena el modelo.
        
        Args:
            X (pd.DataFrame): Features de entrenamiento
            y (pd.Series): Target de entrenamiento
        """
        self.model.fit(X, y)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Realiza predicciones.
        
        Args:
            X (pd.DataFrame): Features para predicción
            
        Returns:
            np.ndarray: Predicciones
        """
        return self.model.predict(X)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Evalúa el modelo.
        
        Args:
            X (pd.DataFrame): Features de prueba
            y (pd.Series): Target de prueba
            
        Returns:
            Dict[str, Any]: Métricas de evaluación
        """
        y_pred = self.predict(X)
        
        return {
            'confusion_matrix': confusion_matrix(y, y_pred),
            'classification_report': classification_report(y, y_pred)
        } 