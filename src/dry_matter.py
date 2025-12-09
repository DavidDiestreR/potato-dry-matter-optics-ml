"""
Mòdul: dry_matter
Funcions relacionades amb la predicció i classificació de la matèria seca.
"""

import numpy as np
import pandas as pd
import pickle
from tensorflow import keras
from typing import Union, Dict, Any


def mape_metric(y_true, y_pred):
    """
    Mètrica MAPE personalitzada per Keras (necessària per carregar el model).
    
    Paràmetres
    ----------
    y_true : tensor
        Valors reals
    y_pred : tensor
        Valors predits
        
    Retorna
    -------
    tensor
        MAPE en percentatge
    """
    import tensorflow as tf
    diff = tf.abs((y_true - y_pred) / tf.clip_by_value(tf.abs(y_true), 1e-7, 1e10))
    return 100. * tf.reduce_mean(diff)


def load_model_and_scaler(model_path: str, scaler_path: str) -> tuple:
    """
    Carrega el model entrenat i el scaler des dels fitxers guardats.
    
    Paràmetres
    ----------
    model_path : str
        Ruta al fitxer .h5 del model
    scaler_path : str
        Ruta al fitxer .pkl del scaler
        
    Retorna
    -------
    tuple
        (model, scaler) carregats
        
    Exemple
    -------
    >>> model, scaler = load_model_and_scaler(
    ...     'data/output/model_prediccio_ms.h5',
    ...     'data/output/scaler_X.pkl'
    ... )
    """
    # Carregar model amb la mètrica personalitzada
    model = keras.models.load_model(
        model_path, 
        custom_objects={'mape_metric': mape_metric}
    )
    
    # Carregar scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler


def predict_dm(
    features: Union[pd.DataFrame, Dict[str, float], np.ndarray],
    scaler: Any,
    model: Any
) -> Union[float, np.ndarray]:
    """
    Prediu la matèria seca (MS) a partir d'un vector de característiques.

    Paràmetres
    ----------
    features : pd.DataFrame, dict o np.ndarray
        Característiques d'entrada. Pot ser:
        - DataFrame amb columnes: ['color_promig_R', 'color_promig_G', 'color_promig_B',
                                   'desviació_R', 'desviació_G', 'desviació_B', 'canal_NIR']
        - Diccionari amb les mateixes claus
        - Array numpy amb forma (n_samples, 7) o (7,) per una sola mostra
    scaler : sklearn.preprocessing.Scaler
        Objecte scaler (StandardScaler o MinMaxScaler) entrenat per normalitzar les features
    model : keras.Model
        Model de xarxa neuronal entrenat per fer la predicció

    Retorna
    -------
    float o np.ndarray
        Predicció de matèria seca en percentatge (%).
        - Si l'entrada és una sola mostra (dict o array 1D): retorna float
        - Si l'entrada són múltiples mostres (DataFrame o array 2D): retorna np.ndarray
    
    Notes
    -----
    - Les features d'entrada han d'estar en el mateix ordre i unitats que les 
      utilitzades durant l'entrenament del model
    - El scaler normalitza automàticament les features abans de la predicció
    - Els valors de matèria seca predits són en percentatge (%)
    """
    
    # Columnes esperades en l'ordre correcte
    feature_cols = [
        'color_promig_R', 'color_promig_G', 'color_promig_B',
        'desviació_R', 'desviació_G', 'desviació_B', 'canal_NIR'
    ]
    
    # Convertir l'entrada a numpy array
    if isinstance(features, dict):
        # Cas 1: Diccionari (una sola mostra)
        X = np.array([[features[col] for col in feature_cols]])
        single_sample = True
        
    elif isinstance(features, pd.DataFrame):
        # Cas 2: DataFrame (una o múltiples mostres)
        # Verificar que té totes les columnes necessàries
        missing_cols = set(feature_cols) - set(features.columns)
        if missing_cols:
            raise ValueError(
                f"El DataFrame no conté les columnes: {missing_cols}\n"
                f"Columnes necessàries: {feature_cols}"
            )
        X = features[feature_cols].values
        single_sample = (len(features) == 1)
        
    elif isinstance(features, np.ndarray):
        # Cas 3: Array numpy
        if features.ndim == 1:
            # Array 1D (una sola mostra)
            if len(features) != 7:
                raise ValueError(
                    f"L'array ha de tenir 7 features, però en té {len(features)}"
                )
            X = features.reshape(1, -1)
            single_sample = True
        elif features.ndim == 2:
            # Array 2D (múltiples mostres)
            if features.shape[1] != 7:
                raise ValueError(
                    f"L'array ha de tenir 7 columnes (features), però en té {features.shape[1]}"
                )
            X = features
            single_sample = (features.shape[0] == 1)
        else:
            raise ValueError(
                f"L'array ha de ser 1D o 2D, però té {features.ndim} dimensions"
            )
    else:
        raise TypeError(
            "El paràmetre 'features' ha de ser un DataFrame, dict o np.ndarray"
        )
    
    # Normalitzar les features amb el scaler
    X_scaled = scaler.transform(X)
    
    # Fer la predicció
    y_pred = model.predict(X_scaled, verbose=0).flatten()
    
    # Retornar float si és una sola mostra, array si són múltiples
    if single_sample:
        return float(y_pred[0])
    else:
        return y_pred


def dry_matter_quality_classification(dm_value: float) -> str:
    """
    Classifica la qualitat de la patata en funció de la matèria seca.

    Paràmetres
    ----------
    dm_value : float
        Valor de matèria seca (%).

    Retorna
    -------
    str
        Categoria de qualitat:
        - "descartada" si dm_value < 19
        - "preu rebaixat" si 19 <= dm_value <= 20
        - "bona" si dm_value > 20
    """
    if dm_value < 19:
        return "descartada"
    elif 19 <= dm_value <= 20:
        return "preu rebaixat"
    else:
        return "bona"