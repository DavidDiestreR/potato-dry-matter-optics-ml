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
        
    Exemples
    --------
    Predicció per una sola mostra (diccionari):
    
    >>> features_dict = {
    ...     'color_promig_R': 123.45,
    ...     'color_promig_G': 98.76,
    ...     'color_promig_B': 87.65,
    ...     'desviació_R': 12.34,
    ...     'desviació_G': 10.23,
    ...     'desviació_B': 9.87,
    ...     'canal_NIR': 234.56
    ... }
    >>> ms_pred = predict_dm(features_dict, scaler, model)
    >>> print(f"Matèria seca predita: {ms_pred:.2f}%")
    
    Predicció per múltiples mostres (DataFrame):
    
    >>> df_test = pd.DataFrame({
    ...     'color_promig_R': [123.45, 130.21],
    ...     'color_promig_G': [98.76, 105.43],
    ...     'color_promig_B': [87.65, 92.11],
    ...     'desviació_R': [12.34, 11.89],
    ...     'desviació_G': [10.23, 9.75],
    ...     'desviació_B': [9.87, 10.12],
    ...     'canal_NIR': [234.56, 241.32]
    ... })
    >>> ms_preds = predict_dm(df_test, scaler, model)
    >>> print(f"Prediccions: {ms_preds}")
    
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


def predict_dm_batch(
    df: pd.DataFrame,
    scaler: Any,
    model: Any,
    output_csv: str = None
) -> pd.DataFrame:
    """
    Prediu la matèria seca per un lot de mostres i retorna/guarda els resultats.
    
    Paràmetres
    ----------
    df : pd.DataFrame
        DataFrame amb les característiques de les mostres
    scaler : sklearn.preprocessing.Scaler
        Scaler entrenat
    model : keras.Model
        Model entrenat
    output_csv : str, opcional
        Ruta on guardar els resultats. Si és None, no es guarda
        
    Retorna
    -------
    pd.DataFrame
        DataFrame original amb columna 'MS_predit' afegida
        
    Exemple
    -------
    >>> df_test = pd.read_csv('data/test_samples.csv')
    >>> df_resultats = predict_dm_batch(
    ...     df_test, scaler, model,
    ...     output_csv='data/prediccions.csv'
    ... )
    """
    # Fer còpia per no modificar l'original
    df_result = df.copy()
    
    # Fer prediccions
    prediccions = predict_dm(df, scaler, model)
    
    # Afegir prediccions al DataFrame
    df_result['MS_predit'] = prediccions
    
    # Guardar si s'especifica ruta
    if output_csv:
        df_result.to_csv(output_csv, index=False)
        print(f"✓ Prediccions guardades a '{output_csv}'")
    
    return df_result


# Exemple d'ús del mòdul
if __name__ == "__main__":
    """
    Exemple d'ús de les funcions de predicció.
    """
    
    # Rutes als fitxers del model
    MODEL_PATH = "data/output/test_run_2/model_prediccio_ms.h5"
    SCALER_PATH = "data/output/test_run_2/scaler_X.pkl"
    
    # Carregar model i scaler
    print("Carregant model i scaler...")
    model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)
    print("✓ Model i scaler carregats correctament\n")
    
    # EXEMPLE 1: Predicció per una sola mostra (diccionari)
    print("=" * 70)
    print("EXEMPLE 1: Predicció per una sola mostra (diccionari)")
    print("=" * 70)
    
    features_sample = {
        'color_promig_R': 123.45,
        'color_promig_G': 98.76,
        'color_promig_B': 87.65,
        'desviació_R': 12.34,
        'desviació_G': 10.23,
        'desviació_B': 9.87,
        'canal_NIR': 234.56
    }
    
    ms_pred = predict_dm(features_sample, scaler, model)
    print(f"Features: {features_sample}")
    print(f"\n→ Matèria seca predita: {ms_pred:.2f}%\n")
    
    # EXEMPLE 2: Predicció per múltiples mostres (DataFrame)
    print("=" * 70)
    print("EXEMPLE 2: Predicció per múltiples mostres (DataFrame)")
    print("=" * 70)
    
    df_test = pd.DataFrame({
        'id_mostra': ['PAT_001', 'PAT_002', 'PAT_003'],
        'color_promig_R': [123.45, 130.21, 115.67],
        'color_promig_G': [98.76, 105.43, 92.34],
        'color_promig_B': [87.65, 92.11, 83.21],
        'desviació_R': [12.34, 11.89, 13.45],
        'desviació_G': [10.23, 9.75, 11.02],
        'desviació_B': [9.87, 10.12, 9.34],
        'canal_NIR': [234.56, 241.32, 228.91]
    })
    
    df_resultats = predict_dm_batch(df_test, scaler, model)
    print("\nResultats:")
    print(df_resultats[['id_mostra', 'MS_predit']])
    print("\n" + "=" * 70)


def dry_matter_quality_classification(dm_value):
    """
    Classifica la qualitat de la patata en funció de la matèria seca.

    Paràmetres
    ----------
    dm_value : float
        Valor de matèria seca (% o la unitat que decidiu).
    """
    pass