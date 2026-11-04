import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack # type: ignore
import numpy as np

from utils.read_ds import read_ds
from utils.extract_text_and_features import extract_features_and_text


def test_data(file_path: str = "test", transformer=None):
    '''
    Charge les données de test, 
    nettoie, 
    extrait les caractéristiques et retourne la matrice X correspondante.

    Paramètres:
    - file_path: str : chemin vers le fichier CSV contenant les données de test.
    - transformer: object : transformateur utilisé pour encoder les données de test (doit être le même que celui utilisé pour l'entraînement).

    Retourne:
    - X_test : matrice des caractéristiques de test
    '''
    X_test_df = read_ds(file_path)

    test_navigateur = X_test_df[["navigateur"]]
    test_actions_df = X_test_df.drop(columns=["navigateur"])
    scaler = transformer['scaler']


    t_ecrans_te, t_mods_te, t_acts_te, num_feats_te = extract_features_and_text(test_actions_df)

    enc_nav = transformer['enc_nav']
    enc_ecrans = transformer['tfidf_ecrans']
    enc_mods = transformer['tfidf_modules']
    enc_acts = transformer['tfidf_actions']

    X_nav_test = enc_nav.transform(test_navigateur)
    X_ecrans_test = enc_ecrans.transform(t_ecrans_te)
    X_mods_test = enc_mods.transform(t_mods_te)
    X_num_test = scaler.transform(num_feats_te)
    X_actions_test = enc_acts.transform(t_acts_te)

    X_test = hstack([X_ecrans_test, X_mods_test, X_actions_test, X_num_test, X_nav_test])

    return X_test




