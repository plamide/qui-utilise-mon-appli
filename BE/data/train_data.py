import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.sparse import hstack # type: ignore
import numpy as np


def train_data(file_path: str = "train"):
    '''
    Charge les données d'entraînement, nettoie, extrait les caractéristiques
    et retourne la matrice X prête pour le modèle et le vecteur y.

    Paramètres:
    - file_path: str : chemin vers le fichier CSV contenant les données d'entraînement.

    Retourne:
    - X: scipy.sparse.csr.csr_matrix : matrice des caractéristiques 
    '''

    def read_ds(ds_name: str):
        path = ds_name if ds_name.endswith('.csv') else f'data/{ds_name}.csv'
        with open(path) as f:
            lines = f.readlines()
            max_actions = max(len(str(c).split(",")) for c in lines)
            _names = ["util", "navigateur"] if "train" in path else ["navigateur"]
            _names.extend(range(max_actions - len(_names)))
            from io import StringIO
            return pd.read_csv(StringIO("".join(lines)), names=_names, dtype=str)

    #Chargement des données 
    df_train = read_ds(file_path)

    y = df_train["util"].values
    X_df = df_train.drop(columns=["util"])

    def clean_data(df: pd.DataFrame):
        # Navigateur
        df_navigateur = df[["navigateur"]]
        # Actions
        df_actions = df.drop(columns=["navigateur"])

        def encode_navigateur(df_nav: pd.DataFrame):
            '''Encode la colonne navigateur en One-Hot.'''
            enc = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
            navigateur_encoded = enc.fit_transform(df_nav)
            return navigateur_encoded, enc

        p_time = re.compile(r"^t(\d{1,6})$")

        def extract_features_and_text(df_act: pd.DataFrame):
            '''Extrait le texte des sessions et les features numériques.'''
            text_sessions = []
            numeric_features = []

            for row in df_act.values:
                t_max = 0
                num_actions = 0
                actions = []

                for val in row:
                    if pd.isna(val):
                        continue
                    val_str = str(val).strip()
                    m = p_time.match(val_str)
                    if m:
                        val_t = int(m.group(1))
                        if val_t > t_max:  # noqa: PLR1730
                            t_max = val_t
                    else:
                        num_actions += 1
                        actions.append(val_str.replace(" ", "_"))

                text_sessions.append(" ".join(actions))
                cadence = (num_actions / (t_max + 1)) * 60
                numeric_features.append([t_max, num_actions, cadence])

            return text_sessions, np.array(numeric_features)

        # Encodage
        nav_encoded, enc_nav = encode_navigateur(df_navigateur)

        #Texte et features numériques
        text_sessions, num_feats = extract_features_and_text(df_actions)

        # Vectorisation TF-IDF sur les actions
        tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
        X_tfidf = tfidf.fit_transform(text_sessions)

        # Standardisation des variables continues (durée, nb actions, cadence)
        scaler = StandardScaler()
        X_num = scaler.fit_transform(num_feats)

        X_final = hstack([X_tfidf, X_num, nav_encoded])

        # Transformers 
        transformers = {
            "tfidf": tfidf,
            "scaler": scaler,
            "enc_nav": enc_nav
        }

        return X_final, transformers

    X, transformers = clean_data(X_df)

    return X, y, transformers

if __name__ == "__main__":
    X, y, transformers = train_data("train")
    print(f"X shape: {X.shape}, y shape: {y.shape}")