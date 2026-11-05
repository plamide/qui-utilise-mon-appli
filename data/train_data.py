import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack # type: ignore
import numpy as np


def train_data(file_path: str = "train", train_split: float = 0.8, random_state: int = 42):
    '''
    Charge les données d'entraînement, 
    effectue le split train/test,
    nettoie, 
    extrait les caractéristiques et retourne les matrices X et vecteurs y correspondants.

    Paramètres:
    - file_path: str : chemin vers le fichier CSV contenant les données d'entraînement.
    - train_split: float : proportion du jeu de données à allouer au test (ex: 0.2 pour 20%).

    Retourne:
    - X_train, X_test, y_train, y_test 
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

    # Chargement global des données brutes
    df_train_full = read_ds(file_path)

    y_full = df_train_full["util"].values
    X_df_full = df_train_full.drop(columns=["util"])

    # 1. SPLIT DES DONNÉES BRUTES AVANT TOUT ENCODAGE 
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df_full, y_full, test_size=train_split, random_state=random_state, stratify=y_full
    )

    def extract_features_and_text(df_act: pd.DataFrame):
        p_time   = re.compile(r"^t(\d{1,6})$")
        p_screen = re.compile(r"\(([^)]+)\)")
        p_module = re.compile(r"\$(\w+)\$")
        p_noise  = re.compile(r"\([^)]*\)|\$[^$]*\$|<[^>]*>")

        text_ecrans  = []
        text_modules = []
        text_actions = []
        numeric_features = []

        for row in df_act.values:
            t_max = 0
            num_actions = 0
            ecrans, modules, actions = [], [], []

            for val in row:
                if pd.isna(val):
                    continue
                val_str = str(val).strip()

                m_time = p_time.match(val_str)
                if m_time:
                    val_t = int(m_time.group(1))
                    if val_t > t_max:
                        t_max = val_t
                    continue

                num_actions += 1

                m_screen = p_screen.search(val_str)
                if m_screen:
                    ecrans.append(m_screen.group(1).replace(" ", "_"))

                for m_mod in p_module.finditer(val_str):
                    modules.append(m_mod.group(1))

                action_clean = p_noise.sub("", val_str).strip().replace(" ", "_")
                if action_clean:
                    actions.append(action_clean)

            text_ecrans.append(" ".join(ecrans))
            text_modules.append(" ".join(modules))
            text_actions.append(" ".join(actions))
            cadence = (num_actions / (t_max + 1)) * 60
            numeric_features.append([t_max, num_actions, cadence])

        return text_ecrans, text_modules, text_actions, np.array(numeric_features)

    # 2. Extraction du texte et des features numériques pour Train et Test séparément
    # Navigation
    train_navigateur = X_train_df[["navigateur"]]
    test_navigateur = X_test_df[["navigateur"]]

    # Actions
    train_actions_df = X_train_df.drop(columns=["navigateur"])
    test_actions_df = X_test_df.drop(columns=["navigateur"])

    t_ecrans_tr, t_mods_tr, t_acts_tr, num_feats_tr = extract_features_and_text(train_actions_df)
    t_ecrans_te, t_mods_te, t_acts_te, num_feats_te = extract_features_and_text(test_actions_df)

    # 3. ENCODAGE & AJUSTEMENT STRICTEMENT SUR LE TRAIN (puis application sur le test)
    
    # OneHotEncoder Navigateur
    enc_nav = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    nav_encoded_train = enc_nav.fit_transform(train_navigateur)
    nav_encoded_test = enc_nav.transform(test_navigateur)

    # TF-IDF Ecrans
    tfidf_ecrans = TfidfVectorizer(max_features=500, sublinear_tf=True, ngram_range=(1, 1))
    X_ecrans_train = tfidf_ecrans.fit_transform(t_ecrans_tr)
    X_ecrans_test = tfidf_ecrans.transform(t_ecrans_te)

    # TF-IDF Modules
    tfidf_modules = TfidfVectorizer(max_features=100, sublinear_tf=True, ngram_range=(1, 1))
    X_modules_train = tfidf_modules.fit_transform(t_mods_tr)
    X_modules_test = tfidf_modules.transform(t_mods_te)

    # TF-IDF Actions
    tfidf_actions = TfidfVectorizer(max_features=2000, sublinear_tf=False, ngram_range=(2, 3))
    X_actions_train = tfidf_actions.fit_transform(t_acts_tr)
    X_actions_test = tfidf_actions.transform(t_acts_te)

    # Standardisation
    scaler = StandardScaler()
    X_num_train = scaler.fit_transform(num_feats_tr)
    X_num_test = scaler.transform(num_feats_te)

    # 4. Assemblage final (hstack)
    X_train_final = hstack([X_ecrans_train, X_modules_train, X_actions_train, X_num_train, nav_encoded_train])
    X_test_final = hstack([X_ecrans_test, X_modules_test, X_actions_test, X_num_test, nav_encoded_test])

    transformers = {
        "tfidf_ecrans":  tfidf_ecrans,
        "tfidf_modules": tfidf_modules,
        "tfidf_actions": tfidf_actions,
        "scaler":        scaler,
        "enc_nav":       enc_nav
    }

    return X_train_final, X_test_final, y_train, y_test, transformers

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, transformers = train_data("train", train_split=0.8)
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
