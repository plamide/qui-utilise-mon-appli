import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import scipy.sparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from utils.reporter import generate_html_report

def train_model(
    X: scipy.sparse.csr_matrix, 
    y: np.ndarray, 
    params: dict,
    model_type: str = "LogisticRegression", 
    train_split: float = 0.8,
    generate_report: bool = False, 
    report_filename: str = "rapport_evaluation.html"   
    ):    
    '''

    Entraîne un modèle de machine learning sur les données d'entraînement fournies. 
    '''
    print('Début de l\'entraînement du modèle...')
    X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=1-train_split, random_state=42, stratify=y
)

    if model_type == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(
            **params,                     
            class_weight='balanced',   
            max_iter=1000,
            solver='lbfgs',
            random_state=42
        )
    # elif model_type == "LogisticRegressionCV":
    #     from sklearn.linear_model import LogisticRegressionCV
    #     model = LogisticRegressionCV(
    #         Cs=[0.1, 1.0, 10.0],       
    #         cv=3,
    #         max_iter=1000,
    #         n_jobs=1,                  
    #         scoring='f1_macro',
    #         class_weight='balanced',
    #         random_state=42
    #     )
    else:
        raise ValueError(f"Modèle non supporté : {model_type}")

    print(f"Entraînement du modèle {model_type} en cours...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    report_dict = classification_report(y_val, y_pred, output_dict=True, zero_division=0)

    if generate_report:
        generate_html_report(report_dict, output_filename=report_filename)
    acc = accuracy_score(y_val, y_pred)
    macro_f1 = report_dict["macro avg"]["f1-score"]
    print("=" * 45)
    print(f" Modèle            : {model_type}")
    print(f" Précision globale : {acc * 100:.2f} %")
    print(f" F1-Score Macro    : {macro_f1:.4f}")
    print(f" Rapport sauvegardé: {report_filename}")
    print("=" * 45)

    return model, classification_report(y_val, y_pred, output_dict=True)

if __name__ == "__main__":
    import data.train_data as td 

    X, y, transformer = td.train_data("train")

    model, report = train_model(X, 
                                y, 
                                model_type="LogisticRegression", 
                                train_split=0.8,
                                generate_report=True,
                                report_filename="rapport_evaluation.html")