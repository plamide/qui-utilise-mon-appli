import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.train_data import train_data
from model.train_model import train_model

def run_pipeline(
        model_type: str, 
        train_split: float, 
        params:dict, 
        generate_report: bool, 
        report_filename: str
        ):
    """
    Exécute le pipeline complet : préparation des données, entraînement du modèle et génération du rapport.
    """
    # Préparation des données
    X_train, X_val, y_train, y_val, transformer = train_data("train")

    # Entraînement du modèle et génération du rapport
    model, report = train_model(
        X_train, 
        y_train, 
        X_val,
        y_val,
        params,
        model_type=model_type, 
        generate_report=generate_report,
        report_filename=report_filename
    )

if __name__ == "__main__":

    model_type = "LogisticRegression"
    train_split = 0.8
    params = {
        'C': 81.66170040525944,
        'tol': 5.171910682373782e-05
    }
    generate_report = False
    report_filename = "rapport_evaluation.html"


    run_pipeline(model_type, train_split, params, generate_report, report_filename)