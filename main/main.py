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
        report_filename: str, 
        predict:bool = True, 
        predict_filename:str = "test.csv"
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

    print("Entrainement terminé.")

    if predict :
        print("Prédiction sur les données de test...")
        from data.test_data import test_data
        # Données de test
        X_test = test_data("test", transformer)

        # Prédiction 
        y_test_pred = model.predict(X_test)

        # Sauvegarde vers un CSV
        import pandas as pd

        df_predictions = pd.DataFrame({
            "RowId": range(1, len(y_test_pred) + 1),
            "prediction": y_test_pred
        })

        df_predictions.to_csv(predict_filename, index=False)
        print(f"Prédictions sauvegardées dans le fichier : {predict_filename}")




if __name__ == "__main__":

    # model_type = "LogisticRegression"
    # train_split = 0.8
    # params = {
    #     'C': 81.66170040525944,
    #     'tol': 5.171910682373782e-05
    # }
    # generate_report = False
    # report_filename = "rapport_evaluation.html"

    model_type = "LinearSVC"
    train_split = 0.8
    params = {
        "C": 6.321627285922138,
        "tol": 0.0008058376899739417, 
        "max_iter": 3000, 
        "loss": "squared_hinge", 
        "dual": "auto"
    }
    generate_report = False
    report_filename = "rapport_evaluation.html"


    run_pipeline(model_type, train_split, params, generate_report, report_filename, predict=False, predict_filename="test.csv")