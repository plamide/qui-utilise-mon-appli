import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import optuna
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from data.train_data import train_data

# Chargement des données brutes
X_train, X_val, y_train, y_val, _ = train_data("train")

# Fonction objectif à maximiser
def objective(trial):
    # Élargissement des fenêtres d'optimisation et des paramètres
    c_val = trial.suggest_float("C", 1.0, 100.0, log=True)
    tol = trial.suggest_float("tol", 1e-6, 1e-3, log=True)
    # solver = trial.suggest_categorical("solver", ["lbfgs", "saga"])
    # max_iter = trial.suggest_int("max_iter", 500, 2000, step=100)
    loss = "squared_hinge"
    max_iter = trial.suggest_int("max_iter", 2000, 4000, step=500)
    
    # model = LogisticRegression(
    #     C=c_val,
    #     tol=tol,
    #     solver=solver,
    #     max_iter=max_iter,
    #     class_weight="balanced",
    #     random_state=42
    # )

    model = LinearSVC(
        C=c_val,
        tol=tol,
        loss=loss,
        max_iter=max_iter,
        class_weight="balanced",
        random_state=42,
        dual = "auto"
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    # Métrique cible à maximiser
    score_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    return score_f1


if __name__ == "__main__":
    # Désactiver les logs verbeux d'Optuna
    optuna.logging.set_verbosity(optuna.logging.INFO)
    
    # Création de l'étude (direction : maximiser le F1)
    study = optuna.create_study(direction="maximize", study_name="Copilote_F1_Optimization")
    
    print(" Lancement de l'optimisation Optuna élargie (30 essais)...")
    study.optimize(objective, n_trials=30, timeout=900)

    print("\n" + "=" * 50)
    print(f" Meilleur F1-Score Macro : {study.best_value:.4f}")
    print(" Meilleurs hyperparamètres :")
    for key, val in study.best_params.items():
        print(f"   - {key}: {val}")
    print("=" * 50)

    # 1. Visualisation de l'historique d'optimisation
    fig_history = optuna.visualization.plot_optimization_history(study)
    fig_history.write_html("optuna_history.html")
    print(" Rapport d'historique sauvegardé : optuna_history.html")

    # 2. Visualisation de l'importance des hyperparamètres
    try:
        fig_importances = optuna.visualization.plot_param_importances(study)
        fig_importances.write_html("optuna_importances.html")
        print(" Rapport d'importance sauvegardé : optuna_importances.html")
    except Exception as e:
        print(f"Impossible de générer le graphe d'importance (nombre d'essais potentiellement insuffisant) : {e}")
