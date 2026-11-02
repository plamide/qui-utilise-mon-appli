import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import optuna
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from data.train_data import train_data

# 1. Chargement des données brutes
X, y, _ = train_data("train")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Définition de la fonction objectif à maximiser
def objective(trial):
    # Espace de recherche des hyperparamètres
    c_val = trial.suggest_float("C", 1e-2, 1e2, log=True)
    tol = trial.suggest_float("tol", 1e-5, 1e-3, log=True)
    
    # Modèle avec les hyperparamètres du trial
    model = LogisticRegression(
        C=c_val,
        tol=tol,
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=42
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
    
    print(" Lancement de l'optimisation Optuna (30 essais)...")
    study.optimize(objective, n_trials=30, timeout=600)

    print("\n" + "=" * 50)
    print(f" Meilleur F1-Score Macro : {study.best_value:.4f}")
    print(" Meilleurs hyperparamètres :")
    for key, val in study.best_params.items():
        print(f"   - {key}: {val}")
    print("=" * 50)

    # 3. Visualisation intégrée d'Optuna (pour votre rapport de BE !)
    # fig = optuna.visualization.plot_optimization_history(study)
    # fig.write_html("optuna_history.html")