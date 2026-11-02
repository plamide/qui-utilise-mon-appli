import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def generate_html_report(report_dict: dict, output_filename: str = "rapport_evaluation.html"):
    """
    Génère un rapport HTML riche et interactif pour l'évaluation multi-classes.
    """
    df = pd.DataFrame(report_dict).T
    
    # 1. Extraction des KPI globaux
    macro_f1 = df.loc["macro avg", "f1-score"]
    accuracy = report_dict.get("accuracy", df.loc["accuracy", "precision"] if "accuracy" in df.index else 0.0)
    
    # 2. DataFrame par utilisateur
    df_users = df.drop(index=["accuracy", "macro avg", "weighted avg"], errors="ignore").reset_index()
    df_users.rename(columns={"index": "utilisateur"}, inplace=True)
    df_users["f1-score"] = pd.to_numeric(df_users["f1-score"], errors="coerce").fillna(0)
    df_users["precision"] = pd.to_numeric(df_users["precision"], errors="coerce").fillna(0)
    df_users["recall"] = pd.to_numeric(df_users["recall"], errors="coerce").fillna(0)
    df_users["support"] = pd.to_numeric(df_users["support"], errors="coerce").fillna(0)

    # -------------------------------------------------------------
    # GRAPHIQUE 1 : Donut Chart par Tranches de Performance
    # -------------------------------------------------------------
    def get_tier(val):
        if val >= 0.8: return "🟢 Excellent (F1 ≥ 0.8)"
        elif val >= 0.5: return "🟡 Bon (0.5 ≤ F1 < 0.8)"
        elif val > 0.0: return "🟠 Faible (0 < F1 < 0.5)"
        else: return "🔴 Non reconnu (F1 = 0)"

    df_users["Tranche"] = df_users["f1-score"].apply(get_tier)
    tier_counts = df_users["Tranche"].value_counts().reset_index()
    tier_counts.columns = ["Tranche", "Nombre"]

    fig_donut = px.pie(
        tier_counts, 
        names="Tranche", 
        values="Nombre", 
        hole=0.45,
        title="Répartition de la Performance sur les 247 Utilisateurs",
        color="Tranche",
        color_discrete_map={
            "🟢 Excellent (F1 ≥ 0.8)": "#00CC96",
            "🟡 Bon (0.5 ≤ F1 < 0.8)": "#636EFA",
            "🟠 Faible (0 < F1 < 0.5)": "#FFA15A",
            "🔴 Non reconnu (F1 = 0)": "#EF553B"
        }
    )
    fig_donut.update_traces(textinfo="percent+value", pull=[0.02, 0.02, 0.02, 0.05])
    fig_donut.update_layout(template="plotly_white", height=420)

    # -------------------------------------------------------------
    # GRAPHIQUE 2 : Boxplots comparatifs (Précision, Rappel, F1)
    # -------------------------------------------------------------
    fig_box = go.Figure()
    fig_box.add_trace(go.Box(y=df_users["precision"], name="Précision", marker_color="#636EFA", boxpoints="outliers"))
    fig_box.add_trace(go.Box(y=df_users["recall"], name="Rappel", marker_color="#00CC96", boxpoints="outliers"))
    fig_box.add_trace(go.Box(y=df_users["f1-score"], name="F1-Score", marker_color="#AB63FA", boxpoints="outliers"))
    fig_box.update_layout(
        title="Dispersion des Métriques (Médianes et Quartiles sur 247 classes)",
        yaxis_title="Score (0 à 1)",
        template="plotly_white",
        height=400
    )

    # -------------------------------------------------------------
    # GRAPHIQUE 3 : Nuage de points F1-Score vs Support
    # -------------------------------------------------------------
    fig_scatter = px.scatter(
        df_users, 
        x="support", 
        y="f1-score", 
        hover_name="utilisateur",
        size="support",
        color="f1-score",
        color_continuous_scale="Viridis",
        title="F1-Score en fonction du Nombre de Sessions en Test (Support)",
        labels={"support": "Nombre de sessions en validation", "f1-score": "F1-Score"}
    )
    fig_scatter.update_layout(template="plotly_white", height=450)

    # -------------------------------------------------------------
    # GRAPHIQUE 4 : Top 10 des Meilleurs vs Top 10 des Plus Difficiles
    # -------------------------------------------------------------
    top10 = df_users.nlargest(10, ["f1-score", "support"])
    flop10 = df_users.nsmallest(10, ["f1-score", "support"])

    fig_top_flop = make_subplots(rows=1, cols=2, subplot_titles=("Top 10 : Utilisateurs les mieux reconnus", "Top 10 : Utilisateurs les plus difficiles"))
    
    fig_top_flop.add_trace(
        go.Bar(x=top10["f1-score"], y=top10["utilisateur"], orientation="h", marker_color="#00CC96", name="Top 10"),
        row=1, col=1
    )
    fig_top_flop.add_trace(
        go.Bar(x=flop10["f1-score"], y=flop10["utilisateur"], orientation="h", marker_color="#EF553B", name="Flop 10"),
        row=1, col=2
    )
    fig_top_flop.update_layout(template="plotly_white", showlegend=False, height=420)
    fig_top_flop.update_xaxes(range=[0, 1.05], title="F1-Score")

    # -------------------------------------------------------------
    # ASSEMBLAGE DE LA PAGE HTML AUTONOME
    # -------------------------------------------------------------
    html_content = f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <title>Rapport d'Évaluation du Modèle - Identification Copilote</title>
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 30px; background-color: #f8f9fa; color: #202124; }}
            .container {{ max-width: 1150px; margin: 0 auto; background: white; padding: 35px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.06); }}
            h1 {{ color: #1a73e8; margin-top: 0; border-bottom: 2px solid #e8eaed; padding-bottom: 12px; }}
            .kpi-container {{ display: flex; gap: 15px; margin: 25px 0 35px 0; }}
            .kpi-card {{ flex: 1; padding: 18px; background: #f1f3f4; border-radius: 8px; text-align: center; border-left: 4px solid #1a73e8; }}
            .kpi-title {{ font-size: 13px; color: #5f6368; text-transform: uppercase; font-weight: 600; }}
            .kpi-value {{ font-size: 30px; font-weight: bold; color: #202124; margin-top: 6px; }}
            .chart-card {{ background: white; padding: 15px; margin-bottom: 30px; border-radius: 8px; border: 1px solid #e8eaed; min-height: 440px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Rapport de Classification des Utilisateurs Copilote</h1>
            
            <div class="kpi-container">
                <div class="kpi-card">
                    <div class="kpi-title">Accuracy Globale</div>
                    <div class="kpi-value">{accuracy * 100:.2f} %</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">F1-Score Macro</div>
                    <div class="kpi-value">{macro_f1:.4f}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">F1 ≥ 0.8 (Excellents)</div>
                    <div class="kpi-value" style="color: #00875a;">{sum(df_users['f1-score'] >= 0.8)} <span style="font-size:16px; color:#5f6368;">({sum(df_users['f1-score'] >= 0.8)/len(df_users)*100:.0f}%)</span></div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">F1 = 0.0 (À améliorer)</div>
                    <div class="kpi-value" style="color: #d93025;">{sum(df_users['f1-score'] == 0.0)} <span style="font-size:16px; color:#5f6368;">({sum(df_users['f1-score'] == 0.0)/len(df_users)*100:.0f}%)</span></div>
                </div>
            </div>

            <div class="chart-card">
                {fig_donut.to_html(full_html=False, include_plotlyjs=False)}
            </div>

            <div class="chart-card">
                {fig_box.to_html(full_html=False, include_plotlyjs=False)}
            </div>
            
            <div class="chart-card">
                {fig_scatter.to_html(full_html=False, include_plotlyjs=False)}
            </div>

            <div class="chart-card">
                {fig_top_flop.to_html(full_html=False, include_plotlyjs=False)}
            </div>
        </div>
    </body>
    </html>
    """

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f" Nouveau rapport généré avec succès : {output_filename}")