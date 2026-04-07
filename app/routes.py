from flask import Blueprint, render_template, request, jsonify
import pandas as pd
from src.predict import ChurnPredictor

main_bp = Blueprint('main', __name__)
predictor = None

# Champs numériques attendus par le formulaire
NUMERIC_FIELDS = [
    'Recency', 'Frequency', 'MonetaryTotal', 'MonetaryAvg',
    'SatisfactionScore', 'SupportTicketsCount',
    'CustomerTenureDays', 'ReturnRatio'
]


@main_bp.before_app_request
def load_predictor():
    """Charge le modèle une seule fois au démarrage de l'application."""
    global predictor
    if predictor is None:
        try:
            predictor = ChurnPredictor()
        except Exception as e:
            print(f"[WARN] Modèle non chargé : {e}")
            print("[WARN] Lancez d'abord : python src/preprocessing.py && python src/train_model.py")


@main_bp.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@main_bp.route('/predict', methods=['POST'])
def predict():
    if predictor is None:
        return render_template(
            'index.html',
            error="Le modèle n'est pas encore chargé. "
                  "Lancez d'abord : python src/preprocessing.py puis python src/train_model.py",
            prediction=None
        )

    try:
        # Récupération des données du formulaire
        data = request.form.to_dict()

        # Conversion des champs numériques (avec valeur par défaut si absent)
        for key in NUMERIC_FIELDS:
            raw = data.get(key, '').strip()
            data[key] = float(raw) if raw else 0.0

        # Création du DataFrame à 1 ligne
        df = pd.DataFrame([data])

        # Prédiction
        preds, probs = predictor.predict(df)

        churn_class = int(preds[0])
        churn_prob  = float(probs[0])

        if churn_class == 1:
            status = "🚨 Risque de Churn (Départ probable)"
        else:
            status = "✅ Client Fidèle (Restera probablement)"

        return render_template(
            'index.html',
            prediction=status,
            probability=round(churn_prob * 100, 1),
            form_data=data
        )

    except Exception as e:
        return render_template(
            'index.html',
            error=f"Erreur lors de la prédiction : {str(e)}",
            prediction=None
        )
