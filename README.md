# Atelier Machine Learning : Analyse Comportementale Clientèle Retail

## Titre et description
Ce projet propose une solution complète pour l'analyse comportementale de la clientèle d'un site de e-commerce de cadeaux. Le but est de prédire le risque de **Churn** (départ d'un client) à partir d'une base de transactions imparfaitement structurée comportant 52 variables.

La chaîne complète inclut : 
- L'exploration et la préparation des données
- La mise en oeuvre d'un pipeline de prétraitement avec Scikit-Learn (Nettoyage, Encodage One-Hot/Ordinal, PCA, Imputation).
- L'optimisation des hyperparamètres (GridSearchCV) sur des modèles de machine learning tels que `LogisticRegression` et `RandomForestClassifier`.
- Une interface Web de démonstration via Flask.

## Structure du projet explicité
```text
E-commerceML/
|-- data/
|   |-- raw/          # Jeu de données original
|   |-- process/      # Données nettoyées avant scission
|   \-- train_test/   # Scissions 80/20 après prétraitement
|-- notebooks/        # Notebooks pour l'expérimentation (Nettoyage initial et prototypes)
|-- src/              # Code métier Python de production
|   |-- preprocessing.py  # Script de nettoyage et d'ingéniérie des caractéristiques
|   |-- train_model.py    # Script entraînant les algorithmes et sauvegardant le modèle final
|   |-- predict.py        # Wrapper contenant la logique d'inférence (Predictor)
|   \-- utils.py          # Fonctions utilitaires (loggings, load/save)
|-- models/           # Fichiers .pkl générés de vos modèles d'apprentissage (pipeline, modèle, meta)
|-- app/              # Déploiement logiciel 
|   |-- templates/
|   |   \-- index.html     # Design HTML/CSS de l'application cliente
|   |-- __init__.py   # Factory de l'application Flask
|   \-- routes.py     # Définition des endpoints MVC
|-- run.py            # Point d'entrée à la racine pour lancer l'API
|-- reports/          # Rapports (si implémenté par la suite)
|-- requirements.txt  # Les dépendances (pandas, scikit-learn, flask, joblib)
\-- README.md         # Documentation de référence
```

## Instructions d'installation

1. **Cloner ou récupérer le dossier.**
2. **Créer un environnement virtuel python**
   ```bash
   python -m venv venv
   ```
3. **Activer l'environnement**
   - Sur Windows : `venv\Scripts\activate`
   - Sur Mac/Linux : `source venv/bin/activate`
4. **Installer les dépendances requises**
   ```bash
   pip install -r requirements.txt
   ```

## Guide d'utilisation

Une fois les paquets installés, exécutez le cycle de vie du modèle de manière consécutive.
Veillez à toujours vous placer à la racine du projet `E-commerceML`.

**Étape 1 : Préparation des données**
```bash
python -m src.preprocessing
```
*Génère les splits train/test et exporte les transformateurs (pipelines).*

**Étape 2 : Entraînement des modèles**
```bash
python -m src.train_model
```
*Effectue la recherche des hyperparamètres sur Logistic Regression et Random Forest (Ceci peut prendre quelques instants en raison de la GridSearch sur 100 n_estimators).*

**Étape 3 : Déploiement de l'API Flask**
```bash
python run.py
```
Ouvrez le lien fourni (généralement `http://127.0.0.1:5000/`) dans n'importe quel navigateur web moderne. Une interface intuitive vous permet de simuler un client en entrant des détails basiques pour y prédire le Churn.
