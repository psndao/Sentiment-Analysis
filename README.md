# 🎬 Sentiment Analysis IMDb avec LSTM et Keras

Ce projet a pour objectif de construire un modèle d'analyse de sentiments à partir des critiques de films IMDb, en utilisant un réseau de neurones LSTM avec des embeddings appris automatiquement.

## 📁 Données

Les données utilisées proviennent du **IMDb Large Movie Review Dataset**, disponible ici :  
🔗 [https://ai.stanford.edu/~amaas/data/sentiment/](https://ai.stanford.edu/~amaas/data/sentiment/)

- 50 000 critiques (format `.txt`)
- Réparties équitablement entre critiques positives et négatives
- Organisation :



## 🎯 Objectifs

Deux tâches sont menées dans ce projet :

### 🔹 Tâche 1 : Apprentissage de la matrice d'embedding Z

- Modèle : `Embedding → GlobalAveragePooling1D → Dense(sigmoid)`
- Objectif : apprendre une représentation sémantique des mots (type Word2Vec)
- Résultat : extraction de la matrice `Z` des embeddings appris

### 🔹 Tâche 2 : Classification avec LSTM

- Modèle : `Embedding(weights=Z) → LSTM → Dense(sigmoid)`
- Objectif : prédire la polarité (positive ou négative) d'une critique textuelle
- Métrique : `accuracy` sur les données de test

## ⚙️ Technologies utilisées

- Python 3
- TensorFlow / Keras
- NumPy, scikit-learn
- Jupyter Notebook

## 🛠️ Installation

```bash
git clone https://github.com/psndao/Sentiment-Analysis.git
cd Sentiment-Analysis
pip install -r requirements.txt
