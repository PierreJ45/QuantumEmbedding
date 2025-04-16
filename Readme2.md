# Quantum Embedding for Advanced Analytics in Finance

Ce dépôt regroupe plusieurs implémentations de modèles hybrides appliqués à la prédiction de séries temporelles financières. Chaque dossier correspond à un papier de recherche reproduit ou réimplémenté.

## Structure du dépôt

### `classificationPaper/`

- Implémente l'article **S. Lloyd etAL. “Quantum Embeddings for Machine Learning”. In :arXiv preprint arXiv:2001.03622 (2020). URL:https://arxiv.org/pdf/2001.03622.**.
- Ce dossier contient le code nécessaire à la reproduction des résultats décrits dans le papier X, portant sur la classification de séries temporelles financières.

### `QLSTMPaper/`

- Implémente l'article **Samuel Yen-Chi Chen et al. “Quantum Long Short-Term Memory”. In :arXiv preprint (2020). URL: https://arxiv.org/pdf/2009.01783**.
- Contenu principal :
  - `lstm_snp_vol.ipynb` : Notebook basé sur le repo GitHub de **DikshantDulal : https://github.com/DikshantDulal/SoftServe_QLSTM/tree/main**, adapté pour la prédiction de la volatilité journalière des rendements du **S&P 500**. Utilise le fichier `features_target.csv` comme dataset.
  - `features_target.csv` : Dataset utilisé pour la prédiction de la volatilité du S&P 500.
  - `Factory.py` : Fichier pyhton auxiliaire qui contient les implémentations des modèles LTSM ET QLSTM.
  - `lstmReimplem.ipynb` : Implémentation **from scratch** de modèles LSTM et QLSTM appliqués à un dataset de taux de change **EUR/USD** obtenu via la libraire yahoofinance.

### `QuLTSF/`

- Contient des expériences en cours sur la **Quantum Long-Term Sequence Forecasting** (QuLTSF).


## 🔧 Environnement recommandé

- Python >= 3.8
- TensorFlow / PyTorch
- Qiskit (pour les modules quantiques)
- Jupyter Notebook

Créer un environnement avec :

```bash
pip install -r requirements.txt
