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

- Implémente l'article **Hari Hara Suthan Chittoor, Paul Robert Griffin, Ariel Neufeld, Jayne Thompson, Mile Gu. "QuLTSF: Long-Term Time Series Forecasting with Quantum Machine Learning" (2024). URL: https://arxiv.org/abs/2412.13769**.
- Contenu principal :
  - `data_provider` : Folder contenant les datasets utilisés ainsi que les script permettant de prétraiter les données.
  - `models` : Folder contenant les modèles implémentés sur `Pytorch` pour les baselines, et sur `Pennylane` et `Pytorch` également pour le modèle hybride QuLTSF.
  - `utils` : Folder contenant heper functions.
  - `linear.py` : Script à exécuter pour lancer la pipeline d'entrainement et d'évaluation d'un modèle linéaire séléctionné, sous une configuration bien choisie.
  - `qultsf.py` : Script à exécuter pour lancer la pipeline d'entrainement et d'évaluation du modèle QuLTSF, sous une configuration bien choisie.


##  Environnement recommandé

- Python >= 3.8
- TensorFlow / PyTorch
- Qiskit
- PennyLane
- Jupyter Notebook

Créer un environnement avec :

```bash
pip install -r requirements.txt
```

##  Contribution
- Pierre El Anati : pierre.el-anati@student-cs.fr
- Pierre Jourdin : pierre.jourdin@student-cs.fr
- Malek Bouhadida : malek.bouhadida@student-cs.fr
