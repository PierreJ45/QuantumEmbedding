# QuLTSF — Modèle hybride pour la Prévision de Séries Temporelles à Long Terme

Cette partie du projet explore l'utilisation d'un modèle hybride **quantique-classique** nommé **QuLTSF** (Quantum Long-Term Series Forecasting) pour la **prédiction à long terme de séries temporelles (LTSF)**.  
L’objectif est de comparer ses performances à celles de modèles linéaires, considérés comme baselines.

---

## Contenu du projet

- **Modèle principal** : `QuLTSF`, architecture inspirée du papier original QuLTSF [1], combinant :
  - Une couche classique d’entrée de taille L, la taille de la séquence d'input
  - Une couche cachée quantique (VQC), avec N qubit et K layers 
  - Une couche de sortie classique de taille T, la taille de la séquence d'output

- **Modèles baseline** :
  - `Linear`
  - `NLinear`
  - `DLinear`
---

## Jeux de données utilisés

- `MRK` (financier – action Merck & Co.), où la target est le `Close price`
- `EUR-USD` (financier – taux de change), où la target est le `Close price`
- `Météo` (météorologique - différentes données sur l'humidité la température...), où la target s'appelle `OT`
---

## Pipeline d’entraînement

- Split : 90% train / 10% test
- Scaling des données​
- Optimisation : algorithme ADAM avec taux d’apprentissage adaptatif​
- Validation croisée à 5 folds
- Optimisation : ADAM
- Métrique : MSE normalisée (en divisant par la variance de l'ensemble considéré train ou test)​
- Environnement : simulation et non sur QPU
---

## Résultats principaux
### Comparaison des performances sur un même dataset, avec différentes configurations:

| Configuration     | QuLTSF (#Param) | QuLTSF (MSE) | Linear (#Param) | Linear (MSE) | NLinear (#Param) | NLinear (MSE) | DLinear (#Param) | DLinear (MSE) |
|------------------|------------------|--------------|------------------|--------------|-------------------|----------------|-------------------|----------------|
| L=30, T=5        | 286              | 1.95         | 155              | 1.14         | 155               | 1.18           | 310               | **1.06**        |
| L=60, T=5        | 526              | 2.14         | 305              | 1.14         | 305               | 1.16           | 610               | **1.13**        |
| L=60, T=10       | 1050             | 2.85         | 610              | 1.96         | 610               | **1.88**        | 1220              | 1.98           |
| L=120, T=5       | 1006             | 4.23         | 605              | 1.20         | 605               | 1.22           | 1210              | **1.14**        |



Points à retenir:
- Augmenter L n'améliore pas les résultats, ce qui contre intuitif, car augmenter la taille du contexte et donc de la séquence d'entrée donnearait plus d'information au modèle pour réaliser ses prédictions. Cependant, cela va en parallèle augmenter le nombre de paramètres, face à une réduction du nombre d'instances disponibles pour l'entrainement. Par conséquent, l'entraînement devient plus compliqué.

- Pour les 4 modèles et pour toutes les configurations, la MSE normalisée est > 1 : Aucun des modèles n'est parvenu à faire mieux qu'un modèle naïf qui prédit la moyenne constante.

- A nombre de paramètres équivalent, les modèles linéaires performent mieux que le QuLTSF.

### Performance par rapport aux jeux de données:

| Dataset   | QuLTSF     | Linear     | NLinear    | DLinear    |
|-----------|------------|------------|------------|------------|
| EUR-USD   | 1.95       | 1.14       | 1.18       | **1.06**   |
| MRK       | 0.0657     | **0.0061** | 0.0059     | **0.0061** |
| Météo     | **0.000316** | 0.000328   | 0.000331   | 0.000328   |

- Les performances des 4 modèles est beaucoup plus meilleure dans le cas du dataset MRK et Météo. Leur avantage para rapport au premier dataset EUR-USD, est que la target est stationnaire. De plus, le dataset Météo contient 10x plus d'instances.

- Le QuLTSF surpasse les baselines, dans le cas du dataset Météo, donnat une MSE normalisée plus faible. Cependant, il faut noter que pour avoir ce résultat, nous avons augmenté le nombre de paramètres du QuLTSF (2055 paramètres pour `nqubits = 6` et `q_layers = 2` vs ~300 paramètres pour les modèles linéaires). Le temps de calcul à aussi augmenté considérablement : plus que 6h. 


# Références

- [1]: https://arxiv.org/abs/2412.13769 "QuLTSF: Quantum Long-Term Series Forecasting"