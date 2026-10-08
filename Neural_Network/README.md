# Réseau neuronal artificiel réalisé sans framework

Ce projet implémente avec NumPy un petit réseau neuronal pour résoudre un problème de classification binaire. La propagation avant, la fonction de coût, la rétropropagation et la descente de gradient sont codées explicitement afin d'illustrer les mécanismes d'apprentissage.

## Architecture

Le modèle utilise deux variables d'entrée, une couche cachée de 8 neurones avec activation ReLU et un neurone de sortie avec activation sigmoid :

```text
2 entrées -> 8 neurones ReLU -> 1 neurone sigmoid -> probabilité
```

Le nombre de neurones cachés est réglable avec le paramètre `hidden_size` de la fonction d'entraînement.

Les calculs effectués par le réseau sont :

```text
Z1 = X @ W1 + b1
A1 = ReLU(Z1) = max(0, Z1)
Z2 = A1 @ W2 + b2
A2 = sigmoid(Z2) = 1 / (1 + exp(-Z2))
```

La sortie `A2` représente la probabilité que l'échantillon appartienne à la classe 1. La classe prédite vaut 1 si cette probabilité est supérieure ou égale à 0,5, et 0 sinon.

## Apprentissage

Le notebook entraîne le réseau sur un jeu de données synthétique généré par `make_blobs` de scikit-learn (100 échantillons, deux caractéristiques et deux classes).

La fonction de coût est l'entropie croisée binaire :

```text
J = -mean(y * log(A2) + (1 - y) * log(1 - A2))
```

Les gradients sont calculés par rétropropagation à travers la couche de sortie puis la couche cachée. Les poids et biais de chaque couche sont ensuite mis à jour par descente de gradient :

```text
paramètre = paramètre - taux_apprentissage * gradient
```

L'initialisation des poids est déterministe pour faciliter la reproduction des résultats. Les paramètres par défaut sont un taux d'apprentissage de `0.05`, `2000` itérations et `8` neurones cachés.

## Contenu du projet

- `neural_netwok.ipynb` : implémentation NumPy, entraînement, prédiction et visualisations du réseau neuronal.
- `deep-learning-mnist.ipynb` : notebook séparé consacré à un modèle de deep learning sur MNIST.
- `requirements.txt` : bibliothèques Python nécessaires aux notebooks.
- `best.keras` : modèle Keras enregistré associé au travail sur le deep learning.

Le notebook du réseau visualise la perte au cours de l'entraînement, la frontière de décision non linéaire et la surface 3D des probabilités prédites.

## Installation et exécution

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Ouvrir `neural_netwok.ipynb` dans VS Code avec l'extension Jupyter, ou dans Jupyter Notebook, puis exécuter les cellules dans l'ordre.

## Bibliothèques

- Python
- NumPy
- Matplotlib
- scikit-learn
- Plotly
