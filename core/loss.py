# core/loss.py
from sklearn.metrics import log_loss
import numpy as np


def compute_log_loss(y_true, y_pred_proba):
    """
    Calcule la log-loss (perte logarithmique) d’un modèle de classification.
    Utile pour mesurer la qualité des probabilités prédites.
    """
    loss = log_loss(y_true, y_pred_proba)
    print(f"📉 Log Loss : {loss:.4f}")
    return loss


def compute_accuracy(y_true, y_pred):
    """
    Calcule simplement la précision (accuracy) du modèle.
    """
    acc = np.mean(y_true == y_pred)
    print(f"✅ Accuracy : {acc:.2f}")
    return acc


