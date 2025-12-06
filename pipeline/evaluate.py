from core.loss import compute_log_loss
from sklearn.metrics import accuracy_score



def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    loss = compute_log_loss(y_test, y_pred_proba)
    
    print(f"✅ Accuracy : {acc:.2f}")
    print(f"📉 Log Loss : {loss:.4f}")
    return acc, loss
