import os
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import joblib

from .preprocessing import load_and_preprocess_data

# Paths
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "model.bin")
TARGET = "SeriousDlqin2yrs"

def train_model(data_path: str = "data/raw/cs-training.csv",
                    save_model: bool = True):
    """
    Train XGBoost model on credit risk data.

    Args:
        data_path (str): Path to raw CSV file.
        save_model (bool): If True, saves the trained model to disk.

    Returns:
        model: Trained XGBoost model.
        auc: ROC-AUC score on validation set.
    """
    df = load_and_preprocess_data(data_path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y
    )

    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        colsample_bytree=0.8,
        subsample=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        verbosity=0  # suppress warnings
    )

    xgb.fit(X_train, y_train)

    y_pred = xgb.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    print(f"Tuned XGBoost ROC-AUC: {auc:.4f}")

    if save_model:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(xgb, MODEL_FILE)
        print(f"Model saved to {MODEL_FILE}")

    return xgb, auc


if __name__ == "__main__":
    train_model()
