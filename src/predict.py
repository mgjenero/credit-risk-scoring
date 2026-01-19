import os
import joblib
import pandas as pd
from .preprocessing import preprocess_data


def predict_credit_default(df: pd.DataFrame):
    """
    df: single-row DataFrame with API input
    """

    MODEL_PATH = "models/model.bin"

    # Preprocess (creates DebtPerDependent, HighUtilization)
    df = preprocess_data(df)

    # Rename API-friendly columns to match training
    rename_map = {
        "NumberOfTime30_59DaysPastDueNotWorse": "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60_89DaysPastDueNotWorse": "NumberOfTime60-89DaysPastDueNotWorse"
    }
    df = df.rename(columns=rename_map)

    # Load model
    model = joblib.load(MODEL_PATH)

    # Predict
    probs = model.predict_proba(df)[:, 1]
    classes = model.predict(df)

    return pd.DataFrame({
        "default_probability": probs,
        "will_default": classes.astype(bool)
    })
