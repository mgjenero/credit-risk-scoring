# src/app.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
from .predict import predict_credit_default

app = FastAPI(title="Credit Risk Prediction API")

# --- Input schema ---
class CreditInput(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float
    age: int
    NumberOfTime30_59DaysPastDueNotWorse: int
    DebtRatio: float
    MonthlyIncome: float
    NumberOfOpenCreditLinesAndLoans: int
    NumberOfTimes90DaysLate: int
    NumberRealEstateLoansOrLines: int
    NumberOfTime60_89DaysPastDueNotWorse: int
    NumberOfDependents: int = Field(0, description="Optional")

# --- Output schema ---
class PredictResponse(BaseModel):
    default_probability: float
    will_default: bool

# --- Endpoint ---
@app.post("/predict", response_model=PredictResponse)
async def predict(input_data: CreditInput):
    # Convert Pydantic input to DataFrame
    df = pd.DataFrame([input_data.dict()])

    # Predict (preprocessing is applied inside predict_credit_default)
    result = predict_credit_default(df)

    # Return first row (single input)
    row = result.iloc[0]
    return PredictResponse(
        default_probability=row['default_probability'],
        will_default=row['will_default']
    )

# --- Run app ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
