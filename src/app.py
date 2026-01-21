from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import pandas as pd
from .predict import predict_credit_default

app = FastAPI(title="Credit Risk Prediction API")

# --- Input schema ---
class CreditInput(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float = Field(
        ..., ge=0.0, le=1.5,
        description="Revolving utilization rate (0.0-1.5)"
    )
    age: int = Field(
        ..., ge=0, le=120,
        description="Customer age (0-120)"
    )
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(
        ..., ge=0, le=100,
        description="Number of times 30-59 days past due (0-100)"
    )
    DebtRatio: float = Field(
        ..., ge=0.0, le=10000.0,
        description="Debt ratio (monthly debt / monthly income, typically 0-100)"
    )
    MonthlyIncome: float = Field(
        ..., ge=0.0, le=3_000_000,
        description="Monthly income in dollars (0-3M, typically 0-30k)"
    )
    NumberOfOpenCreditLinesAndLoans: int = Field(
        ..., ge=0, le=100,
        description="Number of open credit lines and loans (0-100)"
    )
    NumberOfTimes90DaysLate: int = Field(
        ..., ge=0, le=100,
        description="Number of times 90+ days late (0-100)"
    )
    NumberRealEstateLoansOrLines: int = Field(
        ..., ge=0, le=100,
        description="Number of real estate loans or lines (0-100)"
    )
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(
        ..., ge=0, le=100,
        description="Number of times 60-89 days past due (0-100)"
    )
    NumberOfDependents: int = Field(
        0, ge=0, le=20,
        description="Number of dependents (0-20)"
    )

    @field_validator('age')
    @classmethod
    def validate_age(cls, v):
        """Age 0 is unrealistic for credit applications."""
        if v == 0:
            raise ValueError('Age cannot be 0. Must be greater than 0.')
        return v

    @field_validator('MonthlyIncome')
    @classmethod
    def validate_monthly_income(cls, v):
        """Monthly income 0 is unrealistic for credit applications."""
        if v == 0:
            raise ValueError('MonthlyIncome cannot be 0. Must be greater than 0.')
        return v

# --- Output schema ---
class PredictResponse(BaseModel):
    default_probability: float
    will_default: bool

# --- Endpoint ---
@app.post("/predict", response_model=PredictResponse)
async def predict(input_data: CreditInput):
    """
    Predict credit default probability for a customer.
    
    Args:
        input_data: Customer credit and financial information
        
    Returns:
        default_probability: Probability of default (0.0-1.0)
        will_default: Binary prediction (True/False at 0.5 threshold)
    """
    try:
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
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Model file not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# --- Run app ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
