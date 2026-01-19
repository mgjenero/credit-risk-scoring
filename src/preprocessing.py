import pandas as pd


def load_data(data_path: str) -> pd.DataFrame:
    """Load credit risk data."""
    df = pd.read_csv(data_path, index_col=0)
    return df



def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess credit risk data."""
    # This is kept to process new data consistently
    monthly_income_median = 5400.0

    # Handle missing values
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(monthly_income_median)
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(0)

    # Feature engineering
    df['DebtPerDependent'] = df['DebtRatio'] / (df['NumberOfDependents'] + 1)
    df['HighUtilization'] = (df['RevolvingUtilizationOfUnsecuredLines'] > 0.8).astype(int)

    return df


def load_and_preprocess_data(data_path: str) -> pd.DataFrame:
    """Load and preprocess credit risk data."""
    df = load_data(data_path)
    df = preprocess_data(df)
    return df

    # src/preprocessing.py
import pandas as pd



if __name__ == "__main__":
    input_path = "data/raw/cs-training.csv"
    output_path = "data/processed/credit_data_train_processed.csv"

    df = load_and_preprocess_data(input_path)
    df.to_csv(output_path, index=False)
