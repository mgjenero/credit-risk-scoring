# Credit Risk Scoring
App is currently deployed and you can acces here: https://credit-risk-scoring-waky.onrender.com/docs#/ <br>IMPORTANT: App also needs some time to wake up so be patient :) 

---

## 📌 Problem Description

This project solves a **credit risk prediction** problem:
given a customer’s financial and credit history, predict the probability that they will experience **serious delinquency within the next two years**.

The model can be used by:

* Banks and fintech companies
* Credit scoring systems
* Risk assessment pipelines

The output is a **probability of default** (0.0-1.0) and a **binary decision** (`will_default`).

---

## 📊 Dataset

The dataset comes from the **Give Me Some Credit** Kaggle competition.

**Target variable**

* `SeriousDlqin2yrs` – 1 if the person experienced serious delinquency, else 0

**Key features**

* Credit utilization
* Age
* Past due counts (30–59, 60–89, 90+ days)
* Debt ratio
* Monthly income
* Number of open credit lines
* Real estate loans
* Number of dependents

Raw data is stored in:

```
data/raw/
```

Processed data is stored in:

```
data/processed/
```

---

## 🔍 Exploratory Data Analysis (EDA)

EDA is available in:

```
notebooks/eda.ipynb
```

Includes:

* Target distribution
* Missing value analysis
* Feature ranges and outliers
* Correlation analysis

---

## 🧠 Models Trained

Multiple models were trained and compared:

| Model               | ROC-AUC    |
| ------------------- | ---------- |
| Logistic Regression | 0.7885      |
| Random Forest       | 0.8603      |
| LightGBM            | 0.8657      |
| **XGBoost** ✅      | **0.8662** |

The final deployed model is **XGBoost**, tuned using GridSearchCV.

Saved models are stored in:

```
models/
```

---

## ⚙️ Preprocessing & Feature Engineering

Implemented in:

```
src/preprocessing.py
```

Steps include:

* Missing value handling
* Feature engineering:

  * `DebtPerDependent`
  * `HighUtilization`

---

## 🚀 Model Training

Training logic is implemented in:

```
src/train.py
```

To train the model:

```bash
python src/train.py
```

The trained model is saved to:

```
models/model.bin
```

---

## 🌐 Model Deployment (FastAPI)

The model is deployed as a production-ready REST API.

### Quick Start

```bash
# Install dependencies
uv sync

# Activate environment
source .venv/bin/activate

# Run API locally
uvicorn src.app:app --reload
```

API runs on: `http://localhost:8000`

### Interactive Documentation

FastAPI provides automatic API documentation:
- **Swagger UI**: http://localhost:8000/docs – Full interactive API with field hints and constraints
- **ReDoc**: http://localhost:8000/redoc – Alternative documentation UI

### Endpoints

#### `/predict` (POST)
Predict credit default for a customer.

**Request:**

```json
{
  "RevolvingUtilizationOfUnsecuredLines": 0.35,
  "age": 45,
  "NumberOfTime30_59DaysPastDueNotWorse": 1,
  "DebtRatio": 0.25,
  "MonthlyIncome": 5500,
  "NumberOfOpenCreditLinesAndLoans": 7,
  "NumberOfTimes90DaysLate": 0,
  "NumberRealEstateLoansOrLines": 2,
  "NumberOfTime60_89DaysPastDueNotWorse": 0,
  "NumberOfDependents": 2
}
```

**Response:**
```json
{
  "default_probability": 0.13,
  "will_default": false
}
```

**Status Codes:**
- `200` – Success
- `422` – Validation error (invalid inputs)
- `500` – Server error (model not found, prediction error)

---

## 📦 Dependency Management

This project uses **uv** for dependency and environment management.

### Install Dependencies

```bash
# Install all dependencies (including dev tools)
uv sync

# Install production only (for deployment)
uv sync --no-dev
```

### Dependency Groups

**Production** (API runtime):
- fastapi, uvicorn – Web framework
- xgboost, scikit-learn – ML models
- pandas, pydantic – Data processing & validation
- joblib – Model serialization

**Development** (training & notebooks):
- lightgbm, kagglehub – Model training
- matplotlib, seaborn – Plotting
- nbconvert – Notebook tools

Dependencies defined in: `pyproject.toml` and `uv.lock`

---

## 🐳 Docker Deployment

Build and run containerized API:

```bash
# Build image
docker build -t credit-risk-api .

# Run container
docker run -p 8000:8000 credit-risk-api
```

Container details:
- **Base Image**: Python 3.12-slim
- **Port**: 8000
- **Entry Point**: Uvicorn serving `src.app:app`
- **Dependencies**: Production only (--no-dev)

---
