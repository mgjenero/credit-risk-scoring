# Credit Risk Scoring

## 📌 Problem Description

This project solves a **credit risk prediction** problem:
given a customer’s financial and credit history, predict the probability that they will experience **serious delinquency within the next two years**.

The model can be used by:

* Banks and fintech companies
* Credit scoring systems
* Risk assessment pipelines

The output is a **probability of default** and a **binary decision** (`will_default`).

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
| Logistic Regression | ~0.78      |
| Random Forest       | ~0.84      |
| LightGBM            | ~0.86      |
| **XGBoost (final)** | **0.8662** |

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
* Fixed median imputation for inference consistency
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

The model is deployed as a **REST API** using FastAPI.

### Run the API locally

```bash
uvicorn src.app:app --reload
```

### API Endpoint

**POST** `/predict`

Example request:

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

Example response:

```json
{
  "default_probability": 0.13,
  "will_default": false
}
```

---

## 📦 Dependency Management

This project uses **uv** for dependency and environment management.

Install dependencies:

```bash
uv sync
```

Activate environment:

```bash
source .venv/bin/activate
```

Dependencies are defined in:

```
pyproject.toml
uv.lock
```

---

## 🐳 Containerization

Build Docker image:

```bash
docker build -t credit-risk-api .
```

Run container:

```bash
docker run -p 8000:8000 credit-risk-api
```

---

## ☁️ Cloud Deployment

The application can be deployed to:

* Any Docker-compatible cloud (AWS, GCP, Azure)
* Kubernetes
* Render / Fly.io / Railway

The Docker image exposes port `8000`.

---
