# 🛒 Store Sales — Time Series Forecasting

A machine learning project that predicts daily store sales for Corporación Favorita, a large Ecuadorian grocery retailer, using historical sales, store metadata, oil prices, holidays, and transactions data.

Live demo → [Streamlit App](https://share.streamlit.io) *(deploy to get your link)*

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Data Loading & Merging](#3-data-loading--merging)
4. [Data Cleaning](#4-data-cleaning)
5. [Exploratory Data Analysis](#5-exploratory-data-analysis)
6. [Feature Engineering](#6-feature-engineering)
7. [Model Training & Comparison](#7-model-training--comparison)
8. [Best Model & Evaluation](#8-best-model--evaluation)
9. [Streamlit Deployment](#9-streamlit-deployment)
10. [How to Run Locally](#10-how-to-run-locally)

---

## 1. Project Overview

**Goal:** Predict the unit sales for thousands of product families sold at Favorita stores across Ecuador.

**Metric:** RMSLE (Root Mean Squared Log Error) — penalizes under-prediction more than over-prediction.

**Approach:** Tabular regression using a scikit-learn Pipeline (preprocessing + Random Forest), trained on engineered time-series features.

---

## 2. Dataset

Data sourced from the [Kaggle Store Sales Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) competition.

| File | Description |
|---|---|
| `train.csv` | 3M rows of daily sales per store/family (2013–2017) |
| `stores.csv` | Store metadata — city, state, type, cluster |
| `oil.csv` | Daily WTI oil price (Ecuador is oil-dependent) |
| `holidays_events.csv` | National/regional/local holidays |
| `transactions.csv` | Daily transaction counts per store |

---

## 3. Data Loading & Merging

All five datasets are loaded and merged into one master DataFrame:

```python
df = train.copy()
df = df.merge(stores, on='store_nbr', how='left')
df = df.merge(transactions, on=['date', 'store_nbr'], how='left')
df = df.merge(oil, on='date', how='left')
df = df.merge(holidays, on='date', how='left')
```

Final shape after merge: **(3,000,888 rows × 13 columns)**

---

## 4. Data Cleaning

| Column | Missing % | Fix |
|---|---|---|
| `dcoilwtico` | 30.9% | Forward-fill then back-fill |
| `transactions` | 8.2% | Fill with 0 (no transactions recorded) |
| `holiday_type` | 85.6% | Fill with `"None"` |

Additional steps:
- Holidays cleaned: removed `transferred=True` rows, deduplicated by date
- Sorted by `store_nbr → family → date` for correct lag computation
- Dropped duplicate rows (safety check)

---

## 5. Exploratory Data Analysis

- **Sales distribution** is highly right-skewed — most sales are 0 or near-zero; ~9% of rows have sales > 1,000
- **Strategy:** Keep all rows (RMSLE penalizes under-prediction)
- Visualizations: sales histogram, clipped distribution (to 500), actual vs predicted scatter

---

## 6. Feature Engineering

A custom `engineer_features()` function adds:

**Date / Time Features**
| Feature | Description |
|---|---|
| `month`, `day`, `day_of_week` | Calendar components |
| `week_of_year`, `quarter` | Seasonal signals |
| `is_weekend` | 1 if Sat/Sun |
| `trend` | Days since dataset start |

**Domain Knowledge Features**
| Feature | Description |
|---|---|
| `is_holiday` | 1 if any holiday type |
| `is_payday` | 1 if day is 15th or 30th |
| `onpromotion` | Items on promotion |

**Lag & Rolling Features** *(training only)*
| Feature | Description |
|---|---|
| `sales_lag_1/7/14/28` | Sales N days ago |
| `sales_roll_mean_7/28` | Rolling average |
| `sales_roll_std_7/28` | Rolling std deviation |
| `promo_lag_interaction` | `onpromotion × sales_lag_7` |

---

## 7. Model Training & Comparison

Train/test split: **before 2017 = train**, **2017+ = test** (300K sampled rows for speed)

A scikit-learn **Pipeline** was used for all models:
```
Pipeline([
    ('preprocessing', ColumnTransformer([
        ('num', StandardScaler + SimpleImputer),
        ('cat', OneHotEncoder + SimpleImputer)
    ])),
    ('model', <model>)
])
```

| Model | RMSE | MAE | R² |
|---|---|---|---|
| **Random Forest (fast)** | **494.87** | 176.18 | **0.8627** |
| Hist Gradient Boosting | 501.78 | 171.53 | 0.8588 |
| Linear Regression | 869.72 | 274.72 | 0.5758 |
| Ridge Regression | 869.73 | 274.72 | 0.5758 |

---

## 8. Best Model & Evaluation

**Improved Random Forest** (tuned hyperparameters):
```python
RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
```

| Metric | Score |
|---|---|
| RMSE | 482.12 |
| MAE | 156.26 |
| R² | **0.8696** |

Top feature importances: `store_nbr` (31.5%), `family` (11%), `dcoilwtico` (12.7%)

Model saved with:
```python
import joblib
joblib.dump(best_pipe, 'sales_model.pkl')
```

---

## 9. Streamlit Deployment

An interactive web app (`app.py`) was built to serve the model:

- Input all 10 features via form widgets
- Model loaded with `@st.cache_resource` for performance
- Prediction displayed in a styled gradient card
- Color-coded interpretation (Low / Moderate / Strong / Very High)

**Run locally:**
```bash
pip install streamlit scikit-learn pandas numpy joblib
streamlit run app.py
```

**Deploy to Streamlit Cloud:**
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set main file to `app.py` → Deploy

---

## 10. How to Run Locally

```bash
# Clone the repo
git clone https://github.com/OmarEssam-20/Store-Sales---Time-Series-Forecasting.git
cd Store-Sales---Time-Series-Forecasting

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 🛠 Tech Stack

`Python` · `scikit-learn` · `pandas` · `numpy` · `joblib` · `Streamlit` · `matplotlib` · `seaborn`
