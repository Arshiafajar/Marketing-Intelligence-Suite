# Marketing Intelligence Suite

**End-to-end Data Science & Machine Learning solution for marketing analytics**

> **Arshia Fajar** · BSc Data Science · [LinkedIn](https://linkedin.com/in/arshia-fajar)

---

##  Dataset Setup (Required)

This project uses the **Online Retail II UCI** dataset. 1M+ real e-commerce transactions from a UK retailer (2009–2011).

**Download steps:**
1. Go to **https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci**
2. Sign in to Kaggle (free account) and click **Download**
3. Unzip the file — you will get `online_retail_II.xlsx`
4. Place it at: `data/raw/online_retail_II.xlsx`

> All four notebooks will automatically detect and load this file. If it is not present, a `FileNotFoundError` with clear instructions is raised.

---

## Project Overview

This project delivers a complete marketing data science pipeline across four modules, covering the full lifecycle from raw data to deployed interactive dashboard. It was built to demonstrate applied machine learning, statistical experimentation, and NLP on realistic e-commerce marketing data.

**Dataset:** UCI Online Retail Dataset (500K+ transactions, 4,000+ customers) or fully reproducible synthetic data

---

##  Project Structure

```
marketing-intelligence-suite/
├── notebooks/
│   ├── 01_customer_segmentation.ipynb          # Module 1: RFM + Clustering
│   ├── 02_propensity_clv_modeling.ipynb        # Module 2: XGBoost + SHAP
│   ├── 03_ab_testing_statistical_experimentation.ipynb  # Module 3: A/B Testing
│   └── 04_nlp_sentiment_topics_genai.ipynb     # Module 4: NLP + GenAI
├── dashboard/
│   └── app.py                                  # Streamlit interactive dashboard
├── src/
│   ├── xgb_propensity_model.pkl               # Saved propensity model
│   ├── xgb_clv_model.pkl                      # Saved CLV model
│   └── feature_scaler.pkl                     # Feature scaler
├── data/
│   ├── raw/                                   # Input datasets
│   └── processed/                             # Model outputs & predictions
├── reports/                                   # Auto-generated charts & plots
└── requirements.txt
```

---

## Modules

### Module 1 — Customer Segmentation
**File:** [`notebooks/01_customer_segmentation.ipynb`](notebooks/01_customer_segmentation.ipynb)

| Step | Technique |
|---|---|
| Feature Engineering | RFM (Recency, Frequency, Monetary) |
| Log transform + scaling | `StandardScaler`, `np.log1p` |
| Optimal k selection | Elbow Method + **Silhouette Score** |
| Primary clustering | **K-Means** (k=4) |
| Outlier detection | **DBSCAN** density clustering |
| Visualization | **PCA** 2D projection |
| Output | Segment profiles: Champions, Loyal, At-Risk, New |

**Key Result:** 4 distinct customer segments with interpretable RFM profiles, enabling personalized marketing targeting.

---

### Module 2 — Customer Propensity & CLV Prediction
**File:** [`notebooks/02_propensity_clv_modeling.ipynb`](notebooks/02_propensity_clv_modeling.ipynb)

| Step | Technique |
|---|---|
| Feature engineering | RFM + AvgOrderValue, FrequencyRecency, RecencyBucket |
| Baseline classifier | **Logistic Regression** |
| Ensemble classifier | **Random Forest** (200 trees) |
| Optimized classifier | **XGBoost** + GridSearchCV (5-fold StratifiedKFold) |
| Interpretability | **SHAP** values (summary + waterfall plots) |
| CLV regression | **XGBoost Regressor** (MAE, RMSE, R²) |
| Evaluation | ROC-AUC, Precision, Recall, F1 |

**Key Result:** XGBoost propensity model achieves AUC ~0.87. SHAP identifies Recency and Frequency as primary churn drivers.

---

### Module 3 — A/B Testing & Statistical Experimentation
**File:** [`notebooks/03_ab_testing_statistical_experimentation.ipynb`](notebooks/03_ab_testing_statistical_experimentation.ipynb)

| Step | Technique |
|---|---|
| Experiment design | Power analysis, minimum sample size calculation |
| Test 1 (conversion rate) | **Chi-Square Test** |
| Normality check | **Shapiro-Wilk Test** |
| Test 2 (order value) | **Welch's t-test** + **Mann-Whitney U** |
| Effect size | **Cohen's d** |
| Confidence intervals | 95% CI for difference in means |
| Causal inference | **Propensity Score Matching** (PSM) |

**Key Result:** Treatment campaign delivers +3pp conversion lift (p<0.05, χ²=12.4) and +£7 higher average order value — projected £840K annual revenue uplift at scale.

---

### Module 4 — NLP & Generative AI
**File:** [`notebooks/04_nlp_sentiment_topics_genai.ipynb`](notebooks/04_nlp_sentiment_topics_genai.ipynb)

| Step | Technique |
|---|---|
| Text preprocessing | Tokenization, lemmatization, stopword removal |
| Sentiment analysis | **VADER** (rule-based, fast) |
| Topic modeling | **LDA** (Latent Dirichlet Allocation, 6 topics) |
| Topic labeling | Business-readable cluster assignment |
| GenAI integration | **OpenAI GPT / Gemini** API prompt engineering |
| Output | Per-topic sentiment scores + marketing recommendations |

**Key Result:** 6 topics identified (Delivery, Product Quality, Customer Service, Value, Packaging, Accuracy). Delivery has the most negative sentiment — key marketing improvement area.

---

## hhow to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run notebooks in order
```bash
jupyter notebook notebooks/01_customer_segmentation.ipynb
```
> Notebooks work fully with **synthetic data** — no external dataset required. To use real data, place [`online_retail.xlsx`](https://archive.ics.uci.edu/ml/datasets/Online+Retail) in `data/raw/`.

### 3. Launch the interactive dashboard
```bash
streamlit run dashboard/app.py
```

---

## Results Summary

| Module | Metric | Result |
|---|---|---|
| Segmentation | Silhouette Score | 0.42 |
| Propensity | XGBoost AUC-ROC | 0.87 |
| CLV | R² Score | 0.83 |
| A/B Test | Conversion lift | +25% relative |
| NLP | Reviews analyzed | 2,000+ |

---

##  Tech Stack

`Python 3.10` · `scikit-learn` · `XGBoost` · `SHAP` · `NLTK` · `SciPy` · `Streamlit` · `Pandas` · `Matplotlib` · `Seaborn`


