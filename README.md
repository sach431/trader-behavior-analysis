# 📊 Trader Performance vs Market Sentiment Analysis

## -- Project Overview

This project analyzes trader performance across different market sentiment phases:

- Extreme Fear  
- Fear  
- Neutral  
- Greed  
- Extreme Greed  

The objective is to understand how market sentiment impacts:

- Profitability  
- Win Rate  
- Trade Frequency  
- Long/Short Behaviour  
- Risk (PnL Volatility)  
- Predictive Trade Outcomes  

---

## 📈 Key Performance Metrics

| Metric | Value |
|--------|--------|
| Total Trades | 35,864 |
| Total PnL | 3,624,808.47 |
| Win Rate | 42.86% |
| Model Accuracy | 53% | 

---

## 📊 Dashboard Features

The Streamlit dashboard provides:

✔ Overall performance metrics  
✔ Daily PnL trend visualization  
✔ Sentiment-wise average PnL comparison  
✔ Win rate analysis by sentiment  
✔ Trade frequency trend  
✔ Long vs Short behaviour breakdown  
✔ Risk (PnL volatility) comparison  
✔ Predictive model performance report  

---

## 🧠 Key Insights

- **Extreme Greed** phases show highest profitability but also highest volatility.
- **Neutral** sentiment provides stable and balanced performance.
- **Extreme Fear** shows lowest win rate and weak profitability.
- Trade frequency spikes often correlate with increased PnL volatility.
- Model performs better in identifying losses than predicting gains.

---

## 🤖 Predictive Modeling

A classification model was developed to predict trade profitability.

**Model Performance:**

| Metric        | Value |
|--------------|--------|
| Accuracy     | 0.53   |
| Precision (Class 0) | 0.58 |
| Recall (Class 0)    | 0.62 |
| Precision (Class 1) | 0.45 |
| Recall (Class 1)    | 0.40 |

Further improvements can be achieved through:
- Feature engineering
- Ensemble methods
- Hyperparameter tuning

---
## 🛠 Tech Stack

- Python  
- Pandas  
- NumPy  
- Plotly  
- Streamlit  
- Scikit-learn  
---

