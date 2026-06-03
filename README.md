# Financial Market Analysis Using Artificial Intelligence

## Overview

This repository contains my Bachelor's thesis project in Applied Mathematics, focused on Bitcoin market analysis and direction prediction using Machine Learning and Artificial Intelligence techniques.

The project investigates whether historical market data, technical indicators, and external financial signals can be used to identify and predict future market movements. The primary objective is to classify the next-day Bitcoin market direction into three categories:

* **UP**
* **DOWN**
* **NEUTRAL**

The project combines statistical analysis, feature engineering, regime detection, machine learning models, and MLOps components to create an end-to-end predictive pipeline.

---

## Research Motivation

Financial markets are highly dynamic, non-linear, and influenced by numerous interacting factors. Traditional linear approaches often struggle to capture complex market behavior.

This research explores how Machine Learning and regime-aware modeling can improve market direction prediction by incorporating:

* Historical Bitcoin price data
* Market volatility information
* Technical indicators
* External macroeconomic signals
* Market regime identification

---

## Project Workflow

```text
Bitcoin OHLCV Data
        +
External Signals
(Fear & Greed, VIX, S&P 500)
                ↓
         Data Validation
                ↓
      Exploratory Data Analysis
                ↓
        Feature Engineering
                ↓
         Feature Selection
                ↓
         Train/Test Split
                ↓
     Hidden Markov Model (HMM)
        Market Regimes
                ↓
       Machine Learning Models
      ├── Logistic Regression
      ├── LightGBM
      └── LightGBM + HMM
                ↓
        Model Evaluation
                ↓
      Strategy Backtesting
                ↓
         Monitoring
                ↓
          Retraining
                ↓
     FastAPI + Docker API
```

---

## Data Sources

### Market Data

* Binance API (BTC/USDT OHLCV)

### External Signals

* Fear & Greed Index
* S&P 500 Index
* VIX Volatility Index

---

## Feature Engineering

The project generates multiple categories of predictive features, including:

* Trend indicators
* Momentum indicators
* Volatility indicators
* Volume-based features
* Market sentiment features
* Time-based features
* External market signals

Feature selection techniques are then applied to identify the most informative variables for model training.

---

## Models

### Logistic Regression

Used as a baseline linear classification model.

### Random Forest

Used for feature importance estimation and feature selection.

### LightGBM

Gradient boosting model used to capture complex non-linear market relationships.

### Hidden Markov Model (HMM)

Used to identify latent market regimes and volatility states.

### LightGBM + HMM

Final regime-aware model that combines market regime probabilities with engineered features.

---

## Model Monitoring and MLOps

In addition to model development, the project includes several production-oriented components:

* FastAPI inference service
* Docker containerization
* Automated model monitoring
* Performance tracking
* Retraining pipeline
* GitHub Actions Continuous Integration (CI)

These components demonstrate how a research model can be transformed into a maintainable production-ready workflow.

---

## Technologies

### Programming

* Python

### Data Science & Machine Learning

* Pandas
* NumPy
* Scikit-learn
* LightGBM
* hmmlearn

### Visualization

* Matplotlib
* Seaborn

### Data Sources

* Binance API
* yfinance

### MLOps

* FastAPI
* Docker
* GitHub Actions

---

## Results

The experiments showed that:

* LightGBM significantly outperformed the baseline Logistic Regression model.
* Incorporating HMM-derived regime information improved prediction quality and model stability.
* Regime-aware modeling provided more robust performance across different market conditions.
* Monitoring and retraining pipelines enable continuous evaluation of model performance on newly available data.

---

## Academic Information

**Bachelor's Thesis**

**Title:** Financial Market Analysis Using Artificial Intelligence

**Author:** Arsen Hovhannisyan

**Field:** Applied Mathematics

**University:** National Polytechnic University of Armenia

**Year:** 2026
