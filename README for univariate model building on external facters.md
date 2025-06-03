# Time Series Forecasting on Energy Commodities using Holt-Winters Method

This project involves data preprocessing, exploratory data analysis, and time series forecasting using the Holt-Winters Exponential Smoothing method on key external energy market factors.

## 📁 Dataset

The dataset (`externalfactors.xlsx`) contains multiple sheets with information on global energy prices like Natural Gas, Crude Oil (WTI, Dubai), and Dutch Natural Gas. These sheets are merged into a single DataFrame for analysis.

## 🔧 Features

- Merges multiple sheets from Excel into a single dataset.
- Handles missing data using forward-fill.
- Visualizes the data using boxplots and correlation heatmaps.
- Drops non-relevant columns based on data quality and relevance.
- Performs time series forecasting using Holt-Winters exponential smoothing.
- Optimizes hyperparameters (trend, seasonality, seasonal periods) via grid search.
- Splits data into train/test and forecasts for both historical and next 30 months.
- Saves forecasted results to CSV files.

## 📊 Visualizations

- Boxplots for understanding distribution.
- Correlation heatmaps.
- Forecast plots with train, test, and future forecast curves.

## 📁 Output Files

- `All_Test_and_Forecasted_Values.csv` – Merged forecast vs actuals on test set.
- `Future_30_Month_Forecast.csv` – 30-month forecast from last date in data.

## 📦 Libraries Used

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `sklearn`
- `xgboost`
- `statsmodels`

## 🧪 Models Used

- **Holt-Winters Exponential Smoothing** from `statsmodels`
- Evaluated using:
  - MAPE (Mean Absolute Percentage Error)
  - MAE, MSE, R² Score

## 🏁 How to Run

1. Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost statsmodels openpyxl
```

2. Place the Excel file (`externalfactors.xlsx`) in your working directory.

3. Run the Python script to generate visualizations and forecast files.

## 📈 Forecasted Columns

- Natural Gas Price
- US Crude Oil WTI Prices
- Dubai Crude Oil
- Dutch Natural Gas

## 📬 Author

Developed by **Bomma Sonia** as part of a time series forecasting project.

---

