🔥 Coal Price Forecasting - Part 2
This is Part 2 of the Coal Price Forecasting Project using machine learning techniques. It continues from the preprocessing and visualization in Part 1, and focuses on model training, evaluation, and future forecasting using forecasted external factors.

📁 Files Used
final Coal Historical Prices _2020-24.xlsx: Contains historical coal prices and external factor data across multiple sheets.

final external factors predictions.xlsx: Contains future predictions for external economic indicators (natural gas, crude oil, etc.).

🔄 Workflow Steps
1. Data Loading and Preprocessing
Loaded multiple Excel sheets and merged them into one unified dataset.

Forward filled missing values and removed duplicates.

Visualized data distributions using box plots and explored correlations via heatmaps.

Removed irrelevant features such as 'Non ferrous metal price'.

2. Date Feature Engineering
Converted date columns to datetime format.

Transformed the Date column to a numeric Days feature for model compatibility.

Removed any dates overlapping with forecasted data to avoid data leakage.

🎯 Target Variables
Coal prices for multiple grades and markets were selected as target columns:

Coal Richards Bay 4800kcal NAR fob

Coal Richards Bay 5500kcal NAR fob

Coal Richards Bay 5700kcal NAR fob

Coal Richards Bay 6000kcal NAR fob current week avg

Coal India 5500kcal NAR cfr

🧠 Modeling
Algorithm Used: RandomForestRegressor

Split data into training and testing sets (80/20).

Model trained and evaluated for each target variable.

Evaluation Metrics:

R² Score

Mean Absolute Error (MAE)

Results are saved in:

coal_price_predictions.csv – Predictions on the test set

model_evaluation_metrics.csv – R² and MAE scores for each model

🔮 Forecasting Future Coal Prices
Using the final external factors predictions.xlsx, the model predicts future coal prices based on external economic indicators:

Natural Gas Price

US Crude Oil WTI Prices

Dubai Crude Oil

Dutch Natural Gas

Predictions are saved to:

predicted_coal_prices.csv

📊 Output Files
coal_price_predictions.csv: Test set predictions

model_evaluation_metrics.csv: Model performance metrics

predicted_coal_prices.csv: Future price forecasts

✅ How to Run
Install dependencies:
```
pip install pandas numpy matplotlib seaborn scikit-learn xgboost openpyxl
```
Place both Excel files in the correct directory (/content/ or update the paths accordingly).

Run the script:
```
python coal_price_forecasting_part2.py
```
🧩 Dependencies
pandas

numpy

matplotlib

seaborn

scikit-learn

xgboost

openpyxl (for Excel reading)

