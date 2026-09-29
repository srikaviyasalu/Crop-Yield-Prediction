# Crop Yield Prediction

A professional agricultural decision-support web application that helps farmers and agricultural planners estimate crop yields based on historical agricultural data and environmental conditions.

## Product Purpose
The purpose of this application is to estimate crop yield and allow users to explore different crop/condition scenarios. The machine learning model acts as an internal engine, providing users with a simple, intuitive workflow to understand how variables like area, rainfall, fertilizer, and pesticide impact their expected harvest.

## Features
- **🔮 Predict Yield**: Enter your specific conditions to get an estimated crop yield. The primary default year is 2026, which provides an estimate based on historical patterns.
- **🔄 What-If Analysis**: Establish a baseline scenario and adjust individual conditions (such as rainfall or fertilizer) to immediately see the absolute and percentage change in estimated yield.
- **🌾 Compare Crops**: Select multiple crops under the exact same agricultural and environmental conditions to compare their estimated yields side-by-side.
- **📈 Crop Trends**: View historical actual yield trends over time for specific crops in specific regions. (This relies entirely on real historical dataset records).
- **💡 Insights**: Dive deeper into the technical details of the estimation engine, including historical test set performance, feature importance (which variables the model relied on most strongly), and a comprehensive dataset overview.

## Dataset Details
This application relies on historical Indian agricultural data (`data/crop_yield.csv`). 
- **Available Years**: Historical data is available up to 2020. 
- **Target Variable**: The application predicts `Yield`.
- **Target Leakage Prevention**: `Production` is intentionally excluded from the model features, as it is directly related to cultivated area and output.

## ML Approach
The application utilizes a `RandomForestRegressor` within a Scikit-Learn pipeline to estimate yields based on the relationships learned from the historical data. 

## Limitations
- **Estimation, Not Prophecy**: The 2026 results are an estimated yield based on historical patterns and user-entered conditions. They are not predictions validated against actual 2026 outcomes, as the dataset currently only contains historical records through 2020. 
- **Model Features vs Context Year**: To prevent inappropriate time-series extrapolation, the specific `Crop_Year` is intentionally omitted from the Random Forest's predictive features. The model makes estimations based purely on the historical agricultural patterns (Crop, State, Season, Area, Rainfall, Fertilizer, Pesticide).
- **Historical Bounds**: If a user enters conditions (like extreme rainfall or fertilizer amounts) that fall outside the historical ranges present in the dataset, the reliability of the estimate may decrease.

## How to Run

1. Ensure you have Python installed.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the application:
   ```bash
   streamlit run app.py
   ```
4. The application will automatically open in your default web browser.
