"""
Test script to verify model can make predictions
"""
import pandas as pd
import numpy as np
import joblib

print("Loading model and data...")
model = joblib.load('models/xgb_model.pkl')
feature_cols = joblib.load('models/feature_cols.pkl')

print(f"\nModel expects {len(feature_cols)} features:")
print(feature_cols)

# Load data
df = pd.read_csv("data/household_power_consumption.txt", sep=";")
df.replace("?", np.nan, inplace=True)
df = df.dropna()

for col in df.columns:
    if col not in ["Date", "Time"]:
        df[col] = df[col].astype(float)

df["Datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    format="%d/%m/%Y %H:%M:%S"
)
df = df.sort_values("Datetime")
df = df.set_index("Datetime")
df = df.drop(columns=["Date", "Time"])

# Add time features
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

print(f"\nLoaded {len(df)} data points")
print(f"Columns in data: {df.columns.tolist()}")

# Create features for prediction
df_pred = df.copy()

# Create lag features
lag_list = [1, 2, 5, 10, 30, 60]
for lag in lag_list:
    df_pred[f'lag_{lag}'] = df_pred['Global_active_power'].shift(lag)

# Create rolling features
df_pred['rolling_mean'] = df_pred['Global_active_power'].shift(1).rolling(10).mean()
df_pred['rolling_std'] = df_pred['Global_active_power'].shift(1).rolling(10).std()

# Drop leaky features
leaky_features = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]
cols_to_drop = [col for col in leaky_features if col in df_pred.columns]
df_pred = df_pred.drop(columns=cols_to_drop)
df_pred = df_pred.dropna()

print(f"\nProcessed data shape: {df_pred.shape}")
print(f"Columns available: {df_pred.columns.tolist()}")

# Select features for prediction
X = df_pred[feature_cols]
print(f"\nFeatures for prediction shape: {X.shape}")

# Make prediction on last row
latest = X.iloc[[-1]]
print(f"\nMaking prediction on latest data point...")
prediction = model.predict(latest)

print(f"\n✅ SUCCESS! Predicted energy consumption: {prediction[0]:.2f} kW")
print(f"\nActual average in data: {df['Global_active_power'].mean():.2f} kW")
print(f"Prediction is within reasonable range: {0.5 < prediction[0] < 10}")
