# ⚡ OptiWatt - AI Energy Intelligence Platform

An AI-powered energy consumption prediction and optimization dashboard built with Streamlit, XGBoost, and advanced time series analysis.

## 🌟 Features

### 🤖 AI Energy Intelligence Agent
- Natural language query interface
- Contextual energy predictions
- Personalized recommendations
- Risk assessment and alerts

### 📊 Advanced Analytics
- Hourly and daily consumption patterns
- Time series visualization
- Distribution analysis
- Peak usage identification

### 🔮 Predictive Modeling
- XGBoost-based predictions
- Multi-horizon forecasting
- Confidence intervals
- Historical comparison

### 💡 Smart Recommendations
- Time-shifting strategies
- Appliance optimization
- Behavioral insights
- Sustainability impact metrics

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone or navigate to the project directory**
```bash
cd OptiWatt
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Train and save the model (if not already done)**
   - Open and run all cells in `notebooks/01_data_preprocessing.ipynb`
   - This will generate the model files in the `models/` directory

4. **Run the Streamlit dashboard**
```bash
streamlit run app.py
```

5. **Access the dashboard**
   - Open your browser to `http://localhost:8501`

## 📁 Project Structure

```
OptiWatt/
├── app.py                          # Main Streamlit application
├── data/
│   └── household_power_consumption.txt  # Raw energy data
├── models/
│   ├── xgb_model.pkl              # Trained XGBoost model
│   ├── scaler.pkl                 # Feature scaler
│   └── feature_cols.pkl           # Feature column names
├── notebooks/
│   └── 01_data_preprocessing.ipynb  # Data preprocessing & training
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🎯 How to Use the Dashboard

### 1. 🏠 Home & AI Agent
Ask natural language questions like:
- "How much energy will be consumed after 1 hour?"
- "How can I save energy?"
- "When is my peak usage time?"
- "Why is energy consumption high?"

### 2. 📈 Analytics
Explore visualizations:
- **Hourly Pattern**: Average consumption by hour of day
- **Daily Pattern**: Consumption trends across the week
- **Distribution**: Statistical distribution of usage
- **Time Series**: Recent consumption trends

### 3. 🔮 Predictions
Generate forecasts:
- Select prediction horizon (1-120 minutes)
- View predicted consumption
- Assess risk levels
- Compare with historical patterns

### 4. 💡 Recommendations
Get personalized advice:
- Identify peak and off-peak hours
- Review actionable recommendations
- Calculate environmental impact
- Track potential savings

## 🧠 AI Agent Capabilities

The AI Energy Intelligence Agent can handle:

### Future Predictions
```
"How much energy will be consumed after 1 hour?"
"Predict energy usage tomorrow evening"
```

### Energy Saving
```
"How can I save energy?"
"Give recommendations for reducing power usage"
```

### Peak Analysis
```
"When is my peak usage time?"
"Will there be a power spike after 1 hour?"
```

### Explanations
```
"Why is energy usage high?"
"Explain the prediction"
```

## 📊 Model Performance

- **Algorithm**: XGBoost Regressor
- **R² Score**: ~0.89
- **MAE**: ~0.13 kW
- **Features**: Lag features, rolling statistics, time-based features
- **Horizon**: 1-minute ahead predictions

## 🌍 Sustainability Impact

The dashboard includes environmental impact calculations:
- CO₂ emissions reduction
- Tree planting equivalents
- Energy savings potential
- Green energy recommendations

## 🔧 Technical Details

### Features Used
- **Lag features**: 1, 2, 5, 10, 30, 60 minutes
- **Rolling statistics**: Mean and standard deviation (10-minute window)
- **Time features**: Hour of day, day of week, weekend indicator

### Model Configuration
```python
XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror"
)
```

## 📝 Dataset

The project uses the **UCI Household Power Consumption** dataset containing:
- **Period**: 2006-2010
- **Frequency**: 1-minute intervals
- **Features**: Global active power, voltage, intensity, sub-metering
- **Size**: ~2 million observations

## 🛠️ Customization

### Modify Prediction Horizon
Edit the `HORIZON` variable in the notebook or adjust in the dashboard slider.

### Add New Features
Update the feature engineering section in `01_data_preprocessing.ipynb` and retrain the model.

### Customize Recommendations
Modify the `ai_energy_agent()` function in `app.py` to add custom logic.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the dataset
- XGBoost developers for the excellent ML library
- Streamlit team for the amazing framework

## 📧 Support

For questions or issues, please open an issue in the repository.

---

**Built with ❤️ for a sustainable future**
