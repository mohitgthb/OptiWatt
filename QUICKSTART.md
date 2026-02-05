# OptiWatt Dashboard - Quick Start Guide

## ✅ Completed Setup

Your OptiWatt AI Energy Intelligence Platform is ready to use!

### What's Been Set Up:

1. ✅ **XGBoost Model Trained & Saved**
   - Location: `models/xgb_model.pkl`
   - Performance: R² = 0.89, MAE = 0.13 kW

2. ✅ **Streamlit Dashboard Created**
   - File: `app.py`
   - Features: AI Agent, Analytics, Predictions, Recommendations

3. ✅ **Dependencies Updated**
   - `requirements.txt` includes streamlit and joblib

---

## 🚀 How to Run the Dashboard

### Option 1: Using the Batch File (Windows)
Simply double-click: **`run_dashboard.bat`**

### Option 2: Using Command Line
```bash
cd d:\PROJECTS\OptiWatt
streamlit run app.py
```

### Option 3: Install Dependencies First (if needed)
```bash
pip install streamlit joblib
streamlit run app.py
```

---

## 🎯 Using the Dashboard

Once the dashboard starts (opens at `http://localhost:8501`):

### 1. 🏠 Home & AI Agent Tab
**Ask Questions Like:**
- "How much energy will be consumed after 1 hour?"
- "How can I save energy?"
- "When is my peak usage time?"
- "Why is energy consumption high?"
- "Give me recommendations for reducing power usage"

**The AI will provide:**
- 🔮 Predictions with risk levels
- 📖 Clear explanations
- 💡 Actionable recommendations
- 🌍 Sustainability impact

### 2. 📈 Analytics Tab
**Explore:**
- Hourly consumption patterns
- Daily trends
- Distribution analysis
- Time series visualizations

### 3. 🔮 Predictions Tab
**Generate Forecasts:**
- Adjust prediction horizon (1-120 minutes)
- View predicted consumption
- See risk assessment
- Compare with historical data

### 4. 💡 Recommendations Tab
**Get Personalized Advice:**
- Peak vs. off-peak hours
- Energy-saving strategies
- Environmental impact calculator
- Potential savings estimation

---

## 🧠 AI Agent Features

### Query Types Supported:

#### ⚡ Future Predictions
```
Input: "How much energy will be consumed after 1 hour?"
Output: 
- Predicted value (e.g., 2.45 kW)
- Risk level (Low/Medium/High)
- Explanation of factors
- Recommendations
```

#### 💰 Energy Saving
```
Input: "How can I save energy?"
Output:
- Time-shifting strategies
- Appliance optimization tips
- Behavioral recommendations
- Estimated savings
```

#### 📊 Peak Analysis
```
Input: "When is my peak usage time?"
Output:
- Peak hour identification
- Usage comparison
- Prevention strategies
- Cost implications
```

#### ❓ Explanations
```
Input: "Why is energy consumption high?"
Output:
- Pattern analysis
- Contributing factors
- Simple, non-technical language
- Actionable insights
```

---

## 📊 Dashboard Pages Overview

### Page 1: Home & AI Agent
- **Natural Language Interface**: Ask questions in plain English
- **Smart Responses**: Context-aware answers with predictions
- **Risk Assessment**: Visual indicators (🟢 🟡 🔴)
- **Example Queries**: Click to try pre-made questions

### Page 2: Analytics
- **Hourly Pattern**: Bar chart of average consumption by hour
- **Daily Pattern**: Line chart showing weekly trends
- **Distribution**: Histogram with statistical measures
- **Time Series**: Recent consumption trends (last 7 days)

### Page 3: Predictions
- **Horizon Selector**: Choose 1-120 minutes ahead
- **Prediction Display**: Value with comparison to average
- **Risk Indicator**: Visual risk level
- **Visualization**: Prediction in context of recent history

### Page 4: Recommendations
- **Peak Hours**: Identified high-consumption periods
- **Off-Peak Hours**: Best times for energy-intensive tasks
- **Action Plan**: 5 categories of recommendations
- **Impact Calculator**: CO₂ savings and tree equivalents

---

## 💡 Best Practices

### For Accurate Predictions:
1. ✅ Ask about near-term horizons (1-2 hours)
2. ✅ Use specific time frames
3. ✅ Consider current time of day
4. ⚠️ Long-term predictions (>24 hours) are estimates

### For Energy Savings:
1. 🕐 Shift heavy appliance use to off-peak hours
2. 🌡️ Adjust thermostat during peak times
3. 💡 Replace bulbs with LEDs
4. 🔌 Eliminate phantom loads
5. 📊 Monitor regularly using the dashboard

---

## 🌍 Sustainability Metrics

The dashboard calculates:
- **CO₂ Reduction**: kg of carbon dioxide saved
- **Tree Equivalent**: Number of trees that absorb the same CO₂
- **Energy Savings**: kWh saved per month
- **Cost Savings**: Estimated monetary savings

**Example**: 20% reduction = ~500 kg CO₂/year = 23 trees planted

---

## 🔧 Troubleshooting

### Dashboard Won't Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run app.py
```

### Model Not Found Error
```bash
# Run the notebook to generate model files
# Open: notebooks/01_data_preprocessing.ipynb
# Execute all cells
```

### Data Not Loading
- Ensure `data/household_power_consumption.txt` exists
- Check file permissions
- Verify file format (semicolon-separated)

---

## 📈 Model Details

**Algorithm**: XGBoost Regressor
**Training Data**: 2006-2010 household power consumption
**Update Frequency**: Retrain monthly for best accuracy
**Features**: 
- Lag: 1, 2, 5, 10, 30, 60 minutes
- Rolling: mean & std (10-min window)
- Time: hour, day of week, weekend flag

---

## 🎨 Customization

### Change Color Scheme
Edit the CSS in `app.py` (lines 18-52)

### Add New Queries
Modify `ai_energy_agent()` function in `app.py`

### Adjust Prediction Horizon
Change slider range in Predictions page code

### Custom Recommendations
Edit recommendation logic in `ai_energy_agent()`

---

## 📞 Support

**Issues?** Check:
1. Python version (3.8+)
2. All dependencies installed
3. Model files exist in `models/`
4. Data file in `data/`

**Need Help?**
- Review README.md
- Check error messages in terminal
- Verify file paths are correct

---

## 🎯 Next Steps

1. ✅ Install dependencies: `pip install -r requirements.txt`
2. ✅ Run dashboard: `streamlit run app.py` or double-click `run_dashboard.bat`
3. ✅ Explore all 4 pages
4. ✅ Try the AI agent with different queries
5. ✅ Review your energy consumption patterns
6. ✅ Implement recommended energy-saving strategies

---

**Enjoy your AI Energy Intelligence Platform! ⚡🌍**
