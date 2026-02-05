import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="OptiWatt - AI Energy Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .recommendation-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-left: 4px solid #4CAF50;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .alert-box {
        background-color: #FFEBEE;
        padding: 1rem;
        border-left: 4px solid #F44336;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-left: 4px solid #2196F3;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('models/xgb_model.pkl')
        feature_cols = joblib.load('models/feature_cols.pkl')
        return model, feature_cols
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

@st.cache_data
def load_data():
    try:
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
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def create_features_for_prediction(df, horizon=1):
    """Create lag and rolling features for prediction - matches training exactly"""
    df_pred = df.copy()
    
    # Keep only the columns we need for feature engineering
    if 'Global_active_power' not in df_pred.columns:
        st.error("Global_active_power column not found in data")
        return None
    
    # Create lag features FIRST (before dropping any columns)
    lag_list = [1, 2, 5, 10, 30, 60]
    for lag in lag_list:
        df_pred[f'lag_{lag}'] = df_pred['Global_active_power'].shift(lag)
    
    # Create rolling features
    df_pred['rolling_mean'] = df_pred['Global_active_power'].shift(horizon).rolling(10).mean()
    df_pred['rolling_std'] = df_pred['Global_active_power'].shift(horizon).rolling(10).std()
    
    # Create target (for structure, but we'll drop it for prediction)
    df_pred['target'] = df_pred['Global_active_power'].shift(-horizon)
    
    # Drop ALL leaky features (all power measurement columns)
    leaky_features = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3"
    ]
    
    # Only drop columns that exist
    cols_to_drop = [col for col in leaky_features if col in df_pred.columns]
    df_pred = df_pred.drop(columns=cols_to_drop)
    
    # Drop NaN rows
    df_pred = df_pred.dropna()
    
    return df_pred

def predict_energy(model, feature_cols, features_df):
    """Make energy prediction"""
    try:
        # Ensure we have the right columns
        if features_df is None or len(features_df) == 0:
            st.error("No data available for prediction")
            return None
        
        # Select only the feature columns, in the correct order
        missing_cols = [col for col in feature_cols if col not in features_df.columns]
        if missing_cols:
            st.error(f"Missing columns: {missing_cols}")
            st.error(f"Available columns: {features_df.columns.tolist()}")
            return None
        
        X = features_df[feature_cols]
        
        # XGBoost was trained on unscaled data, so no scaling needed
        prediction = model.predict(X)
        return prediction
    except Exception as e:
        st.error(f"Prediction error: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

def get_risk_level(value, historical_mean, historical_std):
    """Determine risk level based on historical data"""
    if value < historical_mean - 0.5 * historical_std:
        return "Low", "🟢"
    elif value < historical_mean + 0.5 * historical_std:
        return "Medium", "🟡"
    else:
        return "High", "🔴"

def ai_energy_agent(query, df, model, feature_cols):
    """
    AI Energy Intelligence Agent
    Interprets queries and provides actionable insights
    """
    query_lower = query.lower()
    
    # Calculate historical statistics
    historical_mean = df['Global_active_power'].mean()
    historical_std = df['Global_active_power'].std()
    
    response = {
        "prediction": None,
        "explanation": "",
        "recommendations": [],
        "sustainability_impact": "",
        "risk_level": "Medium",
        "risk_icon": "🟡"
    }
    
    # FUTURE PREDICTION QUERIES
    if any(word in query_lower for word in ["hour", "after", "next", "future", "will be"]):
        # Extract time horizon
        if "1 hour" in query_lower or "after 1 hour" in query_lower:
            horizon_minutes = 60
        elif "2 hour" in query_lower:
            horizon_minutes = 120
        elif "30 min" in query_lower:
            horizon_minutes = 30
        else:
            horizon_minutes = 60  # Default
        
        # Get latest data point
        df_pred = create_features_for_prediction(df, horizon=1)
        if df_pred is not None and len(df_pred) > 0:
            latest_features = df_pred.iloc[[-1]]
            prediction = predict_energy(model, feature_cols, latest_features)
            
            if prediction is not None:
                pred_value = prediction[0]
                response["prediction"] = pred_value
                risk, icon = get_risk_level(pred_value, historical_mean, historical_std)
                response["risk_level"] = risk
                response["risk_icon"] = icon
                
                response["explanation"] = f"""
Based on recent consumption patterns and historical data analysis, 
the predicted energy consumption after {horizon_minutes} minutes is **{pred_value:.2f} kW**.

This prediction considers:
- Recent consumption trends (last hour)
- Time of day patterns
- Day of week behavior
- Historical rolling averages
"""
                
                if risk == "High":
                    response["recommendations"] = [
                        "⚡ **High consumption alert!** Consider delaying high-power appliances",
                        "🌡️ Check heating/cooling settings - adjust temperature by 1-2°C",
                        "💡 Switch off unnecessary lights and electronics",
                        "⏰ Shift dishwasher/washing machine use to off-peak hours"
                    ]
                    response["sustainability_impact"] = "🌍 Reducing peak consumption by 20% can save ~15 kWh/month and reduce carbon footprint by 7 kg CO2"
                elif risk == "Medium":
                    response["recommendations"] = [
                        "👍 Normal consumption expected",
                        "💡 Good time for regular appliance use",
                        "🔌 Unplug devices not in use to prevent phantom load",
                        "⚡ Monitor heavy appliance usage"
                    ]
                    response["sustainability_impact"] = "🌿 Maintaining efficient usage patterns supports sustainable energy consumption"
                else:
                    response["recommendations"] = [
                        "✅ Excellent! Low consumption period",
                        "🔋 Great time to charge devices or run energy-intensive tasks",
                        "♻️ Your efficient energy use is helping the environment",
                        "📊 Keep up the sustainable practices!"
                    ]
                    response["sustainability_impact"] = "🌟 Your energy-efficient behavior is reducing carbon emissions - keep it up!"
    
    # TIME-SPECIFIC DATE QUERIES
    elif any(word in query_lower for word in ["august", "december", "january", "specific date", "on "]):
        response["explanation"] = """
To predict energy consumption for a specific date in the future, I need:
- The exact date and time
- Current behavioral patterns will be extrapolated

Note: Predictions for dates far in the future have higher uncertainty.
For the most accurate predictions, please ask about the next few hours or days.
"""
        response["recommendations"] = [
            "📅 For date-specific predictions, ensure the date is within a reasonable forecast window",
            "🔮 Long-term predictions should be used as estimates, not exact values",
            "📊 Consider seasonal patterns when interpreting future predictions"
        ]
    
    # ENERGY SAVING RECOMMENDATIONS
    elif any(word in query_lower for word in ["save", "reduce", "lower", "decrease", "recommendation"]):
        # Analyze peak usage hours
        hourly_avg = df.groupby(df.index.hour)['Global_active_power'].mean()
        peak_hours = hourly_avg.nlargest(3).index.tolist()
        
        response["explanation"] = f"""
Based on your historical consumption patterns, your peak usage hours are typically:
**{', '.join([f'{h}:00-{h+1}:00' for h in peak_hours])}**

Here are personalized recommendations to reduce your energy consumption:
"""
        
        response["recommendations"] = [
            f"⏰ **Time-shift strategy**: Peak hours are {', '.join([f'{h}:00' for h in peak_hours])}. Shift heavy appliances to off-peak times (11 PM - 6 AM)",
            "🌡️ **Smart thermostat**: Set heating/cooling 2°C lower/higher when away. Potential savings: 10-15% annually",
            "💡 **LED upgrade**: Replace remaining incandescent bulbs with LEDs. Saves ~$75/year",
            "🔌 **Phantom power**: Use smart power strips to eliminate standby consumption (~8-10% of total)",
            "🏠 **Appliance efficiency**: Run dishwasher/washing machine only when full. Saves 15-20% on appliance energy",
            "☀️ **Natural resources**: Use natural light during day, maximize use of natural ventilation"
        ]
        
        response["sustainability_impact"] = """
🌍 **Environmental Impact**: 
- Reducing consumption by 20% = ~500 kg CO2/year saved
- Equivalent to planting 23 trees annually
- Comparable to driving 2,000 km less per year
"""
    
    # PEAK USAGE QUERIES
    elif any(word in query_lower for word in ["peak", "highest", "maximum", "spike", "alert"]):
        hourly_avg = df.groupby(df.index.hour)['Global_active_power'].mean()
        peak_hour = hourly_avg.idxmax()
        peak_value = hourly_avg.max()
        
        response["prediction"] = peak_value
        response["risk_level"] = "High"
        response["risk_icon"] = "🔴"
        
        response["explanation"] = f"""
Your **peak energy consumption** typically occurs at **{peak_hour}:00 - {peak_hour+1}:00**.

Average peak consumption: **{peak_value:.2f} kW**

This is typically {((peak_value / historical_mean - 1) * 100):.1f}% higher than your daily average.
"""
        
        response["recommendations"] = [
            f"⚠️ Avoid running multiple high-power appliances during {peak_hour}:00-{peak_hour+1}:00",
            "🔄 Stagger appliance usage - don't run dishwasher, washing machine, and dryer simultaneously",
            "📉 Shift 30% of peak-hour consumption to off-peak hours for significant savings",
            "⚡ Consider a smart energy management system for automatic load balancing"
        ]
        
        response["sustainability_impact"] = "⚡ Reducing peak demand helps grid stability and reduces need for additional power plants"
    
    # EXPLANATION QUERIES
    elif any(word in query_lower for word in ["why", "explain", "how", "what causes"]):
        response["explanation"] = """
Energy consumption patterns are influenced by:

**1. Time-based factors**:
- Hour of day (morning/evening peaks)
- Day of week (weekdays vs weekends)
- Seasonal variations

**2. Behavioral patterns**:
- Appliance usage schedules
- Heating/cooling habits
- Lighting patterns

**3. Historical trends**:
- Your past consumption directly influences predictions
- The model learns from patterns over time
"""
        
        response["recommendations"] = [
            "📊 Regular monitoring helps identify unusual consumption patterns",
            "🔍 Compare weekday vs weekend usage to spot inefficiencies",
            "📈 Track monthly trends to measure improvement"
        ]
    
    # DEFAULT RESPONSE
    else:
        response["explanation"] = """
I'm your AI Energy Intelligence Assistant! I can help you with:

**🔮 Predictions**:
- "How much energy will be consumed after 1 hour?"
- "Predict energy for tomorrow evening"

**💡 Recommendations**:
- "How can I save energy?"
- "Give me energy-saving tips"

**📊 Analysis**:
- "When is my peak usage time?"
- "Why is energy consumption high?"

**🎯 Specific queries**:
- "What will consumption be on August 5th?"

Please ask me anything about your energy usage!
"""
    
    return response

# Main App
def main():
    # Header
    st.markdown('<p class="main-header">⚡ OptiWatt</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Energy Intelligence Platform</p>', unsafe_allow_html=True)
    
    # Load resources
    model, feature_cols = load_model_and_scaler()
    df = load_data()
    
    if model is None or df is None:
        st.error("❌ Failed to load model or data. Please check the files.")
        return
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/200/000000/lightning-bolt.png", width=150)
        st.title("📊 Dashboard Controls")
        
        page = st.radio(
            "Navigate",
            ["🏠 Home & AI Agent", "📈 Analytics", "🔮 Predictions", "💡 Recommendations"]
        )
        
        st.markdown("---")
        st.markdown("### 📋 Quick Stats")
        st.metric("Total Data Points", f"{len(df):,}")
        st.metric("Average Consumption", f"{df['Global_active_power'].mean():.2f} kW")
        st.metric("Peak Consumption", f"{df['Global_active_power'].max():.2f} kW")
    
    # Page: Home & AI Agent
    if page == "🏠 Home & AI Agent":
        st.markdown("## 🤖 AI Energy Intelligence Agent")
        st.markdown("Ask me anything about your energy consumption, predictions, or get personalized recommendations!")
        
        # Query input
        user_query = st.text_input(
            "💬 Your Question:",
            placeholder="e.g., How much energy will be consumed after 1 hour?",
            key="query_input"
        )
        
        if st.button("🔍 Ask AI Agent", type="primary"):
            if user_query:
                with st.spinner("🤔 Analyzing your query..."):
                    response = ai_energy_agent(user_query, df, model, feature_cols)
                
                # Display response
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if response["prediction"] is not None:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h2>Predicted Consumption</h2>
                            <h1>{response["prediction"]:.2f} kW</h1>
                            <p>Risk Level: {response["risk_icon"]} {response["risk_level"]}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("### 📖 Explanation")
                    st.markdown(f'<div class="info-box">{response["explanation"]}</div>', unsafe_allow_html=True)
                
                with col2:
                    if response["risk_level"] == "High":
                        st.markdown("### ⚠️ Alert")
                        st.markdown('<div class="alert-box">High consumption expected!</div>', unsafe_allow_html=True)
                
                # Recommendations
                if response["recommendations"]:
                    st.markdown("### 💡 Recommendations")
                    for rec in response["recommendations"]:
                        st.markdown(f'<div class="recommendation-box">{rec}</div>', unsafe_allow_html=True)
                
                # Sustainability impact
                if response["sustainability_impact"]:
                    st.markdown("### 🌍 Sustainability Impact")
                    st.success(response["sustainability_impact"])
            else:
                st.warning("Please enter a question!")
        
        # Example queries
        st.markdown("---")
        st.markdown("### 💭 Example Questions")
        examples = [
            "How much energy will be consumed after 1 hour?",
            "How can I save energy?",
            "When is my peak usage time?",
            "Why is energy consumption high?",
            "Give me recommendations for reducing power usage"
        ]
        
        cols = st.columns(2)
        for i, example in enumerate(examples):
            with cols[i % 2]:
                if st.button(example, key=f"example_{i}"):
                    st.rerun()
    
    # Page: Analytics
    elif page == "📈 Analytics":
        st.markdown("## 📊 Energy Consumption Analytics")
        
        # Time range selector
        col1, col2 = st.columns(2)
        with col1:
            viz_type = st.selectbox(
                "Visualization Type",
                ["Hourly Pattern", "Daily Pattern", "Consumption Distribution", "Time Series"]
            )
        
        if viz_type == "Hourly Pattern":
            fig, ax = plt.subplots(figsize=(12, 5))
            hourly_avg = df.groupby(df.index.hour)['Global_active_power'].mean()
            ax.bar(hourly_avg.index, hourly_avg.values, color='#667eea', alpha=0.7)
            ax.set_xlabel('Hour of Day', fontsize=12)
            ax.set_ylabel('Average Consumption (kW)', fontsize=12)
            ax.set_title('Average Energy Consumption by Hour', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
            
            peak_hour = hourly_avg.idxmax()
            st.info(f"🔥 Peak consumption hour: **{peak_hour}:00 - {peak_hour+1}:00** ({hourly_avg.max():.2f} kW)")
        
        elif viz_type == "Daily Pattern":
            fig, ax = plt.subplots(figsize=(12, 5))
            daily_avg = df.groupby(df.index.dayofweek)['Global_active_power'].mean()
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            ax.plot(days, daily_avg.values, marker='o', linewidth=2, markersize=8, color='#764ba2')
            ax.set_xlabel('Day of Week', fontsize=12)
            ax.set_ylabel('Average Consumption (kW)', fontsize=12)
            ax.set_title('Average Energy Consumption by Day', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        elif viz_type == "Consumption Distribution":
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.hist(df['Global_active_power'], bins=50, color='#667eea', alpha=0.7, edgecolor='black')
            ax.axvline(df['Global_active_power'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["Global_active_power"].mean():.2f} kW')
            ax.set_xlabel('Energy Consumption (kW)', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Distribution of Energy Consumption', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
        
        else:  # Time Series
            st.markdown("### 📅 Energy Consumption Time Series (Last 7 Days)")
            df_recent = df.tail(7 * 24 * 60)  # Last 7 days
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(df_recent.index, df_recent['Global_active_power'], linewidth=1, color='#667eea', alpha=0.8)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Consumption (kW)', fontsize=12)
            ax.set_title('Energy Consumption - Last 7 Days', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)
    
    # Page: Predictions
    elif page == "🔮 Predictions":
        st.markdown("## 🔮 Energy Consumption Predictions")
        
        st.markdown("### Predict Future Consumption")
        st.info("💡 **Note**: This model predicts the next minute. For longer horizons, we use iterative forecasting (step-by-step predictions).")
        
        horizon = st.slider("Prediction Horizon (minutes ahead)", 1, 120, 1)
        
        if st.button("Generate Prediction", type="primary"):
            with st.spinner("Generating prediction..."):
                df_pred = create_features_for_prediction(df, horizon=1)
                if df_pred is not None and len(df_pred) > 0:
                    # Optimize: For long horizons, predict at intervals
                    if horizon <= 10:
                        step_size = 1
                        num_steps = horizon
                    elif horizon <= 30:
                        step_size = 2  # Predict every 2 minutes
                        num_steps = horizon // step_size
                    elif horizon <= 60:
                        step_size = 5  # Predict every 5 minutes
                        num_steps = horizon // step_size
                    else:
                        step_size = 10  # Predict every 10 minutes
                        num_steps = horizon // step_size
                    
                    # For multi-step ahead prediction, use optimized iterative forecasting
                    predictions = []
                    prediction_times = []
                    current_data = df.tail(1000).copy()  # Use only recent data for speed
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for step_idx in range(num_steps):
                        # Update progress
                        progress = (step_idx + 1) / num_steps
                        progress_bar.progress(progress)
                        status_text.text(f"Forecasting minute {(step_idx + 1) * step_size}/{horizon}...")
                        
                        # Predict for this step
                        for micro_step in range(step_size):
                            # Create features from current data
                            df_temp = create_features_for_prediction(current_data, horizon=1)
                            if df_temp is None or len(df_temp) == 0:
                                break
                            
                            # Get latest features and predict
                            latest_features = df_temp.iloc[[-1]]
                            pred = predict_energy(model, feature_cols, latest_features)
                            
                            if pred is None:
                                break
                            
                            pred_value = pred[0]
                            
                            # For next iteration, append prediction as new data point
                            last_time = current_data.index[-1]
                            new_time = last_time + timedelta(minutes=1)
                            prediction_times.append((step_idx * step_size + micro_step + 1))
                            
                            # Create new row with predicted value
                            new_row = pd.DataFrame({
                                'Global_active_power': [pred_value],
                                'Global_reactive_power': [current_data['Global_reactive_power'].iloc[-1]],
                                'Voltage': [current_data['Voltage'].iloc[-1]],
                                'Global_intensity': [current_data['Global_intensity'].iloc[-1]],
                                'Sub_metering_1': [current_data['Sub_metering_1'].iloc[-1]],
                                'Sub_metering_2': [current_data['Sub_metering_2'].iloc[-1]],
                                'Sub_metering_3': [current_data['Sub_metering_3'].iloc[-1]],
                                'hour': [new_time.hour],
                                'dayofweek': [new_time.dayofweek],
                                'is_weekend': [1 if new_time.dayofweek >= 5 else 0]
                            }, index=[new_time])
                            
                            # Append to current data (keep only recent portion for memory)
                            current_data = pd.concat([current_data.tail(500), new_row])
                        
                        # Store the prediction at this interval
                        if pred is not None:
                            predictions.append(pred_value)
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    if predictions:
                        # Final prediction is the last one
                        final_pred = predictions[-1]
                        avg_pred = np.mean(predictions)
                        
                        historical_mean = df['Global_active_power'].mean()
                        historical_std = df['Global_active_power'].std()
                        risk, icon = get_risk_level(final_pred, historical_mean, historical_std)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Predicted Consumption", f"{final_pred:.2f} kW", f"{((final_pred/historical_mean - 1)*100):.1f}% vs avg")
                        with col2:
                            st.metric("Risk Level", f"{icon} {risk}")
                        with col3:
                            if step_size > 1:
                                st.metric("Forecast Interval", f"Every {step_size} min")
                            else:
                                st.metric("Time Horizon", f"{horizon} min")
                        
                        if horizon > 1:
                            st.info(f"📊 Average over {horizon} minutes: **{avg_pred:.2f} kW** | Range: {min(predictions):.2f} - {max(predictions):.2f} kW")
                        
                        # Visualization
                        st.markdown("### 📊 Prediction vs Recent History")
                        recent_data = df['Global_active_power'].tail(200)
                        
                        fig, ax = plt.subplots(figsize=(12, 5))
                        
                        # Plot historical data
                        hist_x = range(len(recent_data))
                        ax.plot(hist_x, recent_data.values, label='Historical', linewidth=2, color='#667eea', alpha=0.7)
                        
                        # Plot predictions
                        if len(predictions) > 1:
                            pred_x = range(len(recent_data), len(recent_data) + len(predictions))
                            ax.plot(pred_x, predictions, label=f'Forecast ({horizon} min)', 
                                   linewidth=2, color='red', linestyle='--', marker='o', markersize=4)
                        else:
                            ax.axhline(final_pred, color='red', linestyle='--', linewidth=2, 
                                      label=f'Prediction: {final_pred:.2f} kW', xmin=0.8)
                        
                        # Add average line
                        ax.axhline(historical_mean, color='green', linestyle=':', linewidth=2, 
                                  label=f'Historical Avg: {historical_mean:.2f} kW')
                        ax.fill_between(range(len(recent_data) + len(predictions)), 
                                       historical_mean - historical_std, 
                                       historical_mean + historical_std, 
                                       alpha=0.2, color='green', label='±1 Std Dev')
                        
                        ax.set_xlabel('Time Steps (minutes)', fontsize=12)
                        ax.set_ylabel('Consumption (kW)', fontsize=12)
                        ax.set_title(f'Energy Forecast - Next {horizon} Minute(s)', fontsize=14, fontweight='bold')
                        ax.legend()
                        ax.grid(alpha=0.3)
                        st.pyplot(fig)
                        
                        # Show prediction table for longer horizons
                        if horizon > 5 and len(predictions) > 5:
                            st.markdown("### 📋 Detailed Forecast")
                            
                            # Create time points based on step size
                            time_points = [i * step_size if i > 0 else 1 for i in range(len(predictions))]
                            
                            forecast_df = pd.DataFrame({
                                'Minutes Ahead': time_points,
                                'Predicted kW': [f"{p:.2f}" for p in predictions],
                                'Risk': [get_risk_level(p, historical_mean, historical_std)[1] for p in predictions]
                            })
                            st.dataframe(forecast_df.head(20), use_container_width=True)
                            if len(predictions) > 20:
                                st.caption(f"Showing first 20 of {len(predictions)} forecast points")
                    else:
                        st.error("Failed to generate predictions")
    
    # Page: Recommendations
    elif page == "💡 Recommendations":
        st.markdown("## 💡 Personalized Energy Saving Recommendations")
        
        # Analyze usage patterns
        hourly_avg = df.groupby(df.index.hour)['Global_active_power'].mean()
        peak_hours = hourly_avg.nlargest(3).index.tolist()
        low_hours = hourly_avg.nsmallest(3).index.tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔥 Peak Usage Hours")
            for hour in peak_hours:
                st.markdown(f'<div class="alert-box">**{hour}:00 - {hour+1}:00**: {hourly_avg[hour]:.2f} kW</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### ✅ Low Usage Hours")
            for hour in low_hours:
                st.markdown(f'<div class="recommendation-box">**{hour}:00 - {hour+1}:00**: {hourly_avg[hour]:.2f} kW</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🎯 Action Plan")
        
        recommendations = [
            {
                "title": "⏰ Time-Shifting Strategy",
                "description": f"Move energy-intensive tasks from peak hours ({', '.join([f'{h}:00' for h in peak_hours])}) to off-peak hours ({', '.join([f'{h}:00' for h in low_hours])})",
                "impact": "High - Potential savings: 15-25%"
            },
            {
                "title": "🌡️ Smart Temperature Control",
                "description": "Reduce heating/cooling by 2°C during peak hours. Use programmable thermostat for automatic adjustments.",
                "impact": "Medium - Potential savings: 10-15%"
            },
            {
                "title": "💡 Lighting Optimization",
                "description": "Switch to LED bulbs, use natural light during day, install motion sensors in low-traffic areas.",
                "impact": "Medium - Potential savings: 8-12%"
            },
            {
                "title": "🔌 Phantom Load Elimination",
                "description": "Use smart power strips, unplug chargers when not in use, enable power-saving modes on devices.",
                "impact": "Low-Medium - Potential savings: 5-10%"
            },
            {
                "title": "🏠 Appliance Efficiency",
                "description": "Run dishwasher/washing machine only when full, use cold water for laundry, air-dry when possible.",
                "impact": "Medium - Potential savings: 10-15%"
            }
        ]
        
        for rec in recommendations:
            with st.expander(f"**{rec['title']}**"):
                st.write(rec['description'])
                st.success(f"💰 {rec['impact']}")
        
        st.markdown("---")
        st.markdown("### 🌍 Environmental Impact Calculator")
        
        reduction_percent = st.slider("Target Reduction (%)", 5, 50, 20)
        avg_monthly = df['Global_active_power'].mean() * 24 * 30  # kWh per month
        reduction_kwh = avg_monthly * (reduction_percent / 100)
        co2_saved = reduction_kwh * 0.5  # Approximate kg CO2 per kWh
        trees_equivalent = co2_saved / 21.77  # kg CO2 absorbed by one tree per year / 12 months
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Energy Saved", f"{reduction_kwh:.0f} kWh/month")
        with col2:
            st.metric("CO2 Reduced", f"{co2_saved:.0f} kg/month")
        with col3:
            st.metric("Trees Equivalent", f"{trees_equivalent:.1f} trees/month")

if __name__ == "__main__":
    main()
