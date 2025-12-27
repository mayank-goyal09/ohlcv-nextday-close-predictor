import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Stock Price Predictor | ML Forecasting",
    page_icon="💹",
    layout="wide"
)

# ========= THEME: WALL STREET GLASS PANEL + DOLLAR GREEN =========
DARK = "#0A0E27"
DARK_PANEL = "#151932"
GLASS = "#1E2442"
GLASS_LIGHT = "#2A3154"
GREEN = "#00C853"
GREEN_DARK = "#00A844"
GREEN_LIGHT = "#4ADE80"
RED = "#FF1744"
GOLD = "#FFD700"
SILVER = "#C0C0C0"
WHITE = "#FFFFFF"
GREY = "#94A3B8"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Orbitron:wght@700;900&display=swap');
html, body, [class*="css"] {{ font-family: 'Rajdhani', sans-serif; }}

/* Dark Wall Street background */
.stApp {{
  background: linear-gradient(135deg, {DARK} 0%, #0F1629 50%, {DARK} 100%);
  background-attachment: fixed;
}}

.block-container {{ max-width: 1500px; padding-top: 1rem; }}
[data-testid="stHeader"] {{ background: transparent; }}

/* ALL TEXT */
html, body, p, li, span, div, label, small {{ color: {GREY} !important; }}
h1, h2, h3, h4 {{ 
  color: {GREEN_LIGHT} !important; 
  font-family: 'Orbitron', sans-serif !important;
  font-weight: 900 !important; 
  text-shadow: 0 0 20px rgba(0, 200, 83, 0.4);
  letter-spacing: 2px;
}}

/* HERO GLASS PANEL */
.hero {{
  position: relative;
  border-radius: 20px;
  padding: 32px 40px;
  background: linear-gradient(135deg, rgba(30, 36, 66, 0.95) 0%, rgba(21, 25, 50, 0.9) 100%);
  border: 2px solid rgba(0, 200, 83, 0.3);
  box-shadow: 0 0 40px rgba(0, 200, 83, 0.2), 0 20px 60px rgba(0,0,0,0.6);
  backdrop-filter: blur(10px);
  overflow: hidden;
  margin-bottom: 24px;
}}

/* Animated ticker tape */
@keyframes scroll {{
  0% {{ transform: translateX(100%); }}
  100% {{ transform: translateX(-100%); }}
}}

.ticker-tape {{
  position: absolute;
  top: 8px;
  left: 0;
  width: 100%;
  height: 4px;
  background: linear-gradient(90deg, transparent, {GREEN}, transparent);
  animation: scroll 3s linear infinite;
}}

.hero-icon {{
  position: absolute;
  right: 30px;
  top: 50%;
  transform: translateY(-50%);
  font-size: 100px;
  opacity: 0.08;
  filter: drop-shadow(0 0 30px {GREEN});
}}

.hero-title {{
  font-size: 2.8rem;
  font-weight: 900;
  color: {GREEN_LIGHT} !important;
  font-family: 'Orbitron', sans-serif;
  margin-bottom: 12px;
  text-transform: uppercase;
}}

.hero-sub {{
  color: {GREY} !important;
  font-size: 1.1rem;
  line-height: 1.7;
  opacity: 0.9;
}}

.badges {{
  display: flex;
  gap: 14px;
  flex-wrap: wrap;
  margin-top: 18px;
}}

.badge {{
  padding: 10px 18px;
  border-radius: 8px;
  font-size: 0.9rem;
  font-weight: 800;
  background: linear-gradient(135deg, {GREEN_DARK} 0%, {GREEN} 100%);
  color: {DARK} !important;
  border: 2px solid {GREEN_LIGHT};
  box-shadow: 0 4px 15px rgba(0, 200, 83, 0.3);
  text-transform: uppercase;
  letter-spacing: 1px;
}}

/* GLASS CARDS */
.card {{
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(30, 36, 66, 0.92) 0%, rgba(21, 25, 50, 0.88) 100%);
  border: 2px solid rgba(0, 200, 83, 0.25);
  box-shadow: 0 10px 40px rgba(0,0,0,0.5);
  backdrop-filter: blur(8px);
  padding: 28px;
  margin-bottom: 20px;
}}

.card-title {{
  font-weight: 900;
  font-size: 1.3rem;
  color: {GREEN_LIGHT} !important;
  margin-bottom: 10px;
  display: flex;
  align-items: center;
  gap: 12px;
  font-family: 'Orbitron', sans-serif;
  text-transform: uppercase;
  letter-spacing: 1.5px;
}}

.pulse-dot {{
  display: inline-block;
  width: 12px;
  height: 12px;
  background: {GREEN};
  border-radius: 50%;
  box-shadow: 0 0 20px {GREEN};
  animation: pulse-glow 1.5s ease-in-out infinite;
}}

@keyframes pulse-glow {{
  0%, 100% {{ opacity: 1; box-shadow: 0 0 20px {GREEN}; }}
  50% {{ opacity: 0.5; box-shadow: 0 0 10px {GREEN}; }}
}}

.card-sub {{
  color: {GREY} !important;
  font-size: 0.95rem;
  margin-bottom: 18px;
  opacity: 0.85;
}}

/* Inputs */
.stNumberInput input,
.stSelectbox select,
.stTextInput input {{
  background: {DARK_PANEL} !important;
  border: 2px solid rgba(0, 200, 83, 0.3) !important;
  border-radius: 12px !important;
  color: {GREEN_LIGHT} !important;
  font-weight: 700 !important;
}}

.stSelectbox div[data-baseweb="select"] > div {{
  background: {DARK_PANEL} !important;
  border: 2px solid rgba(0, 200, 83, 0.3) !important;
}}

.stSelectbox span {{
  color: {GREEN_LIGHT} !important;
  font-weight: 800 !important;
}}

/* Slider */
.stSlider > div > div > div {{
  background: {GREEN} !important;
}}

/* Buttons */
.stButton > button {{
  width: 100%;
  border: none;
  border-radius: 14px;
  padding: 1.1rem 1.6rem;
  font-weight: 900;
  font-size: 1.1rem;
  letter-spacing: 2px;
  color: {DARK} !important;
  background: linear-gradient(135deg, {GREEN_DARK} 0%, {GREEN} 50%, {GREEN_LIGHT} 100%);
  box-shadow: 0 15px 40px rgba(0, 200, 83, 0.4);
  text-transform: uppercase;
  font-family: 'Orbitron', sans-serif;
}}

.stButton > button:hover {{
  transform: translateY(-3px);
  box-shadow: 0 20px 50px rgba(0, 200, 83, 0.6);
}}

/* Sidebar */
[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, {DARK_PANEL} 0%, {DARK} 100%);
  border-right: 2px solid rgba(0, 200, 83, 0.3);
}}

/* Metrics */
[data-testid="stMetric"] {{
  background: {GLASS};
  border: 2px solid rgba(0, 200, 83, 0.3);
  border-radius: 16px;
  padding: 1.4rem;
  box-shadow: 0 8px 25px rgba(0,0,0,0.4);
}}

[data-testid="stMetricValue"] {{
  color: {GREEN_LIGHT} !important;
  font-weight: 900 !important;
  font-size: 2.2rem !important;
  font-family: 'Orbitron', sans-serif;
}}

[data-testid="stMetricLabel"] {{
  color: {GREY} !important;
  font-weight: 700 !important;
  text-transform: uppercase;
  letter-spacing: 1px;
}}

/* Checkbox */
.stCheckbox label {{
  color: {GREY} !important;
  font-weight: 700 !important;
}}

/* Info icon custom */
.info-icon {{
  display: inline-block;
  width: 20px;
  height: 20px;
  background: linear-gradient(135deg, {GREEN_DARK}, {GREEN_LIGHT});
  border-radius: 50%;
  text-align: center;
  line-height: 20px;
  color: {DARK} !important;
  font-weight: 900;
  font-size: 0.75rem;
  cursor: help;
  box-shadow: 0 0 10px rgba(0, 200, 83, 0.5);
}}

/* Expander */
.streamlit-expanderHeader {{
  background: {GLASS} !important;
  border: 1px solid rgba(0, 200, 83, 0.3) !important;
  border-radius: 10px !important;
  color: {GREEN_LIGHT} !important;
  font-weight: 800 !important;
}}

/* Dataframe */
.dataframe {{
  background: {DARK_PANEL} !important;
  color: {GREY} !important;
}}
</style>
""", unsafe_allow_html=True)

# ========= HERO =========
st.markdown(f"""
<div class="hero">
  <div class="ticker-tape"></div>
  <div class="hero-icon">💹</div>
  <div class="hero-title">💹 AI Stock Price Predictor</div>
  <div class="hero-sub">
    Harness the power of machine learning to forecast next-day closing prices. Built with Decision Tree Regression + Time Series Cross-Validation.
  </div>
  <div class="badges">
    <div class="badge">Decision Tree ML</div>
    <div class="badge">Time Series CV</div>
    <div class="badge">Real-Time Data</div>
    <div class="badge">By Mayank Goyal</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ========= HELPERS =========
def flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

@st.cache_data
def load_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False)
    df = flatten_yf_columns(df)
    df = df.dropna().copy()
    return df

def make_supervised(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list]:
    df = df.copy()
    df["ret_1"] = df["Close"].pct_change()
    df["lag_close_1"] = df["Close"].shift(1)
    df["lag_ret_1"] = df["ret_1"].shift(1)
    df["ma_5"] = df["Close"].rolling(5).mean()
    df["ma_20"] = df["Close"].rolling(20).mean()
    df["vol_20"] = df["ret_1"].rolling(20).std()
    df["y"] = df["Close"].shift(-1)
    
    data = df.dropna().copy()
    features = ["lag_close_1", "lag_ret_1", "ma_5", "ma_20", "vol_20", "Volume"]
    X = data[features]
    y = data["y"]
    return X, y, features

@st.cache_resource
def train_model(X_train: pd.DataFrame, y_train: pd.Series, n_splits: int):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    dt = DecisionTreeRegressor(random_state=42)
    
    param_grid = {
        "max_depth": [2, 3, 5, 8, 12, None],
        "min_samples_leaf": [1, 5, 10, 25, 50],
        "min_samples_split": [2, 10, 50, 100],
    }
    
    grid = GridSearchCV(
        dt,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_

def eval_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

# ========= SIDEBAR =========
with st.sidebar:
    st.markdown("### 🎛️ Configuration Panel")
    
    with st.expander("ℹ️ What is a Ticker?"):
        st.write("A stock ticker symbol (e.g., AAPL for Apple, TSLA for Tesla) uniquely identifies a publicly traded company.")
    
    ticker = st.text_input("📊 Stock Ticker", value="AAPL", help="Enter the stock symbol (e.g., AAPL, TSLA, GOOGL)")
    
    col1, col2 = st.columns(2)
    with col1:
        start = st.text_input("📅 Start Date", value="2018-01-01", help="Format: YYYY-MM-DD")
    with col2:
        end = st.text_input("📅 End Date", value="2025-01-01", help="Format: YYYY-MM-DD")
    
    st.markdown("---")
    
    with st.expander("ℹ️ What is Holdout Size?"):
        st.write("Percentage of recent data reserved for testing the model's accuracy. The rest is used for training.")
    
    test_size = st.slider("🔬 Holdout Size (%)", 10, 40, 20, 5, help="% of data for final testing")
    
    with st.expander("ℹ️ What is Time Series CV?"):
        st.write("Splits data into multiple train/test folds chronologically to validate model performance without data leakage.")
    
    n_splits = st.slider("🔄 CV Folds", 3, 10, 5, 1, help="Number of time series cross-validation splits")
    
    st.markdown("---")
    
    with st.expander("ℹ️ What is Hyperparameter Tuning?"):
        st.write("Automatically searches for the best model settings (depth, leaf size) to maximize accuracy.")
    
    do_tune = st.checkbox("⚙️ Enable Hyperparameter Tuning", value=True, help="Uses GridSearchCV for optimal parameters")
    
    st.markdown("---")
    train_btn = st.button("🚀 Train Model")

# ========= MAIN =========
if train_btn:
    with st.spinner("📡 Fetching market data..."):
        raw = load_prices(ticker, start, end)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title"><span class="pulse-dot"></span>Raw Market Data</div>', unsafe_allow_html=True)
    st.dataframe(raw.tail(10), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    X, y, features = make_supervised(raw)
    
    split_idx = int(len(X) * (1 - test_size / 100))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    m1, m2, m3 = st.columns(3)
    m1.metric("📊 Total Samples", f"{len(X):,}")
    m2.metric("🏋️ Training Set", f"{len(X_train):,}")
    m3.metric("🧪 Test Set", f"{len(X_test):,}")
    
    with st.spinner("🤖 Training AI model..."):
        if do_tune:
            model, best_params = train_model(X_train, y_train, n_splits=n_splits)
        else:
            model = DecisionTreeRegressor(max_depth=6, min_samples_leaf=25, random_state=42)
            model.fit(X_train, y_train)
            best_params = {"max_depth": 6, "min_samples_leaf": 25, "min_samples_split": 2}
    
    y_pred = model.predict(X_test)
    y_pred_naive = X_test["lag_close_1"].values
    
    mae_tree, rmse_tree = eval_metrics(y_test, y_pred)
    mae_naive, rmse_naive = eval_metrics(y_test, y_pred_naive)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title"><span class="pulse-dot"></span>Performance Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-sub">Comparing AI model vs simple baseline (yesterday\'s price)</div>', unsafe_allow_html=True)
    
    results = pd.DataFrame({
        "Model": ["🔴 Naive Baseline", "🤖 Decision Tree AI"],
        "MAE (Mean Absolute Error)": [f"${mae_naive:.2f}", f"${mae_tree:.2f}"],
        "RMSE (Root Mean Squared Error)": [f"${rmse_naive:.2f}", f"${rmse_tree:.2f}"],
        "MAE Improvement": ["-", f"${mae_naive - mae_tree:.2f}"],
        "RMSE Improvement": ["-", f"${rmse_naive - rmse_tree:.2f}"],
    })
    st.dataframe(results, use_container_width=True)
    
    with st.expander("ℹ️ What do these metrics mean?"):
        st.write("**MAE**: Average dollar error in predictions (lower is better)")
        st.write("**RMSE**: Penalizes large errors more (lower is better)")
        st.write("**Improvement**: How much better AI is than naive baseline")
    
    st.markdown(f"**🎯 Best Hyperparameters:** `{best_params}`")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Chart
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title"><span class="pulse-dot"></span>Prediction vs Reality (Test Set)</div>', unsafe_allow_html=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test.index,
        y=y_test.values,
        mode='lines',
        name='Actual Price',
        line=dict(color=GREEN, width=3),
    ))
    fig.add_trace(go.Scatter(
        x=y_test.index,
        y=y_pred,
        mode='lines',
        name='AI Prediction',
        line=dict(color=GOLD, width=2, dash='dash'),
    ))
    fig.update_layout(
        paper_bgcolor=DARK_PANEL,
        plot_bgcolor=DARK_PANEL,
        font=dict(color=GREY),
        xaxis=dict(showgrid=True, gridcolor=GLASS),
        yaxis=dict(showgrid=True, gridcolor=GLASS, title="Price ($)"),
        height=450,
        hovermode='x unified',
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Next day prediction
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title"><span class="pulse-dot"></span>🔮 Next-Day Forecast</div>', unsafe_allow_html=True)
    
    latest_X = X.iloc[[-1]][features]
    next_close = model.predict(latest_X)[0]
    last_actual = raw["Close"].iloc[-1]
    change = next_close - last_actual
    change_pct = (change / last_actual) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("📅 Last Trading Day", str(X.index[-1].date()))
    col2.metric("💵 Last Close Price", f"${last_actual:.2f}")
    col3.metric("🔮 Predicted Next Close", f"${next_close:.2f}", f"{change_pct:+.2f}%")
    
    with st.expander("ℹ️ How is this calculated?"):
        st.write("Uses the most recent market data (lag prices, moving averages, volatility) to predict tomorrow's closing price.")
    
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("👈 Configure settings in the sidebar and click **🚀 Train Model** to start!")

# ========= FOOTER =========
st.markdown(f"""
<hr style="margin-top: 2.5rem; border-color: rgba(0, 200, 83, 0.3); border-width: 2px;">
<div style="text-align: center; padding: 1.2rem 0; color: {GREY}; font-size: 0.95rem;">
  © 2025 <span style="font-weight: 900; color: {GREEN_LIGHT};">AI Stock Predictor</span> · Developed by 
  <span style="font-weight: 900;">Mayank Goyal</span><br>
  <a href="https://www.linkedin.com/in/mayank-goyal-4b8756363" target="_blank" 
     style="color: {GREEN_LIGHT}; text-decoration: none; font-weight: 800; margin-right: 18px;">LinkedIn</a>
  <a href="https://github.com/mayank-goyal09" target="_blank" 
     style="color: {GREEN_LIGHT}; text-decoration: none; font-weight: 800;">GitHub</a><br>
  <span style="font-size: 0.8rem; opacity: 0.7;">💹 Decision Tree Regression · Time Series CV · Yahoo Finance API</span>
</div>
""", unsafe_allow_html=True)
