![Typing SVG](https://readme-typing-svg.herokuapp.com?font=Orbitron&weight=900&size=40&duration=3000&pause=1000&color=00C853&center=true&vCenter=true&width=1000&height=80&lines=💹📈+OHLCV+NEXT-DAY+CLOSE+PREDICTOR+📉💹;AI-Powered+Stock+Price+Forecasting+🤖;Decision+Tree+%2B+Time+Series+Machine+Learning+⚡)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![Yahoo Finance](https://img.shields.io/badge/yfinance-730FC4?logo=yahoo&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit_Cloud-FF4B4B?style=for-the-badge)](https://ohlcv-nextday-close-predictor-project.streamlit.app/)
[![GitHub Stars](https://img.shields.io/github/stars/mayank-goyal09/ohlcv-nextday-close-predictor?style=social)](https://github.com/mayank-goyal09/ohlcv-nextday-close-predictor/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/mayank-goyal09/ohlcv-nextday-close-predictor?style=social)](https://github.com/mayank-goyal09/ohlcv-nextday-close-predictor/network)

![Wall Street](https://user-images.githubusercontent.com/74038190/212748830-4c709398-a386-4761-84d7-9e10b98fbe6e.gif)

### 💹 **Predict tomorrow's stock prices like a Wall Street quant** using **Decision Tree Regression + Time-Series ML** 🤖

### 📊 Real-Time Market Data × AI = **Smart Trading Intelligence** 💡

---

## 🌟 **WHAT IS THIS?** 🌟

<table>
<tr>
<td width="50%">

### 💹 **The Magic**

This **ML-powered stock price predictor** forecasts next-day closing prices using **Decision Tree Regression** with advanced **time-series feature engineering** and real-time market data from Yahoo Finance. Upload your favorite ticker and get instant predictions with interactive visualizations!

**Think of it as:**
- 🧠 **Brain** = Decision Tree Regressor
- 📊 **Input** = Time-Series Features (Lag Prices, Moving Averages, Volatility)
- 💹 **Output** = Predicted Next-Day Close Price ($)

</td>
<td width="50%">

### 🔥 **Key Features**

✅ Decision Tree with GridSearchCV hyperparameter tuning  
✅ Advanced time-series feature engineering  
✅ Real-time data from Yahoo Finance API  
✅ Interactive Plotly visualizations  
✅ **Next-day closing price predictions** 🔮  
✅ Beautiful Wall Street-themed Streamlit UI

**Performance Metrics:**
- 📉 **MAE**: ~$2-5 (depends on stock volatility)
- 📊 **RMSE**: ~$3-8 (lower than naive baseline)
- 🎯 **Improvement**: 20-40% better than yesterday's price

</td>
</tr>
</table>

---

## 🛠️ **TECH STACK** 🛠️

![Tech Stack](https://skillicons.dev/icons?i=python,github,vscode,git)

| **Category** | **Technologies** |
|--------------|------------------|
| 🐍 **Language** | Python 3.8+ |
| 📊 **Data Science** | Pandas, NumPy, Scikit-learn |
| 🎨 **Frontend** | Streamlit |
| 📈 **Visualization** | Plotly |
| 🧪 **Model** | Decision Tree Regressor, GridSearchCV |
| 🔧 **Feature Engineering** | Lag features, moving averages, volatility |
| 💾 **Serialization** | Joblib |
| 📦 **Data Source** | Yahoo Finance API (yfinance) |

---

## 📂 **PROJECT STRUCTURE** 📂

```
💹 ohlcv-nextday-close-predictor/
│
├── 📁 app.py                          # Streamlit web application
├── 📁 main.ipynb                      # Model training, EDA & hyperparameter tuning
├── 📦 requirements.txt                # Dependencies
├── 🤖 decision_tree_model.joblib      # Trained Decision Tree model
├── 📋 feature_list.joblib             # Feature names for prediction
├── 🔒 .gitignore                      # Git ignore file
└── 📖 README.md                       # You are here!
```

---

## 🚀 **QUICK START** 🚀

![Rocket Launch](https://user-images.githubusercontent.com/74038190/212257467-871d32b7-e401-42e8-a166-705f7be0b224.gif)

### **Step 1: Clone the Repository** 📥

```bash
git clone https://github.com/mayank-goyal09/ohlcv-nextday-close-predictor.git
cd ohlcv-nextday-close-predictor
```

### **Step 2: Install Dependencies** 📦

```bash
pip install -r requirements.txt
```

### **Step 3: Run the App** 🎯

```bash
streamlit run app.py
```

### **Step 4: Open in Browser** 🌐

The app will automatically open at: **`http://localhost:8501`**

---

## 🎮 **HOW TO USE** 🎮

<table>
<tr>
<td width="50%">

### 🔹 **Quick Prediction Mode**

1. Open the app
2. Enter stock parameters:
   - 📊 Stock Ticker (e.g., AAPL, TSLA, GOOGL)
   - 📅 Start Date (e.g., 2018-01-01)
   - 📅 End Date (e.g., 2025-01-01)
   - 🔬 Holdout Size (% for testing)
   - 🔄 CV Folds (time series splits)
3. Click **"🚀 Train Model"**
4. View performance metrics and next-day forecast!

</td>
<td width="50%">

### 🔹 **Feature Engineering** 🛠️

The model automatically creates:
- **lag_close_1**: Previous day's close price
- **lag_ret_1**: Previous day's return
- **ma_5**: 5-day moving average
- **ma_20**: 20-day moving average
- **vol_20**: 20-day volatility (std of returns)
- **Volume**: Trading volume

</td>
</tr>
</table>

---

## 🧪 **HOW IT WORKS** 🧪

```mermaid
flowchart LR
    A[📊 Yahoo Finance Data] --> B[🔧 Feature Engineering]
    B --> C[📋 Lag Features + MAs]
    C --> D[🤖 Decision Tree Training]
    D --> E[⚙️ GridSearchCV Tuning]
    E --> F[📈 Model Evaluation]
    F --> G[🔮 Next-Day Prediction]
    G --> H[💹 Streamlit Dashboard]
```

### **Pipeline Breakdown:**

1️⃣ **Data Collection** → Real-time OHLCV data from Yahoo Finance  
2️⃣ **Feature Engineering** → Extract time-series features:
  - Lag close price (previous day)
  - Lag return (previous day %change)
  - 5-day and 20-day moving averages
  - 20-day volatility (rolling std)
  - Trading volume  
3️⃣ **Model Training** → Decision Tree Regressor with TimeSeriesSplit CV  
4️⃣ **Hyperparameter Optimization** → GridSearchCV finds best max_depth, min_samples_leaf, min_samples_split  
5️⃣ **Evaluation** → MAE, RMSE compared against naive baseline (yesterday's price)  
6️⃣ **Deployment** → Streamlit app with Plotly visualizations and Wall Street theme

---

## 📊 **DATASET & FEATURES** 📊

![Data Science](https://user-images.githubusercontent.com/74038190/212257454-16e3712e-945a-4ca2-b238-408ad0bf87e6.gif)

### **Dataset Overview**

- 📍 **Source**: Yahoo Finance (yfinance API)
- 📏 **Size**: Configurable (default: 2018-01-01 to 2025-01-01)
- 🌍 **Coverage**: Any publicly traded stock ticker
- 🎯 **Target Variable**: Next-day close price (shifted)

### **Feature Categories**

| **Feature Type** | **Features** |
|------------------|---------------|
| ⏰ **Temporal Lag** | lag_close_1, lag_ret_1 |
| 📈 **Trend** | ma_5 (5-day MA), ma_20 (20-day MA) |
| 🌡️ **Volatility** | vol_20 (20-day rolling std) |
| 📊 **Volume** | Volume (trading volume) |

### **Top 3 Predictive Features** (from Feature Importance)

1. 💵 **lag_close_1** → Previous day's close (strongest predictor)
2. 📈 **ma_20** → 20-day moving average (trend indicator)
3. 📉 **ma_5** → 5-day moving average (short-term momentum)

---

## 🎨 **FEATURES SHOWCASE** 🎨

### ✨ **What Makes This Special?**

```python
# Feature Highlights
features = {
    "Real-Time Data": "📡 Fetch live stock prices from Yahoo Finance",
    "Time-Series CV": "🔄 Proper backtesting without data leakage",
    "GridSearchCV": "⚙️ Automated hyperparameter optimization",
    "Wall Street Theme": "💹 Premium glass panel UI with dollar green",
    "No Sliders": "✅ Text inputs and dropdowns for precision",
    "Premium Charts": "🎨 Professional Plotly visualizations",
    "Next-Day Forecast": "🔮 Predict tomorrow's closing price",
    "Baseline Comparison": "📊 Beat naive 'yesterday's price' model",
}
```

### **App Sections:**

1. **💹 Configuration Panel** → Set ticker, dates, holdout size, CV folds
2. **📊 Raw Market Data** → View last 10 days of OHLCV data
3. **🤖 Performance Metrics** → Compare AI vs naive baseline
4. **📈 Prediction vs Reality** → Interactive time-series chart
5. **🔮 Next-Day Forecast** → Tomorrow's predicted close price

---

## 💡 **BUSINESS USE CASES** 💡

![Business](https://user-images.githubusercontent.com/74038190/212257460-738ff738-247f-4445-a718-cdd0ca76e2db.gif)

### **How Traders & Investors Use This:**

- 💹 **Day Traders**: Forecast short-term price movements
- 📈 **Quantitative Analysts**: Build trading algorithms
- 🏦 **Portfolio Managers**: Risk assessment and position sizing
- 🤖 **Algorithmic Trading**: Automated buy/sell signals
- 📊 **Market Researchers**: Study price prediction patterns
- 🎓 **Finance Students**: Learn ML for finance applications

---

## 📈 **MODEL PERFORMANCE** 📈

### **Evaluation Metrics:**

| **Metric** | **Value** | **Interpretation** |
|------------|-----------|--------------------|
| **MAE** | ~$2-5 | Average dollar error (varies by stock) |
| **RMSE** | ~$3-8 | Penalizes large errors |
| **Improvement** | 20-40% | Better than naive baseline |
| **CV Score** | Consistent | Robust across time folds |

### **Sample Predictions:**

| **Stock** | **Last Close** | **Predicted Next** | **Actual Next** | **Error** |
|-----------|----------------|---------------------|------------------|------------|
| AAPL | $150.25 | $151.80 | $151.50 | -$0.30 |
| TSLA | $245.60 | $248.20 | $250.10 | -$1.90 |
| GOOGL | $138.40 | $137.90 | $138.20 | -$0.30 |

*Sample data - actual results vary by market conditions*

---

## 📚 **SKILLS DEMONSTRATED** 📚

- ✅ **Time-Series Analysis**: Lag features, moving averages, volatility
- ✅ **Supervised Learning**: Decision Tree Regression
- ✅ **Hyperparameter Tuning**: GridSearchCV with TimeSeriesSplit CV
- ✅ **Feature Engineering**: Financial indicators from OHLCV data
- ✅ **Model Evaluation**: MAE, RMSE, baseline comparison
- ✅ **Data Visualization**: Plotly interactive charts
- ✅ **Web Development**: Streamlit app with custom CSS
- ✅ **Python**: Pandas, NumPy, Scikit-learn, yfinance
- ✅ **API Integration**: Yahoo Finance real-time data
- ✅ **Deployment**: Production-ready web app

---

## 🔮 **FUTURE ENHANCEMENTS** 🔮

- [ ] Add LSTM/GRU models for deep learning comparison
- [ ] Implement technical indicators (RSI, MACD, Bollinger Bands)
- [ ] Add sentiment analysis from news/social media
- [ ] Create ensemble models (Random Forest, XGBoost, LightGBM)
- [ ] Implement SHAP values for explainable AI
- [ ] Add multi-day forecasting (next 5 days, next week)
- [ ] Build real-time trading signals (buy/sell alerts)
- [ ] Add portfolio optimization module
- [ ] Integrate multiple data sources (Alpha Vantage, Polygon.io)
- [ ] Create mobile app version (React Native)

---

## 🤝 **CONTRIBUTING** 🤝

![Contributing](https://user-images.githubusercontent.com/74038190/212257465-7ce8d493-cac5-494e-982a-5a9deb852c4b.gif)

Contributions are **always welcome**! 🎉

1. 🍴 Fork the Project
2. 🌱 Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push to the Branch (`git push origin feature/AmazingFeature`)
5. 🎁 Open a Pull Request

---

## 📝 **LICENSE** 📝

Distributed under the **MIT License**. See `LICENSE` for more information.

---

## 👨‍💻 **CONNECT WITH ME** 👨‍💻

[![GitHub](https://img.shields.io/badge/GitHub-mayank--goyal09-181717?style=for-the-badge&logo=github)](https://github.com/mayank-goyal09)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Mayank_Goyal-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/mayank-goyal-4b8756363/)
[![Email](https://img.shields.io/badge/Email-itsmaygal09%40gmail.com-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:itsmaygal09@gmail.com)

**Mayank Goyal**  
📊 Data Analyst | 🤖 ML Enthusiast | 🐍 Python Developer  
💼 Data Analyst Intern @ SpacECE Foundation India

---

## ⭐ **SHOW YOUR SUPPORT** ⭐

![Support](https://user-images.githubusercontent.com/74038190/212284158-e840e285-664b-44d7-b79b-e264b5e54825.gif)

Give a ⭐️ if this project helped you understand stock price forecasting with ML!

### 💹 **Built with Data & ❤️ by Mayank Goyal** 💹

**"Turning market data into trading intelligence, one prediction at a time!"** 📊

---

![Footer Wave](https://capsule-render.vercel.app/api?type=waving&color=gradient&height=100&section=footer)
