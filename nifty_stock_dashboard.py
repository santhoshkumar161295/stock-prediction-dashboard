import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# NIFTY 50 Stocks Tickers
nifty_50_stocks = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "State Bank of India": "SBIN.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Larsen & Toubro": "LT.NS",
    "ITC": "ITC.NS",
    "Axis Bank": "AXISBANK.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "HCL Technologies": "HCLTECH.NS",
    "Wipro": "WIPRO.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Tech Mahindra": "TECHM.NS",
    "Nestle India": "NESTLEIND.NS"
}

# Streamlit App Title
st.title("📈 NIFTY 50 Stock Price Prediction Dashboard")

# Dropdown for Stock Selection
stock_name = st.selectbox("Select a NIFTY 50 Stock:", list(nifty_50_stocks.keys()))
ticker = nifty_50_stocks[stock_name]

# Fetch Stock Data
@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, start="2015-01-01", end="2024-12-31")
    return df

df = load_data(ticker)

# Display Data
st.subheader("Stock Data Overview")
st.write(df.tail())

# Plot Stock Price
st.subheader("Stock Price History")
fig, ax = plt.subplots()
ax.plot(df['Close'], label="Closing Price")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# Feature Engineering for Prediction
df_close = df[['Close']].values
scaler = MinMaxScaler(feature_range=(0,1))
df_scaled = scaler.fit_transform(df_close)

# Prepare Training Data
train_size = int(len(df_scaled) * 0.8)
train_data, test_data = df_scaled[:train_size], df_scaled[train_size:]

def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_data)
X_test, y_test = create_sequences(test_data)

# Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train Model
model.fit(X_train, y_train, batch_size=16, epochs=5, verbose=1)

# Predict Next Day Price
last_60_days = df_scaled[-60:].reshape(1, 60, 1)
predicted_price = model.predict(last_60_days)
predicted_price = scaler.inverse_transform(predicted_price)

st.subheader("📊 Predicted Stock Price for Next Day")
st.write(f"**Predicted Price for {stock_name}:** ₹{predicted_price[0][0]:.2f}")
