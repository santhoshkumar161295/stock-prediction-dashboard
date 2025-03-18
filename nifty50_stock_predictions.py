import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# List of NIFTY 50 Stocks with their Yahoo Finance tickers
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

# Function to predict stock price using LSTM
def predict_stock_price(ticker):
    try:
        # Fetch stock data
        df = yf.download(ticker, start="2015-01-01", end="2025-03-18")

        # Use only the closing price
        df_close = df[['Close']].values

        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_scaled = scaler.fit_transform(df_close)

        # Prepare training data
        def create_sequences(data, seq_length=60):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i + seq_length])
                y.append(data[i + seq_length])
            return np.array(X), np.array(y)

        train_size = int(len(df_scaled) * 0.8)
        train_data, test_data = df_scaled[:train_size], df_scaled[train_size:]

        X_train, y_train = create_sequences(train_data)
        X_test, y_test = create_sequences(test_data)

        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_train, y_train, batch_size=16, epochs=5, verbose=0)

        # Predict the next day's stock price
        last_60_days = df_scaled[-60:].reshape(1, 60, 1)
        predicted_price = model.predict(last_60_days)
        predicted_price = scaler.inverse_transform(predicted_price)

        return round(predicted_price[0][0], 2)

    except Exception as e:
        return f"Error: {e}"

# Predict for all NIFTY 50 stocks
predictions = {}
for stock, ticker in nifty_50_stocks.items():
    print(f"Predicting for {stock} ({ticker})...")
    predictions[stock] = predict_stock_price(ticker)

# Display predictions
print("
📊 Predicted Stock Prices for Tomorrow:")
for stock, price in predictions.items():
    print(f"{stock}: ₹{price}")
