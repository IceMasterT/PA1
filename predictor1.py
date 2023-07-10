
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from ta import add_all_ta_features

# Read the data from CSV file
df = pd.read_csv('bitstamp_btcusd_1h.csv')
print(df.columns.tolist())

# Generate technical indicators
df = add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume_btc")

# Handle missing values
df = df.dropna()

# Normalize the data
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Split data into train and validation sets
train_size = int(len(df_normalized) * 0.8)
train, validation = df_normalized[:train_size], df_normalized[train_size:]

# Choose features and target
features = train[['trend_sma_fast', 'momentum_rsi', 'trend_macd_diff', 'volatility_bbh', 'volatility_bbl']]
target = train['Close']

# Create and train the model
model = LinearRegression()
model.fit(features, target)

# Evaluate the model
validation_features = validation[['trend_sma_fast', 'momentum_rsi', 'trend_macd_diff', 'volatility_bbh', 'volatility_bbl']]
predicted_prices = model.predict(validation_features)

# Calculate the mean squared error
mse = mean_squared_error(validation['Close'], predicted_prices)
print(f'Mean Squared Error: {mse}')

# Use the model to predict future prices
future_features = df_normalized[['trend_sma_fast', 'momentum_rsi', 'trend_macd_diff', 'volatility_bbh', 'volatility_bbl']].tail(1)
predicted_future_price = model.predict(future_features)
print(f'Predicted Future Price: {predicted_future_price}')
