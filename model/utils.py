# Preprocessing, feature engineering
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import ta

def preprocess(data):
    """Preprocess stock data"""
    if isinstance(data, dict):
        df = pd.DataFrame(data).T
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        df.index = pd.to_datetime(df.index)
    else:
        df = data.copy()
    
    # Convert to numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sort by date
    df = df.sort_index()
    
    # Remove any NaN values
    df = df.dropna()
    
    return df

def feature_engineering(data):
    """Create technical indicators and features"""
    df = data.copy()
    
    # Price-based features
    df['price_change'] = df['close'].pct_change()
    df['high_low_ratio'] = df['high'] / df['low']
    df['close_open_ratio'] = df['close'] / df['open']
    
    # Moving averages
    df['sma_5'] = ta.trend.sma_indicator(df['close'], window=5)
    df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
    df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
    
    # RSI
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['close'])
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()
    df['bb_mid'] = bollinger.bollinger_mavg()
    df['bb_width'] = df['bb_high'] - df['bb_low']
    
    # Volume indicators
    df['volume_sma'] = ta.volume.volume_sma(df['close'], df['volume'])
    df['vwap'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])
    
    # Volatility
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
    
    return df

def create_sequences(data, sequence_length=60, target_col='close'):
    """Create sequences for LSTM model"""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i][data.columns.get_loc(target_col)])
    
    return np.array(X), np.array(y), scaler

def prepare_model_data(df, sequence_length=60):
    """Prepare data for model training"""
    # Feature engineering
    df_features = feature_engineering(df)
    
    # Select features for model
    feature_cols = [
        'close', 'volume', 'price_change', 'high_low_ratio', 'close_open_ratio',
        'sma_5', 'sma_10', 'sma_20', 'rsi', 'macd', 'macd_signal', 'macd_diff',
        'bb_high', 'bb_low', 'bb_mid', 'bb_width', 'volume_sma', 'vwap', 'atr'
    ]
    
    # Remove rows with NaN values
    df_model = df_features[feature_cols].dropna()
    
    # Create sequences
    X, y, scaler = create_sequences(df_model, sequence_length)
    
    return X, y, scaler, feature_cols
