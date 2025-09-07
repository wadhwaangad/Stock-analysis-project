# Predict price & signal
import torch
import pandas as pd
import numpy as np
import joblib
import os
from alpha_vantage.timeseries import TimeSeries

from model import StockPredictor
from utils import preprocess, feature_engineering

class StockPredictor_Service:
    def __init__(self, symbol='AAPL', model_path=None, scaler_path=None, features_path=None):
        self.symbol = symbol
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model components
        self.model_path = model_path or f'../model/{symbol}_model.pth'
        self.scaler_path = scaler_path or f'../model/{symbol}_scaler.pkl'
        self.features_path = features_path or f'../model/{symbol}_features.pkl'
        
        self.model = None
        self.scaler = None
        self.feature_cols = None
        
        self.load_model_components()
        
    def load_model_components(self):
        """Load trained model, scaler, and feature columns"""
        try:
            # Load feature columns
            if os.path.exists(self.features_path):
                self.feature_cols = joblib.load(self.features_path)
                input_dim = len(self.feature_cols)
            else:
                # Default feature set if not found
                self.feature_cols = [
                    'close', 'volume', 'price_change', 'high_low_ratio', 'close_open_ratio',
                    'sma_5', 'sma_10', 'sma_20', 'rsi', 'macd', 'macd_signal', 'macd_diff',
                    'bb_high', 'bb_low', 'bb_mid', 'bb_width', 'volume_sma', 'vwap', 'atr'
                ]
                input_dim = len(self.feature_cols)
            
            # Load model
            self.model = StockPredictor(input_dim=input_dim, hidden_dim=128, num_layers=2)
            if os.path.exists(self.model_path):
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                print(f"Model loaded successfully for {self.symbol}")
            else:
                print(f"Model file not found: {self.model_path}")
                self.model = None
            
            # Load scaler
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                print(f"Scaler loaded successfully for {self.symbol}")
            else:
                print(f"Scaler file not found: {self.scaler_path}")
                self.scaler = None
                
        except Exception as e:
            print(f"Error loading model components: {e}")
            self.model = None
            self.scaler = None
    
    def prepare_prediction_data(self, data, sequence_length=60):
        """Prepare data for prediction"""
        # Preprocess data
        df = preprocess(data)
        
        # Feature engineering
        df_features = feature_engineering(df)
        
        # Select required features
        df_model = df_features[self.feature_cols].dropna()
        
        if len(df_model) < sequence_length:
            raise ValueError(f"Need at least {sequence_length} data points, got {len(df_model)}")
        
        # Get the last sequence_length rows
        last_sequence = df_model.tail(sequence_length).values
        
        # Scale the data
        if self.scaler is not None:
            last_sequence_scaled = self.scaler.transform(last_sequence)
        else:
            last_sequence_scaled = last_sequence
        
        # Reshape for model input
        X = last_sequence_scaled.reshape(1, sequence_length, -1)
        
        return torch.FloatTensor(X).to(self.device)
    
    def predict_price(self, data, sequence_length=60):
        """Predict next price"""
        if self.model is None or self.scaler is None:
            return None, "Model or scaler not loaded"
        
        try:
            # Prepare data
            X = self.prepare_prediction_data(data, sequence_length)
            
            # Make prediction
            with torch.no_grad():
                prediction_scaled = self.model(X).cpu().numpy()[0][0]
            
            # Inverse transform to get actual price
            # Create dummy array for inverse transform
            dummy = np.zeros((1, len(self.feature_cols)))
            close_idx = self.feature_cols.index('close')
            dummy[0, close_idx] = prediction_scaled
            
            prediction_actual = self.scaler.inverse_transform(dummy)[0, close_idx]
            
            return prediction_actual, None
            
        except Exception as e:
            return None, str(e)
    
    def generate_signal(self, data, current_price=None):
        """Generate trading signal based on prediction"""
        try:
            predicted_price, error = self.predict_price(data)
            
            if error:
                return "HOLD", 0.0, error
            
            if current_price is None:
                current_price = data['close'].iloc[-1]
            
            # Calculate percentage change
            price_change_pct = ((predicted_price - current_price) / current_price) * 100
            
            # Generate signal based on thresholds
            if price_change_pct > 2.0:  # If predicted to rise by more than 2%
                signal = "BUY"
            elif price_change_pct < -2.0:  # If predicted to fall by more than 2%
                signal = "SELL"
            else:
                signal = "HOLD"
            
            return signal, price_change_pct, None
            
        except Exception as e:
            return "HOLD", 0.0, str(e)
    
    def get_technical_analysis(self, data):
        """Get technical analysis indicators"""
        try:
            df = preprocess(data)
            df_features = feature_engineering(df)
            
            latest = df_features.iloc[-1]
            
            analysis = {
                'current_price': latest['close'],
                'rsi': latest['rsi'],
                'macd': latest['macd'],
                'macd_signal': latest['macd_signal'],
                'sma_5': latest['sma_5'],
                'sma_10': latest['sma_10'],
                'sma_20': latest['sma_20'],
                'bb_position': (latest['close'] - latest['bb_low']) / (latest['bb_high'] - latest['bb_low']),
                'volume_ratio': latest['volume'] / latest['volume_sma'] if latest['volume_sma'] > 0 else 1.0
            }
            
            # Interpret indicators
            interpretation = {
                'rsi_signal': 'OVERSOLD' if analysis['rsi'] < 30 else 'OVERBOUGHT' if analysis['rsi'] > 70 else 'NEUTRAL',
                'macd_signal': 'BULLISH' if analysis['macd'] > analysis['macd_signal'] else 'BEARISH',
                'trend': 'UPTREND' if analysis['current_price'] > analysis['sma_20'] else 'DOWNTREND',
                'bb_signal': 'UPPER' if analysis['bb_position'] > 0.8 else 'LOWER' if analysis['bb_position'] < 0.2 else 'MIDDLE'
            }
            
            return analysis, interpretation, None
            
        except Exception as e:
            return None, None, str(e)

def predict(symbol='AAPL', api_key=None):
    """Main prediction function"""
    # Fetch recent data
    ts = TimeSeries(key=api_key or 'demo', output_format='pandas')
    
    try:
        data, _ = ts.get_daily_adjusted(symbol=symbol, outputsize='compact')
        data.columns = ['open', 'high', 'low', 'close', 'adjusted_close', 'volume', 'dividend', 'split']
        data = data[['open', 'high', 'low', 'close', 'volume']]
        
        # Initialize predictor
        predictor = StockPredictor_Service(symbol)
        
        # Get predictions
        predicted_price, price_error = predictor.predict_price(data)
        signal, price_change_pct, signal_error = predictor.generate_signal(data)
        analysis, interpretation, analysis_error = predictor.get_technical_analysis(data)
        
        results = {
            'symbol': symbol,
            'current_price': data['close'].iloc[-1],
            'predicted_price': predicted_price,
            'signal': signal,
            'price_change_pct': price_change_pct,
            'technical_analysis': analysis,
            'interpretation': interpretation,
            'errors': {
                'price_prediction': price_error,
                'signal_generation': signal_error,
                'technical_analysis': analysis_error
            }
        }
        
        return results
        
    except Exception as e:
        return {'error': str(e)}

if __name__ == "__main__":
    # Test prediction
    api_key = os.getenv('ALPHAVANTAGE_API_KEY', 'demo')
    result = predict('AAPL', api_key)
    print(result)
