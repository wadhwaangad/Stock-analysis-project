# Model training script for stock analysis
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
import os
from alpha_vantage.timeseries import TimeSeries

from model import StockPredictor, StockTrainer
from utils import preprocess, prepare_model_data

class StockDataset:
    def __init__(self, symbol='AAPL', api_key=None):
        self.symbol = symbol
        self.api_key = api_key or 'demo'
        self.ts = TimeSeries(key=self.api_key, output_format='pandas')
        
    def fetch_data(self, outputsize='full'):
        """Fetch stock data from Alpha Vantage"""
        try:
            data, meta_data = self.ts.get_daily_adjusted(symbol=self.symbol, outputsize=outputsize)
            # Rename columns to standard format
            data.columns = ['open', 'high', 'low', 'close', 'adjusted_close', 'volume', 'dividend', 'split']
            data = data[['open', 'high', 'low', 'close', 'volume']]  # Keep only needed columns
            return data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def load_or_fetch_data(self):
        """Load existing data or fetch new data"""
        data_path = f"../data/{self.symbol}_data.csv"
        
        if os.path.exists(data_path):
            print(f"Loading existing data for {self.symbol}")
            data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        else:
            print(f"Fetching new data for {self.symbol}")
            data = self.fetch_data()
            if data is not None:
                os.makedirs("../data", exist_ok=True)
                data.to_csv(data_path)
        
        return data

def train_model(symbol='AAPL', api_key=None, sequence_length=60, epochs=100):
    """Train the stock prediction model"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    dataset = StockDataset(symbol, api_key)
    data = dataset.load_or_fetch_data()
    
    if data is None:
        print("Failed to load data")
        return None
    
    print(f"Data shape: {data.shape}")
    print(f"Data range: {data.index[0]} to {data.index[-1]}")
    
    # Preprocess data
    data = preprocess(data)
    
    # Prepare model data
    X, y, scaler, feature_cols = prepare_model_data(data, sequence_length)
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature columns: {feature_cols}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Further split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, shuffle=False
    )
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    input_dim = X.shape[2]
    model = StockPredictor(input_dim=input_dim, hidden_dim=128, num_layers=2).to(device)
    
    # Initialize trainer
    trainer = StockTrainer(model, learning_rate=0.001)
    
    # Train model
    print("Starting training...")
    history = trainer.fit(train_loader, val_loader, epochs=epochs)
    
    # Evaluate on test set
    test_loss, test_mse, test_mae, test_predictions, test_actuals = trainer.validate(test_loader)
    print(f"Test Loss: {test_loss:.4f}, Test MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f}")
    
    # Save model and scaler
    torch.save(model.state_dict(), f'../model/{symbol}_model.pth')
    joblib.dump(scaler, f'../model/{symbol}_scaler.pkl')
    joblib.dump(feature_cols, f'../model/{symbol}_features.pkl')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(test_actuals[:100], label='Actual', alpha=0.7)
    plt.plot(test_predictions[:100], label='Predicted', alpha=0.7)
    plt.title('Predictions vs Actual (First 100 test samples)')
    plt.xlabel('Sample')
    plt.ylabel('Normalized Price')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'../model/{symbol}_training_results.png')
    plt.show()
    
    return model, scaler, feature_cols, history

if __name__ == "__main__":
    # Get API key from environment or use demo
    api_key = os.getenv('ALPHAVANTAGE_API_KEY', 'demo')
    
    # Train model for AAPL
    model, scaler, features, history = train_model('AAPL', api_key)
    
    print("Training completed!")
