# Stock Analysis App

A comprehensive stock analysis application built with FastAPI, Redis, Docker, and Alpha Vantage API. Features real-time data, AI-powered predictions, and technical analysis with a sleek black and green theme.

## Features

- **Real-time Stock Data**: Live quotes from Alpha Vantage API
- **AI Price Predictions**: LSTM neural network for price forecasting
- **Technical Analysis**: RSI, MACD, Bollinger Bands, and more
- **Trading Signals**: Buy/Sell/Hold recommendations
- **Redis Caching**: Fast data retrieval with intelligent caching
- **Modern UI**: Responsive black and green themed interface
- **Docker Support**: Easy deployment with Docker Compose

## Project Structure

```
├── data/
│   └── stock_data.csv          # Historical price data
├── model/
│   ├── train.py               # Model training script
│   ├── predict.py             # Predict price & signal
│   ├── model.py               # Neural network architecture
│   └── utils.py               # Preprocessing, feature engineering
├── app/
│   ├── main.py                # FastAPI application
│   ├── static/
│   │   └── style.css          # CSS styling
│   └── templates/
│       ├── index.html         # Main dashboard
│       └── analyze.html       # Detailed analysis page
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Quick Start

### Using Docker (Recommended)

1. Clone the repository
2. Set your Alpha Vantage API key:
   ```bash
   echo "ALPHAVANTAGE_API_KEY=your_api_key_here" > .env
   ```
3. Run with Docker Compose:
   ```bash
   docker-compose up --build
   ```
4. Open http://localhost:8000

### Manual Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start Redis server:
   ```bash
   redis-server
   ```

3. Set environment variables:
   ```bash
   export ALPHAVANTAGE_API_KEY=your_api_key_here
   ```

4. Run the application:
   ```bash
   cd app
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

## API Endpoints

- `GET /` - Main dashboard
- `GET /api/quote/{symbol}` - Real-time quote
- `GET /api/predict/{symbol}` - AI prediction and signal
- `GET /api/technical/{symbol}` - Technical analysis
- `GET /api/dashboard/{symbol}` - Complete dashboard data
- `GET /analyze/{symbol}` - Detailed analysis page
- `GET /health` - Health check

## Model Training

To train the AI model with your own data:

```bash
cd model
python train.py
```

This will:
- Fetch historical data from Alpha Vantage
- Prepare features and technical indicators
- Train an LSTM neural network
- Save the trained model, scaler, and feature list

## Configuration

### Environment Variables

- `ALPHAVANTAGE_API_KEY` - Your Alpha Vantage API key
- `REDIS_HOST` - Redis server host (default: localhost)

### Alpha Vantage API

Get your free API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key).

## Technologies Used

- **Backend**: FastAPI, Python
- **AI/ML**: PyTorch, scikit-learn, pandas
- **Database**: Redis for caching
- **API**: Alpha Vantage for stock data
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Docker, Docker Compose

## Technical Indicators

The app calculates various technical indicators:

- Simple Moving Averages (SMA 5, 10, 20)
- Relative Strength Index (RSI)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Volume indicators
- Average True Range (ATR)

## AI Model

The prediction model uses:
- LSTM neural network architecture
- 60-day sequence length for predictions
- Technical indicators as features
- Price change percentage for signals

## Contributing

Feel free to contribute by:
- Adding new technical indicators
- Improving the UI/UX
- Enhancing the AI model
- Adding more data sources

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This application is for educational and informational purposes only. It should not be considered as financial advice. Always do your own research before making investment decisions.
