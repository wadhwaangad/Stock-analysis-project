# Final app to serve predictions

# Final app to serve predictions
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import redis
import os
import json
import asyncio
from datetime import datetime, timedelta
from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv
import sys
sys.path.append('../model')

from predict import predict, StockPredictor_Service

load_dotenv()

app = FastAPI(title="Stock Analysis API", docs_url="/docs")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Redis connection
try:
    r = redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'), port=6379, db=0, decode_responses=True)
    r.ping()
    print("Connected to Redis")
except:
    print("Redis connection failed, using in-memory cache")
    r = None

# Alpha Vantage setup
ALPHA_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "demo")
ts = TimeSeries(key=ALPHA_KEY, output_format='pandas')

# Cache helper functions
def get_cache(key):
    if r:
        try:
            return r.get(key)
        except:
            return None
    return None

def set_cache(key, value, ex=300):  # 5 minutes default
    if r:
        try:
            r.setex(key, ex, value)
        except:
            pass

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with stock analysis interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/quote/{symbol}")
async def get_quote(symbol: str):
    """Get real-time quote for a symbol"""
    cache_key = f"quote:{symbol.upper()}"
    cached = get_cache(cache_key)
    
    if cached:
        return JSONResponse(json.loads(cached))
    
    try:
        data, meta = ts.get_quote_endpoint(symbol=symbol.upper())
        quote_data = {
            "symbol": symbol.upper(),
            "price": float(data['05. price']),
            "change": float(data['09. change']),
            "change_percent": data['10. change percent'].strip('%'),
            "volume": int(data['06. volume']),
            "timestamp": data['07. latest trading day']
        }
        
        # Cache for 1 minute
        set_cache(cache_key, json.dumps(quote_data), 60)
        return JSONResponse(quote_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching quote: {str(e)}")

@app.get("/api/predict/{symbol}")
async def get_prediction(symbol: str):
    """Get price prediction and trading signal"""
    cache_key = f"prediction:{symbol.upper()}"
    cached = get_cache(cache_key)
    
    if cached:
        return JSONResponse(json.loads(cached))
    
    try:
        # Get prediction
        result = predict(symbol.upper(), ALPHA_KEY)
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        # Cache for 5 minutes
        set_cache(cache_key, json.dumps(result, default=str), 300)
        return JSONResponse(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating prediction: {str(e)}")

@app.get("/api/technical/{symbol}")
async def get_technical_analysis(symbol: str):
    """Get technical analysis for a symbol"""
    cache_key = f"technical:{symbol.upper()}"
    cached = get_cache(cache_key)
    
    if cached:
        return JSONResponse(json.loads(cached))
    
    try:
        # Fetch recent data for technical analysis
        data, _ = ts.get_daily_adjusted(symbol=symbol.upper(), outputsize='compact')
        data.columns = ['open', 'high', 'low', 'close', 'adjusted_close', 'volume', 'dividend', 'split']
        data = data[['open', 'high', 'low', 'close', 'volume']]
        
        # Initialize predictor for technical analysis
        predictor = StockPredictor_Service(symbol.upper())
        analysis, interpretation, error = predictor.get_technical_analysis(data)
        
        if error:
            raise HTTPException(status_code=500, detail=error)
        
        result = {
            "symbol": symbol.upper(),
            "analysis": analysis,
            "interpretation": interpretation,
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache for 2 minutes
        set_cache(cache_key, json.dumps(result, default=str), 120)
        return JSONResponse(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating technical analysis: {str(e)}")

@app.get("/api/dashboard/{symbol}")
async def get_dashboard_data(symbol: str):
    """Get complete dashboard data for a symbol"""
    try:
        # Get all data concurrently
        quote_task = asyncio.create_task(get_quote(symbol))
        prediction_task = asyncio.create_task(get_prediction(symbol))
        technical_task = asyncio.create_task(get_technical_analysis(symbol))
        
        quote_response = await quote_task
        prediction_response = await prediction_task
        technical_response = await technical_task
        
        dashboard_data = {
            "quote": quote_response.body.decode() if hasattr(quote_response, 'body') else quote_response,
            "prediction": prediction_response.body.decode() if hasattr(prediction_response, 'body') else prediction_response,
            "technical": technical_response.body.decode() if hasattr(technical_response, 'body') else technical_response,
            "last_updated": datetime.now().isoformat()
        }
        
        return JSONResponse(dashboard_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching dashboard data: {str(e)}")

@app.get("/analyze/{symbol}", response_class=HTMLResponse)
async def analyze_stock(request: Request, symbol: str):
    """Stock analysis page"""
    return templates.TemplateResponse("analyze.html", {
        "request": request, 
        "symbol": symbol.upper()
    })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    redis_status = "connected" if r and r.ping() else "disconnected"
    return {
        "status": "healthy",
        "redis": redis_status,
        "timestamp": datetime.now().isoformat()
    }
