"""
================================================================================
AI Financial Intelligence Platform — Meridian Analytics
FastAPI Backend — Dual Data Source: Alpha Vantage + yFinance
================================================================================
Author: Dilip Chennam
"""

import os
import re
import time
import math
import logging
from collections import defaultdict
from datetime import datetime
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
import anthropic
from prophet import Prophet

# ── Load environment variables
load_dotenv()
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY")
ALPHA_VANTAGE_KEY  = os.getenv("ALPHA_VANTAGE_KEY")
ALLOWED_ORIGINS    = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
RATE_LIMIT         = int(os.getenv("RATE_LIMIT_PER_MIN", "10"))

if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY not set in environment")
if not ALPHA_VANTAGE_KEY:
    raise RuntimeError("ALPHA_VANTAGE_KEY not set in environment")

# ── Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── App
app = FastAPI(title="Meridian Analytics API", version="1.0.0", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ── Rate limiter
request_counts: dict = defaultdict(list)

def rate_limit_check(request: Request):
    ip  = request.client.host
    now = time.time()
    request_counts[ip] = [t for t in request_counts[ip] if now - t < 60]
    if len(request_counts[ip]) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a moment.")
    request_counts[ip].append(now)

# ── Input validation
VALID_TICKER = re.compile(r'^[A-Z0-9.\-]{1,10}$')

def validate_ticker(ticker: str) -> str:
    ticker = ticker.strip().upper()
    if not VALID_TICKER.match(ticker):
        raise HTTPException(status_code=400, detail="Invalid ticker symbol.")
    return ticker

# ── Models
class TickerRequest(BaseModel):
    ticker: str

    @validator('ticker')
    def clean_ticker(cls, v):
        v = v.strip().upper()
        if not re.compile(r'^[A-Z0-9.\-]{1,10}$').match(v):
            raise ValueError("Invalid ticker format")
        return v

class ChatRequest(BaseModel):
    ticker: str
    question: str
    history: list = []

    @validator('ticker')
    def clean_ticker(cls, v):
        v = v.strip().upper()
        if not re.compile(r'^[A-Z0-9.\-]{1,10}$').match(v):
            raise ValueError("Invalid ticker format")
        return v

    @validator('question')
    def clean_question(cls, v):
        v = v.strip()
        if len(v) < 2:
            raise ValueError("Question too short")
        if len(v) > 500:
            raise ValueError("Question too long")
        return v

# ============================================================
# DATA FETCHING — ALPHA VANTAGE (fundamentals)
# ============================================================
def fetch_alpha_vantage_overview(ticker: str) -> dict:
    """Fetch company overview from Alpha Vantage."""
    try:
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_VANTAGE_KEY}"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if not data or 'Symbol' not in data:
            return {}
        return {
            'name':          data.get('Name', ticker),
            'sector':        data.get('Sector', 'N/A'),
            'industry':      data.get('Industry', 'N/A'),
            'description':   data.get('Description', '')[:500],
            'market_cap':    float(data.get('MarketCapitalization', 0) or 0),
            'pe_ratio':      float(data.get('TrailingPE', 0) or 0) or None,
            'beta':          float(data.get('Beta', 0) or 0) or None,
            'dividend_yield':float(data.get('DividendYield', 0) or 0),
            'week_52_high':  float(data.get('52WeekHigh', 0) or 0),
            'week_52_low':   float(data.get('52WeekLow', 0) or 0),
            'avg_volume':    float(data.get('SharesOutstanding', 0) or 0),
        }
    except Exception as e:
        logger.warning(f"Alpha Vantage overview failed for {ticker}: {e}")
        return {}


def fetch_alpha_vantage_quote(ticker: str) -> dict:
    """Fetch real-time quote from Alpha Vantage."""
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={ALPHA_VANTAGE_KEY}"
        resp = requests.get(url, timeout=10)
        data = resp.json().get('Global Quote', {})
        if not data:
            return {}
        current_price = float(data.get('05. price', 0) or 0)
        prev_close    = float(data.get('08. previous close', 0) or 0)
        price_change  = ((current_price - prev_close) / prev_close * 100) if prev_close else 0
        return {
            'current_price': round(current_price, 2),
            'prev_close':    round(prev_close, 2),
            'price_change':  round(price_change, 2),
            'volume':        int(float(data.get('06. volume', 0) or 0)),
        }
    except Exception as e:
        logger.warning(f"Alpha Vantage quote failed for {ticker}: {e}")
        return {}


# ============================================================
# DATA FETCHING — YFINANCE (historical data for charts)
# ============================================================
def fetch_yfinance_history(ticker: str) -> pd.DataFrame:
    """Fetch 1-year historical price data from yFinance."""
    try:
        stock = yf.Ticker(ticker)
        hist  = stock.history(period="1y")
        if hist.empty:
            hist = stock.history(period="6mo")
        return hist
    except Exception as e:
        logger.warning(f"yFinance history failed for {ticker}: {e}")
        return pd.DataFrame()


# ============================================================
# COMBINED DATA FETCH
# ============================================================
def fetch_financial_data(ticker: str) -> dict:
    """Combine Alpha Vantage fundamentals + yFinance historical data."""
    logger.info(f"Fetching data for {ticker}")

    # Fetch fundamentals from Alpha Vantage
    overview = fetch_alpha_vantage_overview(ticker)
    quote    = fetch_alpha_vantage_quote(ticker)

    if not quote or not quote.get('current_price'):
        raise HTTPException(status_code=404, detail=f"No data found for '{ticker}'. Please check the ticker symbol.")

    # Fetch historical data from yFinance
    hist = fetch_yfinance_history(ticker)

    if hist.empty:
        raise HTTPException(status_code=404, detail=f"No historical data for '{ticker}'.")

    # Calculate risk metrics from historical data
    hist['daily_return'] = hist['Close'].pct_change().fillna(0)
    daily_returns = hist['daily_return'].dropna()

    annual_return = daily_returns.mean() * 252
    annual_vol    = daily_returns.std() * np.sqrt(252)
    sharpe        = (annual_return - 0.05) / annual_vol if annual_vol > 0 else 0
    max_drawdown  = ((hist['Close'] / hist['Close'].cummax()) - 1).min()
    var_95        = daily_returns.quantile(0.05)

    # Price data for charts
    price_data = hist.reset_index()[['Date', 'Close', 'Volume', 'daily_return']].tail(252)
    price_data = price_data.fillna(0)
    price_data['Close']        = price_data['Close'].round(4)
    price_data['daily_return'] = price_data['daily_return'].round(6)
    price_data['Volume']       = price_data['Volume'].fillna(0).astype(int)
    price_data['Date']         = price_data['Date'].dt.strftime('%Y-%m-%d')

    # Use Alpha Vantage week 52 if available, else from history
    week_52_high = overview.get('week_52_high') or float(hist['Close'].max())
    week_52_low  = overview.get('week_52_low')  or float(hist['Close'].min())

    return {
        'ticker':        ticker,
        'name':          overview.get('name', ticker),
        'sector':        overview.get('sector', 'N/A'),
        'industry':      overview.get('industry', 'N/A'),
        'description':   overview.get('description', ''),
        'market_cap':    overview.get('market_cap', 0),
        'current_price': quote.get('current_price', 0),
        'price_change':  quote.get('price_change', 0),
        'week_52_high':  round(week_52_high, 2),
        'week_52_low':   round(week_52_low, 2),
        'pe_ratio':      overview.get('pe_ratio'),
        'dividend_yield':overview.get('dividend_yield', 0),
        'beta':          overview.get('beta'),
        'avg_volume':    overview.get('avg_volume', 0),
        'annual_return': round(annual_return * 100, 2),
        'annual_vol':    round(annual_vol * 100, 2),
        'sharpe_ratio':  round(sharpe, 3),
        'max_drawdown':  round(float(max_drawdown) * 100, 2),
        'var_95':        round(float(var_95) * 100, 2),
        'price_data':    price_data.to_dict('records'),
    }


# ============================================================
# ML MODELS
# ============================================================
def run_prophet_forecast(price_data: list, periods: int = 30) -> dict:
    """Run Prophet forecast on price data."""
    try:
        df = pd.DataFrame(price_data)[['Date', 'Close']]
        df.columns = ['ds', 'y']
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.dropna()
        df = df[df['y'] > 0]

        model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
        )
        model.fit(df)
        future   = model.make_future_dataframe(periods=periods, freq='B')
        forecast = model.predict(future)

        result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods + 30)
        result['ds'] = result['ds'].dt.strftime('%Y-%m-%d')

        last_actual  = df['y'].iloc[-1]
        forecast_end = forecast['yhat'].iloc[-1]
        direction    = 'bullish' if forecast_end > last_actual else 'bearish'
        change_pct   = ((forecast_end - last_actual) / last_actual * 100)

        return {
            'forecast_data':   result.to_dict('records'),
            'direction':       direction,
            'forecast_change': round(change_pct, 2),
            'forecast_end':    round(forecast_end, 2),
        }
    except Exception as e:
        logger.warning(f"Prophet forecast failed: {e}")
        return {'forecast_data': [], 'direction': 'neutral', 'forecast_change': 0, 'forecast_end': 0}


def calculate_momentum_score(price_data: list) -> dict:
    """Calculate momentum indicators."""
    try:
        closes  = pd.Series([d['Close'] for d in price_data if d.get('Close') and d['Close'] > 0])
        ma_20   = closes.rolling(20).mean().iloc[-1]
        ma_50   = closes.rolling(50).mean().iloc[-1]
        ma_200  = closes.rolling(200).mean().iloc[-1] if len(closes) >= 200 else None
        current = closes.iloc[-1]

        delta = closes.diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs    = gain / loss
        rsi   = (100 - (100 / (1 + rs))).iloc[-1]

        score = 0
        if current > ma_20:  score += 20
        if current > ma_50:  score += 25
        if ma_20 > ma_50:    score += 20
        if ma_200 and current > ma_200: score += 25
        if 40 < rsi < 70:    score += 10

        signals = []
        if current > ma_20:  signals.append("Price above 20-day MA")
        if current > ma_50:  signals.append("Price above 50-day MA")
        if ma_200 and current > ma_200: signals.append("Price above 200-day MA")
        if rsi > 70: signals.append("Overbought (RSI > 70)")
        if rsi < 30: signals.append("Oversold (RSI < 30)")

        return {
            'momentum_score': round(float(score), 1),
            'rsi':            round(float(rsi), 1),
            'ma_20':          round(float(ma_20), 2),
            'ma_50':          round(float(ma_50), 2),
            'ma_200':         round(float(ma_200), 2) if ma_200 else None,
            'signals':        signals,
        }
    except Exception as e:
        logger.warning(f"Momentum calc failed: {e}")
        return {'momentum_score': 0, 'rsi': 50, 'ma_20': 0, 'ma_50': 0, 'ma_200': None, 'signals': []}


# ============================================================
# CLAUDE AI COMMENTARY
# ============================================================
def generate_commentary(data: dict, forecast: dict, momentum: dict) -> str:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = f"""You are a senior investment analyst at an institutional asset management firm.
Write a concise, professional equity/fund commentary for {data['name']} ({data['ticker']}).

FINANCIAL DATA:
- Current Price: ${data['current_price']} ({data['price_change']:+.2f}% today)
- Market Cap: ${data['market_cap']:,.0f}
- Sector: {data['sector']} | Industry: {data['industry']}
- 52-Week Range: ${data['week_52_low']} - ${data['week_52_high']}
- P/E Ratio: {data['pe_ratio']}
- Beta: {data['beta']}
- Dividend Yield: {(data['dividend_yield'] or 0) * 100:.2f}%

RISK METRICS (1-Year):
- Annual Return: {data['annual_return']}%
- Annual Volatility: {data['annual_vol']}%
- Sharpe Ratio: {data['sharpe_ratio']}
- Max Drawdown: {data['max_drawdown']}%
- VaR (95%): {data['var_95']}%

ML FORECAST (30-day Prophet):
- Direction: {forecast['direction'].upper()}
- Projected Change: {forecast['forecast_change']:+.2f}%

MOMENTUM:
- Score: {momentum['momentum_score']}/100
- RSI: {momentum['rsi']}
- Signals: {', '.join(momentum['signals']) if momentum['signals'] else 'None'}

Write 3 paragraphs: (1) current positioning and drivers, (2) risk and technical outlook, (3) forward-looking view.
Use professional financial language. Reference actual numbers. No bullet points.
End with: 'This commentary is for informational purposes only and does not constitute investment advice.'"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        logger.error(f"Claude API error: {e}")
        return "Commentary generation temporarily unavailable."


def chat_with_analyst(ticker: str, question: str, context: dict, history: list) -> str:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    system = f"""You are a senior financial analyst assistant with expertise in {context.get('name', ticker)} ({ticker}).

Key data points:
- Price: ${context.get('current_price', 'N/A')} ({context.get('price_change', 0):+.2f}% today)
- Sector: {context.get('sector', 'N/A')}
- Annual Return: {context.get('annual_return', 'N/A')}%
- Sharpe Ratio: {context.get('sharpe_ratio', 'N/A')}
- Momentum Score: {context.get('momentum_score', 'N/A')}/100
- RSI: {context.get('rsi', 'N/A')}
- Max Drawdown: {context.get('max_drawdown', 'N/A')}%
- ML Forecast: {context.get('forecast_direction', 'N/A')} ({context.get('forecast_change', 0):+.2f}% projected 30 days)

Answer concisely and professionally. Always end with: 'This is not investment advice.'"""

    messages = []
    for turn in history[-6:]:
        if turn.get('role') and turn.get('content'):
            messages.append({"role": turn['role'], "content": turn['content']})
    messages.append({"role": "user", "content": question})

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            system=system,
            messages=messages
        )
        return response.content[0].text
    except Exception as e:
        logger.error(f"Chat API error: {e}")
        return "Having trouble connecting. Please try again."


# ============================================================
# NaN sanitizer
# ============================================================
def clean(obj):
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean(v) for v in obj]
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return 0
    return obj


# ============================================================
# ROUTES
# ============================================================
@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.post("/analyze")
async def analyze(req: TickerRequest, request: Request, _=Depends(rate_limit_check)):
    ticker = validate_ticker(req.ticker)
    logger.info(f"Analysis requested for: {ticker}")

    data     = fetch_financial_data(ticker)
    forecast = run_prophet_forecast(data['price_data'])
    momentum = calculate_momentum_score(data['price_data'])
    commentary = generate_commentary(data, forecast, momentum)

    data['momentum_score']     = momentum['momentum_score']
    data['rsi']                = momentum['rsi']
    data['forecast_direction'] = forecast['direction']
    data['forecast_change']    = forecast['forecast_change']

    return clean({
        'data':       data,
        'forecast':   forecast,
        'momentum':   momentum,
        'commentary': commentary,
    })


@app.post("/chat")
async def chat(req: ChatRequest, request: Request, _=Depends(rate_limit_check)):
    ticker   = validate_ticker(req.ticker)
    history  = req.history[:12]

    try:
        data     = fetch_financial_data(ticker)
        forecast = run_prophet_forecast(data['price_data'])
        momentum = calculate_momentum_score(data['price_data'])
        context  = {**data, **momentum,
                    'forecast_direction': forecast['direction'],
                    'forecast_change':    forecast['forecast_change']}
    except Exception:
        context = {'name': ticker}

    response = chat_with_analyst(ticker, req.question, context, history)
    return {'response': response}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {str(exc)}")
    return JSONResponse(status_code=500, content={"detail": "An unexpected error occurred."})
