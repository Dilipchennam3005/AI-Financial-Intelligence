"""
================================================================================
AI Financial Intelligence Platform
FastAPI Backend — Secure, Rate-Limited, Production-Ready
================================================================================
Author: Dilip Chennam
"""

import os
import re
import time
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import yfinance as yf
import math
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
import anthropic
from prophet import Prophet

# ── Load environment variables
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ALLOWED_ORIGINS   = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
RATE_LIMIT        = int(os.getenv("RATE_LIMIT_PER_MIN", "10"))

if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY not set in environment")

# ── Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── App
app = FastAPI(
    title="AI Financial Intelligence Platform",
    description="Institutional-grade financial analysis powered by ML and Claude AI",
    version="1.0.0",
    docs_url=None,   # Disable Swagger in production
    redoc_url=None,
)

# ── CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ── Rate limiter (in-memory per IP)
request_counts: dict = defaultdict(list)

def rate_limit_check(request: Request):
    ip  = request.client.host
    now = time.time()
    # Keep only requests from last 60 seconds
    request_counts[ip] = [t for t in request_counts[ip] if now - t < 60]
    if len(request_counts[ip]) >= RATE_LIMIT:
        logger.warning(f"Rate limit exceeded for IP: {ip}")
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
        if not VALID_TICKER.match(v):
            raise ValueError("Invalid ticker format")
        return v

class ChatRequest(BaseModel):
    ticker: str
    question: str
    history: list = []

    @validator('ticker')
    def clean_ticker(cls, v):
        v = v.strip().upper()
        if not VALID_TICKER.match(v):
            raise ValueError("Invalid ticker format")
        return v

    @validator('question')
    def clean_question(cls, v):
        v = v.strip()
        if len(v) < 2:
            raise ValueError("Question too short")
        if len(v) > 500:
            raise ValueError("Question too long (max 500 characters)")
        return v

# ============================================================
# DATA FETCHING
# ============================================================
def fetch_financial_data(ticker: str) -> dict:
    """Fetch comprehensive financial data from yFinance."""
    try:
        import requests
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
        stock = yf.Ticker(ticker, session=session)
        info  = stock.info or {}

        if not info or info.get('regularMarketPrice') is None and info.get('currentPrice') is None:
            raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found or no data available.")

        # Historical price data - 1 year
        hist = stock.history(period="1y")
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No price history for '{ticker}'.")

        # Calculate returns
        hist['daily_return']  = hist['Close'].pct_change()
        hist['cumulative_ret'] = (1 + hist['daily_return']).cumprod() - 1

        # Risk metrics
        daily_returns  = hist['daily_return'].dropna()
        annual_return  = daily_returns.mean() * 252
        annual_vol     = daily_returns.std() * np.sqrt(252)
        sharpe         = (annual_return - 0.05) / annual_vol if annual_vol > 0 else 0
        max_drawdown   = ((hist['Close'] / hist['Close'].cummax()) - 1).min()
        var_95         = daily_returns.quantile(0.05)

        # Price data for charts
        price_data = hist.reset_index()[['Date', 'Close', 'Volume', 'daily_return']].tail(252)
        price_data = price_data.fillna(0)
        price_data['Close'] = price_data['Close'].round(4)
        price_data['daily_return'] = price_data['daily_return'].round(6)
        price_data['Date'] = price_data['Date'].dt.strftime('%Y-%m-%d')

        # Key info
        current_price = info.get('currentPrice') or info.get('regularMarketPrice') or hist['Close'].iloc[-1]
        prev_close    = info.get('previousClose') or hist['Close'].iloc[-2]
        price_change  = ((current_price - prev_close) / prev_close * 100) if prev_close else 0

        return {
            'ticker':       ticker,
            'name':         info.get('longName') or info.get('shortName', ticker),
            'sector':       info.get('sector', 'N/A'),
            'industry':     info.get('industry', 'N/A'),
            'market_cap':   info.get('marketCap', 0),
            'current_price':round(current_price, 2),
            'price_change': round(price_change, 2),
            'week_52_high': info.get('fiftyTwoWeekHigh', 0),
            'week_52_low':  info.get('fiftyTwoWeekLow', 0),
            'pe_ratio':     info.get('trailingPE') or info.get('forwardPE', None),
            'dividend_yield':info.get('dividendYield', 0),
            'beta':         info.get('beta', None),
            'avg_volume':   info.get('averageVolume', 0),
            'description':  info.get('longBusinessSummary', '')[:500] if info.get('longBusinessSummary') else '',
            'annual_return': round(annual_return * 100, 2),
            'annual_vol':    round(annual_vol * 100, 2),
            'sharpe_ratio':  round(sharpe, 3),
            'max_drawdown':  round(max_drawdown * 100, 2),
            'var_95':        round(var_95 * 100, 2),
            'price_data':    price_data.to_dict('records'),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching financial data.")


def run_prophet_forecast(price_data: list, periods: int = 30) -> dict:
    """Run Prophet forecast on price data."""
    try:
        df = pd.DataFrame(price_data)[['Date', 'Close']] if 'Close' in pd.DataFrame(price_data).columns else None
        if df is None:
            df = pd.DataFrame([(d['Date'], d.get('Close', d.get('close'))) for d in price_data], columns=['Date', 'Close'])

        df.columns = ['ds', 'y']
        df['ds']   = pd.to_datetime(df['ds'])
        df         = df.dropna()

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

        last_actual = df['y'].iloc[-1]
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
        logger.warning(f"Prophet forecast failed: {str(e)}")
        return {'forecast_data': [], 'direction': 'neutral', 'forecast_change': 0, 'forecast_end': 0}


def calculate_momentum_score(price_data: list) -> dict:
    """Calculate momentum indicators."""
    try:
        closes = pd.Series([d['Close'] for d in price_data if d.get('Close')])

        ma_20  = closes.rolling(20).mean().iloc[-1]
        ma_50  = closes.rolling(50).mean().iloc[-1]
        ma_200 = closes.rolling(200).mean().iloc[-1] if len(closes) >= 200 else None
        current = closes.iloc[-1]

        rsi_period = 14
        delta = closes.diff()
        gain  = delta.where(delta > 0, 0).rolling(rsi_period).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
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
        if rsi > 70:  signals.append("Overbought (RSI > 70)")
        if rsi < 30:  signals.append("Oversold (RSI < 30)")

        return {
            'momentum_score': round(score, 1),
            'rsi':            round(rsi, 1),
            'ma_20':          round(ma_20, 2),
            'ma_50':          round(ma_50, 2),
            'ma_200':         round(ma_200, 2) if ma_200 else None,
            'signals':        signals,
        }
    except Exception as e:
        logger.warning(f"Momentum calc failed: {str(e)}")
        return {'momentum_score': 0, 'rsi': 50, 'ma_20': 0, 'ma_50': 0, 'ma_200': None, 'signals': []}


# ============================================================
# CLAUDE AI COMMENTARY
# ============================================================
def generate_commentary(data: dict, forecast: dict, momentum: dict) -> str:
    """Generate institutional-grade commentary using Claude."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    prompt = f"""You are a senior investment analyst at an institutional asset management firm. 
Write a concise, professional fund/equity commentary for {data['name']} ({data['ticker']}).

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
- Forecast End Price: ${forecast['forecast_end']}

MOMENTUM ANALYSIS:
- Momentum Score: {momentum['momentum_score']}/100
- RSI: {momentum['rsi']}
- Key Signals: {', '.join(momentum['signals']) if momentum['signals'] else 'No strong signals'}

Write a 3-paragraph institutional commentary:
1. Current positioning and key performance drivers
2. Risk assessment and technical outlook based on the ML signals
3. Forward-looking view for institutional investors

Use professional financial language. Be specific — reference the actual numbers. 
Do not use bullet points. Do not give investment advice. End with a disclaimer."""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        logger.error(f"Claude API error: {str(e)}")
        return "Commentary generation temporarily unavailable."


def chat_with_analyst(ticker: str, question: str, context: dict, history: list) -> str:
    """Chat with Claude about a specific security."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    system = f"""You are a senior financial analyst assistant. You have detailed knowledge of {context.get('name', ticker)} ({ticker}).

Key facts:
- Current Price: ${context.get('current_price', 'N/A')}
- Sector: {context.get('sector', 'N/A')}
- Annual Return (1yr): {context.get('annual_return', 'N/A')}%
- Sharpe Ratio: {context.get('sharpe_ratio', 'N/A')}
- Momentum Score: {context.get('momentum_score', 'N/A')}/100
- RSI: {context.get('rsi', 'N/A')}
- Max Drawdown: {context.get('max_drawdown', 'N/A')}%
- ML Forecast: {context.get('forecast_direction', 'N/A')} ({context.get('forecast_change', 'N/A'):+.2f}% projected)

Answer questions concisely and professionally. Reference specific data when relevant.
Always include: 'This is not investment advice.' at the end of responses."""

    # Build conversation history (max last 6 turns for context window efficiency)
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
        logger.error(f"Chat API error: {str(e)}")
        return "I'm having trouble connecting right now. Please try again."


# ============================================================
# ROUTES
# ============================================================

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.post("/analyze")
async def analyze(req: TickerRequest, request: Request, _=Depends(rate_limit_check)):
    """Full ML analysis + AI commentary for a ticker."""
    ticker = validate_ticker(req.ticker)
    logger.info(f"Analysis requested for: {ticker}")

    # Fetch data
    data     = fetch_financial_data(ticker)
    forecast = run_prophet_forecast(data['price_data'])
    momentum = calculate_momentum_score(data['price_data'])

    # Generate AI commentary
    commentary = generate_commentary(data, forecast, momentum)

    # Merge momentum into data for chat context
    data['momentum_score']      = momentum['momentum_score']
    data['rsi']                 = momentum['rsi']
    data['forecast_direction']  = forecast['direction']
    data['forecast_change']     = forecast['forecast_change']

    import json
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean(v) for v in obj]
        elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return 0
        return obj
    return clean({
        'data':        data,
        'forecast':    forecast,
        'momentum':    momentum,
        'commentary':  commentary,
    })


@app.post("/chat")
async def chat(req: ChatRequest, request: Request, _=Depends(rate_limit_check)):
    """Chat with AI analyst about a specific security."""
    ticker   = validate_ticker(req.ticker)
    question = req.question
    history  = req.history[:12]  # Cap history for security

    logger.info(f"Chat request for {ticker}: {question[:50]}...")

    # Fetch fresh context
    try:
        data     = fetch_financial_data(ticker)
        forecast = run_prophet_forecast(data['price_data'])
        momentum = calculate_momentum_score(data['price_data'])
        context  = {**data, **momentum, 'forecast_direction': forecast['direction'], 'forecast_change': forecast['forecast_change']}
    except Exception:
        context = {'name': ticker}

    response = chat_with_analyst(ticker, question, context, history)
    return {'response': response}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {str(exc)}")
    return JSONResponse(status_code=500, content={"detail": "An unexpected error occurred."})
