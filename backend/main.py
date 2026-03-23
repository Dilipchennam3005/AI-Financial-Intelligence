"""
================================================================================
AI Financial Intelligence Platform — Meridian Analytics
FastAPI Backend — yFinance + Claude API + Prophet
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
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
import anthropic
from prophet import Prophet

# ── Environment
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ALLOWED_ORIGINS   = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
RATE_LIMIT        = int(os.getenv("RATE_LIMIT_PER_MIN", "10"))

if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY not set in environment")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

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
        raise HTTPException(status_code=429, detail="Too many requests. Please wait.")
    request_counts[ip].append(now)

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
    def clean(cls, v):
        v = v.strip().upper()
        if not re.compile(r'^[A-Z0-9.\-]{1,10}$').match(v):
            raise ValueError("Invalid ticker")
        return v

class ChatRequest(BaseModel):
    ticker: str
    question: str
    history: list = []

    @validator('ticker')
    def clean_ticker(cls, v):
        v = v.strip().upper()
        if not re.compile(r'^[A-Z0-9.\-]{1,10}$').match(v):
            raise ValueError("Invalid ticker")
        return v

    @validator('question')
    def clean_question(cls, v):
        v = v.strip()
        if len(v) < 2 or len(v) > 500:
            raise ValueError("Invalid question length")
        return v

# ============================================================
# DATA FETCHING — yFinance only
# ============================================================
def fetch_financial_data(ticker: str) -> dict:
    """Fetch all financial data from yFinance."""
    try:
        stock = yf.Ticker(ticker)

        # Get historical data first — always works
        hist = stock.history(period="1y")
        if hist.empty:
            hist = stock.history(period="6mo")
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for '{ticker}'. Please check the ticker symbol.")

        # Try to get info — may fail for some tickers
        try:
            info = stock.info or {}
        except Exception:
            info = {}

        # Try fast_info as fallback
        try:
            fi = stock.fast_info
            current_price = getattr(fi, 'last_price', None) or info.get('currentPrice') or info.get('regularMarketPrice') or float(hist['Close'].iloc[-1])
            prev_close    = getattr(fi, 'previous_close', None) or info.get('previousClose') or float(hist['Close'].iloc[-2])
            market_cap    = getattr(fi, 'market_cap', None) or info.get('marketCap', 0)
            week_52_high  = getattr(fi, 'year_high', None) or info.get('fiftyTwoWeekHigh') or float(hist['Close'].max())
            week_52_low   = getattr(fi, 'year_low', None) or info.get('fiftyTwoWeekLow') or float(hist['Close'].min())
        except Exception:
            current_price = info.get('currentPrice') or info.get('regularMarketPrice') or float(hist['Close'].iloc[-1])
            prev_close    = info.get('previousClose') or float(hist['Close'].iloc[-2])
            market_cap    = info.get('marketCap', 0)
            week_52_high  = info.get('fiftyTwoWeekHigh') or float(hist['Close'].max())
            week_52_low   = info.get('fiftyTwoWeekLow') or float(hist['Close'].min())

        price_change = ((current_price - prev_close) / prev_close * 100) if prev_close else 0

        # Risk metrics
        hist['daily_return'] = hist['Close'].pct_change().fillna(0)
        daily_returns = hist['daily_return'].dropna()
        annual_return = daily_returns.mean() * 252
        annual_vol    = daily_returns.std() * np.sqrt(252)
        sharpe        = (annual_return - 0.05) / annual_vol if annual_vol > 0 else 0
        max_drawdown  = ((hist['Close'] / hist['Close'].cummax()) - 1).min()
        var_95        = daily_returns.quantile(0.05)

        # Price data for chart
        price_data = hist.reset_index()[['Date', 'Close', 'Volume', 'daily_return']].tail(252)
        price_data = price_data.fillna(0)
        price_data['Close']        = price_data['Close'].round(4)
        price_data['daily_return'] = price_data['daily_return'].round(6)
        price_data['Volume']       = price_data['Volume'].fillna(0).astype(int)
        price_data['Date']         = price_data['Date'].dt.strftime('%Y-%m-%d')

        return {
            'ticker':        ticker,
            'name':          info.get('longName') or info.get('shortName', ticker),
            'sector':        info.get('sector', 'N/A'),
            'industry':      info.get('industry', 'N/A'),
            'description':   (info.get('longBusinessSummary', '') or '')[:500],
            'market_cap':    float(market_cap or 0),
            'current_price': round(float(current_price), 2),
            'price_change':  round(float(price_change), 2),
            'week_52_high':  round(float(week_52_high), 2),
            'week_52_low':   round(float(week_52_low), 2),
            'pe_ratio':      info.get('trailingPE') or info.get('forwardPE'),
            'dividend_yield':float(info.get('dividendYield', 0) or 0),
            'beta':          info.get('beta'),
            'avg_volume':    float(info.get('averageVolume', 0) or 0),
            'annual_return': round(float(annual_return) * 100, 2),
            'annual_vol':    round(float(annual_vol) * 100, 2),
            'sharpe_ratio':  round(float(sharpe), 3),
            'max_drawdown':  round(float(max_drawdown) * 100, 2),
            'var_95':        round(float(var_95) * 100, 2),
            'price_data':    price_data.to_dict('records'),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        raise HTTPException(status_code=500, detail="Error fetching financial data.")


# ============================================================
# ML MODELS
# ============================================================
def run_prophet_forecast(price_data: list, periods: int = 30) -> dict:
    try:
        df = pd.DataFrame(price_data)[['Date', 'Close']]
        df.columns = ['ds', 'y']
        df['ds'] = pd.to_datetime(df['ds'])
        df = df[df['y'] > 0].dropna()

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

        last_actual  = float(df['y'].iloc[-1])
        forecast_end = float(forecast['yhat'].iloc[-1])
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
    try:
        closes  = pd.Series([d['Close'] for d in price_data if d.get('Close') and d['Close'] > 0])
        ma_20   = float(closes.rolling(20).mean().iloc[-1])
        ma_50   = float(closes.rolling(50).mean().iloc[-1])
        ma_200  = float(closes.rolling(200).mean().iloc[-1]) if len(closes) >= 200 else None
        current = float(closes.iloc[-1])

        delta = closes.diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi   = float((100 - (100 / (1 + gain / loss))).iloc[-1])

        score = 0
        if current > ma_20: score += 20
        if current > ma_50: score += 25
        if ma_20 > ma_50:   score += 20
        if ma_200 and current > ma_200: score += 25
        if 40 < rsi < 70:   score += 10

        signals = []
        if current > ma_20: signals.append("Price above 20-day MA")
        if current > ma_50: signals.append("Price above 50-day MA")
        if ma_200 and current > ma_200: signals.append("Price above 200-day MA")
        if rsi > 70: signals.append("Overbought (RSI > 70)")
        if rsi < 30: signals.append("Oversold (RSI < 30)")

        return {
            'momentum_score': round(score, 1),
            'rsi':            round(rsi, 1),
            'ma_20':          round(ma_20, 2),
            'ma_50':          round(ma_50, 2),
            'ma_200':         round(ma_200, 2) if ma_200 else None,
            'signals':        signals,
        }
    except Exception as e:
        logger.warning(f"Momentum calc failed: {e}")
        return {'momentum_score': 0, 'rsi': 50, 'ma_20': 0, 'ma_50': 0, 'ma_200': None, 'signals': []}


# ============================================================
# CLAUDE AI
# ============================================================
def generate_commentary(data: dict, forecast: dict, momentum: dict) -> str:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = f"""You are a senior investment analyst. Write a concise institutional commentary for {data['name']} ({data['ticker']}).

FINANCIAL DATA:
- Price: ${data['current_price']} ({data['price_change']:+.2f}% today)
- Market Cap: ${data['market_cap']:,.0f}
- Sector: {data['sector']} | {data['industry']}
- 52W: ${data['week_52_low']} - ${data['week_52_high']}
- P/E: {data['pe_ratio']} | Beta: {data['beta']} | Dividend: {(data['dividend_yield'] or 0)*100:.2f}%

RISK (1Y): Return {data['annual_return']}% | Vol {data['annual_vol']}% | Sharpe {data['sharpe_ratio']} | Drawdown {data['max_drawdown']}%

ML FORECAST (30d Prophet): {forecast['direction'].upper()} {forecast['forecast_change']:+.2f}%
MOMENTUM: Score {momentum['momentum_score']}/100 | RSI {momentum['rsi']}
Signals: {', '.join(momentum['signals']) if momentum['signals'] else 'None'}

Write 3 paragraphs: positioning/drivers, risk/technical, forward view. Professional tone, reference actual numbers. No bullets.
End: 'This commentary is for informational purposes only and does not constitute investment advice.'"""

    try:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text
    except Exception as e:
        logger.error(f"Claude API error: {e}")
        return "Commentary generation temporarily unavailable."


def chat_with_analyst(ticker: str, question: str, context: dict, history: list) -> str:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    system = f"""You are a senior financial analyst for {context.get('name', ticker)} ({ticker}).
Price: ${context.get('current_price')} ({context.get('price_change', 0):+.2f}%) | Sector: {context.get('sector')}
Return 1Y: {context.get('annual_return')}% | Sharpe: {context.get('sharpe_ratio')} | RSI: {context.get('rsi')}
Momentum: {context.get('momentum_score')}/100 | Forecast: {context.get('forecast_direction')} {context.get('forecast_change', 0):+.2f}%
Answer concisely and professionally. Always end: 'This is not investment advice.'"""

    messages = [{"role": m['role'], "content": m['content']} for m in history[-6:] if m.get('role') and m.get('content')]
    messages.append({"role": "user", "content": question})

    try:
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            system=system,
            messages=messages
        )
        return resp.content[0].text
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return "Having trouble connecting. Please try again."


# ── NaN sanitizer
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

    data       = fetch_financial_data(ticker)
    forecast   = run_prophet_forecast(data['price_data'])
    momentum   = calculate_momentum_score(data['price_data'])
    commentary = generate_commentary(data, forecast, momentum)

    data.update({
        'momentum_score':     momentum['momentum_score'],
        'rsi':                momentum['rsi'],
        'forecast_direction': forecast['direction'],
        'forecast_change':    forecast['forecast_change'],
    })

    return clean({'data': data, 'forecast': forecast, 'momentum': momentum, 'commentary': commentary})


@app.post("/chat")
async def chat(req: ChatRequest, request: Request, _=Depends(rate_limit_check)):
    ticker = validate_ticker(req.ticker)
    try:
        data     = fetch_financial_data(ticker)
        forecast = run_prophet_forecast(data['price_data'])
        momentum = calculate_momentum_score(data['price_data'])
        context  = {**data, **momentum, 'forecast_direction': forecast['direction'], 'forecast_change': forecast['forecast_change']}
    except Exception:
        context = {'name': ticker}

    return {'response': chat_with_analyst(ticker, req.question, context, req.history[:12])}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(status_code=500, content={"detail": "An unexpected error occurred."})
