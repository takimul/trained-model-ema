~~~{"variant":"standard","title":"FastAPI EMA+AI XGBoost Bot with Hourly Summary","id":"54322"}
import os
import json
import requests
import asyncio
import numpy as np
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, Request
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

BOT_TOKEN = os.getenv("BOT_TOKEN")
SYMBOLS = ["OPUSDT", "RENDERUSDT", "BTCUSDT"]
TIMEFRAMES = ["5m", "15m"]

app = FastAPI()
user_ids = set()
sent_signals = set()  # prevent duplicate alerts
sent_hourly = set()   # track hourly alerts

# ---------------- Telegram ----------------
def send_telegram(chat_id, msg):
    if not BOT_TOKEN:
        print("BOT_TOKEN not set — would send:", msg)
        return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": chat_id, "text": msg}, timeout=5)
    except Exception as e:
        print("Telegram send error:", e)

def broadcast(msg):
    for uid in list(user_ids):
        send_telegram(uid, msg)

# ---------------- Indicators ----------------
def get_ema(values, period):
    if len(values) < period:
        return None
    weights = np.exp(np.linspace(-1., 0., period))
    weights /= weights.sum()
    ema = np.convolve(values, weights, mode="full")[:len(values)]
    ema[:period] = ema[period]
    return float(np.round(ema[-1], 8))

def get_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    deltas = np.diff(closes)
    ups = deltas.clip(min=0)
    downs = -1*deltas.clip(max=0)
    roll_up = np.mean(ups[-period:])
    roll_down = np.mean(downs[-period:])
    if roll_down == 0: return 100
    rs = roll_up / roll_down
    return round(100 - (100 / (1 + rs)), 2)

def get_macd(closes, fast=12, slow=26, signal=9):
    if len(closes) < slow + signal:
        return None, None, None
    ema_fast = get_ema(closes, fast)
    ema_slow = get_ema(closes, slow)
    macd_line = ema_fast - ema_slow
    signal_line = get_ema([get_ema(closes[:i], fast) - get_ema(closes[:i], slow) for i in range(slow, len(closes))], signal)
    macd_hist = macd_line - signal_line if signal_line is not None else 0
    return round(macd_line,5), round(signal_line,5) if signal_line else None, round(macd_hist,5)

def compute_features(closes, volumes=None):
    ema9 = get_ema(closes, 9)
    ema26 = get_ema(closes, 26)
    rsi14 = get_rsi(closes, 14)
    macd_line, macd_signal, macd_hist = get_macd(closes)
    last_volume = volumes[-1] if volumes else 0
    return [ema9, ema26, rsi14, macd_line, macd_signal, macd_hist, last_volume]

# ---------------- Binance ----------------
def fetch_klines(symbol, interval, limit=100):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol.upper()}&interval={interval}&limit={limit}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[REST] Error fetching klines {symbol} {interval}: {e}")
        return []

# ---------------- Telegram Webhook ----------------
@app.post("/webhook/{token}")
async def telegram_webhook(token: str, request: Request):
    if token != BOT_TOKEN: return {"ok": False}
    data = await request.json()
    if "message" in data:
        chat_id = data["message"]["chat"]["id"]
        text = data["message"].get("text","")
        if chat_id not in user_ids:
            user_ids.add(chat_id)
            send_telegram(chat_id, f"✅ Subscribed to EMA alerts!\nTracking: {', '.join(SYMBOLS)}")
        if text.lower() == "/start":
            send_telegram(chat_id, "👋 Welcome! EMA + AI alerts are active.")
    return {"ok": True}

# ---------------- ML Model ----------------
def train_ml_model(closes, volumes=None):
    X, y = [], []
    for i in range(30, len(closes)-1):
        feats = compute_features(closes[:i], volumes[:i] if volumes else None)
        X.append(feats)
        y.append(1 if closes[i+1] > closes[i] else 0)
    if len(X) < 20: return None
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(np.array(X), np.array(y))
    return model

def predict_trend(model, closes, volumes=None):
    feats = compute_features(closes, volumes)
    return model.predict_proba([feats])[0][1] if model else None

# ---------------- EMA Monitor ----------------
async def monitor_ema(symbol, interval):
    klines = fetch_klines(symbol, interval, limit=200)
    closes = [float(k[4]) for k in klines]
    volumes = [float(k[5]) for k in klines]
    model = train_ml_model(closes, volumes)
    prev_ema9 = get_ema(closes, 9)
    prev_ema26 = get_ema(closes, 26)
    
    while True:
        await asyncio.sleep(5)
        klines = fetch_klines(symbol, interval, limit=2)
        if not klines: continue
        close_price = float(klines[-1][4])
        closes.append(close_price)
        volumes.append(float(klines[-1][5]))
        ema9 = get_ema(closes, 9)
        ema26 = get_ema(closes, 26)
        if prev_ema9 and prev_ema26 and ema9 and ema26:
            if prev_ema9 < prev_ema26 and ema9 >= ema26:
                prob = predict_trend(model, closes, volumes)
                msg = f"📈 {symbol} ({interval}) EMA9 CROSSED ABOVE EMA26 — BUY 💰\nPrice: {close_price}"
                if prob: msg += f"\n🤖 Uptrend Probability: {round(prob*100,2)}%"
                broadcast(msg)
            elif prev_ema9 > prev_ema26 and ema9 <= ema26:
                prob = predict_trend(model, closes, volumes)
                msg = f"📉 {symbol} ({interval}) EMA9 CROSSED BELOW EMA26 — SELL ⚠️\nPrice: {close_price}"
                if prob: msg += f"\n🤖 Downtrend Probability: {round((1-prob)*100,2)}%"
                broadcast(msg)
        prev_ema9, prev_ema26 = ema9, ema26

# ---------------- Hourly Close Alerts ----------------
async def monitor_hourly():
    while True:
        now = datetime.now(timezone.utc)
        # wait until next hour
        next_hour = (now.replace(minute=0, second=5, microsecond=0) + timedelta(hours=1))
        await asyncio.sleep((next_hour - now).total_seconds())
        for symbol in SYMBOLS:
            klines = fetch_klines(symbol, "1h", limit=2)
            if not klines: continue
            last = klines[-1]
            close_time_ms = int(last[6])
            key = (symbol, "1h", close_time_ms)
            if key in sent_hourly: continue
            close_price = float(last[4])
            ts = datetime.fromtimestamp(close_time_ms/1000, tz=timezone.utc)
            msg = f"🕐 {symbol} 1H Close ({ts.strftime('%Y-%m-%d %H:%M UTC')})\nClose: {close_price}"
            broadcast(msg)
            sent_hourly.add(key)

# ---------------- Startup ----------------
@app.on_event("startup")
async def startup_event():
    # EMA monitors
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            asyncio.create_task(monitor_ema(symbol, tf))
    # Hourly close alerts
    asyncio.create_task(monitor_hourly())
~~~
