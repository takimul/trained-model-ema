# import os
# import json
# import requests
# import asyncio
# import numpy as np
# from datetime import datetime, timedelta, timezone
# from fastapi import FastAPI, Request
# import xgboost as xgb
# from sklearn.preprocessing import StandardScaler

# BOT_TOKEN = os.getenv("BOT_TOKEN")
# SYMBOLS = ["OPUSDT", "RENDERUSDT", "BTCUSDT"]
# TIMEFRAMES = ["5m", "15m"]

# app = FastAPI()
# user_ids = set()
# sent_signals = set()  # prevent duplicate alerts
# sent_hourly = set()   # track hourly alerts

# # ---------------- Telegram ----------------
# def send_telegram(chat_id, msg):
#     if not BOT_TOKEN:
#         print("BOT_TOKEN not set — would send:", msg)
#         return
#     try:
#         url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
#         requests.post(url, data={"chat_id": chat_id, "text": msg}, timeout=5)
#     except Exception as e:
#         print("Telegram send error:", e)

# def broadcast(msg):
#     for uid in list(user_ids):
#         send_telegram(uid, msg)

# # ---------------- Indicators ----------------
# def get_ema(values, period):
#     if len(values) < period:
#         return None
#     weights = np.exp(np.linspace(-1., 0., period))
#     weights /= weights.sum()
#     ema = np.convolve(values, weights, mode="full")[:len(values)]
#     ema[:period] = ema[period]
#     return float(np.round(ema[-1], 8))

# def get_rsi(closes, period=14):
#     if len(closes) < period + 1:
#         return None
#     deltas = np.diff(closes)
#     ups = deltas.clip(min=0)
#     downs = -1*deltas.clip(max=0)
#     roll_up = np.mean(ups[-period:])
#     roll_down = np.mean(downs[-period:])
#     if roll_down == 0: return 100
#     rs = roll_up / roll_down
#     return round(100 - (100 / (1 + rs)), 2)

# def get_macd(closes, fast=12, slow=26, signal=9):
#     if len(closes) < slow + signal:
#         return None, None, None
#     ema_fast = get_ema(closes, fast)
#     ema_slow = get_ema(closes, slow)
#     macd_line = ema_fast - ema_slow
#     signal_line = get_ema([get_ema(closes[:i], fast) - get_ema(closes[:i], slow) for i in range(slow, len(closes))], signal)
#     macd_hist = macd_line - signal_line if signal_line is not None else 0
#     return round(macd_line,5), round(signal_line,5) if signal_line else None, round(macd_hist,5)

# def compute_features(closes, volumes=None):
#     ema9 = get_ema(closes, 9)
#     ema26 = get_ema(closes, 26)
#     rsi14 = get_rsi(closes, 14)
#     macd_line, macd_signal, macd_hist = get_macd(closes)
#     last_volume = volumes[-1] if volumes else 0
#     return [ema9, ema26, rsi14, macd_line, macd_signal, macd_hist, last_volume]

# # ---------------- Binance ----------------
# def fetch_klines(symbol, interval, limit=100):
#     url = f"https://api.binance.com/api/v3/klines?symbol={symbol.upper()}&interval={interval}&limit={limit}"
#     try:
#         r = requests.get(url, timeout=10)
#         r.raise_for_status()
#         return r.json()
#     except Exception as e:
#         print(f"[REST] Error fetching klines {symbol} {interval}: {e}")
#         return []

# # ---------------- Telegram Webhook ----------------
# @app.post("/webhook/{token}")
# async def telegram_webhook(token: str, request: Request):
#     if token != BOT_TOKEN: return {"ok": False}
#     data = await request.json()
#     if "message" in data:
#         chat_id = data["message"]["chat"]["id"]
#         text = data["message"].get("text","")
#         if chat_id not in user_ids:
#             user_ids.add(chat_id)
#             send_telegram(chat_id, f"✅ Subscribed to EMA alerts!\nTracking: {', '.join(SYMBOLS)}")
#         if text.lower() == "/start":
#             send_telegram(chat_id, "👋 Welcome! EMA + AI alerts are active.")
#     return {"ok": True}

# # ---------------- ML Model ----------------
# def train_ml_model(closes, volumes=None):
#     X, y = [], []
#     for i in range(30, len(closes)-1):
#         feats = compute_features(closes[:i], volumes[:i] if volumes else None)
#         X.append(feats)
#         y.append(1 if closes[i+1] > closes[i] else 0)
#     if len(X) < 20: return None
#     model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
#     model.fit(np.array(X), np.array(y))
#     return model

# def predict_trend(model, closes, volumes=None):
#     feats = compute_features(closes, volumes)
#     return model.predict_proba([feats])[0][1] if model else None

# # ---------------- EMA Monitor ----------------
# async def monitor_ema(symbol, interval):
#     klines = fetch_klines(symbol, interval, limit=200)
#     closes = [float(k[4]) for k in klines]
#     volumes = [float(k[5]) for k in klines]
#     model = train_ml_model(closes, volumes)
#     prev_ema9 = get_ema(closes, 9)
#     prev_ema26 = get_ema(closes, 26)
    
#     while True:
#         await asyncio.sleep(5)
#         klines = fetch_klines(symbol, interval, limit=2)
#         if not klines: continue
#         close_price = float(klines[-1][4])
#         closes.append(close_price)
#         volumes.append(float(klines[-1][5]))
#         ema9 = get_ema(closes, 9)
#         ema26 = get_ema(closes, 26)
#         if prev_ema9 and prev_ema26 and ema9 and ema26:
#             if prev_ema9 < prev_ema26 and ema9 >= ema26:
#                 prob = predict_trend(model, closes, volumes)
#                 msg = f"📈 {symbol} ({interval}) EMA9 CROSSED ABOVE EMA26 — BUY 💰\nPrice: {close_price}"
#                 if prob: msg += f"\n🤖 Uptrend Probability: {round(prob*100,2)}%"
#                 broadcast(msg)
#             elif prev_ema9 > prev_ema26 and ema9 <= ema26:
#                 prob = predict_trend(model, closes, volumes)
#                 msg = f"📉 {symbol} ({interval}) EMA9 CROSSED BELOW EMA26 — SELL ⚠️\nPrice: {close_price}"
#                 if prob: msg += f"\n🤖 Downtrend Probability: {round((1-prob)*100,2)}%"
#                 broadcast(msg)
#         prev_ema9, prev_ema26 = ema9, ema26

# # ---------------- Hourly Close Alerts ----------------
# async def monitor_hourly():
#     while True:
#         now = datetime.now(timezone.utc)
#         # wait until next hour
#         next_hour = (now.replace(minute=0, second=5, microsecond=0) + timedelta(hours=1))
#         await asyncio.sleep((next_hour - now).total_seconds())
#         for symbol in SYMBOLS:
#             klines = fetch_klines(symbol, "1h", limit=2)
#             if not klines: continue
#             last = klines[-1]
#             close_time_ms = int(last[6])
#             key = (symbol, "1h", close_time_ms)
#             if key in sent_hourly: continue
#             close_price = float(last[4])
#             ts = datetime.fromtimestamp(close_time_ms/1000, tz=timezone.utc)
#             msg = f"🕐 {symbol} 1H Close ({ts.strftime('%Y-%m-%d %H:%M UTC')})\nClose: {close_price}"
#             broadcast(msg)
#             sent_hourly.add(key)

# # ---------------- Startup ----------------
# @app.on_event("startup")
# async def startup_event():
#     # EMA monitors
#     for symbol in SYMBOLS:
#         for tf in TIMEFRAMES:
#             asyncio.create_task(monitor_ema(symbol, tf))
#     # Hourly close alerts
#     asyncio.create_task(monitor_hourly())
# main.py
import os
import json
import requests
import asyncio
import numpy as np
import pathlib
import time
import joblib
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, Request
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, List

# ---------- Config ----------
BOT_TOKEN = os.getenv("BOT_TOKEN")
SYMBOLS = os.getenv("SYMBOLS", "OPUSDT,RENDERUSDT,BTCUSDT").split(",")  # CSV env override
TIMEFRAMES = ["5m", "15m"]
ML_LOOKBACK = 500            # how many candles to fetch for training/features
TRAIN_WINDOW = 200           # how many last candles to use for training features
PRED_LOOKAHEAD = 5           # label lookahead in candles (during training)
ML_REFRESH_HOURS = 4         # hours between retrains
MODELS_DIR = pathlib.Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# runtime
app = FastAPI()
user_ids = set()
sent_signals = set()   # (symbol, interval, close_time_ms)
sent_hourly = set()    # (symbol, 1h, close_time_ms)
ML_INFO = {}           # symbol -> {"model":..., "scaler":..., "trained_at": unix_ts}

USER_AGENT = "ai-emacross-bot/1.0"

# ---------- Utilities ----------
def send_telegram(chat_id: int, text: str):
    if not BOT_TOKEN:
        print("[TELEGRAM] BOT_TOKEN not set — would send:", text)
        return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=10)
    except Exception as e:
        print("Telegram send error:", e)

def broadcast(text: str):
    for uid in list(user_ids):
        send_telegram(uid, text)

def fetch_klines(symbol: str, interval: str, limit: int = 200) -> List:
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    headers = {"User-Agent": USER_AGENT}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=12)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[REST] fetch_klines {symbol} {interval} error:", e)
        return []

# ---------- Indicators ----------
def get_ema(values: List[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    weights = np.exp(np.linspace(-1., 0., period))
    weights /= weights.sum()
    ema = np.convolve(values, weights, mode="full")[:len(values)]
    ema[:period] = ema[period]
    return float(np.round(ema[-1], 8))

def get_rsi(closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    deltas = np.diff(closes)
    ups = deltas.clip(min=0)
    downs = -1 * deltas.clip(max=0)
    roll_up = np.mean(ups[-period:])
    roll_down = np.mean(downs[-period:])
    if roll_down == 0:
        return 100.0
    rs = roll_up / roll_down
    return round(100 - (100 / (1 + rs)), 2)

def get_macd(closes: List[float], fast=12, slow=26, signal=9) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if len(closes) < slow + signal:
        return None, None, None
    ema_fast = get_ema(closes, fast)
    ema_slow = get_ema(closes, slow)
    if ema_fast is None or ema_slow is None:
        return None, None, None
    macd_line = ema_fast - ema_slow
    # build MACD signal line using list of MACD values
    macd_vals = []
    for i in range(slow, len(closes)):
        f = get_ema(closes[:i+1], fast)
        s = get_ema(closes[:i+1], slow)
        if f is None or s is None:
            continue
        macd_vals.append(f - s)
    if len(macd_vals) < signal:
        return round(macd_line,5), None, 0.0
    signal_line = get_ema(macd_vals, signal)
    macd_hist = macd_line - signal_line if signal_line is not None else 0.0
    return round(macd_line,5), round(signal_line,5) if signal_line else None, round(macd_hist,5)

def compute_features(closes: List[float], volumes: Optional[List[float]] = None) -> List[float]:
    ema9 = get_ema(closes, 9) or 0.0
    ema26 = get_ema(closes, 26) or 0.0
    rsi14 = get_rsi(closes, 14) or 50.0
    macd_line, macd_signal, macd_hist = get_macd(closes)
    macd_line = macd_line or 0.0
    macd_signal = macd_signal or 0.0
    macd_hist = macd_hist or 0.0
    last_volume = volumes[-1] if volumes and len(volumes) > 0 else 0.0
    return [ema9, ema26, rsi14, macd_line, macd_signal, macd_hist, last_volume]

# ---------- ML helpers (train/load/save) ----------
def model_paths(symbol: str) -> Tuple[pathlib.Path, pathlib.Path]:
    sym = symbol.upper()
    model_file = MODELS_DIR / f"{sym}.json"
    scaler_file = MODELS_DIR / f"{sym}_scaler.pkl"
    return model_file, scaler_file

def train_model_for_symbol(symbol: str) -> Optional[Tuple[xgb.XGBClassifier, StandardScaler]]:
    try:
        klines = fetch_klines(symbol, "5m", limit=ML_LOOKBACK)
        if not isinstance(klines, list) or len(klines) < 50:
            print(f"[ML] Not enough data to train {symbol}")
            return None
        closes = [float(k[4]) for k in klines]
        volumes = [float(k[5]) for k in klines]
        X, y = [], []
        for i in range(30, len(closes) - PRED_LOOKAHEAD):
            feats = compute_features(closes[:i+1], volumes[:i+1])
            X.append(feats)
            future = closes[i + PRED_LOOKAHEAD]
            y.append(1 if future > closes[i] else 0)
        if len(X) < 30:
            print(f"[ML] insufficient training samples for {symbol}: {len(X)}")
            return None
        scaler = StandardScaler()
        Xs = scaler.fit_transform(np.array(X))
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_jobs=1)
        model.fit(Xs, np.array(y))
        # save
        model_file, scaler_file = model_paths(symbol)
        try:
            model.save_model(str(model_file))    # XGBoost JSON (or binary depending on version)
            joblib.dump(scaler, scaler_file)
            print(f"[ML] Trained and saved model for {symbol} -> {model_file}, {scaler_file}")
        except Exception as e:
            print("[ML] Warning: failed to save model/scaler:", e)
        return model, scaler
    except Exception as e:
        print("train_model_for_symbol error:", e)
        return None

def load_model_for_symbol(symbol: str) -> Optional[Tuple[xgb.XGBClassifier, StandardScaler]]:
    model_file, scaler_file = model_paths(symbol)
    if model_file.exists() and scaler_file.exists():
        try:
            model = xgb.XGBClassifier()
            model.load_model(str(model_file))
            scaler = joblib.load(scaler_file)
            print(f"[ML] Loaded model for {symbol}")
            return model, scaler
        except Exception as e:
            print("[ML] Error loading model/scaler:", e)
            return None
    return None

def ensure_model(symbol: str) -> Optional[Tuple[xgb.XGBClassifier, StandardScaler]]:
    info = ML_INFO.get(symbol)
    if info and info.get("model") and info.get("scaler"):
        return info["model"], info["scaler"]
    loaded = load_model_for_symbol(symbol)
    if loaded:
        model, scaler = loaded
        ML_INFO[symbol] = {"model": model, "scaler": scaler, "trained_at": int(time.time())}
        return model, scaler
    # train now (blocking) if not present
    res = train_model_for_symbol(symbol)
    if res:
        model, scaler = res
        ML_INFO[symbol] = {"model": model, "scaler": scaler, "trained_at": int(time.time())}
        return model, scaler
    return None

def predict_prob(symbol: str, closes: List[float], volumes: Optional[List[float]] = None) -> Optional[float]:
    info = ML_INFO.get(symbol)
    if not info:
        return None
    model = info["model"]
    scaler = info["scaler"]
    feats = compute_features(closes, volumes)
    Xs = scaler.transform([feats])
    prob = model.predict_proba(Xs)[0][1]
    return float(round(prob * 100, 2))

# ---------- Messaging helpers ----------
def strength_label(prob_pct: float, direction: str = "up") -> str:
    # prob_pct = 0..100 for uptrend probability
    if direction == "up":
        if prob_pct >= 75:
            return "Strong Uptrend 📈"
        if prob_pct >= 60:
            return "Moderate Uptrend ↗️"
        if prob_pct >= 52:
            return "Weak Uptrend ⚠️"
        return "Low confidence Uptrend ⚠️"
    else:
        # direction == "down": use 100 - prob_pct as down probability
        down_pct = 100 - prob_pct
        if down_pct >= 75:
            return "Strong Downtrend 📉"
        if down_pct >= 60:
            return "Moderate Downtrend ↘️"
        if down_pct >= 52:
            return "Weak Downtrend ⚠️"
        return "Low confidence Downtrend ⚠️"

def format_signal_message(symbol: str, interval: str, side: str, price: float, prob_pct: Optional[float]) -> str:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    if prob_pct is None:
        # simple message when no model available
        if side == "buy":
            return f"📈 {symbol} ({interval}) EMA9 crossed ABOVE EMA26 — *BUY*\nPrice: {price}\nTime: {ts}\n(No ML model available)"
        else:
            return f"📉 {symbol} ({interval}) EMA9 crossed BELOW EMA26 — *SELL*\nPrice: {price}\nTime: {ts}\n(No ML model available)"
    # with probability
    if side == "buy":
        label = strength_label(prob_pct, "up")
        return (f"📈 {symbol} ({interval}) EMA9 → EMA26 — *BUY*\n"
                f"Price: {price}\n"
                f"AI Up Probability: {prob_pct}% — {label}\n"
                f"Time: {ts}")
    else:
        # show down probability
        down_pct = round(100 - prob_pct, 2)
        label = strength_label(prob_pct, "down")
        return (f"📉 {symbol} ({interval}) EMA9 → EMA26 — *SELL*\n"
                f"Price: {price}\n"
                f"AI Down Probability: {down_pct}% — {label}\n"
                f"Time: {ts}")

# ---------- Time helper ----------
def interval_to_minutes(interval: str) -> int:
    if interval.endswith("m"):
        return int(interval[:-1])
    if interval.endswith("h"):
        return int(interval[:-1]) * 60
    raise ValueError("unsupported interval")

def seconds_until_next_close(interval_minutes: int) -> float:
    now = datetime.now(timezone.utc)
    minute = now.minute
    next_multiple = ((minute // interval_minutes) + 1) * interval_minutes
    next_hour = now.replace(minute=0, second=0, microsecond=0)
    if next_multiple >= 60:
        next_multiple -= 60
        next_hour = next_hour + timedelta(hours=1)
    next_close = next_hour.replace(minute=next_multiple, second=5, microsecond=0)
    wait = (next_close - now).total_seconds()
    return max(wait, 1.0)

# ---------- Monitor interval (fast detection on new/latest candle) ----------
async def monitor_interval(interval: str):
    minutes = interval_to_minutes(interval)
    print(f"[MON] Starting monitor for {interval} (sync every {minutes}min)")
    # keep a flag to skip the very first observed candle to avoid historical spam
    first_seen = {symbol: True for symbol in SYMBOLS}
    while True:
        wait = seconds_until_next_close(minutes)
        print(f"[SYNC] {interval} next close in {int(wait)}s (UTC {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')})")
        # but we will poll right before close and also shortly after to detect quickly
        await asyncio.sleep(max(wait - 5, 1))
        # poll loop for a short window around close to be responsive
        window_start = time.time()
        while time.time() - window_start < 12:  # ~12s window to catch the close quickly
            for symbol in SYMBOLS:
                klines = fetch_klines(symbol, interval, limit=26)
                if not klines or len(klines) < 2:
                    continue
                # latest closed or latest in-progress candle
                latest = klines[-1]
                prev = klines[-2]
                close_time_ms = int(latest[6])
                key = (symbol, interval, close_time_ms)
                # skip first observed candle to avoid sending historical cross alerts
                if first_seen.get(symbol, True):
                    first_seen[symbol] = False
                    continue
                # prepare closes arrays
                closes = [float(k[4]) for k in klines]
                vols = [float(k[5]) for k in klines]
                # compute prev EMAs using up to prev (closed)
                prev_closes = [float(k[4]) for k in klines[:-1]]
                if len(prev_closes) < 26:
                    continue
                prev_ema9 = get_ema(prev_closes, 9)
                prev_ema26 = get_ema(prev_closes, 26)
                cur_ema9 = get_ema(closes, 9)
                cur_ema26 = get_ema(closes, 26)
                if prev_ema9 is None or prev_ema26 is None or cur_ema9 is None or cur_ema26 is None:
                    continue
                bullish = (prev_ema9 < prev_ema26) and (cur_ema9 >= cur_ema26)
                bearish = (prev_ema9 > prev_ema26) and (cur_ema9 <= cur_ema26)
                if not bullish and not bearish:
                    continue
                # dedupe
                if key in sent_signals:
                    continue
                # ensure model loaded (best-effort)
                ensure_model(symbol)
                prob = predict_prob(symbol, closes, vols)
                side = "buy" if bullish else "sell"
                price = float(closes[-1])
                msg = format_signal_message(symbol, interval, side, price, prob)
                broadcast(msg)
                sent_signals.add(key)
            await asyncio.sleep(1)
        # small sleep until next iteration
        await asyncio.sleep(1)

# ---------- Hourly summaries ----------
async def monitor_hourly():
    # send last closed 1H immediately on startup (previous hour)
    for symbol in SYMBOLS:
        klines = fetch_klines(symbol, "1h", limit=2)
        if isinstance(klines, list) and len(klines) >= 2:
            last = klines[-2]
            close_time_ms = int(last[6])
            key = (symbol, "1h", close_time_ms)
            if key not in sent_hourly:
                o, h, l, c, v = float(last[1]), float(last[2]), float(last[3]), float(last[4]), float(last[5])
                ts = datetime.fromtimestamp(close_time_ms/1000, tz=timezone.utc)
                msg = f"🕐 {symbol} 1H Close ({ts.strftime('%Y-%m-%d %H:%M UTC')})\nOpen: {o} High: {h} Low: {l} Close: {c}\nVolume: {v}"
                broadcast(msg)
                sent_hourly.add(key)
    # sync to real hour closes
    while True:
        now = datetime.utcnow().replace(tzinfo=timezone.utc)
        next_hour = (now + timedelta(hours=1)).replace(minute=0, second=5, microsecond=0)
        wait = (next_hour - now).total_seconds()
        print(f"[HOUR] waiting {int(wait)}s until next 1H close")
        await asyncio.sleep(wait)
        for symbol in SYMBOLS:
            klines = fetch_klines(symbol, "1h", limit=1)
            if not klines:
                continue
            k = klines[-1]
            close_time_ms = int(k[6])
            key = (symbol, "1h", close_time_ms)
            if key in sent_hourly:
                continue
            o, h, l, c, v = float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])
            ts = datetime.fromtimestamp(close_time_ms/1000, tz=timezone.utc)
            msg = f"🕐 {symbol} 1H Close ({ts.strftime('%Y-%m-%d %H:%M UTC')})\nOpen: {o} High: {h} Low: {l} Close: {c}\nVolume: {v}"
            broadcast(msg)
            sent_hourly.add(key)
        await asyncio.sleep(1)

# ---------- ML refresher (retrain every ML_REFRESH_HOURS hours) ----------
async def ml_refresher_loop():
    print("[ML] Initial model load/train for all symbols...")
    # initial load/train
    for symbol in SYMBOLS:
        res = load_model_for_symbol(symbol)
        if res:
            model, scaler = res
            ML_INFO[symbol] = {"model": model, "scaler": scaler, "trained_at": int(time.time())}
        else:
            res2 = train_model_for_symbol(symbol)
            if res2:
                model, scaler = res2
                ML_INFO[symbol] = {"model": model, "scaler": scaler, "trained_at": int(time.time())}
    # periodic retrain
    while True:
        print(f"[ML] Sleeping {ML_REFRESH_HOURS} hours until next retrain...")
        await asyncio.sleep(ML_REFRESH_HOURS * 3600)
        for symbol in SYMBOLS:
            print(f"[ML] Retraining model for {symbol} ...")
            res = train_model_for_symbol(symbol)
            if res:
                model, scaler = res
                ML_INFO[symbol] = {"model": model, "scaler": scaler, "trained_at": int(time.time())}

# ---------- Telegram webhook endpoint ----------
@app.post("/webhook/{token}")
async def telegram_webhook(token: str, request: Request):
    if token != BOT_TOKEN:
        return {"ok": False}
    data = await request.json()
    if "message" in data:
        chat_id = data["message"]["chat"]["id"]
        text = data["message"].get("text", "")
        if chat_id not in user_ids:
            user_ids.add(chat_id)
            send_telegram(chat_id, f"✅ Subscribed to EMA alerts!\nTracking: {', '.join(SYMBOLS)}")
        if text.strip().lower() == "/start":
            send_telegram(chat_id, "👋 Welcome! EMA + AI alerts are active.")
        if text.strip().lower() == "/status":
            lines = [f"Subscribers: {len(user_ids)}"]
            for s in SYMBOLS:
                info = ML_INFO.get(s)
                if info:
                    t = datetime.utcfromtimestamp(info["trained_at"]).strftime("%Y-%m-%d %H:%M UTC")
                    lines.append(f"{s}: model trained at {t}")
                else:
                    lines.append(f"{s}: no model")
            send_telegram(chat_id, "\n".join(lines))
    return {"ok": True}

# ---------- Startup ----------
@app.on_event("startup")
async def startup_event():
    # start ML refresher (initial train/load + periodic retrain)
    asyncio.create_task(ml_refresher_loop())
    # start monitors
    for tf in TIMEFRAMES:
        asyncio.create_task(monitor_interval(tf))
    asyncio.create_task(monitor_hourly())
    print("[STARTUP] background tasks started")
