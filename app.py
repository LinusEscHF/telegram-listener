import os
import re
import time
import json
import logging
import requests
import pandas as pd
from telethon import TelegramClient, events
from telethon.sessions import StringSession

# Config from config_runtime.py
import config_runtime as c

API_ID           = int(c.API_ID)
API_HASH         = c.API_HASH
TELEGRAM_SESSION_STRING = c.TELEGRAM_SESSION_STRING
TELEGRAM_SESSION_NAME = c.TELEGRAM_SESSION_NAME
TARGET_CHAT_ID   = int(c.TARGET_CHAT_ID)
SAFE_MODE        = c.SAFE_MODE
WEBHOOK_URL      = c.WEBHOOK_URL
COIN_LIST        = c.COIN_LIST

HTTP_TIMEOUT     = c.HTTP_TIMEOUT
UA_HEADERS       = {"User-Agent": "telegram-listener/1.0 (+azure-container-apps)"}

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

# ================  HELPER FUNCTIONS  ================

def extract_symbol(message: str) -> str | None:
    match = re.search(r'([A-Za-z0-9\-_]+)\s*/?\s*USDT', message or "", re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None

def get_sentiment_votes(coin_id: str = "bitcoin", timeout: int = HTTP_TIMEOUT) -> dict:
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
    params = {
        "localization": "false",
        "tickers": "false",
        "market_data": "false",
        "community_data": "true",
        "developer_data": "false",
        "sparkline": "false",
    }
    resp = requests.get(url, params=params, timeout=timeout, headers=UA_HEADERS)
    resp.raise_for_status()
    j = resp.json()
    return {
        "up_pct": j.get("sentiment_votes_up_percentage"),
        "down_pct": j.get("sentiment_votes_down_percentage"),
    }

def get_funding_rate(symbol: str = "BTC", product_type: str = "usdt-futures", timeout: int = HTTP_TIMEOUT) -> float:
    url = "https://api.bitget.com/api/v2/mix/market/current-fund-rate"
    params = {"symbol": f"{symbol.upper()}USDT", "productType": product_type}
    resp = requests.get(url, params=params, timeout=timeout, headers=UA_HEADERS)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("code") != "00000":
        raise Exception(f"Bitget API error: {payload}")
    data = payload.get("data", [])
    if not data:
        raise Exception("No data returned from Bitget")
    return float(data[0].get("fundingRate"))

def bitget_candles(base_coin: str, candle_period: str = "1h", queries_limit: int = 1000) -> pd.DataFrame:
    ts = int(time.time())
    date_end_ts = (ts - (ts % 3600)) * 1000
    api_source = "https://api.bitget.com/api/v2/spot/market/candles"
    symbol = f"{base_coin.strip().upper()}USDT"
    url = (
        f"{api_source}?symbol={symbol}"
        f"&granularity={candle_period}"
        f"&endTime={date_end_ts}"
        f"&limit={queries_limit}"
    )
    resp = requests.get(url, timeout=HTTP_TIMEOUT, headers=UA_HEADERS)
    if resp.status_code != 200:
        return pd.DataFrame(columns=["date_ts_ms","date_utc","price_open","price_highest","price_lowest","price_close"])
    payload = resp.json()
    data = payload.get("data", []) or []
    rows = [{
        "date_ts_ms":    int(item[0]),
        "date_utc":      pd.to_datetime(int(item[0]), unit="ms", utc=True),
        "price_open":    float(item[1]),
        "price_highest": float(item[2]),
        "price_lowest":  float(item[3]),
        "price_close":   float(item[4]),
    } for item in data]
    df = pd.DataFrame(rows)
    df = df.sort_values("date_ts_ms").reset_index(drop=True)
    return df

def coingecko_coin(coin_id: str = "bitcoin", vs_currency: str = "usd", days: int = 90, timeout: int = HTTP_TIMEOUT) -> pd.DataFrame:
    api_source = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days}
    resp = requests.get(api_source, params=params, timeout=timeout, headers=UA_HEADERS)
    if resp.status_code != 200:
        return pd.DataFrame(columns=["date_ts_ms","date_ts_hour","date_utc","price_close","volume_24h"])
    payload = resp.json() or {}
    df_p = pd.DataFrame(payload.get("prices", []) or [], columns=["date_ts_ms", "price_close"])
    df_v = pd.DataFrame(payload.get("total_volumes", []) or [], columns=["date_ts_ms", "volume_24h"])
    df = pd.merge(df_p, df_v, on="date_ts_ms", how="outer")
    df["date_ts_ms"] = pd.to_numeric(df["date_ts_ms"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["date_ts_ms"]).astype({"date_ts_ms":"int64"})
    df["date_ts_hour"] = df["date_ts_ms"] - (df["date_ts_ms"] % 3_600_000)
    df["date_utc"] = pd.to_datetime(df["date_ts_hour"], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
    df["price_close"] = pd.to_numeric(df["price_close"], errors="coerce")
    df["volume_24h"] = pd.to_numeric(df["volume_24h"], errors="coerce")
    df = df.sort_values("date_ts_hour").drop_duplicates(subset=["date_ts_hour"]).reset_index(drop=True)
    return df

def atr(df: pd.DataFrame, period: int = 14) -> float:
    df = df.copy()
    if df.empty:
        return float("nan")
    df["prev_close"] = df["price_close"].shift(1)
    df["H-L"]  = df["price_highest"] - df["price_lowest"]
    df["H-PC"] = (df["price_highest"] - df["prev_close"]).abs()
    df["L-PC"] = (df["price_lowest"]  - df["prev_close"]).abs()
    df["TR"]   = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df["ATR"]  = df["TR"].ewm(span=period, adjust=False).mean()
    return float(df["ATR"].iloc[-1])

# =====================  TELEGRAM  ========================
def _build_client() -> TelegramClient:
    if TELEGRAM_SESSION_STRING:
        logging.info("Starting TelegramClient with StringSession")
        return TelegramClient(StringSession(TELEGRAM_SESSION_STRING), API_ID, API_HASH)
    logging.info("Starting TelegramClient with file-based session")
    return TelegramClient(TELEGRAM_SESSION_NAME, API_ID, API_HASH)

def run_telegram_client():
    client = _build_client()

    @client.on(events.NewMessage(incoming=True))
    async def on_new_message(event: events.NewMessage.Event):
        try:
            if event.out or event.chat_id != TARGET_CHAT_ID:
                return

            if SAFE_MODE:
                event.message._client.send_message = lambda *a, **k: None
                event.message._client.send_file = lambda *a, **k: None

            sender = await event.get_sender()
            chat   = await event.get_chat()

            symbol = extract_symbol(event.raw_text)
            if not symbol:
                logging.info("No symbol found in message; skipping")
                return

            try:
                coin_id = next(x["id"] for x in COIN_LIST if x["symbol"].upper() == symbol.upper())
            except StopIteration:
                logging.info("Symbol %s not in COIN_LIST; skipping", symbol)
                return

            # External data
            try:
                sentiment = get_sentiment_votes(coin_id)
            except Exception:
                logging.exception("Sentiment fetch failed")
                sentiment = {"up_pct": None, "down_pct": None}

            try:
                funding_rate = get_funding_rate(symbol)
            except Exception:
                logging.exception("Funding rate fetch failed")
                funding_rate = None

            try:
                df_candles_1h = bitget_candles(symbol, "1h")
                df_candles_4h = bitget_candles(symbol, "4h")
                atr_1h = atr(df_candles_1h)
                atr_4h = atr(df_candles_4h)
            except Exception:
                logging.exception("ATR calc failed")
                atr_1h = atr_4h = None

            try:
                df_cg_coin = coingecko_coin(coin_id=coin_id, vs_currency="usd", days=90)
                df_24h_volume = df_cg_coin["volume_24h"].iloc[::-1][::24][::-1]
                volumen24h_avg14day = float(df_24h_volume[-14:].mean())
                volumen24h_last     = float(df_24h_volume.iloc[-1])
                df_1h_price = df_cg_coin["price_close"]
                df_4h_price = df_cg_coin["price_close"].iloc[::-1][::4][::-1]
                ema50_1h  = float(df_1h_price.ewm(span=50).mean().iloc[-1])
                ema200_1h = float(df_1h_price.ewm(span=200).mean().iloc[-1])
                ema50_4h  = float(df_4h_price.ewm(span=50).mean().iloc[-1])
                ema200_4h = float(df_4h_price.ewm(span=200).mean().iloc[-1])
            except Exception:
                logging.exception("CoinGecko/EMA calc failed")
                volumen24h_avg14day = volumen24h_last = None
                ema50_1h = ema200_1h = ema50_4h = ema200_4h = None

            data = {
                "chat_id": event.chat_id,
                "chat_name": getattr(chat, "title", None),
                "sender_id": sender.id if sender else None,
                "sender_name": getattr(sender, "first_name", None),
                "text": event.raw_text,
                "date": event.date.isoformat(),
                "coin_id": coin_id,
                "symbol": symbol,
                "Price 1h EMA(50)": ema50_1h,
                "Price 1h EMA(200)": ema200_1h,
                "Price 4h EMA(50)": ema50_4h,
                "Price 4h EMA(200)": ema200_4h,
                "Volumen 24h AVG(14)": volumen24h_avg14day,
                "Volumen 24h Last": volumen24h_last,
                "Price 1h ATR(14)": atr_1h,
                "Price 4h ATR(14)": atr_4h,
                "Sentiment Up %": (sentiment.get("up_pct") if isinstance(sentiment, dict) else None),
                "Sentiment Down %": (sentiment.get("down_pct") if isinstance(sentiment, dict) else None),
                "Funding Rate": funding_rate,
            }

            logging.info("[telegram_listener] Sending to webhook: %s",
                         {k: data[k] for k in ("symbol", "chat_id", "date")})
            try:
                resp = requests.post(WEBHOOK_URL, json=data, timeout=HTTP_TIMEOUT, headers=UA_HEADERS)
                resp.raise_for_status()
            except Exception:
                logging.exception("[telegram_listener] Webhook error")

        except Exception:
            logging.exception("Unhandled error in on_new_message")

    logging.info("Telegram listener starting…")
    client.start()
    client.run_until_disconnected()

if __name__ == "__main__":
    logging.info("Booting telegram-listener container…")
    run_telegram_client()
