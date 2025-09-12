import logging
import re
import requests
import pandas as pd
import time
import statistics
import config_runtime as c

### PARSE TRADE MESSAGE

def parse_trade_message(message):
    # Full text of the message
    text = message.strip()

    # --- symbol (e.g. "#CRV/USDT")
    m_sym = re.search(r'#([A-Za-z0-9\-_]+)/USDT', text, re.IGNORECASE)
    symbol = m_sym.group(1).lower() if m_sym else None

    if not symbol:
        return None

    # --- direction (long / short)
    m_dir = re.search(r'\b(long|short)\b', text, re.IGNORECASE)
    direction = m_dir.group(1).lower() if m_dir else None

    # --- Split message into segments based on keywords ---
    keywords = ['Exchanges', 'Leverage', 'BUY ZONE', 'SELL', 'STOP-LOSS']
    pattern = r'\b(' + '|'.join(keywords) + r')\b\s*[:\-–—]?\s*'
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    
    segments = {}
    if len(parts) > 1:
        it = iter(parts[1:])
        for key in it:
            normalized_key = key.lower().replace(' ', '_').replace('-', '_')
            segments[normalized_key] = next(it, '').strip()

    # --- Initialize variables ---
    open_range = None
    take_profit = []
    stop_loss = None
    leverage = None
    num = r'[0-9]+(?:[.,][0-9]+)?'

    # --- Parse BUY ZONE ---
    if 'buy_zone' in segments:
        seg = segments['buy_zone']
        m_buy = re.search(rf'({num})\s*[-–—]\s*({num})', seg, re.IGNORECASE)
        if m_buy:
            left = float(m_buy.group(1).replace(',', '.'))
            right = float(m_buy.group(2).replace(',', '.'))
            open_range = (left, right)

    # --- Parse SELL / Take Profit ---
    if 'sell' in segments:
        seg = segments['sell']
        nums = re.findall(num, seg)
        take_profit = [float(n.replace(',', '.')) for n in nums]

    # --- Parse Stop Loss ---
    if 'stop_loss' in segments:
        seg = segments['stop_loss']
        m_sl = re.search(rf'({num})', seg, re.IGNORECASE)
        if m_sl:
            stop_loss = float(m_sl.group(1).replace(',', '.'))

    # --- Parse Leverage ---
    if 'leverage' in segments:
        seg = segments['leverage']
        m_range = re.search(r'(\d+)\s*[xX]?\s*[-–—]\s*(\d+)\s*[xX]?', seg)
        if m_range:
            leverage = (int(m_range.group(1)), int(m_range.group(2)))
        else:
            m_min = re.search(r'(\d+)\s*[xX]', seg)
            m_max = re.search(r'\bmax\s*(\d+)\b', seg, re.IGNORECASE)
            min_val = int(m_min.group(1)) if m_min else None
            max_val = int(m_max.group(1)) if m_max else None

            if min_val is not None and max_val is not None:
                leverage = (min_val, max_val)
            elif max_val is not None:
                leverage = (max_val, max_val)
            elif min_val is not None:
                leverage = min_val

    # --- return the parsed data as a dict
    return {
        "symbol": symbol,
        "direction": direction,
        "open_range": open_range,
        "take_profit": take_profit,
        "stop_loss": stop_loss,
        "leverage": leverage,
    }


### API SENTIMENT (CG)

def get_sentiment_votes(coin_id: str = "bitcoin", timeout: int = 15) -> dict:

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
    params = {
        "localization": "false",
        "tickers": "false",
        "market_data": "false",
        "community_data": "true",
        "developer_data": "false",
        "sparkline": "false",
    }
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    j = resp.json()
    return {
        "up_pct": j.get("sentiment_votes_up_percentage"),
        "down_pct": j.get("sentiment_votes_down_percentage")
    }

### API FUNDING RATE (CG)

def get_funding_rate(symbol: str = "BTC", product_type: str = "usdt-futures", timeout: int = 10) -> float | None:
    """
    Fetch current funding rate (only the value) for a given symbol from Bitget API.
    Returns None if fetching fails.
    """
    url = "https://api.bitget.com/api/v2/mix/market/current-fund-rate"
    params = {"symbol": f"{symbol.upper()}USDT", "productType": product_type}

    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()

        if payload.get("code") != "00000":
            logging.error(f"Bitget API error for funding rate ({symbol}): {payload}")
            return None

        data = payload.get("data", [])
        if not data:
            logging.warning(f"No funding rate data returned from Bitget for {symbol}")
            return None

        return float(data[0].get("fundingRate"))

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch funding rate for {symbol}: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred in get_funding_rate for {symbol}: {e}")
        return None


### API CANLDES (BG)

def bitget_candles(base_coin: str, candle_period : str = '1h', queries_limit : int = 1000):
    ts = int(time.time())
    date_end_ts = (ts - (ts % 3600)) * 1000

    api_source = 'https://api.bitget.com/api/v2/spot/market/candles'
    symbol = f"{base_coin.strip().upper()}USDT"

    url = (
        f"{api_source}?symbol={symbol}"
        f"&granularity={candle_period}"
        f"&endTime={date_end_ts}"
        f"&limit={queries_limit}"
    )

    resp = requests.get(url, timeout=15)
    if resp.status_code != 200:
        return []

    payload = resp.json()
    data = payload.get('data', []) or []

    rows = [{
        "date_ts_ms":    int(item[0]),
        "date_utc":      pd.to_datetime(int(item[0]), unit="ms", utc=True),
        "price_open":    float(item[1]),
        "price_highest": float(item[2]),
        "price_lowest":  float(item[3]),
        "price_close":   float(item[4])
    } for item in data]


    df = pd.DataFrame(rows)
    df = df.sort_values("date_ts_ms").reset_index(drop=True)
    return df

# API COIN DATA (CG)

def coingecko_coin(
    coin_id: str = "bitcoin",
    vs_currency: str = "usd",
    days: int = 90,
    timeout: int = 15
) -> pd.DataFrame:

    #hourly data for 1-90 days: https://docs.coingecko.com/v3.0.1/reference/coins-id-market-chart

    api_source = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days}

    # Request data from CoinGecko
    resp = requests.get(api_source, params=params, timeout=timeout)
    if resp.status_code != 200:
        return pd.DataFrame(columns=["date_ts_ms", "date_ts_hour", "date_utc", "price_close", "volume_24h"])

    payload = resp.json() or {}
    prices = payload.get("prices", []) or []
    volumes = payload.get("total_volumes", []) or []

    # Convert to DataFrames
    df_p = pd.DataFrame(prices, columns=["date_ts_ms", "price_close"])
    df_v = pd.DataFrame(volumes, columns=["date_ts_ms", "volume_24h"])

    # Merge price and volume by timestamp
    df = pd.merge(df_p, df_v, on="date_ts_ms", how="outer")

    # Convert timestamp to integers
    df["date_ts_ms"] = df["date_ts_ms"].astype("int64")

    # Round timestamps down to the nearest full hour (ms precision)
    df["date_ts_hour"] = df["date_ts_ms"] - (df["date_ts_ms"] % 3600000)

    # Human-readable UTC date
    df["date_utc"] = pd.to_datetime(df["date_ts_hour"], unit="ms", utc=True)

    # Convert to numeric types
    df["price_close"] = pd.to_numeric(df["price_close"], errors="coerce")
    df["volume_24h"] = pd.to_numeric(df["volume_24h"], errors="coerce")

    # Sort by timestamp and drop duplicates
    df = df.sort_values("date_ts_hour").drop_duplicates(subset=["date_ts_hour"]).reset_index(drop=True)

    return df

### ATR CALCULATION

def atr(df, period=14) -> float:
    df = df.copy()
    
    # Previous close
    df["prev_close"] = df["price_close"].shift(1)
    
    # True Range components
    df["H-L"]  = df["price_highest"] - df["price_lowest"]
    df["H-PC"] = (df["price_highest"] - df["prev_close"]).abs()
    df["L-PC"] = (df["price_lowest"] - df["prev_close"]).abs()
    
    # True Range (TR)
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    
    # Average True Range (EMA of TR)
    atr_last = df["TR"].ewm(span=period, adjust=False).mean().iloc[-1]
    
    return float(atr_last)

### TREND

def get_trend(ema50: float, ema200: float, sep_pct: float = 0.001) -> str:
    diff = ema50 - ema200
    thr  = abs(ema200) * sep_pct
    if diff >  thr: return "BULL"
    if diff < -thr: return "BEAR"
    return "NEUTRAL"
    
def get_overall_trend(trend_1h: str, trend_4h: str) -> str:
    if trend_4h == "BULL" and trend_1h == "BULL":
        return "BULL_CONFIRMED"
    elif trend_4h == "BEAR" and trend_1h == "BEAR":
        return "BEAR_CONFIRMED"
    elif trend_4h == "BULL" and trend_1h == "NEUTRAL":
        return "BULL_BIAS"
    elif trend_4h == "BEAR" and trend_1h == "NEUTRAL":
        return "BEAR_BIAS"
    else:
        return "MIXED"
    

# Check if a symbol is available on Bitget futures market
def is_symbol_on_bitget_futures(symbol: str, product_type: str = "usdt-futures", timeout: int = 10) -> bool:
    """
    Check if a symbol is available on the Bitget futures market.
    """
    url = "https://api.bitget.com/api/v2/mix/market/contracts"
    params = {"productType": product_type}

    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()

        if payload.get("code") != "00000":
            logging.error(f"Bitget API error when fetching contracts: {payload}")
            return 0

        data = payload.get("data", [])
        if not data:
            logging.warning("No contract data returned from Bitget.")
            return 0

        # Check if the symbol exists in the list of contracts
        target_symbol = f"{symbol.upper()}USDT"
        is_available = any(contract.get("symbol") == target_symbol for contract in data)

        return 1 if is_available else 0

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch Bitget contracts: {e}")
        return 0
    except Exception as e:
        logging.error(f"An unexpected error occurred in is_symbol_on_bitget_futures: {e}")
        return 0


def signal_data(message: str):

    # SYMBOL AND COIN_ID
    message_data = parse_trade_message(message)
    symbol = message_data['symbol']
    try:
        coin_id = next(x["id"] for x in c.COIN_LIST if x["symbol"] == symbol)
    except StopIteration:
        logging.error(f"Coin ID not found for symbol: {symbol}")
        return None

    # SENTIMENT
    sentiment = get_sentiment_votes(coin_id)
    sentiment_votes_up_percentage = sentiment.get('up_pct', None)
    sentiment_votes_down_percentage = sentiment.get('down_pct', None)

    # FUNDING RATE
    funding_rate = get_funding_rate(symbol)

    # ATR
    df_candles_1h = bitget_candles(symbol, '1h')
    df_candles_4h = bitget_candles(symbol, '4h')
    atr_1h = atr(df_candles_1h)
    atr_4h = atr(df_candles_4h)

    # VOLUME
    df_cg_coin = coingecko_coin(coin_id=coin_id, vs_currency="usd", days=90)
    df_cg_coin = df_cg_coin.set_index("date_utc").sort_index()

    df_24h_volume = df_cg_coin["volume_24h"].resample("1d").last().dropna()
    volumen24h_avg14day = df_24h_volume.tail(14).mean()
    volumen24h_now = df_24h_volume.iloc[-1]
    vol_condition = volumen24h_now >= volumen24h_avg14day

    # PRICES 1h / 4h
    df_1h_price = df_cg_coin['price_close']  # index: 1H
    df_4h_price = df_cg_coin['price_close'].resample("4h").last().dropna()

    ema50_1h  = df_1h_price.ewm(span=50,  adjust=False, min_periods=50).mean().iloc[-1]
    ema200_1h = df_1h_price.ewm(span=200, adjust=False, min_periods=200).mean().iloc[-1]
    ema50_4h  = df_4h_price.ewm(span=50,  adjust=False, min_periods=50).mean().iloc[-1]
    ema200_4h = df_4h_price.ewm(span=200, adjust=False, min_periods=200).mean().iloc[-1]

    trend_1h = get_trend(ema50=ema50_1h, ema200=ema200_1h)
    trend_4h = get_trend(ema50=ema50_4h, ema200=ema200_4h)
    trend_overall = get_overall_trend(trend_1h, trend_4h)
    
    message_data = parse_trade_message(message)

    entry_wgt = (message_data['open_range'][0] + message_data['open_range'][1]) / 2
    risk_unit = entry_wgt - message_data['stop_loss'] if message_data['direction'] == 'long' else message_data['stop_loss'] - entry_wgt
    coins = (c.MARGIN * message_data['leverage'][-1])/entry_wgt # czy max??
    position_value = coins * entry_wgt
    max_loss = coins * risk_unit
    max_loss_pct = max_loss / c.MARGIN
    take_profit = [
        message_data['take_profit'][0],
        statistics.median(message_data['take_profit']),
        message_data['take_profit'][-1]
    ]

    logging.info("Processed message data: %s", message_data)

    sgn = 1 if message_data['direction'].lower() == 'long' else -1
    deltas = [(tp - entry_wgt) * sgn for tp in take_profit]
    profit_tp = [d * coins * w for d, w in zip(deltas, c.WEIGHT_TP)]
    roi = [p / c.MARGIN for p in profit_tp]
    profit_total = sum(profit_tp)
    roi_total = profit_total / c.MARGIN
    risk_to_reward = profit_tp[1]/max_loss
    atr_1h_tp2_ratio = sgn * (take_profit[1] - entry_wgt) / atr_1h
    atr_4h_tp2_ratio = sgn * (take_profit[1] - entry_wgt) / atr_4h
    risk_condition = max_loss_pct <= c.MAX_RISK
    profit_condition = roi_total >= c.MIN_PROFIT

    # Check if the symbol is available on Bitget futures market
    is_available = is_symbol_on_bitget_futures(symbol)

    data = {
        "coin_id": coin_id,
        "symbol": symbol,
        "available_on_bitget": is_available,
        "margin": c.MARGIN,
        "leverage": message_data['leverage'],
        "direction": message_data["direction"],
        "open_range": message_data["open_range"],
        "trend_overall": trend_overall,
        "sentiment_up_pct": sentiment_votes_up_percentage,
        "sentiment_down_pct": sentiment_votes_down_percentage,
        "funding_rate": funding_rate,
        "volume_now": volumen24h_now,
        "volume_avg_14d": volumen24h_avg14day,
        "volume_condition": vol_condition,
        "atr_1h": atr_1h,
        "atr_4h": atr_4h,
        "atr_tp2_ratio_1h": atr_1h_tp2_ratio,
        "atr_tp2_ratio_4h": atr_4h_tp2_ratio,
        "entry": entry_wgt,
        "stop_loss": message_data['stop_loss'],
        "take_profit_1": take_profit[0],
        "take_profit_2": take_profit[1],
        "take_profit_3": take_profit[2],
        "max_loss": max_loss,
        "max_loss_pct": max_loss_pct,
        "profit_tp1": profit_tp[0],
        "profit_tp2": profit_tp[1],
        "profit_tp3": profit_tp[2],
        "roi_tp1": roi[0],
        "roi_tp2": roi[1],
        "roi_tp3": roi[2],
        "profit_total": profit_total,
        "roi_total": roi_total,
        "risk_reward_tp2": risk_to_reward,
        "risk_condition": risk_condition,
        "profit_condition": profit_condition,
    }
    return data