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

    # --- symbol (e.g. "CRV/USDT", "CRV USDT")
    m_sym = re.search(r'([A-Za-z0-9\-_]+)\s*/?\s*USDT', text, re.IGNORECASE)
    symbol = m_sym.group(1).lower() if m_sym else None

    # --- direction (long / short)
    m_dir = re.search(r'\b(long|short)\b', text, re.IGNORECASE)
    direction = m_dir.group(1).lower() if m_dir else None

    # --- BUY ZONE (two values: left and right)
    num = r'[0-9]+(?:[.,][0-9]+)?'  # number with dot or comma
    m_buy = re.search(rf'\bBUY\s*ZONE\b\s*[:\-–—]?\s*({num})\s*[-–—]\s*({num})', text, re.IGNORECASE)
    open_range = None
    if m_buy:
        left = float(m_buy.group(1).replace(',', '.'))
        right = float(m_buy.group(2).replace(',', '.'))
        open_range = (left, right)

    # --- SELL / Take Profit (list of values at the end of the line)
    m_sell_line = re.search(r'\bSELL\b\s*[:\-–—]?\s*([^\n\r]+)', text, re.IGNORECASE)
    take_profit = []
    if m_sell_line:
        nums = re.findall(num, m_sell_line.group(1))
        take_profit = [float(n.replace(',', '.')) for n in nums]

    # --- Stop Loss
    m_sl = re.search(rf'\bSTOP[-\s]?LOSS\b\s*[:\-–—]?\s*({num})', text, re.IGNORECASE)
    stop_loss = float(m_sl.group(1).replace(',', '.')) if m_sl else None

    # --- Leverage (range if "max" present)
    leverage = None
    m_lev_line = re.search(r'Leverage\s*:\s*([^\n\r]+)', text, re.IGNORECASE)
    if m_lev_line:
        seg = m_lev_line.group(1)

        # 1) explicit numeric range like "5-20x" or "5 – 20x"
        m_range = re.search(r'(\d+)\s*[xX]?\s*[-–—]\s*(\d+)\s*[xX]?', seg)
        if m_range:
            leverage = (int(m_range.group(1)), int(m_range.group(2)))
        else:
            # 2) "5x ... max20" (may contain brackets/spaces)
            m_min = re.search(r'(\d+)\s*[xX]\b|\b(\d+)\b', seg)   # captures "5x" or bare "5"
            m_max = re.search(r'\bmax\s*([0-9]+)\b', seg, re.IGNORECASE)

            min_val = None
            if m_min:
                # group(1) if "5x", else group(2) for bare "5"
                min_val = int(m_min.group(1) or m_min.group(2))

            if m_max:
                max_val = int(m_max.group(1))
                leverage = (min_val if min_val is not None else max_val, max_val)
            elif min_val is not None:
                # only single value present -> return a single int
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

def get_funding_rate(symbol: str = "BTC", product_type: str = "usdt-futures", timeout: int = 10) -> float:
    """
    Fetch current funding rate (only the value) for a given symbol from Bitget API.
    """
    url = "https://api.bitget.com/api/v2/mix/market/current-fund-rate"
    params = {"symbol": f"{symbol.upper()}USDT", "productType": product_type}

    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()

    if payload.get("code") != "00000":
        raise Exception(f"Bitget API error: {payload}")

    data = payload.get("data", [])
    if not data:
        raise Exception("No data returned from Bitget")

    return float(data[0].get("fundingRate"))


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
    
def signal_data(message: str):

    # SYMBOL AND COIN_ID
    message_data = parse_trade_message(message)
    symbol = message_data['symbol']
    coin_id = next(x["id"] for x in c.COIN_LIST if x["symbol"] == symbol)

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
    c.WEIGHT_TP = [0.2, 0.6, 0.2]
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

    data = {
        "coin_id": coin_id,
        "symbol": symbol,
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