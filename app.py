import logging
import requests
import pandas as pd
from telethon import TelegramClient, events
from telethon.sessions import StringSession

# Config from config_runtime.py
import config_runtime as c
import utils as u

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

# =====================  TELEGRAM LISTENER  ========================
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

            try:
                signal_data = u.signal_data(event.raw_text)
                data = {
                    "chat_id": event.chat_id,
                    "chat_name": getattr(chat, "title", None),
                    "sender_id": sender.id if sender else None,
                    "sender_name": getattr(sender, "first_name", None),
                    "text": event.raw_text,
                    "date": event.date.isoformat(),
                    'coin_id': signal_data['coin_id'],
                    'symbol': signal_data['symbol'],
                    "margin": signal_data['margin'],
                    "leverage_from": signal_data['leverage'][0],
                    "leverage_to": signal_data['leverage'][-1],
                    "direction": signal_data["direction"],
                    "open_range_from": signal_data["open_range"][0],
                    "open_range_to": signal_data["open_range"][-1],
                    'trend_overall': signal_data['trend_overall'],
                    'sentiment_up_pct': signal_data['sentiment_up_pct'],
                    'sentiment_down_pct': signal_data['sentiment_down_pct'],
                    'funding_rate': signal_data['funding_rate'],
                    'volume_now': signal_data['volume_now'],
                    'volume_avg_14d': signal_data['volume_avg_14d'],
                    'risk_condition': int(signal_data['risk_condition']),
                    'atr_1h': signal_data['atr_1h'],
                    'atr_4h': signal_data['atr_4h'],
                    'atr_tp2_ratio_1h': signal_data['atr_tp2_ratio_1h'],
                    'atr_tp2_ratio_4h': signal_data['atr_tp2_ratio_4h'],
                    'entry': signal_data['entry'],
                    'stop_loss': signal_data['stop_loss'],
                    'take_profit_1': signal_data['take_profit_1'],
                    'take_profit_2': signal_data['take_profit_2'],
                    'take_profit_3': signal_data['take_profit_3'],
                    'max_loss': signal_data['max_loss'],
                    'max_loss_pct': signal_data['max_loss_pct'],
                    'profit_tp1': signal_data['profit_tp1'],
                    'profit_tp2': signal_data['profit_tp2'],
                    'profit_tp3': signal_data['profit_tp3'],
                    'roi_tp1': signal_data['roi_tp1'],
                    'roi_tp2': signal_data['roi_tp2'],
                    'roi_tp3': signal_data['roi_tp3'],
                    'profit_total': signal_data['profit_total'],
                    'roi_total': signal_data['roi_total'],
                    'risk_reward_tp2': signal_data['risk_reward_tp2'],
                    'profit_condition': int(signal_data['profit_condition']),
                    'volume_condition': int(signal_data['volume_condition']),
                }
                
                logging.info("Sending to webhook: %s",
                            {k: data[k] for k in ("symbol", "chat_id", "date")})
                try:
                    resp = requests.post(WEBHOOK_URL, json=data, timeout=HTTP_TIMEOUT, headers=UA_HEADERS)
                    resp.raise_for_status()
                except Exception:
                    logging.exception("Webhook error")

            except Exception as e:
                logging.warning("Could not process message: %s", e)

            logging.info("Done processing message.")

        except Exception as e:
            logging.exception("Unhandled error in on_new_message: %s", e)

    logging.info("Telegram listener starting…")
    client.start()
    client.run_until_disconnected()

if __name__ == "__main__":
    logging.info("Booting telegram-listener container…")
    run_telegram_client()
