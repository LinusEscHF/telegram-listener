import logging
import time
import requests
import pandas as pd
from telethon import TelegramClient, events
from telethon.sessions import StringSession

# Config from config_runtime.py
import config_runtime as c
import utils as u

UA_HEADERS       = {"User-Agent": "telegram-listener/1.0 (+azure-container-apps)"}

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

# --- Cooldown state ---
last_message_time = 0

# =====================  TELEGRAM LISTENER  ========================
def _build_client() -> TelegramClient:
    if c.TELEGRAM_SESSION_STRING:
        logging.info("Starting TelegramClient with StringSession")
        return TelegramClient(StringSession(c.TELEGRAM_SESSION_STRING), c.API_ID, c.API_HASH)
    else:
        logging.error("TELEGRAM_SESSION_STRING is not configured.")
        raise ValueError("TELEGRAM_SESSION_STRING is not configured.")

def run_telegram_client():
    client = _build_client()

    @client.on(events.NewMessage(incoming=True, chats=c.TARGET_CHAT_ID))
    async def on_new_message(event: events.NewMessage.Event):
        global last_message_time
        current_time = time.time()

        if current_time - last_message_time < c.COOLDOWN_SECONDS:
            remaining = c.COOLDOWN_SECONDS - (current_time - last_message_time)
            logging.info(f"Cooldown active. Ignoring message for {remaining:.1f} more seconds.")
            return
        
        try:
            # Update last message time
            last_message_time = current_time

            logging.info(
                f"msg.id={event.message.id} "
                f"grouped_id={getattr(event.message, 'grouped_id', None)} "
                f"via_bot={getattr(event.message, 'via_bot_id', None)} "
                f"fwd={bool(event.message.fwd_from)} "
                f"date={event.date.isoformat()} "
                f"text={event.raw_text[:120]!r}"
            )

            if c.SAFE_MODE:
                event.message._client.send_message = lambda *a, **k: None
                event.message._client.send_file = lambda *a, **k: None

            sender = await event.get_sender()
            chat   = await event.get_chat()

            logging.info(f"New message in {chat.title if hasattr(chat, 'title') else chat.id}  from {sender.id if sender else 'unknown'}: {event.raw_text}")

            # Forward telegram message to Zapier
            try:
                requests.post(
                    "https://hooks.zapier.com/hooks/catch/12655008/u1ba50b/",
                    json={
                        "chat_id": event.chat_id,
                        "chat_name": getattr(chat, "title", None),
                        "sender_id": sender.id if sender else None,
                        "sender_name": getattr(sender, "first_name", None),
                        "text": event.raw_text,
                        "date": event.date.isoformat()
                    },
                    timeout=c.HTTP_TIMEOUT,
                    headers=UA_HEADERS
                )
            except Exception:
                logging.exception("Zapier webhook error")

            try:
                signal_data = u.signal_data(event.raw_text)
                logging.info(f"Parsed signal data: {signal_data}")

                if not signal_data:
                    logging.info("No valid signal data found, skipping.")
                    return

                data = {
                    "chat_id": event.chat_id,
                    "chat_name": getattr(chat, "title", None),
                    "sender_id": sender.id if sender else None,
                    "sender_name": getattr(sender, "first_name", None),
                    "text": event.raw_text,
                    "date": event.date.isoformat(),
                    'coin_id': signal_data.get('coin_id'),
                    'symbol': signal_data.get('symbol'),
                    "margin": signal_data.get('margin'),
                    "leverage_from": signal_data.get('leverage', [None])[0],
                    "leverage_to": signal_data.get('leverage', [None])[-1],
                    "direction": signal_data.get("direction"),
                    "open_range_from": signal_data.get("open_range", [None])[0],
                    "open_range_to": signal_data.get("open_range", [None])[-1],
                    'trend_overall': signal_data.get('trend_overall'),
                    'sentiment_up_pct': signal_data.get('sentiment_up_pct'),
                    'sentiment_down_pct': signal_data.get('sentiment_down_pct'),
                    'funding_rate': signal_data.get('funding_rate'),
                    'volume_now': signal_data.get('volume_now'),
                    'volume_avg_14d': signal_data.get('volume_avg_14d'),
                    'risk_condition': int(signal_data.get('risk_condition', 0)),
                    'atr_1h': signal_data.get('atr_1h'),
                    'atr_4h': signal_data.get('atr_4h'),
                    'atr_tp2_ratio_1h': signal_data.get('atr_tp2_ratio_1h'),
                    'atr_tp2_ratio_4h': signal_data.get('atr_tp2_ratio_4h'),
                    'entry': signal_data.get('entry'),
                    'stop_loss': signal_data.get('stop_loss'),
                    'take_profit_1': signal_data.get('take_profit_1'),
                    'take_profit_2': signal_data.get('take_profit_2'),
                    'take_profit_3': signal_data.get('take_profit_3'),
                    'max_loss': signal_data.get('max_loss'),
                    'max_loss_pct': signal_data.get('max_loss_pct'),
                    'profit_tp1': signal_data.get('profit_tp1'),
                    'profit_tp2': signal_data.get('profit_tp2'),
                    'profit_tp3': signal_data.get('profit_tp3'),
                    'roi_tp1': signal_data.get('roi_tp1'),
                    'roi_tp2': signal_data.get('roi_tp2'),
                    'roi_tp3': signal_data.get('roi_tp3'),
                    'profit_total': signal_data.get('profit_total'),
                    'roi_total': signal_data.get('roi_total'),
                    'risk_reward_tp2': signal_data.get('risk_reward_tp2'),
                    'profit_condition': int(signal_data.get('profit_condition', 0)),
                    'volume_condition': int(signal_data.get('volume_condition', 0)),
                }
                
                logging.info("Sending to webhook: %s",
                            {k: data[k] for k in ("symbol", "chat_id", "date")})
                try:
                    resp = requests.post(c.WEBHOOK_URL, json=data, timeout=c.HTTP_TIMEOUT, headers=UA_HEADERS)
                    resp.raise_for_status()
                except Exception:
                    logging.exception("Webhook error")

            except Exception:
                logging.exception("Could not process message")

            logging.info("Done processing message.")

        except Exception as e:
            logging.exception("Unhandled error in on_new_message: %s", e)

    logging.info("Telegram listener starting…")
    client.start()
    client.run_until_disconnected()

if __name__ == "__main__":
    logging.info("Booting telegram-listener container…")
    run_telegram_client()
