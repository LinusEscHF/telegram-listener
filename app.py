import time
import logging

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    while True:
        logging.info("Telegram listener container is running...")
        time.sleep(30)