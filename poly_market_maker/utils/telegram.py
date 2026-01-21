import os
import requests
import logging

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


class Telegram:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        
        if not self.bot_token or not self.chat_id:
            logger.warning("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set")
        
        self._initialized = True

    def send_message(self, message: str) -> bool:
        """Send a message to the Telegram group."""
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram not configured, skipping message")
            return False
        
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False


def main():
    telegram = Telegram()
    success = telegram.send_message("Test message from poly-market-maker")
    print(f"Message sent: {success}")


if __name__ == "__main__":
    main()
