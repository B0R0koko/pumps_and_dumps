from telethon.sync import TelegramClient
from pathlib import Path
from dotenv import load_dotenv
from typing import *

import re
import os
import json
import pandas as pd


class TelegramMessageParser:
    """Collects messages from telegram chats. Collects data and exports to csv files for manual labelling of pumps"""

    ROOT_DIR = Path(os.getcwd())
    MESSAGE_LIMIT = 5000

    def __init__(self, api_id: str, api_hash: str, output_dir: str) -> Self:
        self.client = TelegramClient("crypto_session", api_id, api_hash)
        self.client.start()

        self.output_dir = os.path.join(self.ROOT_DIR, output_dir)
        # Create output dir if it doesn't exist already
        os.makedirs(self.output_dir, exist_ok=True)

    def parse_messages(self, chat_slug: str) -> None:
        data: List[Dict[str, Any]] = []

        for i, message in enumerate(self.client.iter_messages(chat_slug, reverse=True)):
            msg_text: str | None = message.message
            msg_text: str | None = (
                re.sub(r"[^a-zA-Z0-9\s\,\.]", "", msg_text) if msg_text else None
            )

            data.append(
                {
                    "id": message.id,
                    "time": str(message.date),
                    "views": message.views,
                    "forwards": message.forwards,
                    "from_scheduled": message.from_scheduled,
                    "message": msg_text,
                }
            )

        self.write_to_csv(data=data, chat_slug=chat_slug)

    def write_to_json(self, data: List[Dict[str, Any]], chat_slug: str) -> None:
        """Dump data to json file in data folder in the root folder of the project"""
        output_path = os.path.join(self.output_dir, f"{chat_slug}.json")
        with open(output_path, "w") as file:
            json.dump(data, file)

    def write_to_csv(self, data: List[Dict[str, Any]], chat_slug: str) -> None:
        df: pd.DataFrame = pd.DataFrame.from_dict(data)
        df.to_csv(
            os.path.join(self.output_dir, f"{chat_slug}.csv"), mode="w", index=False
        )


def main() -> int:

    load_dotenv()

    chat = "cryptopumps"

    parser = TelegramMessageParser(
        api_hash=os.environ.get("API_HASH"),
        api_id=os.environ.get("API_ID"),
        # Folder contains chats only, they shouldn't be labeled there
        output_dir="data/telegram/chats",
    )

    parser.parse_messages(chat_slug=chat)


if __name__ == "__main__":
    main()
