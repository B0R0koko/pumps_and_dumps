START_DATE = "2017-07-01"
END_DATE = "2024-02-28"

LOG_LEVEL = "INFO"


# Parser settings
USER_AGENT = "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36"
# Obey robots.txt rules
ROBOTSTXT_OBEY = False

# Configure maximum concurrent requests performed by Scrapy (default: 16)
CONCURRENT_REQUESTS = 32
DOWNLOAD_DELAY = 0

# config file with all pairs traded on Binance
import os
from pathlib import Path

ROOT_DIR = Path(os.getcwd())
CONFIG_PATH = os.path.join(ROOT_DIR, "cfg/tickers.json")

# data folder collected from binance
OUTPUT_DIR = os.path.join(ROOT_DIR, "data/trades")
