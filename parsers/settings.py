LOG_LEVEL = "INFO"

# Parser settings
USER_AGENT = "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36"
# Obey robots.txt rules
ROBOTSTXT_OBEY = False


# Configure maximum concurrent requests performed by Scrapy (default: 16)
GB = 8 * 1024 * 1024 * 1024

CONCURRENT_REQUESTS = 3
CONCURRENT_REQUESTS_PER_DOMAIN = 16
DOWNLOAD_DELAY = 0.2
COOKIES_ENABLED = False
REACTOR_THREADPOOL_MAXSIZE = 16

# Change these to avoid timeouts to download and corresponding warnings
DOWNLOAD_WARNSIZE = 0.5 * GB
DOWNLOAD_MAXSIZE = 15 * GB
DOWNLOAD_TIMEOUT = 60 * 60 * 24
