# XAUT EMA Pullback - Railway deployment
# Mudrex execution, Bybit prices
FROM python:3.11-slim

WORKDIR /app

# git required for pip install from GitHub (mudrex-trading-sdk)
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bot runs as long-lived worker
CMD ["python", "bot.py"]
