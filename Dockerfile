# XAUT EMA Pullback - Railway deployment
# Mudrex execution, Bybit prices
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bot runs as long-lived worker
CMD ["python", "bot.py"]
