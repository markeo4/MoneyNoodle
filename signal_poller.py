import time
import requests
from datetime import datetime
import pandas as pd
from app import fetch_ohlcv, enhanced_money_noodle, get_enhanced_signal

# --------------- CONFIG SECTION -----------------

# Your Make.com webhook URL
WEBHOOK_URL = "https://hook.us1.make.com/otpwqt6e9mb931er1rv2yw5wgfh1psyg"

# Assets to monitor (feel free to add more)
SYMBOLS = [
    ("Yahoo Finance", "SPY", "4h"),
    ("Yahoo Finance", "SPY", "1h"),
    ("Yahoo Finance", "QQQ", "4h"),
    ("Kraken", "BTC/USD", "4h"),
    ("Kraken", "SOL/USD", "4h"),
    ("Kraken", "ETH/USD", "4h"),
]

# Polling interval in seconds (10 min)
POLL_INTERVAL = 600

# How many bars to look back for swing (match your web app)
SWING_LOOKBACK = 15

# --------------- STATE FOR DUPLICATE PREVENTION ---------------

last_signals = {}  # { "Yahoo Finance:SPY:4h": "BUY" }

# --------------- MAIN POLLING LOOP -----------------

def send_webhook(payload):
    """Sends JSON payload to Make.com webhook"""
    try:
        resp = requests.post(WEBHOOK_URL, json=payload, timeout=10)
        resp.raise_for_status()
        print(f"[{datetime.now()}] Webhook sent: {payload['symbol']} {payload['signal']}")
    except Exception as e:
        print(f"ERROR sending webhook: {e}")

while True:
    for source, symbol, timeframe in SYMBOLS:
        try:
            df = fetch_ohlcv(source, symbol, timeframe, limit=100)
            df = enhanced_money_noodle(df, swing_lookback=SWING_LOOKBACK)
            signal, reasons, stop_loss, take_profit, current_price, sl_pct, tp_pct, signal_level, indicators = get_enhanced_signal(df)
            key = f"{source}:{symbol}:{timeframe}"
            prev_signal = last_signals.get(key)

            # Only fire if signal is BUY/SELL and it's new
            if signal in ["BUY", "SELL", "STRONG BUY", "STRONG SELL"] and signal != prev_signal:
                payload = {
                    "source": source,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "signal": signal,
                    "signal_level": signal_level,
                    "current_price": current_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "sl_pct": sl_pct,
                    "tp_pct": tp_pct,
                    "reasons": reasons,
                    "indicators": indicators,
                    "timestamp": str(df['timestamp'].iloc[-1]) if 'timestamp' in df else datetime.now().isoformat(),
                }
                send_webhook(payload)
                last_signals[key] = signal

        except Exception as e:
            print(f"Error on {source} {symbol} {timeframe}: {e}")
    print(f"[{datetime.now()}] Sleeping {POLL_INTERVAL} seconds...\n")
    time.sleep(POLL_INTERVAL)