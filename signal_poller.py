import time
import requests
import pandas as pd
from datetime import datetime
import json
import csv
import os
from app import fetch_ohlcv, enhanced_money_noodle, get_enhanced_signal

WEBHOOK_URL = "https://hook.us1.make.com/otpwqt6e9mb931er1rv2yw5wgfh1psyg"
TICKER_FILE = "tickers.csv"
SIGNALS_LOG_FILE = "last_signals.json"
POLL_INTERVAL = 600
SWING_LOOKBACK = 15

def read_symbols_from_file(filepath=TICKER_FILE):
    symbols = []
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            symbols.append((row["source"], row["symbol"], row["timeframe"]))
    return symbols

def read_previous_signals(filename=SIGNALS_LOG_FILE):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            try:
                data = json.load(f)
                return data.get("current_signals", {})
            except json.JSONDecodeError:
                return {}
    return {}

def write_signals_log(current_signals, previous_signals, filename=SIGNALS_LOG_FILE):
    log = {
        "last_run": datetime.now().isoformat(),
        "current_signals": current_signals,
        "previous_signals": previous_signals
    }
    with open(filename, "w") as f:
        json.dump(log, f, indent=2)

def send_webhook(payload):
    try:
        resp = requests.post(WEBHOOK_URL, json=payload, timeout=10)
        resp.raise_for_status()
        print(f"[{datetime.now()}] Webhook sent: {payload['symbol']} {payload['signal']}")
    except Exception as e:
        print(f"ERROR sending webhook: {e}")

while True:
    SYMBOLS = read_symbols_from_file()
    
    # Read previous signals from the last run
    previous_run_signals = read_previous_signals()
    
    current_run_signals = {}  # signals detected in this run
    
    for source, symbol, timeframe in SYMBOLS:
        try:
            df = fetch_ohlcv(source, symbol, timeframe, limit=100)
            df = enhanced_money_noodle(df, swing_lookback=SWING_LOOKBACK)
            signal, reasons, stop_loss, take_profit, current_price, sl_pct, tp_pct, signal_level, indicators = get_enhanced_signal(df)
            key = f"{source}:{symbol}:{timeframe}"
            
            prev_signal_for_ticker = previous_run_signals.get(key)
            current_run_signals[key] = signal  # store for logging

            if signal in ["BUY", "SELL", "STRONG BUY", "STRONG SELL"] and signal != prev_signal_for_ticker:
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
                # No longer update last_signals here, it's handled by write_signals_log at the end of the loop

        except Exception as e:
            print(f"Error on {source} {symbol} {timeframe}: {e}")

    # Write current and previous signals to log file
    write_signals_log(current_run_signals, previous_run_signals)
    print(f"[{datetime.now()}] Sleeping {POLL_INTERVAL} seconds...\n")
    time.sleep(POLL_INTERVAL)