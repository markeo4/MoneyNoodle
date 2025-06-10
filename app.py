from flask import Flask, render_template, request
import ccxt
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.offline as pyo

app = Flask(__name__)

DATA_SOURCES = {
    'Kraken': {'type': 'crypto', 'client': ccxt.kraken()},
    'Coinbase': {'type': 'crypto', 'client': ccxt.coinbase()},
    'Yahoo Finance': {'type': 'stock', 'client': None} # yfinance doesn't need a client object like ccxt
}

ALLOWED_TIMEFRAMES = ['1h', '4h', '1d', '1w']
DEFAULT_LIMITS = {'1h': 200, '4h': 150, '1d': 100, '1w': 52}

def get_signal(df):
    if len(df) < 34:  # Ensure enough data for EMA34
        return "WAIT", ["Insufficient data for indicators."], None, None, None, None, None

    df = df.copy()
    df['sma20_volume'] = df['volume'].rolling(20).mean()
    sma20_volume = df['sma20_volume'].iloc[-1]
    last = df.iloc[-1]
    prev = df.iloc[-2]

    signal = "WAIT"
    reasons = []
    stop_loss = None
    take_profit = None

    # Loosen Band Conditions: Add proximity check
    price_near_lower = abs(last['close'] - last['lower']) / last['close'] < 0.01
    price_bounce = (prev['close'] < prev['lower'] and last['close'] > last['lower']) or price_near_lower

    # Relax EMA Conditions: Check only the most recent candle
    ema_bullish = last['ema8'] > last['ema21'] > last['ema34']
    # Fine-Tune RSI Thresholds: Widen range
    rsi_not_overbought = last['rsi'] < 70
    # Adjust Volume Threshold: Lower multiplier
    volume_high = last['volume'] > 1.0 * sma20_volume

    price_near_upper = abs(last['close'] - last['upper']) / last['close'] < 0.01
    price_reject = (prev['close'] > prev['upper'] and last['close'] < last['upper']) or price_near_upper

    ema_bearish = last['ema8'] < last['ema21'] < last['ema34']
    rsi_not_oversold = last['rsi'] > 30

    # Debugging Output
    print(f"DEBUG: Price Bounce: {price_bounce}, EMA Bullish: {ema_bullish}, RSI: {last['rsi']}, Volume High: {volume_high}")
    print(f"DEBUG: Price Reject: {price_reject}, EMA Bearish: {ema_bearish}")

    # Always show levels to watch
    stop_loss = min(df['low'].tail(3))  # recent swing low
    take_profit = last['upper']         # upper band as TP

    if price_bounce and ema_bullish and rsi_not_overbought and volume_high:
        signal = "BUY"
        reasons.append("Price bounced above lower band. EMAs bullish for last 3 candles. RSI not overbought. Volume above average.")
        stop_loss = min(df['low'].tail(3))
        risk = last['close'] - stop_loss
        take_profit = last['close'] + 2 * risk

    if price_reject and ema_bearish and rsi_not_oversold and volume_high:
        signal = "SELL"
        reasons.append("Price rejected from upper band. EMAs bearish for last 3 candles. RSI not oversold. Volume above average.")
        stop_loss = max(df['high'].tail(3))
        risk = stop_loss - last['close']
        take_profit = last['close'] - 2 * risk

    if signal == "WAIT":
        reasons.append("No strong signal. Watching for bounce off lower band or rejection from upper band.")

    if last['rsi'] > 75:
        reasons.append("RSI is overbought—risk of pullback.")
    elif last['rsi'] < 25:
        reasons.append("RSI is oversold—risk of reversal.")

    # Calculate % distances
    current_price = last['close']
    sl_pct = None
    tp_pct = None
    if stop_loss and current_price:
        sl_pct = 100.0 * abs(current_price - stop_loss) / current_price
    if take_profit and current_price:
        tp_pct = 100.0 * abs(current_price - take_profit) / current_price

    return signal, reasons, stop_loss, take_profit, current_price, sl_pct, tp_pct

def get_symbols(source_name):
    """Return sorted list of available trading pairs/tickers for the selected source."""
    source_info = DATA_SOURCES.get(source_name)
    if not source_info:
        return []

    if source_info['type'] == 'crypto':
        exchange = source_info['client']
        try:
            markets = exchange.load_markets()
            return sorted(markets.keys())
        except Exception as e:
            print(f"Error fetching crypto symbols from {source_name}: {e}")
            return []
    elif source_info['type'] == 'stock':
        # For yfinance, provide a predefined list of popular tickers
        return sorted(['SPY', 'QQQ', 'DIA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META'])
    return []

def fetch_ohlcv(source_name, symbol, timeframe='1w', limit=100):
    source_info = DATA_SOURCES.get(source_name)
    if not source_info:
        raise ValueError(f"Unknown data source: {source_name}")

    df = pd.DataFrame()
    if source_info['type'] == 'crypto':
        exchange = source_info['client']
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    elif source_info['type'] == 'stock':
        # yfinance uses different timeframe strings and limits
        yf_timeframe_map = {
            '1h': '60m', # yfinance supports 60m for intraday
            '4h': '4h',  # yfinance supports 4h
            '1d': '1d',
            '1w': '1wk'
        }
        yf_period_map = {
            '1h': '7d', # 7 days for 1h data
            '4h': '60d', # 60 days for 4h data
            '1d': '2y', # 2 years for 1d data
            '1w': '5y'  # 5 years for 1w data
        }
        
        interval = yf_timeframe_map.get(timeframe, '1d')
        period = yf_period_map.get(timeframe, '2y')

        ticker = yf.Ticker(symbol)
        # Fetch data using period and interval
        data = ticker.history(period=period, interval=interval)
        if not data.empty:
            df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df['timestamp'] = df.index
            df = df.reset_index(drop=True) # Reset index to make it 0-based
        else:
            raise ValueError(f"No data fetched for {symbol} from Yahoo Finance.")

    return df

def money_noodle(df, length=20, mult=2, band_type='ATR'):
    df = df.copy()
    # EMA as the midline
    df['ema'] = df['close'].ewm(span=length, adjust=False).mean()
    
    # ATR or StdDev for band width
    if band_type == 'ATR':
        df['tr'] = np.maximum(df['high'] - df['low'],
                              np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
        df['band_width'] = df['tr'].rolling(window=length).mean()
    else:  # StdDev
        df['band_width'] = df['close'].rolling(window=length).std()
    
    # RSI for additional band expansion
    delta = df['close'].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    ma_up = up.rolling(length).mean()
    ma_down = down.rolling(length).mean()
    rs = ma_up / (ma_down + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    # Widen bands if RSI is overbought/oversold
    df['band_mult'] = mult
    df.loc[(df['rsi'] > 70) | (df['rsi'] < 30), 'band_mult'] = mult * 1.5

    df['upper'] = df['ema'] + df['band_mult'] * df['band_width']
    df['lower'] = df['ema'] - df['band_mult'] * df['band_width']

    # EMA ribbon for trend strength
    for n in [8, 21, 34]:
        df[f'ema{n}'] = df['close'].ewm(span=n, adjust=False).mean()
    return df

@app.route('/', methods=['GET', 'POST'])
def index():
    chart_div = ""
    error = None
    data_sources = list(DATA_SOURCES.keys())
    selected_source = 'Kraken' # Changed from selected_exchange
    selected_symbol = 'SOL/USD'
    band_type = 'ATR'
    selected_timeframe = '1w'
    timeframes = ALLOWED_TIMEFRAMES

    # Get symbols based on the initial selected source
    symbols = get_symbols(selected_source)

    if request.method == 'POST':
        selected_source = request.form.get('exchange', data_sources[0])
        # Prioritize typed input over dropdown selection
        selected_symbol_input = request.form.get('symbol_input', '').strip().upper()
        selected_symbol_dropdown = request.form.get('symbol_dropdown', '')
        
        if selected_symbol_input:
            selected_symbol = selected_symbol_input
        else:
            selected_symbol = selected_symbol_dropdown

        band_type = request.form.get('band_type', 'ATR')
        selected_timeframe = request.form.get('timeframe', '1w')
        symbols = get_symbols(selected_source) # Update symbols based on new source

    # No explicit check here, let the try-except block handle invalid symbols during data fetch.
    # Ensure selected_symbol is passed to template for persistence in input field
    
    try:
        # Limit is less relevant for yfinance with period/interval, but keep for crypto
        limit = DEFAULT_LIMITS.get(selected_timeframe, 100)
        df = fetch_ohlcv(selected_source, selected_symbol, selected_timeframe, limit)
        df = money_noodle(df, band_type=band_type)
        signal, reasons, stop_loss, take_profit, current_price, sl_pct, tp_pct = get_signal(df)
        # Create plotly chart
        trace_candles = go.Candlestick(
            x=df['timestamp'], open=df['open'], high=df['high'],
            low=df['low'], close=df['close'], name='Candles'
        )
        trace_ema = go.Scatter(
            x=df['timestamp'], y=df['ema'], line=dict(color='blue'), name='EMA'
        )
        trace_upper = go.Scatter(
            x=df['timestamp'], y=df['upper'], line=dict(color='green', dash='dash'), name='Upper Band'
        )
        trace_lower = go.Scatter(
            x=df['timestamp'], y=df['lower'], line=dict(color='red', dash='dash'), name='Lower Band'
        )
        # EMA Ribbon
        ribbon_traces = []
        for n, color in zip([8, 21, 34], ['orange', 'purple', 'teal']):
            ribbon_traces.append(go.Scatter(
                x=df['timestamp'], y=df[f'ema{n}'], line=dict(color=color, dash='dot'), name=f'EMA{n}'
            ))

        trace_volume = go.Bar(
            x=df['timestamp'], y=df['volume'],
            name='Volume', marker_color='lightgray',
            yaxis='y2', opacity=0.5
        )

        layout = go.Layout(
            title=f"{selected_symbol} Money Noodle",
            xaxis=dict(title="Date"),
            yaxis=dict(title="Price"),
            yaxis2=dict(
                title="Volume", overlaying='y', side='right', showgrid=False,
                range=[0, df['volume'].max() * 2]  # auto-scale
            ),
            height=700
        )

        data = [trace_candles, trace_ema, trace_upper, trace_lower] + ribbon_traces + [trace_volume]
        fig = go.Figure(data=data, layout=layout)
        chart_div = pyo.plot(fig, output_type='div', include_plotlyjs=True)
    except Exception as e:
        error = f"Could not fetch or plot data for {selected_symbol} from {selected_source}. Error: {e}"
        # Provide fallback values
        signal = None
        reasons = None
        stop_loss = None
        take_profit = None
        current_price = None
        sl_pct = None
        tp_pct = None
        return render_template(
            'index.html',
            chart_div=chart_div,
            error=error,
            exchanges=data_sources,
            selected_exchange=selected_source,
            symbols=symbols,
            selected_symbol=selected_symbol, # Pass selected_symbol to persist typed value
            band_type=band_type,
            timeframes=timeframes,
            selected_timeframe=selected_timeframe,
            signal=signal,
            reasons=reasons,
            stop_loss=stop_loss,
            take_profit=take_profit,
            current_price=current_price,
            sl_pct=sl_pct,
            tp_pct=tp_pct
        )


    return render_template(
        'index.html',
        chart_div=chart_div,
        error=error,
        exchanges=data_sources,
        selected_exchange=selected_source,
        symbols=symbols,
        selected_symbol=selected_symbol, # Pass selected_symbol to persist typed value
        band_type=band_type,
        timeframes=timeframes,
        selected_timeframe=selected_timeframe,
        signal=signal,
        reasons=reasons,
        stop_loss=stop_loss,
        take_profit=take_profit,
        current_price=current_price,
        sl_pct=sl_pct,
        tp_pct=tp_pct
    )

    try:
        # Limit is less relevant for yfinance with period/interval, but keep for crypto
        limit = DEFAULT_LIMITS.get(selected_timeframe, 100)
        df = fetch_ohlcv(selected_source, selected_symbol, selected_timeframe, limit)
        df = money_noodle(df, band_type=band_type)
        signal, reasons, stop_loss, take_profit, current_price, sl_pct, tp_pct = get_signal(df)
        # Create plotly chart
        trace_candles = go.Candlestick(
            x=df['timestamp'], open=df['open'], high=df['high'],
            low=df['low'], close=df['close'], name='Candles'
        )
        trace_ema = go.Scatter(
            x=df['timestamp'], y=df['ema'], line=dict(color='blue'), name='EMA'
        )
        trace_upper = go.Scatter(
            x=df['timestamp'], y=df['upper'], line=dict(color='green', dash='dash'), name='Upper Band'
        )
        trace_lower = go.Scatter(
            x=df['timestamp'], y=df['lower'], line=dict(color='red', dash='dash'), name='Lower Band'
        )
        # EMA Ribbon
        ribbon_traces = []
        for n, color in zip([8, 21, 34], ['orange', 'purple', 'teal']):
            ribbon_traces.append(go.Scatter(
                x=df['timestamp'], y=df[f'ema{n}'], line=dict(color=color, dash='dot'), name=f'EMA{n}'
            ))

        trace_volume = go.Bar(
            x=df['timestamp'], y=df['volume'],
            name='Volume', marker_color='lightgray',
            yaxis='y2', opacity=0.5
        )

        layout = go.Layout(
            title=f"{selected_symbol} Money Noodle",
            xaxis=dict(title="Date"),
            yaxis=dict(title="Price"),
            yaxis2=dict(
                title="Volume", overlaying='y', side='right', showgrid=False,
                range=[0, df['volume'].max() * 2]  # auto-scale
            ),
            height=700
        )

        data = [trace_candles, trace_ema, trace_upper, trace_lower] + ribbon_traces + [trace_volume]
        fig = go.Figure(data=data, layout=layout)
        chart_div = pyo.plot(fig, output_type='div', include_plotlyjs=True)
    except Exception as e:
        error = f"Could not fetch or plot data for {selected_symbol} from {selected_source}. Error: {e}"
        # Provide fallback values
        signal = None
        reasons = None
        stop_loss = None
        take_profit = None
        current_price = None
        sl_pct = None
        tp_pct = None
        return render_template(
            'index.html',
            chart_div=chart_div,
            error=error,
            exchanges=data_sources,
            selected_exchange=selected_source,
            symbols=symbols,
            selected_symbol=selected_symbol,
            band_type=band_type,
            timeframes=timeframes,
            selected_timeframe=selected_timeframe,
            signal=signal,
            reasons=reasons,
            stop_loss=stop_loss,
            take_profit=take_profit,
            current_price=current_price,
            sl_pct=sl_pct,
            tp_pct=tp_pct
        )


    return render_template(
        'index.html',
        chart_div=chart_div,
        error=error,
        exchanges=data_sources,
        selected_exchange=selected_source,
        symbols=symbols,
        selected_symbol=selected_symbol,
        band_type=band_type,
        timeframes=timeframes,
        selected_timeframe=selected_timeframe,
        signal=signal,
        reasons=reasons,
        stop_loss=stop_loss,
        take_profit=take_profit,
        current_price=current_price,
        sl_pct=sl_pct,
        tp_pct=tp_pct
    )


if __name__ == '__main__':
    app.run(debug=True)