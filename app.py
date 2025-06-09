from flask import Flask, render_template, request
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.offline as pyo

app = Flask(__name__)

EXCHANGES = {
    'Kraken': ccxt.kraken(),
    'Coinbase': ccxt.coinbase()
}

ALLOWED_TIMEFRAMES = ['1h', '4h', '1d', '1w']
DEFAULT_LIMITS = {'1h': 200, '4h': 150, '1d': 100, '1w': 52}

def get_signal(df):
    if len(df) < 20:
        return "WAIT", ["Not enough data to compute indicators."], None, None, None, None

    df = df.copy()
    df['sma20_volume'] = df['volume'].rolling(20).mean()
    sma20_volume = df['sma20_volume'].iloc[-1]
    last = df.iloc[-1]
    prev = df.iloc[-2]

    signal = "WAIT"
    reasons = []
    stop_loss = None
    take_profit = None

    price_bounce = prev['close'] < prev['lower'] and last['close'] > last['lower']
    ema_bullish = all(df['ema8'].iloc[i] > df['ema21'].iloc[i] > df['ema34'].iloc[i] for i in range(-3, 0))
    rsi_not_overbought = last['rsi'] < 65
    volume_high = last['volume'] > 1.2 * sma20_volume

    price_reject = prev['close'] > prev['upper'] and last['close'] < last['upper']
    ema_bearish = all(df['ema8'].iloc[i] < df['ema21'].iloc[i] < df['ema34'].iloc[i] for i in range(-3, 0))
    rsi_not_oversold = last['rsi'] > 35

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

def get_symbols(exchange):
    """Return sorted list of available trading pairs for the selected exchange."""
    try:
        markets = exchange.load_markets()
        return sorted(markets.keys())
    except Exception as e:
        return []

def fetch_ohlcv(exchange, symbol, timeframe='1w', limit=100):
    bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
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
    exchanges = list(EXCHANGES.keys())
    selected_exchange = 'Kraken'
    selected_symbol = 'SOL/USD'
    band_type = 'ATR'
    selected_timeframe = '1w'
    timeframes = ALLOWED_TIMEFRAMES

    symbols = get_symbols(EXCHANGES[selected_exchange])

    if request.method == 'POST':
        selected_exchange = request.form.get('exchange', exchanges[0])
        selected_symbol = request.form.get('symbol', symbols[0])
        band_type = request.form.get('band_type', 'ATR')
        selected_timeframe = request.form.get('timeframe', '1w')
        symbols = get_symbols(EXCHANGES[selected_exchange])

    if selected_symbol not in symbols:
        error = "Selected pair not available. Please select a different one."
        return render_template(
            'index.html',
            chart_div=chart_div,
            error=error,
            exchanges=exchanges,
            selected_exchange=selected_exchange,
            symbols=symbols,
            selected_symbol=selected_symbol,
            band_type=band_type,
            timeframes=timeframes,                  # <-- ADD THIS!
            selected_timeframe=selected_timeframe   # <-- AND THIS!
        )

    try:
        limit = DEFAULT_LIMITS.get(selected_timeframe, 100)
        df = fetch_ohlcv(EXCHANGES[selected_exchange], selected_symbol, selected_timeframe, limit)
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
        error = f"Could not fetch or plot data for {selected_symbol} on {selected_exchange}. Error: {e}"
        # Provide fallback values
        signal = None
        reasons = None
        return render_template(
            'index.html',
            chart_div=chart_div,
            error=error,
            exchanges=exchanges,
            selected_exchange=selected_exchange,
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
        exchanges=exchanges,
        selected_exchange=selected_exchange,
        symbols=symbols,
        selected_symbol=selected_symbol,
        band_type=band_type,
        timeframes=timeframes,                  # <-- ADD THIS!
        selected_timeframe=selected_timeframe,   # <-- AND THIS!
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