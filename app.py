from flask import Flask, render_template, request
import ccxt
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.offline as pyo
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

app = Flask(__name__)

DATA_SOURCES = {
    'Kraken': {'type': 'crypto', 'client': ccxt.kraken()},
    'Coinbase': {'type': 'crypto', 'client': ccxt.coinbase()},
    'Yahoo Finance': {'type': 'stock', 'client': None}
}

ALLOWED_TIMEFRAMES = ['1h', '4h', '1d', '1w']
DEFAULT_LIMITS = {'1h': 200, '4h': 150, '1d': 100, '1w': 52}

def calculate_awesome_oscillator(df, short_period=5, long_period=34):
    """Calculate Awesome Oscillator"""
    hl2 = (df['high'] + df['low']) / 2
    ao_short = hl2.rolling(window=short_period).mean()
    ao_long = hl2.rolling(window=long_period).mean()
    ao = ao_short - ao_long
    return ao

def calculate_dmi_adx(df, period=14):
    """Calculate DMI and ADX"""
    df = df.copy()
    
    # Calculate True Range
    df['tr'] = np.maximum(df['high'] - df['low'],
                         np.maximum(abs(df['high'] - df['close'].shift(1)),
                                   abs(df['low'] - df['close'].shift(1))))
    
    # Calculate directional movements
    df['plus_dm'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                            np.maximum(df['high'] - df['high'].shift(1), 0), 0)
    df['minus_dm'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                             np.maximum(df['low'].shift(1) - df['low'], 0), 0)
    
    # Smooth the values using Wilder's smoothing
    df['atr'] = df['tr'].ewm(alpha=1/period, adjust=False).mean()
    df['plus_di_raw'] = df['plus_dm'].ewm(alpha=1/period, adjust=False).mean()
    df['minus_di_raw'] = df['minus_dm'].ewm(alpha=1/period, adjust=False).mean()
    
    # Calculate DI values
    df['plus_di'] = 100 * df['plus_di_raw'] / df['atr']
    df['minus_di'] = 100 * df['minus_di_raw'] / df['atr']
    
    # Calculate DX and ADX
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = df['dx'].ewm(alpha=1/period, adjust=False).mean()
    
    return df['plus_di'], df['minus_di'], df['adx']

def calculate_ttm_scalper(df, fast_period=8, slow_period=21):
    """Calculate TTM Scalper momentum"""
    fast_ema = df['close'].ewm(span=fast_period, adjust=False).mean()
    slow_ema = df['close'].ewm(span=slow_period, adjust=False).mean()
    momentum = fast_ema - slow_ema
    return momentum

def enhanced_money_noodle(df, atr_length=14, atr_multiplier=2.0, rsi_length=14, 
                         volume_length=20, swing_lookback=15, dmi_length=14):
    """Enhanced Money Noodle with 10-level signal system from ThinkOrSwim"""
    df = df.copy()
    
    # Basic price calculations
    df['hl2'] = (df['high'] + df['low']) / 2
    
    # EMA calculations
    df['ema8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema34'] = df['close'].ewm(span=34, adjust=False).mean()
    
    # RSI calculation
    df['rsi'] = RSIIndicator(df['close'], window=rsi_length).rsi()
    
    # Awesome Oscillator
    df['ao'] = calculate_awesome_oscillator(df)
    df['ao_prev'] = df['ao'].shift(1)
    df['ao_prev2'] = df['ao'].shift(2)
    
    # DMI and ADX
    df['plus_di'], df['minus_di'], df['adx'] = calculate_dmi_adx(df, dmi_length)
    
    # TTM Scalper
    df['ttm_momentum'] = calculate_ttm_scalper(df)
    df['ttm_prev'] = df['ttm_momentum'].shift(1)
    
    # ATR for dynamic bands
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=atr_length).average_true_range()
    
    # Dynamic price bands with RSI adjustment
    df['rsi_adjustment'] = np.where((df['rsi'] > 70) | (df['rsi'] < 30), 1.5, 1.0)
    df['adjusted_band_width'] = df['atr'] * atr_multiplier * df['rsi_adjustment']
    df['upper'] = df['ema21'] + df['adjusted_band_width']
    df['lower'] = df['ema21'] - df['adjusted_band_width']
    
    # Volume analysis
    df['volume_sma'] = df['volume'].rolling(window=volume_length).mean()
    
    # Calculate signal conditions
    # EMA trend conditions
    df['ema_bullish'] = (df['ema8'] > df['ema21']) & (df['ema21'] > df['ema34'])
    df['ema_bearish'] = (df['ema8'] < df['ema21']) & (df['ema21'] < df['ema34'])
    
    # Price band conditions
    df['near_lower_band'] = abs(df['close'] - df['lower']) / df['lower'] <= 0.01
    df['bounce_above_lower'] = (df['close'].shift(1) < df['lower'].shift(1)) & (df['close'] > df['lower'])
    df['price_bullish'] = df['near_lower_band'] | df['bounce_above_lower']
    
    df['near_upper_band'] = abs(df['close'] - df['upper']) / df['upper'] <= 0.01
    df['rejection_from_upper'] = (df['close'].shift(1) > df['upper'].shift(1)) & (df['close'] < df['upper'])
    df['price_bearish'] = df['near_upper_band'] | df['rejection_from_upper']
    
    # RSI conditions
    df['rsi_oversold'] = df['rsi'] < 30
    df['rsi_bullish'] = (df['rsi'] >= 30) & (df['rsi'] < 50)
    df['rsi_neutral'] = (df['rsi'] >= 50) & (df['rsi'] <= 70)
    df['rsi_bearish'] = (df['rsi'] > 70) & (df['rsi'] <= 80)
    df['rsi_overbought'] = df['rsi'] > 80
    
    # Awesome Oscillator signals
    df['ao_bullish'] = (df['ao'] > 0) & (df['ao'] > df['ao_prev'])
    df['ao_bearish'] = (df['ao'] < 0) & (df['ao'] < df['ao_prev'])
    
    # DMI signals
    df['dmi_bullish'] = (df['plus_di'] > df['minus_di']) & (df['adx'] > 25)
    df['dmi_bearish'] = (df['minus_di'] > df['plus_di']) & (df['adx'] > 25)
    df['dmi_weak'] = df['adx'] <= 25
    
    # TTM Scalper signals
    df['ttm_bullish'] = (df['ttm_momentum'] > 0) & (df['ttm_momentum'] > df['ttm_prev'])
    df['ttm_bearish'] = (df['ttm_momentum'] < 0) & (df['ttm_momentum'] < df['ttm_prev'])
    
    # Volume conditions
    df['volume_high'] = df['volume'] > 1.5 * df['volume_sma']
    df['volume_normal'] = (df['volume'] > 1.0 * df['volume_sma']) & (df['volume'] <= 1.5 * df['volume_sma'])
    df['volume_low'] = df['volume'] <= 1.0 * df['volume_sma']
    
    # Calculate scores for each component
    df['ema_score'] = np.where(df['ema_bullish'], 2, np.where(df['ema_bearish'], -2, 0))
    
    df['rsi_score'] = np.where(df['rsi_oversold'], 2,
                      np.where(df['rsi_bullish'], 1,
                      np.where(df['rsi_neutral'], 0,
                      np.where(df['rsi_bearish'], -1, -2))))
    
    df['ao_score'] = np.where(df['ao_bullish'], 2, np.where(df['ao_bearish'], -2, 0))
    
    df['dmi_score'] = np.where(df['dmi_bullish'], 2, np.where(df['dmi_bearish'], -2, 0))
    
    df['ttm_score'] = np.where(df['ttm_bullish'], 1, np.where(df['ttm_bearish'], -1, 0))
    
    df['price_score'] = np.where(df['price_bullish'], 1, np.where(df['price_bearish'], -1, 0))
    
    df['volume_score'] = np.where(df['volume_high'], 1, np.where(df['volume_normal'], 0, -1))
    
    # Total score calculation
    df['total_score'] = (df['ema_score'] + df['rsi_score'] + df['ao_score'] + 
                        df['dmi_score'] + df['ttm_score'] + df['price_score'] + df['volume_score'])
    
    # Convert to 10-level signal (scale -12 to +11 into 1 to 10)
    df['signal_10_level'] = np.round((df['total_score'] + 12) * 10 / 23, 0)
    df['signal_10_level'] = np.clip(df['signal_10_level'], 1, 10)
    
    # Signal categories
    df['strong_sell'] = df['signal_10_level'] <= 2
    df['sell'] = (df['signal_10_level'] >= 3) & (df['signal_10_level'] <= 4)
    df['hold'] = (df['signal_10_level'] >= 5) & (df['signal_10_level'] <= 6)
    df['buy'] = (df['signal_10_level'] >= 7) & (df['signal_10_level'] <= 8)
    df['strong_buy'] = df['signal_10_level'] >= 9
    
    # Calculate swing highs and lows for risk management
    df['swing_low'] = df['low'].rolling(window=swing_lookback).min()
    df['swing_high'] = df['high'].rolling(window=swing_lookback).max()
    
    return df

def get_enhanced_signal(df):
    """Get enhanced signal based on 10-level system"""
    if len(df) < 34:
        return "WAIT", ["Insufficient data for indicators."], None, None, None, None, None, 1, {}
    
    last = df.iloc[-1]
    
    # Determine signal based on 10-level system
    if last['strong_buy']:
        signal = "STRONG BUY"
        signal_level = int(last['signal_10_level'])
    elif last['buy']:
        signal = "BUY"
        signal_level = int(last['signal_10_level'])
    elif last['hold']:
        signal = "HOLD"
        signal_level = int(last['signal_10_level'])
    elif last['sell']:
        signal = "SELL"
        signal_level = int(last['signal_10_level'])
    elif last['strong_sell']:
        signal = "STRONG SELL"
        signal_level = int(last['signal_10_level'])
    else:
        signal = "WAIT"
        signal_level = int(last['signal_10_level'])
    
    # Generate reasons based on component scores
    reasons = []
    reasons.append(f"Signal Level: {signal_level}/10 (Total Score: {last['total_score']:.0f})")
    
    # Add component analysis
    if last['ema_bullish']:
        reasons.append("✓ EMA trend is bullish (EMA8 > EMA21 > EMA34)")
    elif last['ema_bearish']:
        reasons.append("✗ EMA trend is bearish (EMA8 < EMA21 < EMA34)")
    else:
        reasons.append("~ EMA trend is mixed")
    
    if last['rsi'] < 30:
        reasons.append("✓ RSI is oversold - potential bounce")
    elif last['rsi'] > 70:
        reasons.append("✗ RSI is overbought - potential pullback")
    
    if last['ao_bullish']:
        reasons.append("✓ Awesome Oscillator is bullish")
    elif last['ao_bearish']:
        reasons.append("✗ Awesome Oscillator is bearish")
    
    if last['dmi_bullish']:
        reasons.append("✓ DMI shows strong bullish trend")
    elif last['dmi_bearish']:
        reasons.append("✗ DMI shows strong bearish trend")
    
    if last['volume_high']:
        reasons.append("✓ Volume is above average")
    elif last['volume_low']:
        reasons.append("✗ Volume is below average")
    
    # Calculate stop loss and take profit
    current_price = last['close']
    
    if signal in ["BUY", "STRONG BUY"]:
        stop_loss = last['swing_low']
        risk = current_price - stop_loss
        take_profit = current_price + 2 * risk
    elif signal in ["SELL", "STRONG SELL"]:
        stop_loss = last['swing_high']
        risk = stop_loss - current_price
        take_profit = current_price - 2 * risk
    else:
        stop_loss = last['swing_low']
        take_profit = last['upper']
    
    # Calculate percentages
    sl_pct = 100.0 * abs(current_price - stop_loss) / current_price if stop_loss else None
    tp_pct = 100.0 * abs(current_price - take_profit) / current_price if take_profit else None
    
    # Additional indicator values for display
    indicators = {
        'rsi': round(last['rsi'], 1),
        'ao': round(last['ao'], 4),
        'adx': round(last['adx'], 1),
        'plus_di': round(last['plus_di'], 1),
        'minus_di': round(last['minus_di'], 1),
        'total_score': int(last['total_score']),
        'ema8': round(last['ema8'], 4),
        'ema21': round(last['ema21'], 4),
        'ema34': round(last['ema34'], 4),
        'ema_score': int(last['ema_score']),
        'rsi_score': int(last['rsi_score']),
        'ao_score': int(last['ao_score']),
        'dmi_score': int(last['dmi_score']),
        'ttm_score': int(last['ttm_score']),
        'price_score': int(last['price_score']),
        'volume_score': int(last['volume_score']),
    }
    
    return signal, reasons, stop_loss, take_profit, current_price, sl_pct, tp_pct, signal_level, indicators

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
        yf_timeframe_map = {
            '1h': '60m',
            '4h': '4h',
            '1d': '1d',
            '1w': '1wk'
        }
        yf_period_map = {
            '1h': '7d',
            '4h': '60d',
            '1d': '2y',
            '1w': '5y'
        }
        
        interval = yf_timeframe_map.get(timeframe, '1d')
        period = yf_period_map.get(timeframe, '2y')

        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        if not data.empty:
            df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df['timestamp'] = df.index
            df = df.reset_index(drop=True)
        else:
            raise ValueError(f"No data fetched for {symbol} from Yahoo Finance.")

    return df

@app.route('/', methods=['GET', 'POST'])
def index():
    chart_div = ""
    error = None
    data_sources = list(DATA_SOURCES.keys())
    selected_source = 'Kraken'
    selected_symbol = 'SOL/USD'
    selected_timeframe = '1w'
    timeframes = ALLOWED_TIMEFRAMES

    symbols = get_symbols(selected_source)

    if request.method == 'POST':
        selected_source = request.form.get('exchange', data_sources[0])
        selected_symbol_input = request.form.get('symbol_input', '').strip().upper()
        selected_symbol_dropdown = request.form.get('symbol_dropdown', '')
        
        if selected_symbol_input:
            selected_symbol = selected_symbol_input
        else:
            selected_symbol = selected_symbol_dropdown

        selected_timeframe = request.form.get('timeframe', '1w')
        symbols = get_symbols(selected_source)
    
    try:
        limit = DEFAULT_LIMITS.get(selected_timeframe, 100)
        df = fetch_ohlcv(selected_source, selected_symbol, selected_timeframe, limit)
        df = enhanced_money_noodle(df)
        signal, reasons, stop_loss, take_profit, current_price, sl_pct, tp_pct, signal_level, indicators = get_enhanced_signal(df)
        
        # Create plotly chart with enhanced indicators
        trace_candles = go.Candlestick(
            x=df['timestamp'], open=df['open'], high=df['high'],
            low=df['low'], close=df['close'], name='Candles'
        )
        
        # EMA lines
        trace_ema8 = go.Scatter(
            x=df['timestamp'], y=df['ema8'], line=dict(color='orange', width=1), name='EMA8'
        )
        trace_ema21 = go.Scatter(
            x=df['timestamp'], y=df['ema21'], line=dict(color='blue', width=2), name='EMA21'
        )
        trace_ema34 = go.Scatter(
            x=df['timestamp'], y=df['ema34'], line=dict(color='purple', width=1), name='EMA34'
        )
        
        # Dynamic bands
        trace_upper = go.Scatter(
            x=df['timestamp'], y=df['upper'], line=dict(color='green', dash='dash'), name='Upper Band'
        )
        trace_lower = go.Scatter(
            x=df['timestamp'], y=df['lower'], line=dict(color='red', dash='dash'), name='Lower Band'
        )
        
        # Signal markers
        buy_signals = df[df['buy'] | df['strong_buy']]
        sell_signals = df[df['sell'] | df['strong_sell']]
        
        trace_buy = go.Scatter(
            x=buy_signals['timestamp'], 
            y=buy_signals['low'] - buy_signals['atr'] * 0.5,
            mode='markers',
            marker=dict(symbol='triangle-up', size=10, color='green'),
            name='Buy Signals'
        )
        
        trace_sell = go.Scatter(
            x=sell_signals['timestamp'], 
            y=sell_signals['high'] + sell_signals['atr'] * 0.5,
            mode='markers',
            marker=dict(symbol='triangle-down', size=10, color='red'),
            name='Sell Signals'
        )
        
        trace_volume = go.Bar(
            x=df['timestamp'], y=df['volume'],
            name='Volume', marker_color='lightgray',
            yaxis='y2', opacity=0.5
        )

        layout = go.Layout(
            title=f"{selected_symbol} Enhanced Money Noodle - Signal Level: {signal_level}/10",
            xaxis=dict(title="Date"),
            yaxis=dict(title="Price"),
            yaxis2=dict(
                title="Volume", overlaying='y', side='right', showgrid=False,
                range=[0, df['volume'].max() * 2]
            ),
            height=700
        )

        data = [trace_candles, trace_ema8, trace_ema21, trace_ema34, trace_upper, trace_lower, 
                trace_buy, trace_sell, trace_volume]
        fig = go.Figure(data=data, layout=layout)
        chart_div = pyo.plot(fig, output_type='div', include_plotlyjs=True)
        
    except Exception as e:
        error = f"Could not fetch or plot data for {selected_symbol} from {selected_source}. Error: {e}"
        signal = None
        reasons = None
        stop_loss = None
        take_profit = None
        current_price = None
        sl_pct = None
        tp_pct = None
        signal_level = 1
        indicators = {}

    return render_template(
        'index.html',
        chart_div=chart_div,
        error=error,
        exchanges=data_sources,
        selected_exchange=selected_source,
        symbols=symbols,
        selected_symbol=selected_symbol,
        timeframes=timeframes,
        selected_timeframe=selected_timeframe,
        signal=signal,
        reasons=reasons,
        stop_loss=stop_loss,
        take_profit=take_profit,
        current_price=current_price,
        sl_pct=sl_pct,
        tp_pct=tp_pct,
        signal_level=signal_level,
        indicators=indicators
    )

if __name__ == '__main__':
    app.run(debug=True)