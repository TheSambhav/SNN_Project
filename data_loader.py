import yfinance as yf
import pandas as pd

def load_price_data(ticker="SPY", start="2017-01-01", end="2022-01-01"):
    # Download with adjusted close
    data = yf.download(ticker, start=start, end=end, auto_adjust=True)
    print("Downloaded columns:", data.columns)

    # Drop multi-index if present (e.g., ('Close', 'SPY') → 'Close')
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel('Ticker')

    # Compute returns from adjusted Close
    ret = data['Close'].pct_change().dropna()

    # Keep aligned data only for valid return rows
    df = data.loc[ret.index].copy()
    df['Return'] = ret

    return df
