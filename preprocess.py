import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def prepare_labels(data, threshold=0.015):
    df = data.copy()
    df['Signal'] = 1  # hold
    df.loc[df['Return'] > threshold, 'Signal'] = 2  # buy
    df.loc[df['Return'] < -threshold, 'Signal'] = 0  # sell
    df['Signal'] = df['Signal'].shift(-1)
    return df.dropna()

def create_features(df):
    df['Ret_1'] = df['Return']
    df['Ret_3'] = df['Return'].rolling(3).mean()
    df['Momentum'] = df['Return'].rolling(3).sum()
    df['Volatility'] = df['Return'].rolling(5).std()
    df['EMA_Momentum'] = df['Return'].ewm(span=5).mean()

    df = df.dropna()

    features = ['Ret_1', 'Ret_3', 'Momentum', 'Volatility', 'EMA_Momentum']
    X = df[features].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    Y = df['Signal'].astype(int).values
    return X, Y