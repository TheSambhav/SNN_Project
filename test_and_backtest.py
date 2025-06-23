import torch
import numpy as np
import matplotlib.pyplot as plt

def predict(model, X_test, time_steps=20):
    model.eval()
    with torch.no_grad():
        X_test = torch.tensor(X_test, dtype=torch.float32)
        output = model(X_test, time_steps=time_steps)
        preds = torch.argmax(output, dim=1).numpy()
    return preds

def backtest(prices, signals):
    equity = 100000
    equity_curve = [equity]
    for i in range(len(signals) - 1):
        position = 1 if signals[i] == 2 else -1 if signals[i] == 0 else 0
        profit = position * (prices[i+1] - prices[i])
        equity += profit
        equity_curve.append(equity)
    return equity_curve

def plot_equity(equity_curve):
    plt.plot(equity_curve)
    plt.title("Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.show()