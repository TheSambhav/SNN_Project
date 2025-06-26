# 📈 Trading Using Spiking Neural Networks (SNN)

This repository presents a novel application of **Spiking Neural Networks (SNNs)** to financial trading. Unlike traditional artificial neural networks, SNNs emulate the dynamic behavior of biological neurons, making them a promising frontier for modeling temporal dependencies in financial time series. This project builds an end-to-end pipeline—from data acquisition to backtesting—to predict **buy**, **sell**, and **hold** trading signals for financial instruments.

---

## 🧠 Project Highlights

- **Neural Model**: Custom SNN architecture built using the `snntorch` library, incorporating leaky integrate-and-fire (LIF) neurons.
- **Features Engineered**: Returns, rolling means, volatility, momentum, and exponential moving averages derived from historical prices.
- **Labels Generated**: Multi-class trading signals based on future return thresholds.
- **Backtesting Framework**: Simulates trading based on predicted signals and plots equity curve to assess strategy performance.
- **Asset**: Default dataset uses SPY ETF from Yahoo Finance (2017–2022), configurable to other assets.

---

## 📁 Repository Structure

| File | Description |
|------|-------------|
| `data_loader.py` | Downloads historical stock/ETF price data using `yfinance`. Calculates daily returns. |
| `preprocess.py` | Generates features and labels for model input. Applies standard scaling. |
| `model.py` | Defines the SNN model with fully connected layers and LIF neurons using `snntorch`. |
| `train.py` | Trains the SNN on labeled financial data with cross-entropy loss. |
| `test_and_backtest.py` | Makes predictions and runs a rule-based backtest with equity curve plotting. |
| `SNN_Trading_Workflow.ipynb` | A Jupyter notebook that ties all steps together for interactive exploration. |

---

## 🔧 Installation

```bash
pip install torch snntorch yfinance pandas numpy scikit-learn matplotlib
```

---

## 🚀 Getting Started

You can follow the full pipeline either through the Jupyter notebook or manually using the scripts.

### Step-by-Step (Python Scripts):

```python
# 1. Load Data
from data_loader import load_price_data
data = load_price_data("SPY", start="2017-01-01", end="2022-01-01")

# 2. Generate Features and Labels
from preprocess import prepare_labels, create_features
labeled_data = prepare_labels(data)
X, Y = create_features(labeled_data)

# 3. Train SNN Model
from train import train_model
model = train_model(X, Y, num_epochs=20)

# 4. Predict and Backtest
from test_and_backtest import predict, backtest, plot_equity
signals = predict(model, X)
prices = data['Close'].iloc[-len(signals):].values
equity_curve = backtest(prices, signals)
plot_equity(equity_curve)
```

---

## 📊 Output Explanation

- **Trading Signals**:
  - `0`: Sell
  - `1`: Hold
  - `2`: Buy

- **Backtest**:
  - Simulates a trading strategy with a starting equity of $100,000.
  - Applies long/short/hold positions based on signals.
  - Displays an equity curve showing performance over time.

---

## 🧪 Methodology

- **Labeling Strategy**: Labels are based on whether the return exceeds a threshold (default: ±1.5%). This encourages early detection of directional moves.
- **Neural Dynamics**: The SNN is run over multiple time steps (default: 20) to accumulate spikes for final classification.
- **Feature Engineering**: The feature set captures short-term momentum, volatility, and smoothing, which are useful in financial forecasting.

---

## 📌 Limitations & Future Work

- ⚠️ No transaction costs, slippage, or position sizing logic are included.
- 🔄 Can be extended to intraday data or multiple assets.
- 🧠 Could be improved using unsupervised SNN pretraining or reinforcement learning.

---

## 📬 Contributions

Pull requests and suggestions for extending the architecture or backtesting framework are welcome!