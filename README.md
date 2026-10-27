# Next-Day Gold Data Predictions

A Python-based forecasting project for predicting the next-day XAU/USD (gold) price using a suite of machine learning and time-series models, ensemble techniques, and a Streamlit dashboard for interactive exploration.

## Overview

This repository combines historical gold market data with modern forecasting approaches to estimate the next trading day's Open, High, Low, and Close prices. It includes:

- historical data ingestion and preprocessing
- multiple predictive models
- hold-out backtesting and validation
- ensemble weighting and model comparison
- market regime analysis
- an interactive dashboard for forecasts and diagnostics

The project is designed for research, experimentation, and practical forecasting workflows in the commodity markets.

## Why this project?

Gold is a highly liquid and volatile asset, and forecasting its next-day movement is a classic time-series problem. This project explores whether a blend of classical machine learning, deep learning, and ensemble strategies can provide useful directional and price-level predictions.

While the outputs are educational and not financial advice, the workflow is built to evaluate model quality using rigorous backtesting and performance metrics.

## Features

- Multi-model forecasting pipeline for XAU/USD
- Backtest evaluation on unseen hold-out periods
- Ensemble model selection and weighting
- Market regime detection using statistical patterns
- Model leaderboard based on RMSE and MAE
- Up/Down classification probabilities for next-day movement
- Streamlit dashboard for exploring predictions, accuracy, and market indicators

## Repository structure

```text
.
├── app.py                          # Streamlit dashboard for forecast visualization
├── train_predict.py               # Data preparation, model training, backtesting, and export
├── xau_usd_using_genetic_algorithm.py
├── requirements.txt                # Python dependencies
├── data/                          # Processed or supporting datasets
├── outputs/                       # Generated forecasts, metrics, and backtest artifacts
├── XAU_USD Historical Data 1980_1994.csv
├── XAU_USD Historical Data 1995-2013.csv
├── XAU_USD Historical Data  2014-2026.csv
├── __pycache__/                   # Python cache files
├── catboost_info/                 # CatBoost training metadata
├── .claude/                       # Local tooling metadata
├── README.md                      # Project documentation
└── .gitignore                     # Repository ignore rules
```

## Data

The repository uses historical XAU/USD datasets spanning multiple periods:

- 1980–1994
- 1995–2013
- 2014–2026

These files provide the raw historical price data used to train and evaluate the forecasting models.

## Tech stack

- Python
- pandas
- NumPy
- scikit-learn
- PyTorch
- LightGBM
- Streamlit
- Plotly
- SciPy
- Chronos Forecasting
- EMD signal analysis tools

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Ferdeen-Bakht/Next-day-gold-data-predictions.git
cd Next-day-gold-data-predictions
```

2. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Train and generate forecasts

```bash
python train_predict.py
```

This performs the training workflow, builds predictive models, evaluates backtest performance, and saves outputs into the `outputs/` folder.

### Launch the dashboard

```bash
streamlit run app.py
```

The dashboard visualizes:

- forecasted next-day OHLC values
- model comparison metrics
- backtest accuracy
- market statistics
- ensemble behavior and probability estimates

## Outputs

The project writes generated artifacts to `outputs/`, including:

- forecast results
- backtest predictions
- OHLC historical comparisons
- evaluation metrics
- summary statistics for model performance

## Model approach

The workflow is built around a comparative modeling strategy, where several models are evaluated on the same task and then ranked using hold-out performance.

The project includes components related to:

- classical statistical feature engineering
- learned temporal patterns from market data
- ensemble aggregation
- evaluation against naive forecasting baselines
- probability-based Up/Down classification

## Performance focus

The project emphasizes:

- minimizing RMSE and MAE for next-day price prediction
- measuring directional accuracy
- comparing models fairly on hold-out data
- explaining what drove the forecast through summary metrics and charts

## Important note

This project is intended for educational and research use. Forecasting financial markets is inherently uncertain, and no model can guarantee future price movements. The outputs should be interpreted as model-based estimates rather than trading advice.

## Contributing

Contributions are welcome. If you would like to improve the modeling pipeline, add new evaluation metrics, refine the dashboard, or improve the documentation, feel free to open an issue or submit a pull request.

## License

This repository does not currently declare a license. If you intend to distribute or reuse the code publicly, consider adding an explicit license file such as MIT or Apache 2.0.

## Contact

For questions or collaboration, reach out via the GitHub repository or the project owner.

---

Built for next-day XAU/USD forecasting with a focus on model evaluation, visualization, and experimentation.
