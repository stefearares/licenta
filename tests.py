import sys
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def aggregate_by_year(df, year_col):
    return df.groupby(year_col).mean().sort_index()


def backtest(vals, window, model_func):
    actuals, preds = [], []
    for i in range(len(vals) - window):
        train = vals[i : i + window]
        test  = vals[i + window]
        pred  = model_func(train)
        actuals.append(test)
        preds.append(pred)
    return np.array(actuals), np.array(preds)


def fit_arima(train):
    m = ARIMA(train, order=(1,1,1),
              enforce_stationarity=False,
              enforce_invertibility=False).fit()
    return m.forecast()[0]


def fit_sarima(train):
    m = SARIMAX(train, order=(2,1,0), trend='t',
                enforce_stationarity=False,
                enforce_invertibility=False).fit(disp=False)
    return m.forecast()[0]


def fit_auto(train):
    m = auto_arima(train, seasonal=False, suppress_warnings=True)
    return m.predict(n_periods=1)[0]


def compute_metrics(actuals, preds):
    mape = np.nanmean(np.abs((actuals - preds) / actuals)) * 100
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    return round(mape,2), round(rmse,2)


def holdout_evaluate(years, vals, train_until, test_until):

    idx = {y: i for i, y in enumerate(years)}
    train_years = [y for y in years if y <= train_until]
    test_years  = [y for y in years if train_until < y <= test_until]
    if len(train_years) < 3 or not test_years:
        return None
    train_vals = [vals[idx[y]] for y in train_years]
    # forecast steps
    steps = len(test_years)
    res = {}
    # ARIMA
    try:
        m1 = ARIMA(train_vals, order=(1,1,1),
                   enforce_stationarity=False,
                   enforce_invertibility=False).fit()
        fc1 = m1.forecast(steps)
    except:
        fc1 = [np.nan]*steps
    # SARIMA
    try:
        m2 = SARIMAX(train_vals, order=(2,1,0), trend='t',
                     enforce_stationarity=False,
                     enforce_invertibility=False).fit(disp=False)
        fc2 = m2.forecast(steps)
    except:
        fc2 = [np.nan]*steps
    # Auto-ARIMA
    try:
        m3 = auto_arima(train_vals, seasonal=False, suppress_warnings=True)
        fc3 = m3.predict(n_periods=steps)
    except:
        fc3 = [np.nan]*steps
    y_true = np.array([vals[idx[y]] for y in test_years])
    def met(fc):
        return compute_metrics(y_true, np.array(fc))
    res['years'] = test_years
    res['ARIMA'] = met(fc1)
    res['SARIMA'] = met(fc2)
    res['Auto-ARIMA'] = met(fc3)
    return res


def main():
    p = argparse.ArgumentParser(
        description="Backtest rolling-window or hold-out evaluation for ARIMA, SARIMA, Auto-ARIMA"
    )
    p.add_argument("csv", help="path to your CSV")
    p.add_argument("--window",   type=int, default=3, help="rolling window size")
    p.add_argument("--train-until", type=int,
                   help="last year to include in training for hold-out")
    p.add_argument("--test-until",  type=int,
                   help="last year to forecast/test for hold-out")
    args = p.parse_args()

    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        print(f"Error! CSV: {e}")
        sys.exit(1)

    year_col = df.columns[0]
    out = []

    # decide mode
    holdout_mode = args.train_until is not None and args.test_until is not None

    for col in df.columns.drop(year_col):
        if 'user_defined' in col:
            continue
        series_df = df[[year_col, col]].dropna()
        agg = aggregate_by_year(series_df, year_col)
        years = agg.index.tolist()
        vals  = agg[col].values.tolist()
        if holdout_mode:
            res = holdout_evaluate(years, vals, args.train_until, args.test_until)
            if not res:
                continue
            for model in ('ARIMA','SARIMA','Auto-ARIMA'):
                mape, rmse = res[model]
                out.append({
                    'series': col,
                    'model': model,
                    'years': f"{args.train_until+1}-{args.test_until}",
                    'MAPE%': mape,
                    'RMSE': rmse
                })
        else:
            if len(vals) < args.window + 1:
                continue
            for name, fn in [
                ("ARIMA(1,1,1)", fit_arima),
                ("SARIMA(2,1,0)+t", fit_sarima),
                ("Auto-ARIMA", fit_auto)
            ]:
                actuals, preds = backtest(vals, args.window, fn)
                mape, rmse = compute_metrics(actuals, preds)
                if name.startswith("ARIMA("):
                    aic = ARIMA(vals, order=(1,1,1), enforce_stationarity=False,
                                 enforce_invertibility=False).fit().aic
                elif name.startswith("SARIMA("):
                    aic = SARIMAX(vals, order=(2,1,0), trend='t', enforce_stationarity=False,
                                  enforce_invertibility=False).fit(disp=False).aic
                else:
                    aic = auto_arima(vals, seasonal=False, suppress_warnings=True).aic()
                out.append({
                    'series': col,
                    'model': name,
                    'AIC': round(aic,2),
                    'MAPE%': mape,
                    'RMSE': rmse
                })

    result_df = pd.DataFrame(out)
    if result_df.empty:
        print("No series to evaluate.")
        sys.exit(1)

    if holdout_mode:
        pivot = result_df.pivot_table(
            index=['series','years'], columns='model', values=['MAPE%','RMSE']
        )
    else:
        pivot = result_df.pivot_table(
            index='series', columns='model', values=['AIC','MAPE%','RMSE']
        )

    print(pivot)

if __name__ == "__main__":
    main()
