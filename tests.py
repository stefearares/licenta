import sys
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def aggregate_by_year(df, year_col):
    """Group a dataframe by ``year_col`` and compute the mean for each year."""
    return df.groupby(year_col).mean().sort_index()


def backtest(vals, window, model_func):
    """Rolling window backtest using ``model_func`` to forecast one step."""
    actuals, preds = [], []
    for i in range(len(vals) - window):
        train = vals[i : i + window]
        test  = vals[i + window]
        pred  = model_func(train)
        actuals.append(test)
        preds.append(pred)
    return np.array(actuals), np.array(preds)


def fit_arima(train):
    """Fit a simple ARIMA(1,1,1) model and forecast one step."""
    m = ARIMA(train, order=(1,1,1),
              enforce_stationarity=False,
              enforce_invertibility=False).fit()
    return m.forecast()[0]


def fit_sarima(train):
    """Fit a SARIMA(2,1,0)+trend model and forecast one step."""
    m = SARIMAX(train, order=(2,1,0), trend='t',
                enforce_stationarity=False,
                enforce_invertibility=False).fit(disp=False)
    return m.forecast()[0]


def fit_auto(train):
    """Fit an Auto-ARIMA model and forecast one step."""
    m = auto_arima(train, seasonal=False, suppress_warnings=True)
    return m.predict(n_periods=1)[0]


def compute_metrics(actuals, preds):
    """Return MAPE and RMSE between arrays of actual and predicted values."""
    mape = np.nanmean(np.abs((actuals - preds) / actuals)) * 100
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    return round(mape,2), round(rmse,2)


def holdout_evaluate(years, vals, train_until, test_until):
    """Evaluate ARIMA variants on a hold-out period."""

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


def test_stationarity(series, name=None):
    """Run ADF and KPSS tests on ``series`` and return the results."""

    # Convert to numpy array if it's not already
    series = np.array(series)
    
    # Skip if too few points
    if len(series) < 8:
        print(f"Series '{name}' has too few points for stationarity testing")
        return {
            'series': name,
            'adf_stationary': None,
            'kpss_stationary': None,
            'adf_pvalue': None,
            'kpss_pvalue': None
        }
    
    result = {'series': name}
    
    # ADF Test (null hypothesis: series is non-stationary)
    try:
        adf_result = adfuller(series, autolag='AIC')
        result['adf_statistic'] = round(adf_result[0], 4)
        result['adf_pvalue'] = round(adf_result[1], 4)
        # If p-value < 0.05, we reject the null hypothesis (series is stationary)
        result['adf_stationary'] = result['adf_pvalue'] < 0.05
        
        # Critical values for reference
        result['adf_critical_values'] = {f'{key}%': round(val, 4) for key, val in adf_result[4].items()}
    except Exception as e:
        print(f"ADF Test failed for '{name}': {str(e)}")
        result['adf_stationary'] = None
        result['adf_pvalue'] = None
    
    # KPSS Test (null hypothesis: series is stationary)
    try:
        kpss_result = kpss(series, regression='c', nlags='auto')
        result['kpss_statistic'] = round(kpss_result[0], 4)
        result['kpss_pvalue'] = round(kpss_result[1], 4)
        # If p-value > 0.05, we fail to reject the null hypothesis (series is stationary)
        result['kpss_stationary'] = result['kpss_pvalue'] > 0.05
        
        # Critical values for reference
        result['kpss_critical_values'] = {f'{key}%': round(val, 4) for key, val in kpss_result[3].items()}
    except Exception as e:
        print(f"KPSS Test failed for '{name}': {str(e)}")
        result['kpss_stationary'] = None
        result['kpss_pvalue'] = None
    
    # Overall assessment
    if result.get('adf_stationary') and result.get('kpss_stationary'):
        result['conclusion'] = "Series is stationary (both tests agree)"
    elif not result.get('adf_stationary') and not result.get('kpss_stationary'):
        result['conclusion'] = "Series is non-stationary (both tests agree)"
    elif result.get('adf_stationary') and not result.get('kpss_stationary'):
        result['conclusion'] = "Conflicting results: ADF says stationary, KPSS says non-stationary"
    elif not result.get('adf_stationary') and result.get('kpss_stationary'):
        result['conclusion'] = "Conflicting results: ADF says non-stationary, KPSS says stationary"
    else:
        result['conclusion'] = "Test results inconclusive"
    
    return result


def analyze_stationarity(file_path, date_col=0):
    """Analyze stationarity of all numeric series in the CSV file."""
    import pandas as pd
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
    
    year_col = df.columns[date_col]
    data_cols = df.columns.drop(year_col)
    df_grouped = df.groupby(year_col)[data_cols].mean().sort_index()
    
    results = []
    for col in data_cols:
        if 'user_defined' in col:
            continue
            
        ts = df_grouped[col].astype(float)
        if ts.size < 8:
            print(f"Series '{col}' has too few points for stationarity testing, minimum 8 required")
            continue
            
        values = ts.tolist()
        result = test_stationarity(values, col)
        
        # Check if differencing improves stationarity
        if not result.get('adf_stationary') or not result.get('kpss_stationary'):
            diff_values = np.diff(values)
            diff_result = test_stationarity(diff_values, f"{col} (1st diff)")
            result['diff_conclusion'] = diff_result['conclusion']
        
        results.append(result)
    

    if results:
        results_df = pd.DataFrame(results)

        # Select and reorder columns for better presentation
        display_columns = ['series', 'adf_stationary', 'kpss_stationary', 
                          'adf_pvalue', 'kpss_pvalue', 'conclusion']

        # Add differencing column if it exists
        if 'diff_conclusion' in results_df.columns:
            display_columns.append('diff_conclusion')
            
        return results_df[display_columns]
    else:
        print("No series suitable for stationarity testing")
        return None



def main():
    """Entry point for running backtests or stationarity analysis from the
    command line."""
    p = argparse.ArgumentParser(
        description="Backtest rolling-window or hold-out evaluation for ARIMA, SARIMA, Auto-ARIMA"
    )
    p.add_argument("csv", help="path to your CSV")
    p.add_argument("--window", type=int, default=3, help="rolling window size")
    p.add_argument("--train-until", type=int,
                   help="last year to include in training for hold-out")
    p.add_argument("--test-until", type=int,
                   help="last year to forecast/test for hold-out")
    p.add_argument("--test-stationarity", action="store_true",
                   help="perform stationarity tests on time series")
    args = p.parse_args()

    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        print(f"Error! CSV: {e}")
        sys.exit(1)
        

    if args.test_stationarity:
        print("\n--- Stationarity Test Results ---")
        stationarity_results = analyze_stationarity(args.csv)
        if stationarity_results is not None:
            print(stationarity_results)
            print("\nNote about stationarity tests:")
            print("ADF Test: H0 = non-stationary, p<0.05 means stationary")
            print("KPSS Test: H0 = stationary, p>0.05 means stationary")
            print("These results can help inform appropriate differencing order for ARIMA models.")
        sys.exit(0)

    year_col = df.columns[0]
    out = []


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