import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def arima_for_all_columns(
    file_path: str,
    date_col: int = 0,
    order: tuple = (1, 1, 1),
    forecast_steps: int = 5
):
    """Fit an ARIMA model for every column in the CSV and forecast."""
    df = pd.read_csv(file_path)

    year_col = df.columns[date_col]
    data_cols = df.columns.drop(year_col)
    df_grouped = df.groupby(year_col)[data_cols].mean().sort_index()

    results = {}
    for col in data_cols:
        ts = df_grouped[col].astype(float)
        if ts.size < 3:
            continue
        values = ts.tolist()
        years = ts.index.tolist()

        try:
            model = ARIMA(
                values,
                order=order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            res = model.fit()
            forecast = res.get_forecast(steps=forecast_steps)
        except Exception as e:
            print(f"Skipping '{col}' due to model error: {e}")
            continue

        original = list(zip(years, values))
        fc_vals = forecast.predicted_mean.tolist()
        last_year = years[-1]
        forecast_list = [(last_year + i, fc_vals[i - 1]) for i in range(1, forecast_steps + 1)]

        ci = forecast.conf_int()
        lowers = ci[:, 0].tolist()
        uppers = ci[:, 1].tolist()
        conf_int = [(last_year + i, lowers[i - 1], uppers[i - 1]) for i in range(1, forecast_steps + 1)]

        results[col] = {
            'order': order,
            'original': original,
            'forecast': forecast_list,
            'conf_int': conf_int
        }

    for col, data in results.items():
        forecast_vals = data['forecast']
        print(f"\n--- {col} ---")
        for i in range(1, len(forecast_vals)):
            year_prev, val_prev = forecast_vals[i - 1]
            year_curr, val_curr = forecast_vals[i]
            growth = ((val_curr - val_prev) / val_prev) * 100 if val_prev else 0
            print(f"{year_prev} → {year_curr}: {growth:.2f}% (de la {val_prev:.2f} la {val_curr:.2f})")

    return results
