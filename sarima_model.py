import pandas as pd
import numpy as np
from itertools import product
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def sarima_for_all_columns(
    file_path: str,
    year_col: str = None,
    order_grid = [(0,1,1), (1,1,1), (1,0,1), (2,1,0)],
    forecast_years: int = 5
):
    """Foloseste un model SARIMA pentru predictie."""
    df = pd.read_csv(file_path)
    if year_col is None:
        year_col = df.columns[0]

    years = df[year_col].astype(int).values
    data = df.drop(columns=[year_col]).astype(float)

    results = {}
    for col in data.columns:
        ts = data[col].dropna().values
        if len(ts) < 8 or ts.sum() == 0:
            continue

        best_aic = np.inf
        best_order = None
        #Incearca toate gridurile pentru a gasi cel mai bun AIC
        for order in order_grid:
            try:
                m = SARIMAX(ts,
                            order=order,
                            trend='t',
                            enforce_stationarity=False,
                            enforce_invertibility=False)
                r = m.fit(disp=False)
                if r.aic < best_aic:
                    best_aic, best_order = r.aic, order
            except Exception:
                continue

        model = SARIMAX(ts,
                        order=best_order,
                        trend='t',
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        res = model.fit(disp=False)

        original = list(zip(years.tolist(), ts.tolist()))
        steps = len(years)
        avg_ppy = max(1, int(len(ts) / len(np.unique(years))))
        fut_steps = forecast_years * avg_ppy

        fc = res.get_forecast(steps=fut_steps)
        fc_mean = fc.predicted_mean

        fc_blocks = [
            fc_mean[i * avg_ppy:(i + 1) * avg_ppy].mean()
            for i in range(forecast_years)
        ]
        last_year = years.max()
        forecast = [(last_year + i + 1, fc_blocks[i]) for i in range(forecast_years)]

        ci = fc.conf_int()
        lo = ci[:, 0]
        hi = ci[:, 1]
        lo_blocks = [
            lo[i * avg_ppy:(i + 1) * avg_ppy].mean()
            for i in range(forecast_years)
        ]
        hi_blocks = [
            hi[i * avg_ppy:(i + 1) * avg_ppy].mean()
            for i in range(forecast_years)
        ]
        conf_int = [
            (last_year + i + 1, lo_blocks[i], hi_blocks[i])
            for i in range(forecast_years)
        ]

        results[col] = {
            'order': best_order,
            'original': original,
            'forecast': forecast,
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
