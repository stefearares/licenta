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
    """
    - file_path: your CSV, must have a column with the year (possibly repeated).
    - year_col: name of that year column (if None we use the first column).
    - order_grid: small list of p,d,q to try.
    - forecast_years: how many years out to predict.
    """
    df = pd.read_csv(file_path)
    if year_col is None:
        year_col = df.columns[0]

    # pull out the time index (just the year integer) and drop it from the data
    years = df[year_col].astype(int).values
    data = df.drop(columns=[year_col]).astype(float)

    results = {}
    for col in data.columns:
        ts = data[col].dropna().values
        if len(ts) < 8 or ts.sum()==0:
            # not enough points or entirely zero
            continue

        # pick best (p,d,q) by AIC
        best_aic = np.inf
        best_order = None
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

        # fit final
        model = SARIMAX(ts,
                        order=best_order,
                        trend='t',
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        res = model.fit(disp=False)

        # 1) Original: list of (year,value) for each observation
        original = list(zip(years.tolist(), ts.tolist()))

        # 2) Forecast step-level out for each calendar day (or step)
        steps = len(years)  # one step for each past point per year
        # we actually want forecast_years * average_points_per_year steps
        avg_ppy = max(1, int(len(ts) / len(np.unique(years))))
        fut_steps = forecast_years * avg_ppy

        fc = res.get_forecast(steps=fut_steps)
        fc_mean = fc.predicted_mean

        # now bundle those back into years by taking the mean of each block of avg_ppy
        fc_blocks = [
            fc_mean[i*avg_ppy:(i+1)*avg_ppy].mean()
            for i in range(forecast_years)
        ]
        last_year = years.max()
        forecast = [(last_year + i + 1, fc_blocks[i]) for i in range(forecast_years)]

        # Conf Int similarly
        ci = fc.conf_int()  # now a NumPy array of shape (steps, 2)
        lo = ci[:, 0]  # lower bounds
        hi = ci[:, 1]
        lo_blocks = [
            lo[i*avg_ppy:(i+1)*avg_ppy].mean()
            for i in range(forecast_years)
        ]
        hi_blocks = [
            hi[i*avg_ppy:(i+1)*avg_ppy].mean()
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

    return results