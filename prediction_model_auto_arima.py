import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from PySide6.QtWidgets import QApplication, QFileDialog

warnings.filterwarnings("ignore", category=FutureWarning)

def plot_bar_evolution_auto(blocks_ahead: int = 5):
    app = QApplication([])
    csv_path, _ = QFileDialog.getOpenFileName(
        None, "Select CSV result file", "", "CSV Files (*.csv);;All Files (*)"
    )
    if not csv_path:
        print("No file selected, aborting.")
        return

    df = pd.read_csv(csv_path)
    year_col = df.columns[0]
    years = df[year_col].astype(int)
    data = df.drop(columns=[year_col]).astype(float)

    results = {}
    for col in data.columns:
        ts = data[col].dropna().values
        if len(ts) < 8 or ts.sum() == 0:
            continue
        model = auto_arima(ts,
                           start_p=0, start_q=0,
                           max_p=3, max_q=3,
                           seasonal=False,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True,
                           trend='t')
        results[col] = {
            'model': model,
            'hist': list(zip(years.tolist(), ts.tolist()))
        }

    if not results:
        print("No series to model.")
        return

    n = len(results)
    cols = (n + 1) // 2
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), constrained_layout=True)
    axes = axes.flatten()

    for ax, (col, info) in zip(axes, results.items()):
        hist = info['hist']
        by_year = {}
        for y, v in hist:
            by_year.setdefault(y, []).append(v)
        hist_years = sorted(by_year)
        hist_vals = [np.mean(by_year[y]) for y in hist_years]

        blocks_per_year = max(1, int(len(hist) / len(set(hist_years))))
        fcast = info['model'].predict(n_periods=blocks_ahead * blocks_per_year)
        fcast_years = [hist_years[-1] + i + 1 for i in range(blocks_ahead)]
        fcast_vals = [
            np.mean(fcast[i * blocks_per_year:(i + 1) * blocks_per_year])
            for i in range(blocks_ahead)
        ]

        # --- Afișare în consolă a creșterilor procentuale ---
        print(f"\n--- {col} (AutoARIMA) ---")
        for i in range(1, len(fcast_vals)):
            year_prev = fcast_years[i - 1]
            year_curr = fcast_years[i]
            val_prev = fcast_vals[i - 1]
            val_curr = fcast_vals[i]
            growth = ((val_curr - val_prev) / val_prev) * 100 if val_prev else 0
            print(f"{year_prev} → {year_curr}: {growth:.2f}% (de la {val_prev:.2f} la {val_curr:.2f})")
        # ------------------------------------------------------

        all_vals = hist_vals + fcast_vals
        M = max(all_vals) or 1
        hist_pct = [v / M * 100 for v in hist_vals]
        fc_pct = [v / M * 100 for v in fcast_vals]

        x_hist = np.arange(len(hist_years))
        x_fc = x_hist[-1] + 1 + np.arange(len(fcast_years))

        ax.bar(x_hist, hist_pct)
        ax.bar(x_fc, fc_pct, color='green')

        ax.set_title(col, fontsize=8)
        ax.set_ylabel('% of max', fontsize=7)
        ax.set_xticks(np.concatenate([x_hist, x_fc]))
        ax.set_xticklabels(
            [str(y) for y in hist_years] + [str(y) for y in fcast_years],
            rotation=90, fontsize=6
        )

    for ax in axes[len(results):]:
        ax.axis('off')

    plt.show()
