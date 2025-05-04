import sys
from PySide6.QtWidgets import QApplication
from algoritmi_licenta import *
from main_window import MainWindow
from main_widget import Widget
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox
from prediction_model import *
from prediction_modelv2 import *

def processing_new_folder_with_safe_files():

    app = QApplication(sys.argv)
    src = QFileDialog.getExistingDirectory(
        None,
        "Select folder with .SAFE products",
        ""
    )
    if not src:
        print("No source folder selected.")
        sys.exit(0)

    # 2) Ask for the user-defined threshold
    threshold, ok = QInputDialog.getDouble(
        None,
        "User-defined Threshold",
        "Enter a user-defined float threshold (e.g. 75.0):",
        75.0,
        1.0,
        1e9,
        2
    )
    if not ok:
        print("Threshold input canceled.")
        sys.exit(0)

    process_folder(src, threshold)

    print("\n>>> RESULTS LIST:", results)


    dest = QFileDialog.getExistingDirectory(
        None,
        "Select export folder",
        ""
    )
    if dest:
        export_results(results, dest)
    else:
        print("Export canceled.")

def gui_app():
    app = QApplication(sys.argv)
    widget = Widget()
    window = MainWindow(widget)
    window.show()
    sys.exit(app.exec())

def processing_normal_image():
    blue_band_path = "C:\\Users\\rares\\OneDrive\\Desktop\\Sentinel-2\\B02-Olt.jpg"  # Blue band
    red_band_path = "C:\\Users\\rares\\OneDrive\\Desktop\\Sentinel-2\\B04-Olt.jpg"  # Red band (or VRE if available)
    swir1_band_path = "C:\\Users\\rares\\OneDrive\\Desktop\\Sentinel-2\\B11-Olt.jpg"  # SWIR1 band
    swir2_band_path = "C:\\Users\\rares\\OneDrive\\Desktop\\Sentinel-2\\B12-Olt.jpg"  # SWIR2 band
    green_band_path = "C:\\Users\\rares\\OneDrive\\Desktop\\Sentinel-2\\B03-Olt.jpg"
    nir_band_path = "C:\\Users\\rares\\OneDrive\\Desktop\\Sentinel-2\\B08-Olt.jpg"
    swir_band_path = "C:\\Users\\rares\\OneDrive\\Desktop\\Sentinel-2\\SWIR-Olt.jpg"
    blue_array, red_array, swir1_array, swir2_array, green_array, nir_array, swir_array = initialize_bands(
        blue_band_path,
        green_band_path,
        red_band_path,
        swir1_band_path,
        swir2_band_path,
        nir_band_path, swir_band_path)
    nsi_index = compute_nsi(green_array, red_array, swir1_array)
    ndesi_index = compute_ndesi(blue_array, red_array, swir1_array, swir2_array)
    normalized_nsi = normalize_arrays(nsi_index)
    binary_nsi = create_binary_image_mean_threshold(nsi_index)
    normalized_ndesi = normalize_arrays(ndesi_index)
    binary_ndesi = create_binary_image_mean_threshold(ndesi_index)
    plotting(binary_ndesi, "Mean NDESI")
    plotting(binary_nsi, "Mean NSI")
    desert_mask = kmeans_clustering_random_centers(normalized_ndesi, n_clusters=2)
    desert_mask2 = kmeans_clustering_pp_centers(normalized_ndesi, n_clusters=2)
    desert_mask3 = kmeans_clustering_pp_centers(normalized_nsi, n_clusters=2)
    desert_mask4 = kmeans_clustering_random_centers(normalized_nsi, n_clusters=2)
    otsu_ndesi = create_binary_image_otsu_threshold(normalized_ndesi)
    otsu_nsi = create_binary_image_otsu_threshold(normalized_nsi)
    user_ndesi = create_binary_image_user_defined_threshold(normalized_ndesi, 79)
    user_nsi = create_binary_image_user_defined_threshold(normalized_nsi, 85)
    plotting(user_ndesi, "User defined threshold NDESI")
    plotting(user_nsi, "User defined threshold NSI")
    plotting(otsu_ndesi, "Otsu NDESI")
    plotting(otsu_nsi, "Otsu NSI")
    plotting(desert_mask, "NDESI Desert Regions Detected via K-Means random")
    plotting(desert_mask2, "NDESI Desert Regions Detected via K-Means pp")
    plotting(desert_mask3, "NSI Desert Regions Detected via K-Means random")
    plotting(desert_mask4, "NSI Desert Regions Detected via K-Means pp")
    print("K NSI random threshold", pixel_count(desert_mask3))
    print("K NSI pp threshold", pixel_count(desert_mask4))
    print("Mean threshold", pixel_count(binary_ndesi))
    print("NSI index", pixel_count(binary_nsi))
    print("K random threshold", pixel_count(desert_mask))
    print("K pp threshold", pixel_count(desert_mask2))

def process_csv_file_with_arima():
    app = QApplication(sys.argv)
    file_path, _ = QFileDialog.getOpenFileName(
        None,
        "Select CSV File for ARIMA",
        "",
        "CSV Files (*.csv);;All Files (*)"
    )
    if not file_path:
        QMessageBox.information(None, "No File", "No CSV selected. Exiting.")
        sys.exit(0)

    results = arima_for_all_columns(file_path)
    if not results:
        QMessageBox.warning(None, "No Models", "No series could be modeled.")
        sys.exit(1)

    for col, data in results.items():
        print(f"\n=== Series: {col} (ARIMA order={data['order']}) ===")
        print("Original data (year, value):", data['original'])
        print("Forecast (year, predicted):", data['forecast'])
        print("Confidence intervals (year, lower, upper):", data['conf_int'])

    sys.exit(0)

def process_csv_file_with_sarima():
    app = QApplication(sys.argv)
    file_path, _ = QFileDialog.getOpenFileName(
        None,
        "Select CSV File for ARIMA",
        "",
        "CSV Files (*.csv);;All Files (*)"
    )
    if not file_path:
        QMessageBox.information(None, "No File", "No CSV selected. Exiting.")
        sys.exit(0)

    results = sarima_for_all_columns(file_path)
    if not results:
        QMessageBox.warning(None, "No Models", "No series could be modeled.")
        sys.exit(1)

    for col, data in results.items():
        print(f"\n=== Series: {col} (SARIMA order={data['order']}) ===")
        print("Original data (year, value):", data['original'])
        print("Forecast (year, predicted):", data['forecast'])
        print("Confidence intervals (year, lower, upper):", data['conf_int'])

    sys.exit(0)
def plot_bar_evolution_arima(file_path: str):

    results_dict = arima_for_all_columns(file_path)
    filtered = {k: v for k, v in results_dict.items() if 'user_defined' not in k}
    if not filtered:
        print("No series to plot.")
        return

    n = len(filtered)
    cols = (n + 1) // 2
    rows = 2
    width = min(cols * 3, 12)
    height = min(rows * 3, 6)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(width, height), constrained_layout=True)
    axes = axes.flatten()

    for ax, (col, data) in zip(axes, filtered.items()):
        orig_years, orig_vals = zip(*data['original'])
        fc_years, fc_vals = zip(*data['forecast'])
        years = list(orig_years) + list(fc_years)
        vals = list(orig_vals) + list(fc_vals)
        x = np.arange(len(years))

        max_val = max(vals) if vals else 1
        pct_vals = [v / max_val * 100 for v in vals]

        hist_len = len(orig_years)

        ax.bar(x[:hist_len], pct_vals[:hist_len])
        ax.bar(x[hist_len:], pct_vals[hist_len:], color='green')

        ax.set_title(col)
        ax.set_ylabel('Percent of Max (%)')

        ax.set_xticks(x)
        ax.set_xticklabels([str(y) for y in years], rotation=90, fontsize=8)

    for ax in axes[n:]:
        ax.axis('off')

    plt.show()

def plot_bar_evolution_sarima(file_path: str):
    results_dict = sarima_for_all_columns(file_path)
    filtered = {k: v for k, v in results_dict.items() if 'user_defined' not in k}
    if not filtered:
        print("No series to plot.")
        return

    n = len(filtered)
    cols = (n + 1) // 2
    rows = 2
    width = min(cols * 3, 12)
    height = min(rows * 3, 6)

    fig, axes = plt.subplots(rows, cols, figsize=(width, height), constrained_layout=True)
    axes = axes.flatten()

    for ax, (col, data) in zip(axes, filtered.items()):
        orig = data['original']
        by_year = {}
        for y, v in orig:
            by_year.setdefault(y, []).append(v)
        hist_years = sorted(by_year)
        hist_vals  = [np.mean(by_year[y]) for y in hist_years]

        fc_years, fc_vals = zip(*data['forecast'])

        all_vals = hist_vals + list(fc_vals)
        max_val  = max(all_vals) or 1
        hist_pct = [v/max_val*100 for v in hist_vals]
        fc_pct   = [v/max_val*100 for v in fc_vals]

        x_hist = np.arange(len(hist_years))
        x_fc   = x_hist[-1] + 1 + np.arange(len(fc_years))

        # draw bars
        ax.bar(x_hist, hist_pct)
        ax.bar(x_fc,   fc_pct, color='green')

        ax.set_title(col)
        ax.set_ylabel('Percent of Max (%)')
        ax.set_xticks(np.concatenate([x_hist, x_fc]))
        ax.set_xticklabels(
            [str(y) for y in hist_years] + [str(y) for y in fc_years],
            rotation=90, fontsize=8
        )

    for ax in axes[n:]:
        ax.axis('off')

    plt.show()

def plot_bar_evolution_flow():

    app = QApplication(sys.argv)
    csv_path, _ = QFileDialog.getOpenFileName(
        None,
        "Select CSV result file",
        "",
        "CSV Files (*.csv);;All Files (*)"
    )
    if not csv_path:
        QMessageBox.information(None, "No File", "No CSV selected.")
        sys.exit(0)

    plot_bar_evolution_sarima(csv_path)
    sys.exit(0)


def compare_arima_sarima(file_path: str,
                         year_col: str=None,
                         sarima_orders=[(0,1,1),(1,1,1),(1,0,1),(2,1,0)],
                         holdout_years: int=2):
    df = pd.read_csv(file_path)
    if year_col is None:
        year_col = df.columns[0]
    df[year_col] = df[year_col].astype(int)

    # 1) collapse to one value per year (mean)
    yearly = df.groupby(year_col).mean()
    years = yearly.index.values
    data = yearly.values  # shape (n_years, n_series)
    cols = yearly.columns

    # define train/test split
    train_idx = slice(0, len(years)-holdout_years)
    test_idx  = slice(len(years)-holdout_years, None)
    train_years, test_years = years[train_idx], years[test_idx]

    for i, col in enumerate(cols):
        ts = yearly[col].values
        train_ts = ts[train_idx]
        test_ts  = ts[test_idx]

        print(f"\n=== Series '{col}' ===")
        # --- ARIMA(1,1,1) ---
        try:
            m1 = ARIMA(train_ts, order=(1,1,1),
                       enforce_stationarity=False,
                       enforce_invertibility=False).fit()
            aic1 = m1.aic
            fc1  = m1.forecast(steps=holdout_years)
            mae1 = np.mean(np.abs(fc1 - test_ts))
        except Exception as e:
            print(" ARIMA ERROR:", e)
            aic1 = np.nan; mae1 = np.nan

        # --- SARIMAX trend + best order by AIC ---
        best_aic, best_order = np.inf, None
        for order in sarima_orders:
            try:
                m = SARIMAX(train_ts,
                            order=order,
                            trend='t',
                            enforce_stationarity=False,
                            enforce_invertibility=False).fit(disp=False)
                if m.aic < best_aic:
                    best_aic, best_order = m.aic, order
            except:
                continue

        try:
            m2 = SARIMAX(train_ts,
                         order=best_order,
                         trend='t',
                         enforce_stationarity=False,
                         enforce_invertibility=False).fit(disp=False)
            aic2 = m2.aic
            fc2  = m2.forecast(steps=holdout_years)
            mae2 = np.mean(np.abs(fc2 - test_ts))
        except Exception as e:
            print(" SARIMA ERROR:", e)
            aic2 = np.nan; mae2 = np.nan

        print(f" ARIMA(1,1,1):     AIC = {aic1:.1f},  MAE = {mae1:.1f}")
        print(f" SARIMA{best_order}+t: AIC = {aic2:.1f},  MAE = {mae2:.1f}")

def process_csv_file_compare():
    app = QApplication(sys.argv)
    csv_path, _ = QFileDialog.getOpenFileName(
        None,
        "Select CSV file for ARIMA vs SARIMA comparison",
        "",
        "CSV Files (*.csv);;All Files (*)"
    )
    if not csv_path:
        QMessageBox.information(None, "No file", "No CSV selected. Exiting.")
        sys.exit(0)

    # run the comparison and print to console
    compare_arima_sarima(csv_path)

    sys.exit(0)
if __name__ == '__main__':
    # plot_bar_evolution_flow()
    # processing_normal_image()
    # processing_new_folder_with_safe_files()
    process_csv_file_compare()
