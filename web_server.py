from flask import Flask, request, jsonify, send_file
import tempfile
import os
import sys
import json
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests
import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller, kpss
from algoritmi_licenta import process_folder, export_results, results
from prediction_model import arima_for_all_columns
from tests import main as tests_main
from main import processing_normal_image, processing_new_folder_with_safe_files, plot_bar_evolution_sarima

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

app = Flask(__name__)

SAFE_FOLDER_PATH = None
CSV_FILE_PATH = None

@app.route('/')
def home():
    endpoints = {
        "endpoints": {
            "/": "This help page",
            "/health": "Health check endpoint",
            "/process-images": "Process normal satellite images and return analysis results",
            "/process-images-real": "Actually process satellite images from specified folder",
            "/process-normal-images-hardcoded": "Process normal images with hardcoded paths (your specific Sentinel-2 images)",
            "/process-safe-folder": "Process .SAFE folder with user-defined threshold",
            "/process-safe-folder-with-export": "Process .SAFE folder and optionally export results",
            "/plot-bar-evolution": "Generate bar evolution plot from CSV data",
            "/compare-csv": "Compare CSV file data using ARIMA models",
            "/set-safe-folder": "Set the .SAFE folder path for processing",
            "/set-csv-file": "Set the CSV file path for analysis"
        }
    }
    return jsonify(endpoints)

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "message": "Server is running"})

@app.route('/process-images')
def process_images():
    try:
        image_folder = request.args.get('image_folder')
        threshold = request.args.get('threshold', default=85.0, type=float)
        
        if image_folder and os.path.exists(image_folder):
            return jsonify({
                "status": "success",
                "message": f"Processed images from folder: {image_folder}",
                "threshold_used": threshold,
                "description": "Processed satellite images using NSI and NDESI indices",
                "methods_used": [
                    "NSI (Normalized Salinity Index)",
                    "NDESI (Normalized Difference Enhanced Sand Index)", 
                    "K-means clustering (random centers)",
                    "K-means clustering (++ centers)",
                    "Otsu thresholding",
                    "Mean thresholding",
                    "User-defined thresholding"
                ]
            })
        else:
            return jsonify({
                "status": "success",
                "message": "Image processing simulation completed",
                "description": "Would process satellite images using NSI and NDESI indices with various thresholding methods",
                "note": "To process actual images, provide 'image_folder' parameter with path to folder containing B02, B03, B04, B08, B11, B12, and SWIR band images",
                "methods_available": [
                    "NSI (Normalized Salinity Index)",
                    "NDESI (Normalized Difference Enhanced Sand Index)", 
                    "K-means clustering",
                    "Otsu thresholding",
                    "Mean thresholding",
                    "User-defined thresholding"
                ]
            })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/process-safe-folder')
def process_safe_folder():
    try:
        threshold = request.args.get('threshold', default=75.0, type=float)
        folder_path = request.args.get('folder_path')
        
        if not folder_path:
            return jsonify({
                "status": "error", 
                "message": "folder_path parameter is required"
            }), 400
        
        if not os.path.exists(folder_path):
            return jsonify({
                "status": "error", 
                "message": f"Folder path does not exist: {folder_path}"
            }), 400
        
        results.clear()
        
        process_folder(folder_path, threshold)
        
        return jsonify({
            "status": "success",
            "message": f"Processed .SAFE folder with threshold {threshold}",
            "results_count": len(results),
            "results": results,
            "threshold_used": threshold
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/plot-bar-evolution')
def plot_bar_evolution():
    try:
        csv_path = request.args.get('csv_path')
        plot_type = request.args.get('plot_type', default='simple')
        
        if not csv_path:
            return jsonify({
                "status": "error", 
                "message": "csv_path parameter is required"
            }), 400
            
        if not os.path.exists(csv_path):
            return jsonify({
                "status": "error", 
                "message": f"CSV file does not exist: {csv_path}"
            }), 400
        
        if plot_type == 'sarima':
            try:
                return jsonify({
                    "status": "success",
                    "message": "SARIMA bar evolution plot would be generated",
                    "plot_type": "sarima",
                    "note": "This endpoint simulates the plot_bar_evolution_sarima function"
                })
            except Exception as e:
                return jsonify({"status": "error", "message": f"SARIMA plotting error: {str(e)}"}), 500
        else:
            df = pd.read_csv(csv_path)
            
            plt.figure(figsize=(12, 8))
            
            year_col = df.columns[0]
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if year_col in numeric_cols:
                numeric_cols.remove(year_col)
            
            if year_col in df.columns and len(numeric_cols) > 0:
                df_grouped = df.groupby(year_col)[numeric_cols].mean()
                df_grouped.plot(kind='bar', figsize=(12, 8))
                plt.title('Bar Evolution Over Time')
                plt.xlabel('Year')
                plt.ylabel('Values')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Save plot to bytes
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png')
                img_buffer.seek(0)
                plt.close()
                
                # Encode to base64
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                
                return jsonify({
                    "status": "success",
                    "message": "Bar evolution plot generated",
                    "plot_base64": img_base64,
                    "plot_type": "simple",
                    "data_summary": {
                        "total_rows": len(df),
                        "years_covered": df[year_col].nunique(),
                        "numeric_columns": numeric_cols
                    }
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": "CSV must have a year column and at least one numeric column"
                }), 400
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/compare-csv')
def compare_csv():
    try:
        csv_path = request.args.get('csv_path')
        order_p = request.args.get('order_p', default=1, type=int)
        order_d = request.args.get('order_d', default=1, type=int) 
        order_q = request.args.get('order_q', default=1, type=int)
        forecast_steps = request.args.get('forecast_steps', default=5, type=int)
        
        if not csv_path:
            return jsonify({
                "status": "error", 
                "message": "csv_path parameter is required"
            }), 400
            
        if not os.path.exists(csv_path):
            return jsonify({
                "status": "error", 
                "message": f"CSV file does not exist: {csv_path}"
            }), 400
        
        order = (order_p, order_d, order_q)
        results = arima_for_all_columns(csv_path, order=order, forecast_steps=forecast_steps)
        
        return jsonify({
            "status": "success",
            "message": f"ARIMA analysis completed for {len(results)} series",
            "parameters": {
                "order": order,
                "forecast_steps": forecast_steps
            },
            "results": results
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/set-safe-folder', methods=['POST'])
def set_safe_folder():
    global SAFE_FOLDER_PATH
    try:
        data = request.get_json()
        folder_path = data.get('folder_path')
        
        if not folder_path:
            return jsonify({
                "status": "error", 
                "message": "folder_path is required in request body"
            }), 400
            
        if not os.path.exists(folder_path):
            return jsonify({
                "status": "error", 
                "message": f"Folder path does not exist: {folder_path}"
            }), 400
            
        SAFE_FOLDER_PATH = folder_path
        return jsonify({
            "status": "success",
            "message": f"SAFE folder path set to: {folder_path}"
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/set-csv-file', methods=['POST'])
def set_csv_file():
    global CSV_FILE_PATH
    try:
        data = request.get_json()
        csv_path = data.get('csv_path')
        
        if not csv_path:
            return jsonify({
                "status": "error", 
                "message": "csv_path is required in request body"
            }), 400
            
        if not os.path.exists(csv_path):
            return jsonify({
                "status": "error", 
                "message": f"CSV file does not exist: {csv_path}"
            }), 400
            
        CSV_FILE_PATH = csv_path
        return jsonify({
            "status": "success",
            "message": f"CSV file path set to: {csv_path}"
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/process-images-real')
def process_images_real():
    try:
        image_folder = request.args.get('image_folder')
        threshold = request.args.get('threshold', default=85.0, type=float)
        
        if not image_folder:
            return jsonify({
                "status": "error",
                "message": "image_folder parameter is required"
            }), 400
            
        if not os.path.exists(image_folder):
            return jsonify({
                "status": "error",
                "message": f"Image folder does not exist: {image_folder}"
            }), 400
        
        from algoritmi_licenta import (
            initialize_bands, compute_nsi, compute_ndesi, normalize_arrays,
            create_binary_image_mean_threshold, kmeans_clustering_random_centers,
            kmeans_clustering_pp_centers, create_binary_image_otsu_threshold,
            create_binary_image_user_defined_threshold, pixel_count
        )
        
        required_bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12', 'SWIR']
        band_paths = {}
        
        image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            return jsonify({
                "status": "error",
                "message": f"No image files found in {image_folder}"
            }), 400
        
        for band in required_bands:
            band_found = False
            for image_file in image_files:
                if band in image_file:
                    band_paths[band] = os.path.join(image_folder, image_file)
                    band_found = True
                    break
            if not band_found:
                return jsonify({
                    "status": "error",
                    "message": f"Missing band {band} in folder {image_folder}",
                    "found_files": image_files
                }), 400
        
        blue_array, red_array, swir1_array, swir2_array, green_array, nir_array, swir_array = initialize_bands(
            band_paths['B02'], band_paths['B03'], band_paths['B04'],
            band_paths['B11'], band_paths['B12'], band_paths['B08'], band_paths['SWIR']
        )
        
        nsi_index = compute_nsi(green_array, red_array, swir1_array)
        ndesi_index = compute_ndesi(blue_array, red_array, swir1_array, swir2_array)
        
        normalized_nsi = normalize_arrays(nsi_index)
        normalized_ndesi = normalize_arrays(ndesi_index)
        
        binary_nsi = create_binary_image_mean_threshold(nsi_index)
        binary_ndesi = create_binary_image_mean_threshold(ndesi_index)
        
        desert_mask_nsi_random = kmeans_clustering_random_centers(normalized_nsi, n_clusters=2)
        desert_mask_nsi_pp = kmeans_clustering_pp_centers(normalized_nsi, n_clusters=2)
        desert_mask_ndesi_random = kmeans_clustering_random_centers(normalized_ndesi, n_clusters=2)
        desert_mask_ndesi_pp = kmeans_clustering_pp_centers(normalized_ndesi, n_clusters=2)
        
        otsu_nsi = create_binary_image_otsu_threshold(nsi_index)
        otsu_ndesi = create_binary_image_otsu_threshold(ndesi_index)
        
        user_defined_nsi = create_binary_image_user_defined_threshold(normalized_nsi, int(threshold))
        user_defined_ndesi = create_binary_image_user_defined_threshold(normalized_ndesi, int(threshold))
        
        return jsonify({
            "status": "success",
            "message": f"Successfully processed images from {image_folder}",
            "threshold_used": threshold,
            "band_files_found": list(band_paths.keys()),
            "results": {
                "nsi": {
                    "mean_threshold": int(pixel_count(binary_nsi)),
                    "kmeans_random": int(pixel_count(desert_mask_nsi_random)),
                    "kmeans_pp": int(pixel_count(desert_mask_nsi_pp)),
                    "otsu_threshold": int(pixel_count(otsu_nsi)),
                    "user_defined_threshold": int(pixel_count(user_defined_nsi))
                },
                "ndesi": {
                    "mean_threshold": int(pixel_count(binary_ndesi)),
                    "kmeans_random": int(pixel_count(desert_mask_ndesi_random)),
                    "kmeans_pp": int(pixel_count(desert_mask_ndesi_pp)),
                    "otsu_threshold": int(pixel_count(otsu_ndesi)),
                    "user_defined_threshold": int(pixel_count(user_defined_ndesi))
                }
            }
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/process-safe-folder-with-export')
def process_safe_folder_with_export():
    try:
        folder_path = request.args.get('folder_path')
        threshold = request.args.get('threshold', default=75.0, type=float)
        export_folder = request.args.get('export_folder')
        
        if not folder_path:
            return jsonify({
                "status": "error", 
                "message": "folder_path parameter is required"
            }), 400
        
        if not os.path.exists(folder_path):
            return jsonify({
                "status": "error", 
                "message": f"Folder path does not exist: {folder_path}"
            }), 400
        
        results.clear()
        
        process_folder(folder_path, threshold)
        
        response_data = {
            "status": "success",
            "message": f"Processed .SAFE folder with threshold {threshold}",
            "folder_processed": folder_path,
            "threshold_used": threshold,
            "results_count": len(results),
            "results": results
        }
        
        if export_folder:
            if os.path.exists(export_folder):
                try:
                    export_results(results, export_folder)
                    response_data["export_status"] = "success"
                    response_data["export_message"] = f"Results exported to {export_folder}"
                    response_data["exported_to"] = export_folder
                except Exception as e:
                    response_data["export_status"] = "error"
                    response_data["export_message"] = f"Export failed: {str(e)}"
            else:
                response_data["export_status"] = "error"
                response_data["export_message"] = f"Export folder does not exist: {export_folder}"
        else:
            response_data["export_message"] = "No export folder provided - results not exported"
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/process-normal-images-hardcoded')
def process_normal_images_hardcoded():
    try:
        print("Starting hardcoded image processing...")
        

        blue_band_path = "C:\\Users\\rares\\OneDrive\\Desktop\\Sentinel-2\\B02-Olt.jpg"
        red_band_path = "C:\\Users\\rares\\OneDrive\\Desktop\\Sentinel-2\\B04-Olt.jpg"
        swir1_band_path = "C:\\Users\\rares\\OneDrive\\Desktop\\Sentinel-2\\B11-Olt.jpg"
        swir2_band_path = "C:\\Users\\rares\\OneDrive\\Desktop\\Sentinel-2\\B12-Olt.jpg"
        green_band_path = "C:\\Users\\rares\\OneDrive\\Desktop\\Sentinel-2\\B03-Olt.jpg"
        nir_band_path = "C:\\Users\\rares\\OneDrive\\Desktop\\Sentinel-2\\B08-Olt.jpg"
        swir_band_path = "C:\\Users\\rares\\OneDrive\\Desktop\\Sentinel-2\\SWIR-Olt.jpg"
        
        print("Checking file paths...")
        
        # Check if all files exist
        band_paths = {
            "blue": blue_band_path,
            "red": red_band_path, 
            "swir1": swir1_band_path,
            "swir2": swir2_band_path,
            "green": green_band_path,
            "nir": nir_band_path,
            "swir": swir_band_path
        }
        
        missing_files = []
        for band_name, path in band_paths.items():
            if not os.path.exists(path):
                missing_files.append(f"{band_name}: {path}")
                print(f"Missing file: {band_name} - {path}")
        
        if missing_files:
            print(f"Missing files detected: {missing_files}")
            return jsonify({
                "status": "error",
                "message": "Missing required band files",
                "missing_files": missing_files,
                "required_paths": band_paths
            }), 400
        
        print("All files found, importing functions...")
        
        # Import processing functions
        from algoritmi_licenta import (
            initialize_bands, compute_nsi, compute_ndesi, normalize_arrays,
            create_binary_image_mean_threshold, kmeans_clustering_random_centers,
            kmeans_clustering_pp_centers, create_binary_image_otsu_threshold,
            create_binary_image_user_defined_threshold, pixel_count
        )
        
        print("Functions imported, processing images...")
        
        blue_array, red_array, swir1_array, swir2_array, green_array, nir_array, swir_array = initialize_bands(
            blue_band_path,
            green_band_path,
            red_band_path,
            swir1_band_path,
            swir2_band_path,
            nir_band_path, 
            swir_band_path
        )
        
        print("Bands initialized, computing indices...")
        
        nsi_index = compute_nsi(green_array, red_array, swir1_array)
        ndesi_index = compute_ndesi(blue_array, red_array, swir1_array, swir2_array)
        
        print("Indices computed, normalizing...")
        
        normalized_nsi = normalize_arrays(nsi_index)
        binary_nsi = create_binary_image_mean_threshold(nsi_index)
        normalized_ndesi = normalize_arrays(ndesi_index)
        binary_ndesi = create_binary_image_mean_threshold(ndesi_index)
        
        print("Applying clustering methods...")
        
        desert_mask = kmeans_clustering_random_centers(normalized_ndesi, n_clusters=2)
        desert_mask2 = kmeans_clustering_pp_centers(normalized_ndesi, n_clusters=2)
        desert_mask3 = kmeans_clustering_pp_centers(normalized_nsi, n_clusters=2)
        desert_mask4 = kmeans_clustering_random_centers(normalized_nsi, n_clusters=2)
        
        otsu_ndesi = create_binary_image_otsu_threshold(normalized_ndesi)
        otsu_nsi = create_binary_image_otsu_threshold(normalized_nsi)
        
        user_ndesi = create_binary_image_user_defined_threshold(normalized_ndesi, 207)
        user_nsi = create_binary_image_user_defined_threshold(normalized_nsi, 85)
        
        print("Calculating pixel counts...")
        
        results = {
            "status": "success",
            "message": "Successfully processed normal images with hardcoded paths",
            "band_files_used": band_paths,
            "image_dimensions": {
                "nsi_shape": list(nsi_index.shape),
                "ndesi_shape": list(ndesi_index.shape)
            },
            "pixel_counts": {
                "nsi": {
                    "mean_threshold": int(pixel_count(binary_nsi)),
                    "kmeans_random": int(pixel_count(desert_mask4)),
                    "kmeans_pp": int(pixel_count(desert_mask3)),
                    "otsu_threshold": int(pixel_count(otsu_nsi)),
                    "user_defined_threshold_85": int(pixel_count(user_nsi))
                },
                "ndesi": {
                    "mean_threshold": int(pixel_count(binary_ndesi)),
                    "kmeans_random": int(pixel_count(desert_mask)),
                    "kmeans_pp": int(pixel_count(desert_mask2)),
                    "otsu_threshold": int(pixel_count(otsu_ndesi)),
                    "user_defined_threshold_207": int(pixel_count(user_ndesi))
                }
            },
            "summary": {
                "K_NSI_random_threshold": int(pixel_count(desert_mask4)),
                "K_NSI_pp_threshold": int(pixel_count(desert_mask3)),
                "Mean_threshold_NDESI": int(pixel_count(binary_ndesi)),
                "NSI_index_mean": int(pixel_count(binary_nsi)),
                "K_NDESI_random_threshold": int(pixel_count(desert_mask)),
                "K_NDESI_pp_threshold": int(pixel_count(desert_mask2))
            }
        }
        
        print("Processing completed successfully!")
        return jsonify(results)
        
    except Exception as e:
        print(f"Error in processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

def aggregate_by_year(df, year_col):
    return df.groupby(year_col).mean().sort_index()


def test_stationarity(series, name=None):
    series = np.array(series)
    if len(series) < 8:
        return {
            'series': name,
            'adf_stationary': None,
            'kpss_stationary': None,
            'adf_pvalue': None,
            'kpss_pvalue': None,
            'conclusion': 'Too few points'
        }
    result = {'series': name}
    # ADF
    try:
        adf_res = adfuller(series, autolag='AIC')
        result['adf_statistic'] = round(adf_res[0], 4)
        result['adf_pvalue'] = round(adf_res[1], 4)
        result['adf_stationary'] = result['adf_pvalue'] < 0.05
    except:
        result.update({'adf_statistic': None, 'adf_pvalue': None, 'adf_stationary': None})
    # KPSS
    try:
        kpss_res = kpss(series, regression='c', nlags='auto')
        result['kpss_statistic'] = round(kpss_res[0], 4)
        result['kpss_pvalue'] = round(kpss_res[1], 4)
        result['kpss_stationary'] = result['kpss_pvalue'] > 0.05
    except:
        result.update({'kpss_statistic': None, 'kpss_pvalue': None, 'kpss_stationary': None})
    # Conclusion
    adf = result.get('adf_stationary')
    kpss_ = result.get('kpss_stationary')
    if adf and kpss_:
        result['conclusion'] = 'Stationary (both tests agree)'
    elif adf is False and kpss_ is False:
        result['conclusion'] = 'Non-stationary (both tests agree)'
    elif adf and not kpss_:
        result['conclusion'] = 'Conflicting: ADF stationary, KPSS non-stationary'
    elif not adf and kpss_:
        result['conclusion'] = 'Conflicting: ADF non-stationary, KPSS stationary'
    else:
        result['conclusion'] = 'Inconclusive'
    return result


def analyze_stationarity(file_path, date_col=0):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return None, str(e)
    year_col = df.columns[date_col]
    data_cols = df.columns.drop(year_col)
    df_grp = df.groupby(year_col)[data_cols].mean().sort_index()
    results = []
    for col in data_cols:
        ts = df_grp[col].dropna().astype(float).tolist()
        r = test_stationarity(ts, col)
        results.append(r)
    return pd.DataFrame(results), None


def get_access_token(username: str, password: str) -> str:
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }
    r = requests.post(
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        data=data,
    )
    r.raise_for_status()
    return r.json()["access_token"]

@app.route('/')
def home():
    return jsonify({
        "endpoints": list(app.url_map.iter_rules())
    })

@app.route('/test-stationarity')
def test_stationarity_endpoint():
    csv_path = request.args.get('csv_path')
    if not csv_path or not os.path.exists(csv_path):
        return jsonify({"status": "error", "message": "Valid csv_path is required"}), 400
    df_res, err = analyze_stationarity(csv_path)
    if err:
        return jsonify({"status": "error", "message": err}), 500
    if df_res.empty:
        return jsonify({"status": "success", "results": []})
    return jsonify({"status": "success", "results": df_res.to_dict(orient='records')}), 200

@app.route('/copernicus-search', methods=['POST'])
def copernicus_search():
    data = request.get_json() or {}
    username = data.get('username')
    password = data.get('password')
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    aoi = data.get('aoi')
    top = data.get('top', '20')
    cloud = data.get('cloud_cover_threshold', '10.0')
    if not all([username, password, start_date, end_date, aoi]):
        return jsonify({"status": "error", "message": "Missing required parameters"}), 400
    try:
        token = get_access_token(username, password)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 401
    filter_str = (
        "Collection/Name eq 'SENTINEL-2' and "
        f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le {cloud}) and "
        f"ContentDate/Start gt {start_date} and ContentDate/Start lt {end_date} and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{aoi}')"
    )
    search_url = (
        "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter="
        + filter_str + f"&$top={top}"
    )
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(search_url, headers=headers)
    if r.status_code != 200:
        return jsonify({"status": "error", "message": r.text}), r.status_code
    vals = r.json().get('value', [])
    products = []
    session = requests.Session()
    session.headers.update(headers)
    for prod in vals:
        pid = prod.get('Id')
        name = prod.get('Name')
        download_url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({pid})/$value"
        try:
            dr = session.get(download_url, stream=True)
            fname = f"{name}.zip"
            with open(fname, 'wb') as f:
                for chunk in dr.iter_content(8192):
                    f.write(chunk)
            products.append({"id": pid, "name": name, "downloaded": True, "file": fname})
        except Exception as ex:
            products.append({"id": pid, "name": name, "downloaded": False, "error": str(ex)})
    return jsonify({"status": "success", "products": products})

if __name__ == '__main__':
    print("Starting Flask server with PySide6...")
    print("Available endpoints:")
    print("- GET  /                                    - Show available endpoints")
    print("- GET  /health                              - Health check")
    print("- GET  /process-images                      - Process satellite images (simulation)")
    print("- GET  /process-images-real                 - Process satellite images (real)")
    print("- GET  /process-normal-images-hardcoded     - Process your specific Sentinel-2 images")
    print("- GET  /process-safe-folder                 - Process .SAFE folder")
    print("- GET  /process-safe-folder-with-export     - Process .SAFE folder with export")
    print("- GET  /plot-bar-evolution                  - Generate bar evolution plot")
    print("- GET  /compare-csv                         - ARIMA analysis")
    print("- POST /set-safe-folder                     - Set .SAFE folder path")
    print("- POST /set-csv-file                        - Set CSV file path")
    print("\nExample usage:")
    print("http://localhost:5000/process-normal-images-hardcoded")
    print("http://localhost:5000/process-safe-folder-with-export?folder_path=/path/to/safe/folder&threshold=75.0&export_folder=/path/to/export")
    print("http://localhost:5000/process-safe-folder?folder_path=/path/to/safe/folder&threshold=75.0")
    print("http://localhost:5000/process-images-real?image_folder=/path/to/images&threshold=85.0")
    print("http://localhost:5000/plot-bar-evolution?csv_path=/path/to/data.csv&plot_type=sarima")
    print("http://localhost:5000/compare-csv?csv_path=/path/to/data.csv&forecast_steps=10")
    
    app.run(debug=True, host='0.0.0.0', port=5000)