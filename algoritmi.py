import os
import zipfile
import cv2
import numpy as np
import matplotlib
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime
import csv
import json
import tempfile

matplotlib.use('QtAgg')
import matplotlib.pyplot as plt


def initialize_bands(blue_band_path, green_band_path, red_band_path, swir1_band_path, swir2_band_path, nir_band_path,
                     swir_band_path):
    """Incarca si converteste imaginile benzilor Sentinel-2.

    Parameters
    ----------
    blue_band_path, green_band_path, red_band_path : str
        Paths la B02, B03 si B04.
    swir1_band_path, swir2_band_path : str
        Paths la B11 si B12.
    nir_band_path : str
        Path la B08.
    swir_band_path : str
        Path la SWIR(B8A).

    Returns
    -------
    tuple[np.ndarray, ...]
        Arrays pentru fiecare banda convertita la ``float32``. Valorile mai mici sau egale
        cu zero sunt inlocuite cu valori mai mici.
    """

    blue_band = cv2.imread(blue_band_path, cv2.IMREAD_GRAYSCALE)
    red_band = cv2.imread(red_band_path, cv2.IMREAD_GRAYSCALE)
    swir1_band = cv2.imread(swir1_band_path, cv2.IMREAD_GRAYSCALE)
    swir2_band = cv2.imread(swir2_band_path, cv2.IMREAD_GRAYSCALE)
    green_band = cv2.imread(green_band_path, cv2.IMREAD_GRAYSCALE)
    nir_band = cv2.imread(nir_band_path, cv2.IMREAD_GRAYSCALE)
    swir_band = cv2.imread(swir_band_path, cv2.IMREAD_GRAYSCALE)


    blue_array = blue_band.astype(np.float32)
    red_array = red_band.astype(np.float32)
    green_array = green_band.astype(np.float32)
    swir1_array = swir1_band.astype(np.float32)
    swir2_array = swir2_band.astype(np.float32)
    nir_array = nir_band.astype(np.float32)
    swir_band_array = swir_band.astype(np.float32)

    blue_array[blue_array <= 0] = 0.1
    red_array[red_array <= 0] = 0.1
    swir1_array[swir1_array <= 0] = 0.1
    swir2_array[swir2_array <= 0] = 0.1
    green_array[green_array <= 0] = 0.1
    nir_array[nir_array <= 0] = 0.1
    swir_band_array[swir_band_array <= 0] = 0.1
    return blue_array, red_array, swir1_array, swir2_array, green_array, nir_array, swir_band_array


def compute_nsi(green_array, red_array, swir1_array):
    """Calculeaza  Normalized Sand Index (NSI).

    Parameters
    ----------
    green_array, red_array, swir1_array : np.ndarray
        Arrays for the green, red si SWIR1.

    Returns
    -------
    np.ndarray
        Valoarea NSI cu  scaling logaritmic aplicat.
    """
    swir1_safe = np.where(swir1_array > 1, swir1_array, 1.01)

    #nsi_index = (green_array + red_array) / np.log1p(swir1_array)
    nsi_index = (green_array + red_array) / np.log(swir1_safe)
    nsi_index = np.nan_to_num(nsi_index, nan=0.1, posinf=0.1, neginf=0.1)
    #O trebuit sa dau clip la valori ca erau un range prea urias pt Kmeans si le am redus logaritmic
    nsi_index = np.log1p(nsi_index)
    return nsi_index


def compute_ndesi(blue_array, red_array, swir1_array, swir2_array):
    """Calculeaza Normalized Difference Enhanced Sand Index (NDESI)."

    Parameters
    ----------
    blue_array, red_array : np.ndarray
    swir1_array, swir2_array : np.ndarray

    Returns
    -------
    np.ndarray
        Vaolirle NDESI cu scaling logarithic aplicat.
    """
    ndesi = ((blue_array - red_array) * (swir1_array - swir2_array)) / \
            ((blue_array + red_array) * (swir1_array + swir2_array))
    ndesi = np.nan_to_num(ndesi, nan=0.1, posinf=0.1, neginf=0.1)

    #Reduce variatiile mari de date la Kmeans si la modele
    ndesi = np.log1p(ndesi)

    return ndesi

def normalize_arrays(index_array):
    """Normalizeaza un array la ``0-255`` pentru vizualizare sau clustering."""
    normalized_index = ((index_array - np.min(index_array)) / (np.max(index_array) - np.min(index_array)) * 255).astype(
        np.uint8)

    return normalized_index


def create_binary_image_user_defined_threshold(array_index, threshold):
    """Threshold pentru array utilizand o valoare aleasa de utilizator."""
    binary_mask = np.where(array_index > threshold, 1, 0)

    return binary_mask.astype(np.uint8) * 255


def create_binary_image_mean_threshold(array_index):
    """Masca binara utilizand valoarea media a unui array."""
    mean_value = np.mean(array_index)
    binary_mask = np.where(array_index > mean_value, 1, 0)

    return binary_mask.astype(np.uint8) * 255


def create_binary_image_otsu_threshold(array_index):
    """Masca binara utilizand algoritmul Otsu."""
    normalized_index = cv2.normalize(array_index, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, binary_mask = cv2.threshold(normalized_index, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_mask


#Normalized pentru ca imaginea e mai granulata si are variatii in desert, in timp ce cea binara nu are variatii si e mai greu de grupat
def kmeans_clustering_pp_centers(normalized_index, n_clusters):
    """Foloseste K-means++ pe imaginea normalizata"""
    pixels = normalized_index.reshape(-1, 1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, n_clusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    segmented_image = labels.reshape(normalized_index.shape)
    desert_cluster = np.argmax(centers)
    desert_regions = (segmented_image == desert_cluster).astype(np.uint8) * 255
    return desert_regions


def kmeans_clustering_random_centers(normalized_index, n_clusters):
    """Foloseste K-means cu centroide aleatoare pe imaginea normalizata."""
    pixels = normalized_index.reshape(-1, 1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    segmented_image = labels.reshape(normalized_index.shape)
    desert_cluster = np.argmax(centers)
    desert_regions = (segmented_image == desert_cluster).astype(np.uint8) * 255
    return desert_regions


def plotting(array_to_plot, title):
    """Vizualizare a unui array cu Matplotlib."""
    plt.imshow(array_to_plot, cmap='gray')
    plt.title(title)
    plt.show()


def pixel_count(array_to_count):
    """Numarul de pixeli albi(cei ce reprezinta nisipul din desert) dintr-un array."""
    white_pixels = np.sum(array_to_count == 255)

    return int(white_pixels)

results = []

def unzip_safe(zip_path, output_root):
    """Extract o arhiva ``.SAFE`` intr-un fisier ``output_root`` si returneaza path-ul extractat."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_ref.extractall(temp_dir)

            for entry in os.listdir(temp_dir):
                if entry.endswith(".SAFE"):
                    extracted_path = os.path.join(temp_dir, entry)
                    final_path = os.path.join(output_root, entry)

                    if os.path.exists(final_path):
                        shutil.rmtree(final_path)

                    shutil.move(extracted_path, final_path)
                    print(f"Unzipped and moved: {zip_path} → {final_path}")
                    return final_path

            raise FileNotFoundError("No .SAFE folder found in the ZIP archive.")


def find_metadata_file(root_folder):
    """Localizeaza fisierul de metadate din ``root_folder`` si returneaza path-ul."""
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.startswith('MTD_MSIL') and file.endswith('.xml'):
                return os.path.join(root, file)
    raise FileNotFoundError("Metadata file not found.")

def parse_metadata(metadata_path):
    """Returneaza achizita ``datetime`` obtinuta din  metadata file."""
    try:
        tree = ET.parse(metadata_path)
        root = tree.getroot()

        ns_match = root.tag[root.tag.find("{")+1:root.tag.find("}")]
        ns = {'n1': ns_match}

        start_elem = root.find('.//n1:PRODUCT_START_TIME', ns)
        if start_elem is not None and start_elem.text:
            return datetime.strptime(start_elem.text, "%Y-%m-%dT%H:%M:%S.%fZ")

        safe_folder = os.path.dirname(metadata_path)
        basename = os.path.basename(safe_folder)
        date_str = basename.split('_')[2]  #  20161031T092122
        year = int(date_str[:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        return datetime(year, month, day)

    except Exception as e:
        print(f" Error parsing  at {metadata_path}: {e}")


def find_band_paths(root_folder):
    """Mapeaza numele benzilor in file paths intr-un folder cu imagini .jp2."""
    band_paths = {
        'B02': None, 'B03': None, 'B04': None,
        'B8A': None, 'B11': None, 'B12': None, 'B08': None
    }
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.jp2'):
                for band in band_paths.keys():
                    if f'_{band}_' in file:
                        band_paths[band] = os.path.join(root, file)
    if None in band_paths.values():
        missing = [k for k, v in band_paths.items() if v is None]
        raise ValueError(f"Missing bands: {missing}")
    return band_paths

def process_sentinel_product(safe_folder, user_defined_threshold):
    """Proceseaza un singur fisier de tipul ``.SAFE`` si updateaza lista globala ``results``."""
    try:
        metadata_path = find_metadata_file(safe_folder)
        acquisition_date = parse_metadata(metadata_path)
    except FileNotFoundError as e:
        raise RuntimeError(f"Metadata not found in {safe_folder}") from e

    year = acquisition_date.year

    try:
        band_paths = find_band_paths(safe_folder)
    except ValueError as e:
        raise RuntimeError(f"Band paths incomplete in {safe_folder}: {e}") from e

    blue, green, red, swir1, swir2, nir, swir = initialize_bands(
        band_paths['B02'], band_paths['B03'], band_paths['B04'],
        band_paths['B11'], band_paths['B12'], band_paths['B08'],
        band_paths['B8A']
    )


    nsi   = compute_nsi(green, red, swir1)
    ndesi = compute_ndesi(blue, red, swir1, swir2)

    # normalize pentru k-means
    norm_nsi   = normalize_arrays(nsi)
    norm_ndesi = normalize_arrays(ndesi)

    # threshold definit de utilizator
    ud_nsi   = pixel_count(create_binary_image_user_defined_threshold(nsi,   user_defined_threshold))
    ud_ndesi = pixel_count(create_binary_image_user_defined_threshold(ndesi, user_defined_threshold))

    results.append({
        'year': year,
        # NSI
        'nsi_mean':          pixel_count(create_binary_image_mean_threshold(nsi)),
        'nsi_otsu':          pixel_count(create_binary_image_otsu_threshold(nsi)),
        'nsi_kmeans_random': pixel_count(kmeans_clustering_random_centers(norm_nsi,   2)),
        'nsi_kmeans_pp':     pixel_count(kmeans_clustering_pp_centers(    norm_nsi,   2)),
        'nsi_user_defined':  ud_nsi,
        # NDESI
        'ndesi_mean':          pixel_count(create_binary_image_mean_threshold(ndesi)),
        'ndesi_otsu':          pixel_count(create_binary_image_otsu_threshold(ndesi)),
        'ndesi_kmeans_random': pixel_count(kmeans_clustering_random_centers(norm_ndesi, 2)),
        'ndesi_kmeans_pp':     pixel_count(kmeans_clustering_pp_centers(    norm_ndesi, 2)),
        'ndesi_user_defined':  ud_ndesi,
    })

    print(f"Processed {safe_folder}: Year {year}")

def process_folder(folder_path, user_defined_threshold):
    """Itereaza prin fiecare subfolder din ``folder_path`` si proceseaza fiecare ``SAFE`` intr-un folder."""
    print(f"Walking folder: {folder_path}")
    for file_name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file_name)

        # --- ZIP handling temporarily disabled ---
        # if file_name.lower().endswith('.safe.zip'):
        #     print(f"   Found zipped SAFE: {file_name}")
        #     try:
        #         extracted_path = unzip_safe(full_path, folder_path)
        #         process_sentinel_product(extracted_path, user_defined_threshold)
        #     except Exception as e:
        #         print(f"  ERROR while processing {file_name}: {e}")

        # se ocupa de foldere deja extracted
        if file_name.lower().endswith('.safe') and os.path.isdir(full_path):
            print(f"   Found extracted SAFE folder: {file_name}")
            try:
                process_sentinel_product(full_path, user_defined_threshold)
            except Exception as e:
                print(f"     Error trying to process {file_name}: {e}")

        else:
            print(f"   Skipped: {file_name} not .SAFE")

    print("\nFinal Results contents:")
    for entry in results:
        print(
            f"Year: {entry['year']}, "
            f"NSI_mean: {entry['nsi_mean']}, NSI_otsu: {entry['nsi_otsu']}, "
            f"NSI_km_pp: {entry['nsi_kmeans_pp']}, NSI_km_rand: {entry['nsi_kmeans_random']}, NSI_ud: {entry['nsi_user_defined']}; "
            f"NDESI_mean: {entry['ndesi_mean']}, NDESI_otsu: {entry['ndesi_otsu']}, "
            f"NDESI_km_pp: {entry['ndesi_kmeans_pp']}, NDESI_km_rand: {entry['ndesi_kmeans_random']}, NDESI_ud: {entry['ndesi_user_defined']}"
        )



def export_results(results, output_folder):
    """Exporteaza lista ``results`` in fisiere format CSV si JSON."""
    os.makedirs(output_folder, exist_ok=True)

    csv_path = os.path.join(output_folder, 'results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=(list(results[0].keys()) if results else []))
        writer.writeheader()
        writer.writerows(results)

    json_path = os.path.join(output_folder, 'results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    print(f"\nResults exported to:\n- {csv_path}\n- {json_path}")
