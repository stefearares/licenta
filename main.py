# Sys processes CLI and sends them to the QApplication
import sys
from PySide6.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton, QLabel, QLineEdit, QFileDialog, \
    QMessageBox
from button_holder import ButtonHolder
from algoritmi_licenta import *


app = QApplication(sys.argv)
window = ButtonHolder()


window.show()

app.exec()

blue_band_path = "C:\\Users\\rares\\OneDrive\\Desktop\\Sentinel-2\\B02-Dolj.jpg"  # Blue band
red_band_path = "C:\\Users\\rares\\OneDrive\\Desktop\\Sentinel-2\\B04-Dolj.jpg"  # Red band (or VRE if available)
swir1_band_path = "C:\\Users\\rares\\OneDrive\\Desktop\\Sentinel-2\\B11-Dolj.jpg"  # SWIR1 band
swir2_band_path = "C:\\Users\\rares\\OneDrive\\Desktop\\Sentinel-2\\B12-Dolj.jpg"  # SWIR2 band
green_band_path = "C:\\Users\\rares\\OneDrive\\Desktop\\Sentinel-2\\B03-Dolj.jpg"
nir_band_path = "C:\\Users\\rares\\OneDrive\\Desktop\\Sentinel-2\\B08-Dolj.jpg"
blue_array, red_array, swir1_array, swir2_array, green_array, nir_array = initialize_bands(blue_band_path,
                                                                                           green_band_path,
                                                                                           red_band_path,
                                                                                           swir1_band_path,
                                                                                           swir2_band_path,
                                                                                           nir_band_path)
#nsi_index = compute_nsi(green_array, red_array, swir1_array)
ndesi_index = compute_ndesi(blue_array, red_array, swir1_array, swir2_array)
#normalized_nsi = normalize_arrays(nsi_index)
normalized_ndesi = normalize_arrays(ndesi_index)
binary_ndesi = create_binary_image_mean_threshold(ndesi_index)
plotting(binary_ndesi, "Binary NDESI")
desert_mask = kmeans_clustering_random_centers(normalized_ndesi, n_clusters=2)
desert_mask2 = kmeans_clustering_pp_centers(normalized_ndesi, n_clusters=2)
plotting(desert_mask, "Desert Regions Detected via K-Means random")
plotting(desert_mask2, "Desert Regions Detected via K-Means pp")
print("Mean threshold", pixel_count(binary_ndesi))
print("K random threshold", pixel_count(desert_mask))
print("K pp threshold", pixel_count(desert_mask2))
