import cv2
import numpy as np
import matplotlib

matplotlib.use('QtAgg')
import matplotlib.pyplot as plt


def initialize_bands(blue_band_path, green_band_path, red_band_path, swir1_band_path, swir2_band_path, nir_band_path):
    # Load images using OpenCV
    blue_band = cv2.imread(blue_band_path, cv2.IMREAD_GRAYSCALE)
    red_band = cv2.imread(red_band_path, cv2.IMREAD_GRAYSCALE)
    swir1_band = cv2.imread(swir1_band_path, cv2.IMREAD_GRAYSCALE)
    swir2_band = cv2.imread(swir2_band_path, cv2.IMREAD_GRAYSCALE)
    green_band = cv2.imread(green_band_path, cv2.IMREAD_GRAYSCALE)
    nir_band = cv2.imread(nir_band_path, cv2.IMREAD_GRAYSCALE)

    # Convert bands to float for computation
    blue_array = blue_band.astype(np.float32)
    red_array = red_band.astype(np.float32)
    green_array = green_band.astype(np.float32)
    swir1_array = swir1_band.astype(np.float32)
    swir2_array = swir2_band.astype(np.float32)
    nir_array = nir_band.astype(np.float32)

    # Avoid division issues by setting any zero or negative values to a small positive number
    blue_array[blue_array <= 0] = 0.1
    red_array[red_array <= 0] = 0.1
    swir1_array[swir1_array <= 0] = 0.1
    swir2_array[swir2_array <= 0] = 0.1
    green_array[green_array <= 0] = 0.1
    nir_array[nir_array <= 0] = 0.1
    return blue_array, red_array, swir1_array, swir2_array, green_array, nir_array


def compute_nsi(green_array, red_array, swir1_array):
    nsi_index = (green_array + red_array) / np.log(swir1_array)
    nsi_index = np.nan_to_num(nsi_index, nan=0.1, posinf=0.1, neginf=0.1)

    return nsi_index


def compute_ndesi(blue_array, red_array, swir1_array, swir2_array):
    ndesi = ((blue_array - red_array) * (swir1_array - swir2_array)) / \
            ((blue_array + red_array) * (swir1_array + swir2_array))
    ndesi = np.nan_to_num(ndesi, nan=0.1, posinf=0.1, neginf=0.1)

    ndesi = np.clip(ndesi, -2, 2)

    return ndesi


#Mask for water detection
def compute_ndwi(green_array, nir_array):
    ndwi_index = (green_array - nir_array) / (green_array + nir_array)
    ndwi_array = np.where(ndwi_index > 0, 1, 0)
    return ndwi_array


def ndesi_minus_ndwi(ndwi_array, binary_ndesi):
    final_mask = np.where(ndwi_array == 0, 1, binary_ndesi)
    return final_mask


def normalize_arrays(index_array):
    normalized_index = ((index_array - np.min(index_array)) / (np.max(index_array) - np.min(index_array)) * 255).astype(
        np.uint8)

    return normalized_index


def create_binary_image_user_defined_threshold(array_index, threshold):
    # Apply a threshold to get a binary mask: values above the threshold are 1 (black), others are 0 (white)
    binary_mask = np.where(ndesi_index > threshold, 1, 0)

    # Invert the binary mask so that 0 becomes white and 1 becomes black
    binary_mask_inverted = 1 - binary_mask

    return binary_mask_inverted.astype(np.uint8) * 255


def create_binary_image_mean_threshold(array_index):
    # Calculate the mean value of NDESI
    mean_value = np.mean(ndesi_index)

    # Apply the threshold based on the mean value
    binary_mask = np.where(ndesi_index > mean_value, 1, 0)

    # Invert the binary mask: 0 becomes white, 1 becomes black
    binary_mask_inverted = 1 - binary_mask

    return binary_mask_inverted.astype(np.uint8) * 255


def kmeans_clustering_pp_centers(normalized_index, n_clusters):
    pixels = normalized_index.reshape(-1, 1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, n_clusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    segmented_image = labels.reshape(normalized_index.shape)
    # Identify the cluster corresponding to desert regions
    # Assuming the desert regions are the brighter cluster
    desert_cluster = np.argmax(centers)
    desert_regions = (segmented_image == desert_cluster).astype(np.uint8) * 255
    return desert_regions


def kmeans_clustering_random_centers(normalized_index, n_clusters):
    pixels = normalized_index.reshape(-1, 1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    segmented_image = labels.reshape(normalized_index.shape)
    # Identify the cluster corresponding to desert regions
    # Assuming the desert regions are the brighter cluster
    desert_cluster = np.argmax(centers)
    desert_regions = (segmented_image == desert_cluster).astype(np.uint8) * 255
    return desert_regions


def plotting(array_to_plot, title):
    plt.imshow(array_to_plot, cmap='gray')
    plt.title(title)
    plt.show()


def pixel_count(array_to_count):
    # Count the number of black pixels (value 1)
    black_pixels = np.sum(array_to_count == 0)

    return black_pixels


# Paths to the necessary bands
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
