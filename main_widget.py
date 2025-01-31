import gc
import os

from PySide6.QtWidgets import QWidget, QTableWidget, QHeaderView, QTableWidgetItem, QHBoxLayout, QPushButton, \
    QVBoxLayout, QLineEdit, QFormLayout, QMessageBox, QLabel
from algoritmi_licenta import *
from dialog_box import DialogBox


class Widget(QWidget):
    def __init__(self):
        super().__init__()

        # Left (Tables for NSI and NDESI)
        self.table_layout = QVBoxLayout()

        self.nsi_title = QLabel("NSI Table")
        self.table_layout.addWidget(self.nsi_title)

        self.table_nsi = QTableWidget()
        self.table_nsi.setColumnCount(6)
        self.table_nsi.setHorizontalHeaderLabels(["Year", "Mean", "K++", "K-rand", "Manual", "Threshold"])
        self.table_nsi.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_layout.addWidget(self.table_nsi)

        self.ndesi_title = QLabel("NDESI Table")
        self.table_layout.addWidget(self.ndesi_title)

        self.table_ndesi = QTableWidget()
        self.table_ndesi.setColumnCount(6)
        self.table_ndesi.setHorizontalHeaderLabels(["Year", "Mean", "K++", "K-rand", "Manual", "Threshold"])
        self.table_ndesi.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_layout.addWidget(self.table_ndesi)

        # Right (Controls & Plotting)
        self.year_input = QLineEdit()
        self.manual_threshold = QLineEdit()
        self.future_year = QLineEdit()
        self.future_year.setPlaceholderText("5")
        self.directory_button = QPushButton("Add New Year")
        self.delete_button = QPushButton("Delete Selected Row")
        self.export = QPushButton("Export Data")

        # Connect Button Click
        self.directory_button.clicked.connect(self.button_clicked)
        self.delete_button.clicked.connect(self.delete_selected_row)

        # Text Field Layout
        form_layout = QFormLayout()
        form_layout.addRow("No. years forecast:", self.future_year)

        # Buttons Layout (Current & Future)
        buttons_layout = QHBoxLayout()
        self.current_button = QPushButton("Current")
        self.future_button = QPushButton("Future")
        self.future_button.setCheckable(True)
        self.current_button.setCheckable(True)
        self.current_button.clicked.connect(self.toggle_buttons)
        self.future_button.clicked.connect(self.toggle_buttons)
        buttons_layout.addWidget(self.current_button)
        buttons_layout.addWidget(self.future_button)

        # Right Layout (Controls)
        self.right = QVBoxLayout()
        self.right.addLayout(form_layout)  # Add text fields
        self.right.addLayout(buttons_layout)  # Add Current & Future buttons
        self.right.addWidget(self.directory_button)  # "Add New Year" button
        self.right.addWidget(self.delete_button)  # Delete Selected Row button
        self.right.addWidget(self.export)  # Export Data button

        # Main Layout (Left and Right sides)
        self.layout = QHBoxLayout(self)
        self.layout.addLayout(self.table_layout)  # Add vertical table layout on the left
        self.layout.addLayout(self.right)  # Add controls and plotting on the right

    def check_and_load_images(self, directory):
        required_bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12', 'SWIR']
        band_paths = {}

        # Check if directory contains only image files
        image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not image_files:
            print(f"No image files found in {directory}")
            return None  # No image files found

        # Check if all required bands are in the filenames
        for band in required_bands:
            band_found = False
            for image_file in image_files:
                if band in image_file:  # Check if the band is in the filename
                    band_paths[band] = os.path.join(directory, image_file)  # Store the full path for that band
                    band_found = True
                    break  # Stop once we find the band in the directory
            if not band_found:
                print(f"Missing band {band} in directory {directory}")
                return None  # Missing a required band

        # Load the images for the bands
        blue_band = band_paths.get('B02')
        green_band = band_paths.get('B03')
        red_band = band_paths.get('B04')
        nir_band = band_paths.get('B08')
        swir1_band = band_paths.get('B11')
        swir2_band = band_paths.get('B12')
        swir_band = band_paths.get('SWIR')

        if None in [blue_band, green_band, red_band, nir_band, swir1_band, swir2_band, swir_band]:
            print("One or more required bands could not be loaded.")
            return None

        blue_array, red_array, swir1_array, swir2_array, green_array, nir_array, swir_array = initialize_bands(
            blue_band,
            green_band,
            red_band,
            swir1_band,
            swir2_band,
            nir_band, swir_band)
        return blue_array, red_array, swir1_array, swir2_array, green_array, nir_array, swir_array

    import time

    def button_clicked(self):
        dlg = DialogBox()
        if dlg.exec():
            year = dlg.year_input.text().strip()
            threshold = dlg.manual_threshold.text().strip()
            directory = dlg.dir_label.text().strip()

            if (year and directory != "No directory selected") and (int(year) > 1900) and (float(threshold) > 0):
                band_arrays = self.check_and_load_images(directory)

                if band_arrays:
                    blue_array, red_array, swir1_array, swir2_array, green_array, nir_array, swir_array = band_arrays
                    for i in range(3):

                        nsi_index = compute_nsi(green_array, red_array, swir1_array)
                        binary_nsi = create_binary_image_mean_threshold(nsi_index)
                        normalized_nsi = normalize_arrays(nsi_index)
                        desert_mask_nsi_1 = kmeans_clustering_random_centers(normalized_nsi, n_clusters=2)
                        desert_mask_nsi_2 = kmeans_clustering_pp_centers(normalized_nsi, n_clusters=2)
                        user_defined_nsi = create_binary_image_user_defined_threshold(normalized_nsi, int(threshold))
                        otsu_nsi = create_binary_image_otsu_threshold(nsi_index)

                        ndesi_index = compute_ndesi(blue_array, red_array, swir1_array, swir2_array)
                        binary_ndesi = create_binary_image_mean_threshold(ndesi_index)
                        normalized_ndesi = normalize_arrays(ndesi_index)
                        user_defined_ndesi = create_binary_image_user_defined_threshold(normalized_ndesi, int(threshold))
                        otsu_ndesi = create_binary_image_otsu_threshold(ndesi_index)
                        desert_mask_ndesi_1 = kmeans_clustering_random_centers(normalized_ndesi, n_clusters=2)
                        desert_mask_ndesi_2 = kmeans_clustering_pp_centers(normalized_ndesi, n_clusters=2)


                    # Insert into NSI Table
                    row_position_nsi = self.table_nsi.rowCount()
                    self.table_nsi.insertRow(row_position_nsi)
                    self.table_nsi.setItem(row_position_nsi, 0, QTableWidgetItem(year))
                    self.table_nsi.setItem(row_position_nsi, 1, QTableWidgetItem(str(int(pixel_count(binary_nsi)))))
                    self.table_nsi.setItem(row_position_nsi, 2, QTableWidgetItem(str(int(pixel_count(desert_mask_nsi_2)))))
                    self.table_nsi.setItem(row_position_nsi, 3, QTableWidgetItem(str(int(pixel_count(desert_mask_nsi_1)))))
                    self.table_nsi.setItem(row_position_nsi, 4, QTableWidgetItem(str(int(pixel_count(user_defined_nsi)))))
                    self.table_nsi.setItem(row_position_nsi, 5, QTableWidgetItem(threshold))

                    gc.collect()

                    # Insert into NDESI Table
                    row_position_ndesi = self.table_ndesi.rowCount()
                    self.table_ndesi.insertRow(row_position_ndesi)
                    self.table_ndesi.setItem(row_position_ndesi, 0, QTableWidgetItem(year))
                    self.table_ndesi.setItem(row_position_ndesi, 1, QTableWidgetItem(str(int(pixel_count(binary_ndesi)))))
                    self.table_ndesi.setItem(row_position_ndesi, 2, QTableWidgetItem(str(int(pixel_count(desert_mask_ndesi_2)))))
                    self.table_ndesi.setItem(row_position_ndesi, 3, QTableWidgetItem(str(int(pixel_count(desert_mask_ndesi_1)))))
                    self.table_ndesi.setItem(row_position_ndesi, 4, QTableWidgetItem(str(int(pixel_count(user_defined_ndesi)))))
                    self.table_ndesi.setItem(row_position_ndesi, 5, QTableWidgetItem(threshold))

                    print(f"Added Year: {year} | Threshold: {threshold} | Directory: {directory}")
                else:
                    print("Error loading images from the directory.")
            else:
                print("Invalid input: Year or directory missing/ invalid input")
        else:
            print("Cancel!")

    def delete_selected_row(self):
        """Deletes the selected rows in both tables"""
        selected_rows_nsi = self.table_nsi.selectionModel().selectedRows()
        selected_rows_ndesi = self.table_ndesi.selectionModel().selectedRows()

        if selected_rows_nsi:
            for row in reversed(selected_rows_nsi):
                self.table_nsi.removeRow(row.row())

        if selected_rows_ndesi:
            for row in reversed(selected_rows_ndesi):
                self.table_ndesi.removeRow(row.row())

        print("Selected rows deleted!")

    def toggle_buttons(self):
        sender = self.sender()

        forecast_value = self.future_year.text().strip()

        if forecast_value.isdigit() and int(forecast_value) > 1:
            if sender == self.current_button:
                self.future_button.setChecked(False)
            elif sender == self.future_button:
                self.current_button.setChecked(False)
        else:
            if sender == self.future_button:
                self.future_button.setChecked(False)
                print("The forecast value must be a number greater than 1 to select 'Future'.")
