import os
from PySide6.QtWidgets import QWidget, QTableWidget, QHeaderView, QTableWidgetItem, QHBoxLayout, QPushButton, \
    QVBoxLayout, QLineEdit, QFormLayout, QMessageBox
from algoritmi_licenta import *
from dialog_box import DialogBox


class Widget(QWidget):
    def __init__(self):
        super().__init__()

        # Left (Table)
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Year", "Mean Pixels", "NSI", "NDESI", "Threshold"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # Right (Controls)
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
        self.right.addWidget(self.export)
        # Main Layout
        self.layout = QHBoxLayout(self)
        self.layout.addWidget(self.table)
        self.layout.addLayout(self.right)

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
        print(band_paths.get('B02'))

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

    def button_clicked(self):
        print("Button clicked!")

        dlg = DialogBox()
        if dlg.exec():
            year = dlg.year_input.text().strip()  # Get year from dialog
            threshold = dlg.manual_threshold.text().strip()
            directory = dlg.dir_label.text().strip()  # Get directory path

            if (year and directory != "No directory selected") and (int(year) > 1900) and (float(threshold) > 0):
                # Check the directory for images and load the bands
                band_arrays = self.check_and_load_images(directory)

                if band_arrays:
                    blue_array,red_array, swir1_array, swir2_array,green_array,nir_array, swir_array = band_arrays
                    # Now you can compute the indices, e.g., NSI, NDESI
                    nsi_index = compute_nsi(green_array, red_array, swir1_array)
                    ndesi_index = compute_ndesi(blue_array, red_array, swir1_array, swir2_array)
                    binary_ndesi= create_binary_image_mean_threshold(ndesi_index)
                    print("Mean threshold", pixel_count(binary_ndesi))
                    # Insert new row into the table
                    row_position = self.table.rowCount()
                    self.table.insertRow(row_position)

                    # Set the year in the first column
                    self.table.setItem(row_position, 0, QTableWidgetItem(year))
                    self.table.setItem(row_position, 4, QTableWidgetItem(threshold))
                    print(f"Added Year: {year} | Threshold: {threshold}| Directory: {directory}")
                else:
                    print("Error loading images from the directory.")
            else:
                print("Invalid input: Year or directory missing/ invalid input")
        else:
            print("Cancel!")


    def delete_selected_row(self):
        """Deletes the selected rows in the table with a confirmation dialog"""
        selected_rows = self.table.selectionModel().selectedRows()

        if selected_rows:
            # Show a confirmation dialog
            reply = QMessageBox.question(self, 'Confirm Delete',
                                         'Are you sure you want to delete the selected row(s)?',
                                         QMessageBox.Yes | QMessageBox.No,
                                         QMessageBox.No)

            if reply == QMessageBox.Yes:
                for row in reversed(selected_rows):  # Remove rows in reverse order to avoid shifting rows
                    self.table.removeRow(row.row())
                print("Selected rows deleted!")
            else:
                print("Delete action canceled.")
        else:
            print("No row selected!")

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
