import sys

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QLabel

from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QDialogButtonBox, QFileDialog


class DialogBox(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Add a New Year")

        # Year Input Field
        self.year_input = QLineEdit()
        self.year_input.setPlaceholderText("Enter Year")

        # Year Input Field
        self.manual_threshold = QLineEdit()
        self.manual_threshold.setPlaceholderText("Enter Threshold")

        # Select Directory Button
        self.dir_button = QPushButton("Select Directory")
        self.dir_button.clicked.connect(self.select_directory)

        # Label to show selected directory
        self.dir_label = QLabel("No directory selected")

        # OK / Cancel Buttons
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        # Layout
        self.layout = QVBoxLayout()

        self.layout.addWidget(self.year_input)
        self.layout.addWidget(self.manual_threshold)
        self.layout.addWidget(self.dir_button)
        self.layout.addWidget(self.dir_label)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

    def select_directory(self):
        """Opens a file dialog to select a directory and updates the label."""
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.dir_label.setText(directory)
