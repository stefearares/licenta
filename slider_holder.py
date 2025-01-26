from PySide6.QtGui import Qt
from PySide6.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton, QLabel, QLineEdit, QFileDialog, \
    QMessageBox, QSlider

class SliderHolder(QSlider):
    def __init__(self):
        super().__init__()
        self.setOrientation(Qt.Horizontal)

