# Sys processes CLI and sends them to the QApplication
import sys
from PySide6.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton, QLabel, QLineEdit, QFileDialog, \
    QMessageBox
from button_holder import ButtonHolder


app = QApplication(sys.argv)
window = ButtonHolder()


window.show()

app.exec()
