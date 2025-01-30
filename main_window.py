from PySide6.QtCore import Slot
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QMainWindow


class MainWindow(QMainWindow):
    def __init__(self,widget):
        super().__init__()
        self.setWindowTitle("Desert Analysis")

        self.menu = self.menuBar()
        self.file_menu = self.menu.addMenu("File")

        exit_action = self.file_menu.addAction("Exit", self.close)

        self.setCentralWidget(widget)

        self.resize(800, 600)