from PySide6.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton, QLabel, QLineEdit, QFileDialog, \
    QMessageBox


def button_pressed(button):
    print('I was clicked! checked: ', button)


class ButtonHolder(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Button Holder App')
        button = QPushButton('Press Me!')
        button.setCheckable(True)
        button.clicked.connect(button_pressed)
        self.setCentralWidget(button)
