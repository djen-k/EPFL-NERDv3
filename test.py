import logging

import cv2
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout
from duallog import duallog


class App(QMainWindow):
    """
    Main Window: root
    """

    def __init__(self, args):
        print("Hello App")
        super().__init__()
        self.args = args
        self.title = 'PowerNERD'
        self.left = 0
        self.top = 0
        self.width = 2000
        self.height = 800
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.mainW = MainWindow(self, args)
        self.setCentralWidget(self.mainW)
        self.show()

    def closeEvent(self, event):
        """
        Handle user closing window
        """
        self.mainW.close()


class MainWindow(QWidget):
    """
    THE main window widget. Includes everything else
    """

    def __init__(self, parent, args):
        super(QWidget, self).__init__(parent)

        self.args = args

        # main window layout
        self.layout = QVBoxLayout(self)


if __name__ == '__main__':
    print("Hello world")
    duallog.setup("logs")
    logging.info("Started application")

    IMG_NOT_AVAILABLE = cv2.imread("res/images/no_image.png")
    IMG_WAITING = cv2.imread("res/images/waiting.jpeg")
    cv2.imshow("not", IMG_NOT_AVAILABLE)
    cv2.imshow("wait", IMG_WAITING)
    cv2.waitKey(0)
