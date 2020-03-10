from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtWidgets import QWidget


class CameraImagePanel(QWidget):
    """
    This class is a panel showing six images in a 2-by-3 grid
    """

    def __init__(self, parent=None):
        QWidget.__init__(self, parent=parent)


class ImagePanel(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent=parent)
        self.p = QPixmap()

    def setPixmap(self, p):
        self.p = p
        self.update()

    def paintEvent(self, event):
        if not self.p.isNull():
            painter = QPainter(self)
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            rect = self.rect()
            resized_image = self.p.scaled(rect.width(), rect.height(), Qt.KeepAspectRatio)
            painter.drawPixmap(rect, resized_image)
