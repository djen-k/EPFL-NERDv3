import PyQt5.QtGui
import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap


def conv_Qt(rawImage, dimensions=None):
    if dimensions is None:
        dimensions = rawImage.shape[0:2]
    # Convert BGR to RGB
    rgbImage = cv2.cvtColor(rawImage, cv2.COLOR_BGR2RGB)

    # Convert to QT format because: read the doc, I've no idea
    convert_to_qt_format = PyQt5.QtGui.QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0],
                                              PyQt5.QtGui.QImage.Format_RGB888)
    convert_to_qt_format = PyQt5.QtGui.QPixmap.fromImage(convert_to_qt_format)

    pixmap = QPixmap(convert_to_qt_format)
    resized_image = pixmap.scaled(dimensions[0], dimensions[1], Qt.KeepAspectRatio, Qt.SmoothTransformation)

    return resized_image
