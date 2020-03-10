from PyQt5 import QtWidgets, QtSerialPort, QtCore


class COMSelectionDialog(QtWidgets.QDialog):
    """
    This class is a dialog window asking the user for COM port
    """

    def __init__(self, parent=None):
        super(COMSelectionDialog, self).__init__(parent)

        # add combo box
        self.portname_comboBox_comport1 = QtWidgets.QComboBox()

        # list all available com port in the combo boxes
        comports = QtSerialPort.QSerialPortInfo.availablePorts()
        for info in comports:
            self.portname_comboBox_comport1.addItem(info.portName())

        self.portname_comboBox_comport1.setCurrentIndex(0)

        # add a OK button to validate
        buttonbox = QtWidgets.QDialogButtonBox()
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setStandardButtons(QtWidgets.QDialogButtonBox.Ok)
        buttonbox.accepted.connect(self.accept)

        # some GUI
        lay = QtWidgets.QFormLayout(self)
        lay.addWidget(QtWidgets.QLabel("Please select COM port for HVPS/Switchboard"))
        lay.addRow("Switchboard port name:", self.portname_comboBox_comport1)
        lay.addRow(buttonbox)
        self.setGeometry(100, 100, 500, 120)

    def get_results(self):
        # return values of fields when OK
        return self.portname_comboBox_comport1.currentText()
