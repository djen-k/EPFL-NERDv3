import logging
import math
import os
import time

import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from src.gui import QtImageTools, Screen
from src.hvps.Switchboard import Switchboard
from src.image_processing import ImageCapture, StrainDetection
from src.measurement.keithley import DAQ6510


class SetupDialog(QtWidgets.QDialog):
    """
    This class is a dialog window asking the user to select a COM port and assign a camera to each DEA
    """

    def __init__(self, defaults=None, parent=None):
        super(SetupDialog, self).__init__(parent)
        # Create logger
        self.logging = logging.getLogger("SetupDialog")
        self._startup = True  # flag to indicate that the dialog is still loading

        if defaults is not None:
            self._defaults = defaults
        else:
            self._defaults = {}

        n_deas = 6  # not accepting anything else at the moment
        self._n_deas = n_deas
        n_cols = 3

        self.com_ports = []
        self.daq_ids = []

        # self._preview_img_size = [640, 360]  # [720, 405]  # [800, 450]
        img_size = [1920, 1080]
        self._preview_img_size = Screen.get_max_size_on_screen(img_size, (2, 3), (100, 800))
        self._adjustment_view_size = Screen.get_max_size_on_screen(img_size, margin=(20, 100))

        self.selection_view_mouse_click = None

        # register callback
        # image_capture.set_new_image_callback(self.updateImage)
        # image_capture.set_new_set_callback(self.updateImages)
        self._image_capture = ImageCapture.SharedInstance
        self._strain_detector = None  # will be set when a reference is recorded
        self.n_cams = 0  # we don't know of any available cameras yet

        self._hvps = Switchboard()  # create switchboard instance. not connected yet
        self._daq = DAQ6510()  # create DAQ instance. not yet connected

        # return values
        self._camorder = [-1] * n_deas
        self._image_buffer = [QtImageTools.conv_Qt(ImageCapture.ImageCapture.IMG_WAITING,
                                                   self._preview_img_size)] * n_deas

        if "cam_order" in self._defaults:
            co = self._defaults["cam_order"]
            assert len(co) == n_deas, "Must define a camera for each DEA! (use -1 if not used)"
            self._cam_order_default = co
        else:
            self._cam_order_default = range(n_deas)

        # Set window's title
        self.setWindowTitle("Camera selection dialog")

        ##############################################################################
        #  Layout
        ##############################################################################

        # panel for all instrument settings
        form_instruments = QtWidgets.QFormLayout()
        form_instruments.setLabelAlignment(Qt.AlignRight)

        # Switchboard ####################################

        # add combo box to select the COM port
        self.cbb_switchboard = QtWidgets.QComboBox()
        self.cbb_switchboard.currentTextChanged.connect(self.cbb_comport_changed)

        # button to refresh com ports
        self.btn_refresh_com = QtWidgets.QPushButton("Refresh switchboards")
        self.btn_refresh_com.clicked.connect(self.refresh_comports)

        # label to show what switchboard we're connected to
        self.lbl_switchboard_status = QtWidgets.QLabel("no switchboard")

        form_instruments.addRow("Switchboard:", self.cbb_switchboard)
        form_instruments.addRow("Status:", self.lbl_switchboard_status)
        form_instruments.addRow("", self.btn_refresh_com)

        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        form_instruments.addRow(" ", separator)

        # Multimeter ########################################

        # add combo box to select the COM port
        self.cbb_daq = QtWidgets.QComboBox()
        self.cbb_daq.currentTextChanged.connect(self.cbb_daq_changed)

        # button to refresh com ports
        self.btn_refresh_daq = QtWidgets.QPushButton("Refresh multimeters")
        self.btn_refresh_daq.clicked.connect(self.refresh_multimeters)

        # button to test resistance measurement
        self.btn_test_res = QtWidgets.QPushButton("Measure resistance")
        self.btn_test_res.clicked.connect(self.test_resistance)

        # label to show what switchboard we're connected to
        self.lbl_daq_status = QtWidgets.QLabel("no multimeter")

        form_instruments.addRow("Multimeter:", self.cbb_daq)
        form_instruments.addRow("Status:", self.lbl_daq_status)
        form_instruments.addRow("", self.btn_refresh_daq)
        form_instruments.addRow("", self.btn_test_res)

        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        form_instruments.addRow(" ", separator)

        # create voltage selector
        self.num_voltage = self.create_num_selector(0, 5000, "voltage", 1000)
        form_instruments.addRow("Test voltage [V]:", self.num_voltage)

        # toggle button to apply voltage (to check strain detection results)
        self.btn_apply_voltage = QtWidgets.QPushButton("Apply voltage now!")
        self.btn_apply_voltage.setCheckable(True)
        self.btn_apply_voltage.setEnabled(False)  # only enable if connected to an HVPS
        self.btn_apply_voltage.clicked.connect(self.btnVoltageClicked)
        form_instruments.addRow("", self.btn_apply_voltage)

        # Test parameters ########################################

        # panel for all test parameters

        form_parameters = QtWidgets.QFormLayout()
        form_parameters.setLabelAlignment(Qt.AlignRight)

        # textbox to set a title/label/description for the test
        self.test_title = QtWidgets.QLineEdit()
        form_parameters.addRow("Title:", self.test_title)
        if "title" in self._defaults:
            self.test_title.setText(self._defaults["title"])

        # checkbox to enable AC mode
        self.chk_ac = QtWidgets.QCheckBox("AC mode")
        self.chk_ac.clicked.connect(self.chkACClicked)
        form_parameters.addRow("", self.chk_ac)

        # checkbox to enable reverse polarity mode
        self.chk_reverse_polarity = QtWidgets.QCheckBox("Reverse polarity")
        self.chk_reverse_polarity.clicked.connect(self.chkRPClicked)
        form_parameters.addRow("", self.chk_reverse_polarity)
        msg = "Polarity will be reversed after each\nmeasurement in the HIGH phase."
        form_parameters.addRow("", QtWidgets.QLabel(msg))

        # create ramp step selector
        self.num_steps = self.create_num_selector(0, 100, "steps", 10)
        form_parameters.addRow("Ramp steps:", self.num_steps)

        # create step duration selector
        self.num_step_duration = self.create_num_selector(0, 10000, "step_duration_s", 10)
        form_parameters.addRow("Step duration (s):", self.num_step_duration)

        # create high duration selector
        self.num_high_duration = self.create_num_selector(0, 100000, "high_duration_min", 60)
        self.num_high_duration.valueChanged.connect(self.updateCycles)
        form_parameters.addRow("High duration (min):", self.num_high_duration)

        # create low duration selector
        self.num_low_duration = self.create_num_selector(0, 100000, "low_duration_s", 30)
        form_parameters.addRow("Low duration (s):", self.num_low_duration)

        # create measurement period selector
        self.num_measurement_period = self.create_num_selector(0, 1000, "measurement_period_s", 10)
        form_parameters.addRow("Measurement interval (s):", self.num_measurement_period)

        # create image save period selector
        self.num_save_image_period = self.create_num_selector(0, 1000, "save_image_period_min", 30)
        form_parameters.addRow("Save image interval (min):", self.num_save_image_period)

        # create AC frequency selector
        self.num_ac_frequency = self.create_num_selector(1, 1000, "ac_frequency_hz", 50)
        self.num_ac_frequency.valueChanged.connect(self.updateCycles)
        form_parameters.addRow("Cycle frequency [Hz]:", self.num_ac_frequency)

        # create measurement period selector
        self.num_ac_wait = self.create_num_selector(0, 1000, "ac_wait_before_measurement_s", 10)
        form_parameters.addRow("Pause before measurement (s):", self.num_ac_wait)

        # create number of cycles indicator
        self.lbl_cycles = QtWidgets.QLabel("")
        form_parameters.addRow("", self.lbl_cycles)

        # create grid layout to show all the images
        gridLay = QtWidgets.QGridLayout()
        self.lbl_image = []
        self.cbb_camera_select = []
        self.chk_active_DEA = []
        self.btn_adjust = []
        self.btn_select = []
        for i in range(n_deas):
            groupBox = QtWidgets.QGroupBox("DEA {}".format(i + 1), self)  # number groups/DEAs from 1 to 6
            grpLay = QtWidgets.QVBoxLayout()
            # add label to show image
            lbl = QtWidgets.QLabel()
            # lbl.setScaledContents(True)
            # lbl.setFixedSize(self._preview_img_size[0], self._preview_img_size[1])
            lbl.setPixmap(self._image_buffer[i])  # initialize with waiting image
            self.lbl_image.append(lbl)

            # add a camera  selection Combobox and fill the with camera names
            cbb = QtWidgets.QComboBox()
            cbb.addItem("Not available")
            # cb.setCurrentIndex(0)
            cbb.currentIndexChanged.connect(self.camSelIndexChanged)
            self.cbb_camera_select.append(cbb)

            # add a checkbox to enable or disable each sample
            chk = QtWidgets.QCheckBox("Active")
            chk.clicked.connect(self.chkActiveClicked)
            if "active_DEAs" in self._defaults:
                chk.setChecked(self._defaults["active_DEAs"][i] == 1)
            else:
                chk.setChecked(True)
            self.chk_active_DEA.append(chk)

            # add adjust button
            btnSel = QtWidgets.QPushButton("Select...")
            btnSel.clicked.connect(self.btnSelectClicked)
            self.btn_select.append(btnSel)

            # add adjust button
            btn = QtWidgets.QPushButton("Adjust...")
            btn.clicked.connect(self.btnAdjustClicked)
            self.btn_adjust.append(btn)

            rowLay = QtWidgets.QHBoxLayout()
            rowLay.addWidget(chk, 0, alignment=Qt.AlignLeft)
            rowLay.addWidget(cbb, 0, alignment=Qt.AlignLeft)
            rowLay.addWidget(btnSel, 0, alignment=Qt.AlignLeft)
            rowLay.addWidget(btn, 1, alignment=Qt.AlignLeft)
            # rowLay.addSpacerItem(QtWidgets.QSpacerItem(1, 1, hPol`icy=Qt.MaximumSize))

            # add image and combobox to group box layout
            grpLay.addLayout(rowLay)
            grpLay.addWidget(lbl, stretch=9)
            groupBox.setLayout(grpLay)  # apply layout to group box

            # put group in the box layout
            row = math.floor(i / n_cols)
            col = i % n_cols
            gridLay.addWidget(groupBox, row, col)

        # OK button to close the dialog and return results
        self.btnStart = QtWidgets.QPushButton("Start!")
        self.btnStart.clicked.connect(self.btnStartClicked)

        # button to take new images
        self.btn_capture = QtWidgets.QPushButton("Take new image")
        self.btn_capture.clicked.connect(self.btnCaptureClicked)

        # button to take new images
        self.chkStrain = QtWidgets.QCheckBox("Show strain")
        self.chkStrain.setEnabled(False)  # only enable once a strain reference has been set
        self.chkStrain.clicked.connect(self.btnCaptureClicked)

        # button to record strain reference
        self.btn_reference = QtWidgets.QPushButton("Record strain reference")
        self.btn_reference.clicked.connect(self.record_strain_reference)

        # create number of average images for strain ref selector
        self.num_avg_img = self.create_num_selector(1, 100, "average_images", 10)
        self.num_avg_img.setMaximumWidth(40)

        # button to load strain reference from previous test
        self.btnLoadReference = QtWidgets.QPushButton("Load reference from previous test...")
        self.btnLoadReference.clicked.connect(self.load_strain_reference)

        # button to load strain reference from previous test
        self.lblStrainRef = QtWidgets.QLabel("No strain reference set!")
        self.lblStrainRef.setStyleSheet("QLabel { color : red }")

        buttonLay = QtWidgets.QHBoxLayout()
        buttonLay.setAlignment(Qt.AlignLeft)
        buttonLay.addWidget(self.btn_capture)
        buttonLay.addWidget(self.chkStrain)
        buttonLay.addWidget(self.btn_reference)
        buttonLay.addWidget(QtWidgets.QLabel("Average images:"))
        buttonLay.addWidget(self.num_avg_img)
        buttonLay.addSpacerItem(QtWidgets.QSpacerItem(20, 1))
        buttonLay.addWidget(self.btnLoadReference)
        buttonLay.addSpacerItem(QtWidgets.QSpacerItem(20, 1))
        buttonLay.addWidget(self.lblStrainRef)
        buttonLay.addSpacerItem(QtWidgets.QSpacerItem(20, 1))
        buttonLay.addWidget(self.btnStart)

        # some GUI (layout stuff)
        # formLay = QtWidgets.QFormLayout(self)

        # create the main layout
        mainLay = QtWidgets.QVBoxLayout(self)

        # protocol schematic panel
        topLay = QtWidgets.QHBoxLayout()
        topLay.setAlignment(Qt.AlignLeft)
        # topLay.setContentsMargins(50, 5, 50, 5)

        # instrument settings panel
        grp_instruments = QtWidgets.QGroupBox("Instruments", self)
        grp_instruments.setLayout(form_instruments)
        topLay.addWidget(grp_instruments)

        # test parameters panel
        grp_parameters = QtWidgets.QGroupBox("Test parameters", self)
        grp_parameters.setLayout(form_parameters)
        topLay.addWidget(grp_parameters)

        self.schematic = QtWidgets.QLabel()
        # self.pix_schematic_DC = QtGui.QPixmap("res/images/schematic.png").scaledToHeight(300, Qt.SmoothTransformation)
        self.pix_schematic_DC = QtGui.QPixmap("res/images/NERD protocol schematic DC.png")
        self.pix_schematic_AC = QtGui.QPixmap("res/images/NERD protocol schematic AC.png")
        self.schematic.setPixmap(self.pix_schematic_DC)
        self.schematic.setAlignment(Qt.AlignTop)
        self.schematic.setContentsMargins(20, 0, 0, 0)
        topLay.addWidget(self.schematic)

        # spacer = QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        # topLay.addSpacerItem(spacer)

        # logo = QtWidgets.QLabel()
        # logo.setPixmap(QtGui.QPixmap("res/images/epfl_logo.png"))
        # logo.setAlignment(Qt.AlignTop)
        # topLay.addWidget(logo)

        mainLay.addLayout(topLay)

        # add a separator between settings and camera images
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        mainLay.addWidget(separator)

        mainLay.addLayout(buttonLay)
        mainLay.addLayout(gridLay, stretch=9)

        ##############################################################################
        #  layout done
        ##############################################################################

        self.refresh_comports()  # fill with available ports
        self.refresh_multimeters()  # fill with available multimeters

        # apply default to AC checkbox and enable or disable frequency selector accordingly
        if "ac_mode" in self._defaults:
            checked = self._defaults["ac_mode"]
        else:
            checked = False
        self.chk_ac.setChecked(checked)
        self.chkACClicked()  # enable/disableAC mode settings and show correct schematic

        # apply default to reverse polarity checkbox
        if "reverse_polarity" in self._defaults:
            checked = self._defaults["reverse_polarity"]
        else:
            checked = False
        self.chk_reverse_polarity.setChecked(checked)
        self.chkRPClicked()

        # self.setWindowFlags(Qt.Window)
        self.show()

        top_size = topLay.totalMinimumSize()
        self._preview_img_size = Screen.get_max_size_on_screen(img_size, (2, 3), margin=(50, top_size.height() + 200))

        # register callback to receive an image immediately when a camera is found
        ImageCapture.SharedInstance.set_new_image_callback(self.new_image_callback)
        self.logging.debug("Registered callback")
        self.logging.debug("Init image capture...")
        self.btnCaptureClicked()  # captures images and updates the labels
        ImageCapture.SharedInstance.set_new_image_callback(None)  # de-register so it doesn't get refreshed twice
        self.logging.debug("Unregistered callback")

        self._startup = False

    def create_num_selector(self, min_val, max_val, config_key, default_value):
        """
        Creates a QSpinBox for setting numeric values.
        :param min_val: Minimum value
        :param max_val: Maximum value
        :param config_key: Key to look up default value in the defaults dict
        :param default_value: Value to use if the key is not in the defaults dict
        :return: A QSpinBox with the given range and value
        """
        num = QtWidgets.QSpinBox()
        num.setMinimum(min_val)
        num.setMaximum(max_val)
        if config_key in self._defaults:
            value = self._defaults[config_key]
        else:
            value = default_value
        num.setValue(value)

        return num

    def updateCycles(self):
        if self.chk_ac.isChecked():
            n_cycles = self.num_high_duration.value() * 60 * self.num_ac_frequency.value()
            self.lbl_cycles.setText("{:,} cycles per high-phase".format(n_cycles))
        else:
            self.lbl_cycles.setText("")

    def btnStartClicked(self):
        # only proceed if a strain reference has been set
        if self._strain_detector is None:
            QtWidgets.QMessageBox.warning(self, "No strain reference",
                                          "To start a measurement, please set a strain reference!",
                                          QtWidgets.QMessageBox.Ok)
            return

        if self.cbb_switchboard.currentText() == "":
            QtWidgets.QMessageBox.warning(self, "No Switchboard/HVPS specified",
                                          "To start a measurement, please select a switchboard/HVPS!",
                                          QtWidgets.QMessageBox.Ok)
            return

        if self.chk_reverse_polarity.isChecked():
            res = QtWidgets.QMessageBox.warning(self, "CAUTION: Reverse polarity mode!",
                                                "CAUTION: Make sure that the setup is wired for reverse polarity mode\n"
                                                "(resistance board disconnected, current sensing via front panel)!\n"
                                                "If not wired correctly, reversing polarity "
                                                "WILL RESULT IN DAMAGE TO THE MULTIMETER!",
                                                QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel,
                                                QtWidgets.QMessageBox.Cancel)
            if res != QtWidgets.QMessageBox.Ok:
                return

        self.accept()

    def chkACClicked(self):
        if self.chk_ac.isChecked():
            self.num_ac_frequency.setEnabled(True)
            self.num_ac_wait.setEnabled(True)
            self.schematic.setPixmap(self.pix_schematic_AC)
        else:
            self.num_ac_frequency.setEnabled(False)
            self.num_ac_wait.setEnabled(False)
            self.schematic.setPixmap(self.pix_schematic_DC)
        self.updateCycles()

    def chkRPClicked(self):
        if self.chk_reverse_polarity.isChecked():
            self.btn_test_res.setText("Measure current")
        else:
            self.btn_test_res.setText("Measure resistance")

    def refresh_comports(self):
        self.cbb_switchboard.blockSignals(True)  # block signals to avoid excessive reconnecting to the switchboard
        self.cbb_switchboard.clear()  # remove all items

        # list all available com ports
        ports = self._hvps.detect()
        display_names = ["{} ({})".format(p.name, p.port) for p in ports]
        port_names = [p.port for p in ports]
        print(display_names)
        self.cbb_switchboard.addItems(display_names)

        # select the default COM port, if it is available
        if "com_port" in self._defaults and self._defaults["com_port"] in port_names:
            idx = port_names.index(self._defaults["com_port"])
            if idx != self.cbb_switchboard.currentIndex():  # only set if the index is different to avoid reconnect
                self.cbb_switchboard.setCurrentIndex(idx)
        # if no default or default not available --> use index 0 which is selected by default

        self.com_ports = port_names

        self.cbb_switchboard.blockSignals(False)  # turn signals back on
        self.cbb_comport_changed()  # call comport changed once to connect to the newly selected com port

    def cbb_comport_changed(self):
        try:
            port_idx = self.cbb_switchboard.currentIndex()
            print(port_idx)
            if port_idx < 0:
                self.lbl_switchboard_status.setText("No switchboard found!")
                self.btn_apply_voltage.setEnabled(False)
                self.btn_apply_voltage.setChecked(False)
            else:
                self._hvps.open(port_idx, reconnect_timeout=0)
                self.lbl_switchboard_status.setText("Connected")
                self.btn_apply_voltage.setEnabled(True)
        except Exception as ex:
            self.logging.debug("Could not connect to switchboard: {}".format(ex))
            self.lbl_switchboard_status.setText("Connection failed!")
            self.btn_apply_voltage.setEnabled(False)
            self.btn_apply_voltage.setChecked(False)

    def refresh_multimeters(self):
        self.cbb_daq.blockSignals(True)  # block signals to avoid excessive reconnecting
        self.cbb_daq.clear()  # remove all items
        self.cbb_daq.addItem("None")  # always add None as the first option

        # list all available DAQs
        instruments = self._daq.list_instruments()
        display_names = [res_info.alias for res_info in instruments.values()]
        self.daq_ids = list(instruments.keys())
        self.cbb_daq.addItems(display_names)

        # select the default DAQ, if it is available
        if self.daq_ids:
            if "daq_id" in self._defaults and self._defaults["daq_id"] in self.daq_ids:
                idx = self.daq_ids.index(self._defaults["daq_id"])
                if idx != self.cbb_daq.currentIndex():  # only set if the index is different to avoid reconnect
                    self.cbb_daq.setCurrentIndex(idx)
            else:
                self.cbb_daq.setCurrentIndex(1)  # set to first DAQ (index 0 is "None")

        self.cbb_daq.blockSignals(False)  # turn signals back on
        self.cbb_daq_changed()  # call DAQ changed once to connect to the newly selected com port

    def test_resistance(self):
        if self.chk_reverse_polarity.isChecked():
            res = self._daq.measure_current(front=True)
            if res is not None:
                QtWidgets.QMessageBox.information(self,
                                                  "Current measurement results",
                                                  "Measured current (nA):\n{}".format(res * 1000000000))
        else:
            res = self._daq.measure_DEA_resistance(deas=range(6))
            if res is not None:
                QtWidgets.QMessageBox.information(self,
                                                  "Resistance measurement results",
                                                  "Measured resistance (kΩ):\n{}".format(res / 1000))

    def cbb_daq_changed(self):
        daq_idx = self.cbb_daq.currentIndex()
        if daq_idx <= 0:  # "None"
            self._daq.disconnect()
            self.lbl_daq_status.setText("Disconnected")
            self._defaults["daq_id"] = "None"  # update default so refresh won't reset selection
        else:
            daq_id = self.daq_ids[daq_idx - 1]  # -1 because 0 in ComboBox is "None"
            success = self._daq.connect(daq_id)
            if success:
                self.lbl_daq_status.setText("Connected")
                self._defaults["daq_id"] = daq_id  # update default so refresh won't reset selection
            else:
                self.logging.debug("Could not connect to instrument: {}".format(daq_id))
                self.lbl_daq_status.setText("Connection failed!")

    def btnAdjustClicked(self):
        sender = self.sender()
        for i_dea in range(self._n_deas):
            if sender == self.btn_adjust[i_dea]:
                self.logging.debug("Clicked adjust button for DEA {}".format(i_dea))
                self.showAdjustmentView(i_dea)
                self.btnCaptureClicked()
                return

    def chkActiveClicked(self):

        if self._strain_detector is not None:
            sender = self.sender()
            sender_id = -1
            for i_dea in range(self._n_deas):
                if sender == self.chk_active_DEA[i_dea]:
                    sender_id = i_dea
                    break

            if sender_id == -1:  # not found. should never happen
                self.logging.error("Could not determine the sender of the checkbox event")
                return

            if self.chk_active_DEA[sender_id].isChecked():  # just been switched on -> strain reference no longer valid
                self.invalidate_strain_reference()
            else:  # was on, now been switched off  --> keep reference but remove the de-selected sample
                self._strain_detector.remove_reference(sender_id)

        self.refreshImages()

    def btnSelectClicked(self):
        sender = self.sender()
        for i_dea in range(self._n_deas):
            if sender == self.btn_select[i_dea]:
                self.logging.debug("Clicked select button for DEA {}".format(i_dea))
                selection = self.showSelectionView(i_dea)
                if selection is not None:
                    self.cbb_camera_select[i_dea].setCurrentIndex(selection + 1)  # first item is "not used"
                    self.chk_active_DEA[i_dea].setChecked(True)
                    self.chk_active_DEA[i_dea].setEnabled(True)

                self.btnCaptureClicked()
                return

    def record_strain_reference(self):
        # protect against impatient clicks
        self.btn_reference.setFixedSize(self.btn_reference.size())
        self.btn_reference.setText("Busy...")
        self.btn_reference.setEnabled(False)

        QApplication.processEvents()

        cap = self._image_capture
        images = cap.read_average_images(self.num_avg_img.value())
        order = self.getCamOrder()
        order_filt = [i for i in order if i >= 0]  # filter to remove unused cameras (-1)
        images = [images[i] for i in order_filt]  # put images in the right order
        active = self.getActiveSamples()
        active_filt = [active[i] for i in range(self._n_deas) if order[i] >= 0]  # filter to remove unused cameras (-1)
        images = [images[i] for i in range(len(images)) if active_filt[i]]  # pick only the active samples

        self._set_strain_reference(images)

        self.btn_reference.setText("Record strain reference")
        self.btn_reference.setEnabled(True)

    def _set_strain_reference(self, images):
        self._strain_detector = StrainDetection.StrainDetector()
        self._strain_detector.set_reference(images)
        self.logging.info("New strain reference has been set.")

        # self.logging.info("Double-checking that the reference is OK: Running strain detection on same set of images")
        # strain_res = self._strain_detector.get_dea_strain(images, False)[0]
        # self.logging.info("Strain result:")
        # self.logging.info("\n{}".format(strain_res))

        self.chkStrain.setEnabled(True)
        self.chkStrain.setChecked(True)

        self.lblStrainRef.setText("Strain reference OK")
        self.lblStrainRef.setStyleSheet("QLabel { color : green }")

        self.refreshImages()

    def load_strain_reference(self):
        folder = "output"
        caption = "Select the output folder of a previous NERD test"
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(self, caption, folder)

        refs = []
        for i in range(self._n_deas):
            fpath = os.path.join(dir_path, "DEA {}".format(i + 1), "Images")
            if os.path.isdir(fpath):
                try:
                    files = os.listdir(fpath)
                    for f in files:
                        if f.endswith("reference.png"):
                            f = os.path.join(fpath, f)
                            refs.append(cv2.imread(f))
                            break
                except Exception as ex:
                    self.logging.debug("Error loading reference image: {}".format(ex))

        active_samples = self.getActiveSamples()
        cams_in_use = active_samples.count(1)
        if len(refs) == cams_in_use:
            self._set_strain_reference(refs)
        else:
            QtWidgets.QMessageBox.warning(self, "Reference does not match",
                                          "The selected reference does not match the selected number of active cameras",
                                          QtWidgets.QMessageBox.Ok)

    def btnVoltageClicked(self):
        if self._hvps is not None:
            try:
                if self.btn_apply_voltage.isChecked():
                    # turn on and set voltage
                    self._hvps.set_output_on()  # set to DC mode
                    self._hvps.set_voltage_no_overshoot(self.num_voltage.value())
                    self._hvps.set_relay_auto_mode()
                    self.btn_apply_voltage.setText("Turn voltage off!")
                    self.btn_apply_voltage.setStyleSheet("QPushButton{ color: red }")
                else:
                    self._hvps.set_output_off()
                    self._hvps.set_voltage(0, block_until_reached=True)
                    self._hvps.set_relays_off()
                    self.btn_apply_voltage.setText("Apply voltage now!")
                    self.btn_apply_voltage.setStyleSheet("QPushButton{ color: black }")
            except TimeoutError:
                self.logging.warning("Unable to set voltage. Refreshing com ports to check connection.")
                self.refresh_comports()

            self.logging.debug("Waiting 1 s after voltgae change to give DEAs time to respond")
            time.sleep(1)  # wait another second for DEA to respond
            # capture new images to show strain
            self.btnCaptureClicked()
        else:
            self.btn_apply_voltage.setChecked(False)
            self.btn_apply_voltage.setEnabled(False)

    def showAdjustmentView(self, i_dea):
        i_cam = self._camorder[i_dea]
        if i_cam == -1:
            return

        ret = -1
        while ret == -1:
            try:
                img = self._image_capture.read_single_image(i_cam)
                img = cv2.resize(img, self._adjustment_view_size)
                cv2.imshow("DEA {} - Press any key to close".format(i_dea + 1), img)
                ret = cv2.waitKey(100)
                self.logging.debug("Key pressed {}".format(ret))
            except Exception as ex:
                self.logging.debug("Exception in showAdjustmentView: {}".format(ex))
                break

        cv2.destroyAllWindows()

    def selection_view_mouse_clicked(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            self.selection_view_mouse_click = (x, y)

    def showSelectionView(self, i_dea):
        selection = None
        try:
            nr = int(round(np.sqrt(self.n_cams)))
            nc = int(np.ceil(np.sqrt(self.n_cams)))
            img_shape = self._image_capture.read_single_image(0).shape
            img_size = (img_shape[1], img_shape[0])  # convert array shape (rows, cols, c), to img size (width, height)
            sel_img_size = Screen.get_max_size_on_screen(img_size, (nr, nc), (40, 100))
            sel_img_shape = (sel_img_size[1], sel_img_size[0], 3)

            wname = "Click image of DEA {} (press ESC to exit)".format(i_dea + 1)
            cv2.namedWindow(wname)
            cv2.setMouseCallback(wname, self.selection_view_mouse_clicked)
            self.selection_view_mouse_click = None
            key = 0
            while key != 27:  # loop until ESC is pressed (or a mouse click detected)
                imgs = self._image_capture.read_images()
                i = -1
                montage = None
                indices = np.zeros((nr, nc), dtype=np.int)  # grid of camera indices to pick the right one easily later
                indices -= 1  # make all indices -1 initially, then we fill in the proper value in the loop
                for r in range(nr):
                    montage_row = None
                    for c in range(nc):
                        i += 1

                        if i < self.n_cams:
                            indices[r, c] = i
                            img = cv2.resize(imgs[i], sel_img_size)
                        else:
                            img = np.zeros(sel_img_shape, dtype=np.uint8)

                        if montage_row is None:
                            montage_row = img
                        else:
                            montage_row = np.hstack((montage_row, img))
                    if montage is None:
                        montage = montage_row
                    else:
                        montage = np.vstack((montage, montage_row))

                cv2.imshow(wname, montage)
                key = cv2.waitKey(100)
                # self.logging.debug("Key pressed {}".format(ret))
                if self.selection_view_mouse_click is not None:
                    x, y = self.selection_view_mouse_click
                    c = int(np.floor(x / sel_img_size[0]))
                    r = int(np.floor(y / sel_img_size[1]))

                    selection = int(indices[r, c])
                    if selection >= 0:
                        cam_name = self._image_capture.get_camera(selection).name
                        self.logging.debug("Selected {} for DEA {}.".format(cam_name, i_dea + 1))
                        self.selection_view_mouse_click = None
                        break
        except Exception as ex:
            self.logging.error("Exception in showAdjustmentView: {}".format(ex))

        cv2.destroyAllWindows()
        return selection

    def camSelIndexChanged(self):
        if self._startup:
            return

        try:
            # get new selected index and previous index
            sender = self.sender()
            i_sender = self.cbb_camera_select.index(sender)
            sender = self.cbb_camera_select[i_sender]
            idx_cam = sender.currentIndex()
            prev_idx = self._camorder[i_sender] + 1

            # self.logging.debug("camera selection index for DEA {} changed to {}".format(src, idx))
            if idx_cam > 0:  # valid camera selected
                self.chk_active_DEA[i_sender].setEnabled(True)
                # check if another slot is using this camera (and swap if so)
                for cb in self.cbb_camera_select:
                    if cb is not sender and cb.currentIndex() == idx_cam:
                        cb.setCurrentIndex(prev_idx)  # set to the previous selection of the sender
            else:  # "not used"
                self.chk_active_DEA[i_sender].setChecked(False)
                self.chk_active_DEA[i_sender].setEnabled(False)
            self._camorder = self.getCamOrder()  # update stored cam order

            # disable strain display because if the camera selection changes, the strain reference is no longer valid
            if self._strain_detector is not None:
                self.invalidate_strain_reference()

            self.refreshImages()
        except Exception as ex:
            self.logging.error("Exception in combobox callback: {}".format(ex))

    def invalidate_strain_reference(self):
        self.chkStrain.setEnabled(False)
        self.chkStrain.setChecked(False)
        self._strain_detector = None
        self.lblStrainRef.setText("No strain reference set!")
        self.lblStrainRef.setStyleSheet("QLabel { color : red }")
        dtitle = "Strain reference no longer valid"
        dmsg = "The strain reference is no longer valid.\n" \
               "Please set a new strain reference!"
        QtWidgets.QMessageBox.information(self, dtitle, dmsg, QtWidgets.QMessageBox.Ok)

    def refreshImages(self):
        self.logging.debug("Recording new set of images")
        images = self._image_capture.get_images_from_buffer()
        order = self.getCamOrder()
        order_filt = [i for i in order if i >= 0]  # filter to remove unused cameras (-1)
        images = [images[i] for i in order_filt]  # put images in the right order
        active = self.getActiveSamples()
        active_filt = [active[i] for i in range(self._n_deas) if order[i] >= 0]  # filter to remove unused cameras (-1)

        if self.chkStrain.isChecked():
            try:
                v = self._hvps.get_current_voltage()
                sv = "{} V".format(v)
            except Exception as ex:
                self.logging.debug("Couldn't read voltage: {}".format(ex))
                sv = ""
            active_images = [images[i] for i in range(len(images)) if active_filt[i]]
            images_res = self._strain_detector.get_dea_strain(active_images, True, True, sv)[2]
            # fill in the result images for all active samples
            k = 0
            for i in range(len(images)):
                if active_filt[i]:
                    images[i] = images_res[k]
                    k += 1

        i_img = 0
        for i_dea in range(self._n_deas):
            i_cam = order[i_dea]
            if i_cam >= 0:
                self.setImage(i_dea, images[i_img])
                i_img += 1
            else:
                self.setImage(i_dea, ImageCapture.ImageCapture.IMG_NOT_AVAILABLE)

    def setImage(self, i_dea, opencv_image, img_size=None):
        if i_dea > 5:
            return  # can't show more than 6 deas for now
        label = self.lbl_image[i_dea]

        if img_size is None:
            # img_size = self._default_img_size
            img_size = (label.size().width(), label.size().height())

        if not self.chk_active_DEA[i_dea].isChecked():
            gray_img = opencv_image.copy()
            gray_img[:] = 127
            opencv_image = cv2.addWeighted(opencv_image, 0.35, gray_img, 0.65, 0.0)

        label.setPixmap(QtImageTools.conv_Qt(opencv_image, img_size))
        self.logging.debug("Set new image for DEA {}".format(i_dea + 1))
        self.repaint()

    def new_image_callback(self, image, timestamp, cam_id):
        self.setImage(int(cam_id[-1]) - 1, image)  # convert last character of name to 0-based index

    def btnCaptureClicked(self):
        # protect against impatient clicks
        self.btn_capture.setFixedSize(self.btn_capture.size())
        self.btn_capture.setText("Busy...")
        self.btn_capture.setEnabled(False)
        QApplication.processEvents()

        self.update_images(self._image_capture.read_images(), self._image_capture.get_timestamps())

        self.btn_capture.setText("Take new image")
        self.btn_capture.setEnabled(True)

    def getCamOrder(self):
        return [cb.currentIndex() - 1 for cb in self.cbb_camera_select]

    def getActiveSamples(self):
        return [int(chb.isChecked()) for chb in self.chk_active_DEA]  # save active state as 0 or 1

    def get_results(self):
        """
        Get sonfiguration for the NERD test as set in the SetupDialog
        :return: A config dict with all the parameters, as well as a strain detector with pre-set strain reference
        """
        daq_ind = self.cbb_daq.currentIndex() - 1  # -1 because first is "None"
        daq_id = self.daq_ids[daq_ind] if daq_ind >= 0 else "None"  # specifically selected no multimeter
        sb_ind = self.cbb_switchboard.currentIndex()
        sb_id = self.com_ports[sb_ind] if sb_ind >= 0 else "None"  # so it doesn't crash if none is selected
        config = {
            "com_port": sb_id,
            "daq_id": daq_id,
            "title": self.test_title.text(),
            "cam_order": self.getCamOrder(),
            "active_DEAs": self.getActiveSamples(),
            "voltage": self.num_voltage.value(),
            "steps": self.num_steps.value(),
            "step_duration_s": self.num_step_duration.value(),
            "high_duration_min": self.num_high_duration.value(),
            "low_duration_s": self.num_low_duration.value(),
            "measurement_period_s": self.num_measurement_period.value(),
            "save_image_period_min": self.num_save_image_period.value(),
            "ac_mode": self.chk_ac.isChecked(),
            "ac_frequency_hz": self.num_ac_frequency.value(),
            "ac_wait_before_measurement_s": self.num_ac_wait.value(),
            "reverse_polarity": self.chk_reverse_polarity.isChecked(),
            "average_images": self.num_avg_img.value()
        }

        return config, self._strain_detector

    def updateImage(self, image, timestamp, cam_id):

        if cam_id >= self.n_cams:  # new camera added
            self.logging.debug("new camera added: {}".format(cam_id))
            self.n_cams = cam_id + 1
            # add to all combo boxes
            cname = self._image_capture.get_camera_names()[cam_id]
            for cb in self.cbb_camera_select:
                cb.addItem(cname)

            # figure out which DEA to assign camera to
            try:
                # see if the newly found camera is defined in the default cam order
                dea_id = self._cam_order_default.index(cam_id)
                self.cbb_camera_select[dea_id].setCurrentIndex(cam_id + 1)  # should be the index of the new cam
                self.lbl_image[dea_id].setPixmap(QtImageTools.conv_Qt(image, self._preview_img_size))
            except ValueError:
                pass  # not found, so we leave target DEA index at -1
        else:
            # show the new image in each frame where the respective camera is selected (should only ever be one)
            for i in range(self._n_deas):
                i_sel = self._camorder[i]
                if i_sel == cam_id:
                    self.lbl_image[i].setPixmap(QtImageTools.conv_Qt(image, self._preview_img_size))

                # if camera index for any dea is -1 ("not used"), show the not available image
                elif i_sel == -1:
                    self.lbl_image[i].setPixmap(
                        QtImageTools.conv_Qt(ImageCapture.ImageCapture.IMG_NOT_AVAILABLE, self._preview_img_size))

    def update_images(self, images, timestamps):

        if self.n_cams != len(images):  # new camera added
            self.logging.debug("new cameras added")
            self.n_cams = len(images)
            # add to all combo boxes
            cnames = self._image_capture.get_camera_names()
            for cb in self.cbb_camera_select:
                cb.addItems(cnames)

            # assign cameras using defaults if available
            for i in range(self._n_deas):
                # see if the newly found camera is defined in the default cam order
                cam_id = self._cam_order_default[i]
                if cam_id in range(self.n_cams):
                    self._camorder[i] = cam_id
                    self.cbb_camera_select[i].setCurrentIndex(cam_id + 1)  # plus 1 because first is 'not used'

        self.refreshImages()
