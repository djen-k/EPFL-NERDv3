import logging
import math
import os
import time

import cv2
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from src.gui import QtImageTools, Screen
from src.hvps.Switchboard import SwitchBoard
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

        # register callback
        # image_capture.set_new_image_callback(self.updateImage)
        # image_capture.set_new_set_callback(self.updateImages)
        self._image_capture = ImageCapture.SharedInstance
        self._strain_detector = None  # will be set when a reference is recorded
        self.n_cams = 0  # we don't know of any available cameras yet

        self._hvps = SwitchBoard()  # create switchboard instance. not connected yet
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

        # panel for all the settings
        formLay = QtWidgets.QFormLayout()
        formLay.setLabelAlignment(Qt.AlignRight)

        # Switchboard ####################################

        # add combo box to select the COM port
        self.cbb_switchboard = QtWidgets.QComboBox()
        self.cbb_switchboard.currentTextChanged.connect(self.cbb_comport_changed)

        # button to refresh com ports
        self.btn_refresh_com = QtWidgets.QPushButton("Refresh switchboards")
        self.btn_refresh_com.clicked.connect(self.refresh_comports)

        # label to show what switchboard we're connected to
        self.lbl_switchboard_status = QtWidgets.QLabel("no switchboard")

        formLay.addRow("Switchboard:", self.cbb_switchboard)
        formLay.addRow("Status:", self.lbl_switchboard_status)
        formLay.addRow("", self.btn_refresh_com)

        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        formLay.addRow(" ", separator)

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

        formLay.addRow("Multimeter:", self.cbb_daq)
        formLay.addRow("Status:", self.lbl_daq_status)
        formLay.addRow("", self.btn_refresh_daq)
        formLay.addRow("", self.btn_test_res)

        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        formLay.addRow(" ", separator)

        # Test parameters ########################################

        # create voltage selector
        self.num_voltage = self.create_num_selector(0, 5000, "voltage", 1000)
        formLay.addRow("Test voltage [V]:", self.num_voltage)

        # toggle button to apply voltage (to check strain detection results)
        self.btn_apply_voltage = QtWidgets.QPushButton("Apply voltage now!")
        self.btn_apply_voltage.setCheckable(True)
        self.btn_apply_voltage.setEnabled(False)  # only enable if connected to an HVPS
        self.btn_apply_voltage.clicked.connect(self.btnVoltageClicked)
        formLay.addRow("", self.btn_apply_voltage)

        # create ramp step selector
        self.num_steps = self.create_num_selector(0, 100, "steps", 10)
        formLay.addRow("Ramp steps:", self.num_steps)

        # create step duration selector
        self.num_step_duration = self.create_num_selector(0, 10000, "step_duration_s", 10)
        formLay.addRow("Step duration (s):", self.num_step_duration)

        # create high duration selector
        self.num_high_duration = self.create_num_selector(0, 100000, "high_duration_min", 60)
        self.num_high_duration.valueChanged.connect(self.updateCycles)
        formLay.addRow("High duration (min):", self.num_high_duration)

        # create low duration selector
        self.num_low_duration = self.create_num_selector(0, 100000, "low_duration_s", 30)
        formLay.addRow("Low duration (s):", self.num_low_duration)

        # create measurement period selector
        self.num_measurement_period = self.create_num_selector(0, 1000, "measurement_period_s", 10)
        formLay.addRow("Measurement interval (s):", self.num_measurement_period)

        # create image save period selector
        self.num_save_image_period = self.create_num_selector(0, 1000, "save_image_period_min", 30)
        formLay.addRow("Save image interval (min):", self.num_save_image_period)

        # checkbox to enable AC mode
        self.chk_ac = QtWidgets.QCheckBox("AC mode")
        self.chk_ac.clicked.connect(self.chkACClicked)
        formLay.addRow("", self.chk_ac)

        # create AC frequency selector
        self.num_ac_frequency = self.create_num_selector(1, 1000, "ac_frequency_hz", 50)
        self.num_ac_frequency.valueChanged.connect(self.updateCycles)
        formLay.addRow("Cycle frequency [Hz]:", self.num_ac_frequency)

        # create number of cycles indicator
        self.lbl_cycles = QtWidgets.QLabel("")
        formLay.addRow("", self.lbl_cycles)

        # apply default to AC checkbox and enable or disable frequency selector accordingly
        if "AC_mode" in self._defaults:
            checked = self._defaults["AC_mode"]
        else:
            checked = False
        self.chk_ac.setChecked(checked)
        self.num_ac_frequency.setEnabled(checked)
        self.updateCycles()

        # create grid layout to show all the images
        gridLay = QtWidgets.QGridLayout()
        self.lbl_image = []
        self.cbb_camera_select = []
        self.btn_adjust = []
        for i in range(n_deas):
            groupBox = QtWidgets.QGroupBox("DEA {}".format(i + 1), self)  # number groups/DEAs from 1 to 6
            grpLay = QtWidgets.QVBoxLayout()
            # add label to show image
            lbl = QtWidgets.QLabel()
            lbl.setFixedSize(self._preview_img_size[0], self._preview_img_size[1])
            lbl.setPixmap(self._image_buffer[i])  # initialize with waiting image
            self.lbl_image.append(lbl)

            # add a camera  selection Combobox and fill the with camera names
            cb = QtWidgets.QComboBox()
            cb.addItem("Not used")
            # cb.setCurrentIndex(0)
            cb.currentIndexChanged.connect(self.camSelIndexChanged)
            self.cbb_camera_select.append(cb)

            # add adjust button
            btn = QtWidgets.QPushButton("Adjust...")
            btn.clicked.connect(self.btnAdjustClicked)
            self.btn_adjust.append(btn)

            rowLay = QtWidgets.QHBoxLayout()
            rowLay.addWidget(cb)
            rowLay.addWidget(btn, alignment=Qt.AlignRight)

            # add image and combobox to group box layout
            grpLay.addWidget(lbl)
            grpLay.addLayout(rowLay)
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
        topLay.addLayout(formLay)
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
        mainLay.addLayout(gridLay)

        ##############################################################################
        #  layout done
        ##############################################################################

        self.refresh_comports()  # fill with available ports
        self.refresh_multimeters()  # fill with available multimeters

        # self.setWindowFlags(Qt.Window)
        self.show()

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

        self.accept()

    def chkACClicked(self):
        if self.chk_ac.isChecked():
            self.num_ac_frequency.setEnabled(True)
            self.schematic.setPixmap(self.pix_schematic_AC)
        else:
            self.num_ac_frequency.setEnabled(False)
            self.schematic.setPixmap(self.pix_schematic_DC)
        self.updateCycles()

    def refresh_comports(self):
        self.cbb_switchboard.blockSignals(True)  # block signals to avoid excessive reconnecting to the switchboard
        self.cbb_switchboard.clear()  # remove all items

        # list all available com ports
        ports = self._hvps.detect()
        display_names = ["{} ({})".format(p.name.decode("ASCII"), p.port) for p in ports]
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
                self._hvps.open(port_idx, connection_timeout=0)
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
        self.logging.debug("clicked adjust")
        sender = self.sender()
        for i_dea in range(self._n_deas):
            if sender == self.btn_adjust[i_dea]:
                self.logging.debug("sender: button for DEA {}".format(i_dea))
                self.showAdjustmentView(i_dea)
                self.btnCaptureClicked()

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
        images = [images[i] for i in order_filt]  # put image sets in the right order

        self._set_strain_reference(images)

        self.btn_reference.setText("Record strain reference")
        self.btn_reference.setEnabled(True)

    def _set_strain_reference(self, images):
        self._strain_detector = StrainDetection.StrainDetector()
        self._strain_detector.set_reference(images)

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
            fpath = os.path.join(dir_path, "DEA {}".format(i), "Images")
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

        camorder = self.getCamOrder()
        cams_in_use = len(camorder) - camorder.count(-1)
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
                    self._hvps.set_switching_mode(1)  # set to DC mode
                    self._hvps.set_relay_auto_mode()
                    self._hvps.set_voltage_no_overshoot(self.num_voltage.value())
                    self.btn_apply_voltage.setText("Turn voltage off!")
                    self.btn_apply_voltage.setStyleSheet("QPushButton{ color: red }")
                else:
                    self._hvps.set_switching_mode(0)
                    self._hvps.set_voltage(0, block_until_reached=True)
                    self._hvps.set_relays_off()
                    self.btn_apply_voltage.setText("Apply voltage now!")
                    self.btn_apply_voltage.setStyleSheet("QPushButton{ color: black }")
            except TimeoutError:
                self.logging.warning("Unable to set voltage. Refreshing com ports to check connection.")
                self.refresh_comports()
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

    def camSelIndexChanged(self):
        if self._startup:
            return

        try:
            # see if any other combobox has the same selected index
            sender = self.sender()
            i_sender = self.cbb_camera_select.index(sender)
            sender = self.cbb_camera_select[i_sender]
            idx_cam = sender.currentIndex()
            prev_idx = self._camorder[i_sender] + 1

            # self.logging.debug("camera selection index for DEA {} changed to {}".format(src, idx))
            if idx_cam > 0:  # if it's 0 ("not used"), we don't need to do anything
                for cb in self.cbb_camera_select:
                    if cb is not sender and cb.currentIndex() == idx_cam:
                        cb.setCurrentIndex(prev_idx)  # set to the previous selection of the sender
            self._camorder = self.getCamOrder()  # update stored cam order

            # disable strain display because if the camera selection changes, the strain reference is no longer valid
            if self._strain_detector is not None:
                self.chkStrain.setEnabled(False)
                self.chkStrain.setChecked(False)
                self._strain_detector = None
                self.lblStrainRef.setText("No strain reference set!")
                self.lblStrainRef.setStyleSheet("QLabel { color : red }")
                dtitle = "Strain reference no longer valid"
                dmsg = "Since the camera order changed, the strain reference is no longer valid!\n" \
                       "Please set a new strain reference"
                QtWidgets.QMessageBox.information(self, dtitle, dmsg, QtWidgets.QMessageBox.Ok)

            self.refreshImages()
        except Exception as ex:
            self.logging.error("Exception in combobox callback: {}".format(ex))

    def refreshImages(self):
        images = self._image_capture.get_images_from_buffer()
        order = self.getCamOrder()
        order_filt = [i for i in order if i >= 0]  # filter to remove unused cameras (-1)
        images = [images[i] for i in order_filt]  # put images in the right order

        if self.chkStrain.isChecked():
            try:
                v = self._hvps.get_current_voltage()
                sv = "{} V".format(v)
            except Exception as ex:
                self.logging.debug("Couldn't read voltage: {}".format(ex))
                sv = ""
            images = self._strain_detector.get_dea_strain(images, True, True, sv)[2]

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
            "cam_order": self.getCamOrder(),
            "voltage": self.num_voltage.value(),
            "steps": self.num_steps.value(),
            "step_duration_s": self.num_step_duration.value(),
            "high_duration_min": self.num_high_duration.value(),
            "low_duration_s": self.num_low_duration.value(),
            "measurement_period_s": self.num_measurement_period.value(),
            "save_image_period_min": self.num_save_image_period.value(),
            "ac_mode": self.chk_ac.isChecked(),
            "ac_frequency_hz": self.num_ac_frequency.value(),
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
