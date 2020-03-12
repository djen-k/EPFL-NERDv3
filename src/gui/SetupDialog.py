import logging
import math
import os
import time

import cv2
from PyQt5 import QtWidgets, QtSerialPort, QtGui
from PyQt5.QtCore import Qt

from src.gui import QtImageTools
from src.hvps import NERDHVPS
from src.image_processing import ImageCapture, StrainDetection


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

        self._default_img_size = [720, 405]  # [800, 450]  [640, 360]

        # register callback
        # image_capture.set_new_image_callback(self.updateImage)
        # image_capture.set_new_set_callback(self.updateImages)
        self._image_capture = ImageCapture.SharedInstance
        self._strain_detector = None  # will be set when a reference is recorded
        self.n_cams = 0  # we don't know of any available cameras yet

        self._hvps = None

        # return values
        self._camorder = [-1] * n_deas
        self._image_buffer = [QtImageTools.conv_Qt(ImageCapture.ImageCapture.IMG_WAITING,
                                                   self._default_img_size)] * n_deas

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

        # add combo box to select the COM port
        self.cbb_port_name = QtWidgets.QComboBox()
        self.cbb_port_name.currentTextChanged.connect(self.cbb_comport_changed)

        # button to refresh com ports
        self.btn_refresh_com = QtWidgets.QPushButton("Refresh")
        self.btn_refresh_com.clicked.connect(self.refresh_comports)

        # label to show what switchboard we're connected to
        self.lbl_switcbox_status = QtWidgets.QLabel("nothing")

        formLay.addRow("Switchboard:", self.cbb_port_name)
        formLay.addRow("Connected to:", self.lbl_switcbox_status)
        formLay.addRow("", self.btn_refresh_com)

        # create voltage selector
        self.num_voltage = self.create_num_selector(300, 5000, "voltage", 1000)
        formLay.addRow("Voltage [V]:", self.num_voltage)

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
        formLay.addRow("High duration (min):", self.num_high_duration)

        # create low duration selector
        self.num_low_duration = self.create_num_selector(0, 100000, "low_duration_s", 30)
        formLay.addRow("Low duration (s):", self.num_low_duration)

        # create measurement period selector
        self.num_measurement_period = self.create_num_selector(0, 1000, "measurement_period_s", 10)
        formLay.addRow("Measure every (s):", self.num_measurement_period)

        # create image save period selector
        self.num_save_image_period = self.create_num_selector(0, 1000, "image_save_period_min", 30)
        formLay.addRow("Save images every (min):", self.num_save_image_period)

        # checkbox to enable AC mode
        self.chk_ac = QtWidgets.QCheckBox("AC mode")
        self.chk_ac.clicked.connect(self.chkACClicked)
        formLay.addRow("", self.chk_ac)

        # create image save period selector
        self.num_ac_frequency = self.create_num_selector(0, 1000, "ac_frequency", 50)
        formLay.addRow("Switching frequency [Hz]:", self.num_ac_frequency)

        # apply default to AC checkbox and enable or disable frequency selector accordingly
        if "AC_mode" in self._defaults:
            checked = self._defaults["AC_mode"]
        else:
            checked = False
        self.chk_ac.setChecked(checked)
        self.chk_ac.setEnabled(False)  # TODO: re-enable once AC mode is implemented
        self.num_ac_frequency.setEnabled(checked)

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
            lbl.setFixedSize(self._default_img_size[0], self._default_img_size[1])
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
        self.btnCapture = QtWidgets.QPushButton("Take new image")
        self.btnCapture.clicked.connect(self.btnCaptureClicked)

        # button to take new images
        self.chkStrain = QtWidgets.QCheckBox("Show strain")
        self.chkStrain.setEnabled(False)  # only enable once a strain reference has been set
        self.chkStrain.clicked.connect(self.btnCaptureClicked)

        # button to record strain reference
        self.btnReference = QtWidgets.QPushButton("Record strain reference")
        self.btnReference.clicked.connect(self.record_strain_reference)

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
        buttonLay.addWidget(self.btnCapture)
        buttonLay.addWidget(self.chkStrain)
        buttonLay.addWidget(self.btnReference)
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

        # create the main layout: vertical box with description at the top, then grid of images, then Next button
        mainLay = QtWidgets.QVBoxLayout(self)

        topLay = QtWidgets.QHBoxLayout()
        topLay.setAlignment(Qt.AlignLeft)
        # topLay.setContentsMargins(50, 5, 50, 5)
        topLay.addLayout(formLay)
        schematic = QtWidgets.QLabel()
        pix_schematic = QtGui.QPixmap("res/images/schematic_voltage.png").scaledToHeight(250, Qt.SmoothTransformation)
        schematic.setPixmap(pix_schematic)
        schematic.setAlignment(Qt.AlignTop)
        schematic.setContentsMargins(20, 5, 20, 5)
        topLay.addWidget(schematic)

        spacer = QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        topLay.addSpacerItem(spacer)

        logo = QtWidgets.QLabel()
        logo.setPixmap(QtGui.QPixmap("res/images/epfl_logo.png"))
        logo.setAlignment(Qt.AlignTop)
        topLay.addWidget(logo)

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

    def create_num_selector(self, min_val, max_val, default_key, default_value):
        """
        Creates a QSpinBox for setting numeric values.
        :param min_val: Minimum value
        :param max_val: Maximum value
        :param default_key: Key to look up default value in the defaults dict
        :param default_value: Value to use if the key is not in the defaults dict
        :return: A QSpinBox with the given range and value
        """
        num = QtWidgets.QSpinBox()
        num.setMinimum(min_val)
        num.setMaximum(max_val)
        if default_key in self._defaults:
            value = self._defaults[default_key]
        else:
            value = default_value
        num.setValue(value)

        return num

    def btnStartClicked(self):
        # only proceed if a strain reference has been set
        if self._strain_detector is None:
            QtWidgets.QMessageBox.warning(self, "No strain reference",
                                          "To start a measurement, please set a strain reference!",
                                          QtWidgets.QMessageBox.Ok)
        else:
            self.accept()

    def chkACClicked(self):
        self.num_ac_frequency.setEnabled(self.chk_ac.isChecked())

    def refresh_comports(self):
        self.cbb_port_name.clear()  # remove all items
        # list all available com ports
        comports = QtSerialPort.QSerialPortInfo().availablePorts()
        portnames = [info.portName() for info in comports]
        # comports = serial.tools.list_ports.comports()
        # portnames = comports[0].name
        self.cbb_port_name.addItems(portnames)

        # select the default COM port, if it is available
        if "com_port" in self._defaults and self._defaults["com_port"] in portnames:
            self.cbb_port_name.setCurrentText(self._defaults["com_port"])
        else:  # no default or default not available --> pick last one
            self.cbb_port_name.setCurrentIndex(len(comports) - 1)

    def cbb_comport_changed(self):
        if self._hvps is not None:
            del self._hvps

        try:
            port = self.cbb_port_name.currentText()
            hvps = NERDHVPS.init_hvps(port)
            self.lbl_switcbox_status.setText(hvps.get_name().decode("ASCII"))
            self._hvps = hvps
            self.btn_apply_voltage.setEnabled(True)
        except Exception as ex:
            self.logging.debug("Could not connect to switchboard: {}".format(ex))
            self.lbl_switcbox_status.setText("nothing")
            self.btn_apply_voltage.setEnabled(False)
            self.btn_apply_voltage.setChecked(False)
            self._hvps = None

    def btnAdjustClicked(self):
        self.logging.debug("clicked adjust")
        sender = self.sender()
        for i_dea in range(self._n_deas):
            if sender == self.btn_adjust[i_dea]:
                self.logging.debug("sender: button for DEA {}".format(i_dea))
                self.showAdjustmentView(i_dea)
                self.btnCaptureClicked()

    def record_strain_reference(self):
        # TODO: protect against impatient clicks

        cap = self._image_capture
        images = cap.read_average(self.num_avg_img.value())
        order = self.getCamOrder()
        order_filt = [i for i in order if i >= 0]  # filter to remove unused cameras (-1)
        images = [images[i] for i in order_filt]  # put image sets in the right order

        self._set_strain_reference(images)

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
            if self.btn_apply_voltage.isChecked():
                # turn on and set voltage
                self._hvps.set_relay_auto_mode()
                self._hvps.set_voltage(self.num_voltage.value(), wait=True)
                self.btn_apply_voltage.setText("Turn voltage off!")
                self.btn_apply_voltage.setStyleSheet("QPushButton{ color: red }")
            else:
                self._hvps.set_voltage(0, wait=True)
                self._hvps.set_relays_off()
                self.btn_apply_voltage.setText("Apply voltage now!")
                self.btn_apply_voltage.setStyleSheet("QPushButton{ color: black }")

            time.sleep(3)
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
                cv2.imshow("DEA {} - Press any key to close".format(i_dea), self._image_capture.read_single(i_cam))
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
                self.logging.debug("Couldn't read vooltage: {}".format(ex))
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
        # TODO: protect against impatient clicks
        self.updateImages(self._image_capture.read_images(), self._image_capture.get_timestamps())

    def btnNextClicked(self):
        print("next")
        self.btnStart.clicked.connect(self.accept)
        # self._image_capture.read_images()
        # # Get index of selected DEA, add it to the camOrderList and disable it form ComboBox
        # if self.camSelW.currentText() != "Not used":
        #     print("{}".format(self.camSelW.currentText()))
        #     # selectedDea = int(self.camSelW.currentText().replace("DEA ", ""))
        #     selectedDea = self.camSelW.currentText()
        #     # self.camOrder.append(selectedDea)
        #     self.camorder[selectedDea]=self.camIndex
        #     self.initial_images[selectedDea] = self.imageReaderT.getAveragedImage()
        #     self.mesh_coordinates[selectedDea] = self.getMeshCoordinates(self.imageReaderT.getAveragedImage())
        #
        #     self.camSelW.model().item(self.camSelW.currentIndex()).setEnabled(False)
        #
        #     # update counter for next acquisition
        #     self.deaIndex += 1
        #
        #
        # # If finished: hide everything and change button OnClick function to close dialog
        # if self.deaIndex == (self.nbDea):
        #     # Close image reader thread
        #     self.deaIndex += 1
        #     self.imageReaderT.close()
        #
        #
        #     # Change layout to ask anything ang show "Finished"
        #     self.nextBtnW.setText("Finish")
        #     self.descTextW.hide()
        #     self.title2W.setText("Finished: Order is: {}".format(self.camorder))
        #     self.camSelW.hide()
        #     self.ImageW.hide()
        #
        #     # Change button's OnClick function to finish dialog successfully
        #     self.nextBtnW.clicked.connect(self.accept)
        #
        # # let's increment camera index counter for next camera
        # self.camIndex+=1
        #
        # # ask thread for new image
        # self.imageReaderT.waitImage()
        # self.imageReaderT.changeIndex(self.camIndex)
        # self.descTextW.setText("Camera with index {}".format(self.camIndex))
        #
        # # Preselect DEA for next
        # self.camSelW.setCurrentIndex(self.camOrderDefault[self.camIndex])

    def getCamOrder(self):
        return [cb.currentIndex() - 1 for cb in self.cbb_camera_select]

    def get_results(self):
        """
        Get sonfiguration for the NERD test as set in the SetupDialog
        :return: A config dict with all the parameters, as well as a strain detector with pre-set strain reference
        """
        config = {
            "com_port": self.cbb_port_name.currentText(),
            "cam_order": self.getCamOrder(),
            "voltage": self.num_voltage.value(),
            "steps": self.num_steps.value(),
            "step_duration_s": self.num_step_duration.value(),
            "high_duration_min": self.num_high_duration.value(),
            "low_duration_s": self.num_low_duration.value(),
            "measurement_period_s": self.num_measurement_period.value(),
            "save_image_period_min": self.num_save_image_period.value(),
            "ac_mode": self.chk_ac.isChecked(),
            "ac_frequency": self.num_ac_frequency.value(),
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
                self.lbl_image[dea_id].setPixmap(QtImageTools.conv_Qt(image, self._default_img_size))
            except ValueError:
                pass  # not found, so we leave target DEA index at -1
        else:
            # show the new image in each frame where the respective camera is selected (should only ever be one)
            for i in range(self._n_deas):
                i_sel = self._camorder[i]
                if i_sel == cam_id:
                    self.lbl_image[i].setPixmap(QtImageTools.conv_Qt(image, self._default_img_size))

                # if camera index for any dea is -1 ("not used"), show the not available image
                elif i_sel == -1:
                    self.lbl_image[i].setPixmap(
                        QtImageTools.conv_Qt(ImageCapture.ImageCapture.IMG_NOT_AVAILABLE, self._default_img_size))

    def updateImages(self, images, timestamps):

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
