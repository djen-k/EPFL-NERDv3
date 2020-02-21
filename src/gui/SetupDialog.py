import logging
import math

import cv2
from PyQt5 import QtWidgets, QtSerialPort, QtGui
from PyQt5.QtCore import Qt

from src.gui import QtImageTools
from src.image_processing import ImageCapture
from src.image_processing import StrainDetection


class SetupDialog(QtWidgets.QDialog):
    """
    This class is a dialog window asking the user to select a COM port and assign a camera to each DEA
    """

    def __init__(self, parent=None, com_port_default=None, cam_order_default=None):
        super(SetupDialog, self).__init__(parent)
        # Create logger
        self.logging = logging.getLogger("SetupDialog")
        self._startup = True  # flag to indicate that the dialog is still loading

        n_deas = 6  # not accepting anything else at the moment
        self._n_deas = n_deas
        n_rows = 2
        n_cols = 3

        self._default_img_size = [640, 360]

        # register callback
        # image_capture.set_new_image_callback(self.updateImage)
        # image_capture.set_new_set_callback(self.updateImages)
        self._image_capture = ImageCapture.SharedInstance
        self.n_cams = 0  # we don't know of any available cameras yet

        # return values
        self._camorder = [-1] * n_deas
        self._image_buffer = [QtImageTools.conv_Qt(ImageCapture.ImageCapture.IMG_WAITING,
                                                   self._default_img_size)] * n_deas

        if cam_order_default is None:
            cam_order_default = range(n_deas)
        elif len(cam_order_default) != n_deas:
            raise ValueError("Default camera order must define a camera for each of {} DEAs. "
                             "(Set -1 if not used)".format(n_deas))
        self._cam_order_default = cam_order_default

        # Set window's title
        self.setWindowTitle("Camera selection dialog")

        # create the main layout: vertical box with description at the top, then grid of images, then Next button
        mainLay = QtWidgets.QVBoxLayout(self)

        # list all available com ports
        comports = QtSerialPort.QSerialPortInfo.availablePorts()
        # add combo box to select the COM port
        self.cbb_port_name = QtWidgets.QComboBox()
        for info in comports:
            self.cbb_port_name.addItem(info.portName())

        # select the default COM port, if it is available
        if com_port_default is not None:
            for i in range(len(comports)):
                if comports[i].portName() == com_port_default:
                    self.cbb_port_name.setCurrentIndex(i)
                    break
        else:  # no default --> pick first one
            self.cbb_port_name.setCurrentIndex(0)

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
        self.btnNext = QtWidgets.QPushButton("OK")
        self.btnNext.clicked.connect(self.accept)

        # button to take new images
        self.btnCapture = QtWidgets.QPushButton("Take new image")
        self.btnCapture.clicked.connect(self.btnCaptureClicked)

        # toggle button to show the strain detection results
        self.btnPreview = QtWidgets.QPushButton("Show strain detection")
        self.btnPreview.setCheckable(True)
        self.btnPreview.clicked.connect(self.btnPreviewClicked)

        buttonLay = QtWidgets.QHBoxLayout()
        buttonLay.addWidget(self.btnCapture, alignment=Qt.AlignLeft)
        buttonLay.addWidget(self.btnPreview, alignment=Qt.AlignLeft)
        buttonLay.addWidget(self.btnNext, alignment=Qt.AlignRight)

        # some GUI (layout stuff)
        # formLay = QtWidgets.QFormLayout(self)

        mainLay.addWidget(QtWidgets.QLabel("Please select COM port for HVPS/Switchboard"))
        mainLay.addWidget(self.cbb_port_name, alignment=Qt.AlignLeft)

        # add a separator between COM selection and camera images
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        mainLay.addWidget(separator)

        mainLay.addWidget(QtWidgets.QLabel("Select the camera to use for each DEA. Make sure the images are sharp "
                                           "and that the DEAs are centered in the images"))
        mainLay.addLayout(gridLay)
        mainLay.addLayout(buttonLay)
        # mainLay.addLayout(formLay)

        # self.setWindowFlags(Qt.Window)
        self.show()

        self.btnCaptureClicked()  # captures images and updates the labels

    def btnAdjustClicked(self):
        self.logging.debug("clicked adjust")
        sender = self.sender()
        for i_dea in range(self._n_deas):
            if sender == self.btn_adjust[i_dea]:
                self.logging.debug("sender: button for DEA {}".format(i_dea))
                self.showAdjustmentView(i_dea)
                self.btnCaptureClicked()

    def btnPreviewClicked(self):
        self.refreshImages()

    def showAdjustmentView(self, i_dea):
        i_cam = self._camorder[i_dea]
        if i_cam == -1:
            return

        ret = -1
        while ret == -1:
            cv2.imshow("DEA {} - Press any key to close".format(i_dea), self._image_capture.get_single_image(i_cam))
            try:
                ret = cv2.waitKey(100)
                self.logging.debug("Key pressed {}".format(ret))
            except Exception as ex:
                self.logging.debug("Exception in showAdjustmentView: {}".format(ex))
                break

        cv2.destroyAllWindows()

    def camSelIndexChanged(self):
        try:
            # find which combo box it came from
            # src = -1
            # camorder = self.getCamOrder()
            # for i in range(len(camorder)):
            #     if camorder[i] != self._camorder[i]:
            #         src = i
            #         break
            #
            # if src == -1:
            #     self.logging.debug("Could not identify source of combobox index changed event")
            #
            # self._camorder = camorder  # update stored cam order
            #
            # if src == -1:
            #     return
            sender = self.sender()
            # self.logging.debug("Combobox callback: {}".format(sender))
            # src = -1
            # try:
            #     src = self.cbb_camera_select.index(sender)
            # except ValueError:
            #     self.logging.debug("Combobox caller not known")

            # see if any other combobox has the same selected index
            assert type(sender) == QtWidgets.QComboBox
            idx = sender.currentIndex()
            # self.logging.debug("camera selection index for DEA {} changed to {}".format(src, idx))
            if idx > 0:  # if it's 0 ("not used"), we don't need to do anything
                for i in range(self._n_deas):
                    cb = self.cbb_camera_select[i]
                    if cb is not sender and cb.currentIndex() == idx:
                        cb.setCurrentIndex(0)  # set to "not used"
            self._camorder = self.getCamOrder()  # update stored cam order
            self.refreshImages()
        except Exception as ex:
            self.logging.error("Exception in combobox callback: {}".format(ex))

    def refreshImages(self):
        for i_dea in range(self._n_deas):
            i_cam = self._camorder[i_dea]
            if i_cam >= 0:
                if self.btnPreview.isChecked():
                    img = self._image_capture.get_single_image_from_buffer(i_cam)
                    try:
                        ellipse = StrainDetection.dea_fit_ellipse(img, 5)
                        img = StrainDetection.draw_ellipse(img, ellipse)
                        self.logging.debug("Electrode area for DEA {}: {}".format(i_dea, ellipse))
                    except Exception as ex:
                        self.logging.warning("Failed to detect electrode area for DEA {}! ({})".format(i_dea, ex))
                        pass
                    self.setImage(i_dea, img)
                else:
                    self.setImage(i_dea, self._image_capture.get_single_image_from_buffer(i_cam))
            else:
                self.setImage(i_dea, ImageCapture.ImageCapture.IMG_NOT_AVAILABLE)

    def setImage(self, i_dea, opencv_image, img_size=None):
        if img_size is None:
            img_size = self._default_img_size
        self.lbl_image[i_dea].setPixmap(QtImageTools.conv_Qt(opencv_image, img_size))

    def btnCaptureClicked(self):
        self._image_capture.read_images()
        self.updateImages(self._image_capture.get_images_from_buffer(), self._image_capture.get_timestamps())

    def btnNextClicked(self):
        print("next")
        self.btnNext.clicked.connect(self.accept)
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
        # return values of fields when OK
        return self.cbb_port_name.currentText(), self.getCamOrder()  # , self.initial_images, self.mesh_coordinates

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
        # show the new image in each frame where the respective camera is selected (should only ever be one)
        # for i_dea in range(self._n_deas):
        #     i_cam = self._camorder[i_dea]
        #
        #     # if camera index for any dea is -1 ("not used"), show the not available image
        #     if i_cam == -1:
        #         self.lbl_image[i_dea].setPixmap(QtImageTools.conv_Qt(ImageCapture.IMG_NOT_AVAILABLE, self._default_img_size))
        #     else:
        #         self.lbl_image[i_dea].setPixmap(QtImageTools.conv_Qt(images[i_cam], self._default_img_size))

    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        try:
            super(SetupDialog, self).paintEvent(a0)
        except Exception as ex:
            self.logging.error("Exception in CamerSelection.paintEvent: {}".format(ex))
