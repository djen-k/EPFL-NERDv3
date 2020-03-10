import logging
import logging.config
import sys
import time
from datetime import datetime, timedelta

import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QApplication

from libs import duallog
from src.fileio.DataSaver import DataSaver
from src.fileio.ImageSaver import ImageSaver
from src.fileio.config import read_config, write_config
from src.gui import SetupDialog
from src.hvps import NERDHVPS
from src.image_processing import ImageCapture, StrainDetection


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

        # logger
        self.logging = logging.getLogger("mainWindow")


def _setup(_comport, _camorder, _voltage):
    config_file_name = "config.txt"

    try:
        _comport, _camorder, _voltage = read_config(config_file_name)
        logging.info("Read config file")
    except Exception as ex:
        logging.warning("Unable to read config file: {}".format(ex))

    # Show setup dialog to get COM port and camera order
    setup_dialog = SetupDialog.SetupDialog(com_port_default=_comport, cam_order_default=_camorder,
                                           voltage_default=_voltage)

    if setup_dialog.exec_():
        _comport, _camorder, _voltage = setup_dialog.get_results()
        logging.info("User selection for HVPS COM port: {}".format(_comport))
        logging.info("Test voltage set to {} V".format(_voltage))
        logging.info("User selection for cam order: {}".format(_camorder))

        try:
            write_config(config_file_name, _comport, _camorder, _voltage)
            logging.info("Wrote config file")
        except Exception:
            logging.warning("Unable to write config file")
    else:
        logging.critical("No COM port or cam order selected")
        raise Exception("No COM port or cam order selected")

    return _comport, _camorder, _voltage


class NERD:

    def __init__(self, _comport, _camorder, _voltage):
        self.logging = logging.getLogger("NERD")
        self.comport = _comport
        self.camorder = _camorder
        self.voltage = _voltage
        self.image_cap = ImageCapture.SharedInstance
        # TODO: implement robust system for getting number of DEAs and dealing with unexpected number of images etc.
        self.n_dea = self.image_cap.get_camera_count()  # this works if we previously selected the cameras

        self.hvpsInst = NERDHVPS.init_hvps(self.comport)
        self.hvpsInst.set_relay_auto_mode()  # enable auto mode by default
        self.hvpsInst.set_switching_mode(1)  # set to DC mode by default to make sure that the HV indicator LED works
        self.shutdown_flag = False
        self.time_started = None
        self.time_paused = timedelta(0)
        self.reference_radii = None
        self.reference_center = None
        self.reference_angles = None

    def run_nogui(self):
        self.logging.info("Running NERD protocol with {} DEAs in main thread without dedicated GUI "
                          "(using openCV windows for visual output".format(self.n_dea))

        # create an image saver for this session to store the recorded images
        session_name = "NERD test {}".format(datetime.now().strftime("%Y%m%d-%H%M%S"))
        dir_name = "output/{}".format(session_name)

        imsaver = ImageSaver(dir_name, self.n_dea, save_result_images=True)

        # TODO: handle varying numbers of DEAs
        save_file_name = "{}/{} data.csv".format(dir_name, session_name)
        saver = DataSaver(self.n_dea, save_file_name)

        strain_detector = StrainDetection.StrainDetector()

        # get settings/values from user
        max_voltage = self.voltage
        min_voltage = 300
        # TODO: check why min voltage is 300 V (officially 50, but doesn't always seem to work)
        nsteps = 10
        voltage_step = max_voltage / nsteps

        duration_low_s = 5
        duration_high_s = 1 * 60 * 60  # 1 h = 3600 s
        step_duration_s = 5
        update_period_s = 1
        image_capture_period = 60

        # prepare / compute what we need before applying voltage
        current_target_voltage = 0

        # states for state machine
        STATE_STARTUP = -1
        STATE_WAITING_LOW = 0
        STATE_STEPPING = 1
        STATE_WAITING_HIGH = 2

        # initial state
        current_state = STATE_STARTUP
        current_step = 0

        # record start time
        t = datetime.now()
        self.time_started = t
        time_last_state_change = t
        time_last_image_taken = t
        start_pause_time = t
        last_image_time = t

        # power on the HVPS
        self.hvpsInst.set_voltage(0)
        self.hvpsInst.set_output_on()

        while not self.shutdown_flag:
            now = datetime.now()

            # ------------------------------
            # state machine: actions (actually just outputs since all the interesting stuff happens on state change)
            # ------------------------------
            if current_state == STATE_STARTUP:
                self.logging.info("Starting up state machine")
                # wait for voltage to be 0 (in case it wasn't at start and needs time to decay)
                measuredVoltage = self.hvpsInst.get_current_voltage()
                while measuredVoltage > 0:
                    self.logging.warning("Current voltage is {}. Waiting for it to decay to 0.".format(measuredVoltage))
                    time.sleep(0.1)
                    measuredVoltage = self.hvpsInst.get_current_voltage()
            elif current_state == STATE_WAITING_LOW:
                msg = "waiting low: {:.0f}/{}s".format((now - time_last_state_change).total_seconds(), duration_low_s)
                self.logging.info(msg)
            elif current_state == STATE_STEPPING:
                dt = (now - time_last_state_change).total_seconds()
                msg = "Step {}/{}, current voltage: {} V, next step in {:.0f}/{} s"
                msg = msg.format(current_step, nsteps, current_target_voltage, dt, step_duration_s)
                self.logging.info(msg)
            elif current_state == STATE_WAITING_HIGH:
                msg = "Waiting high: {:.0f}/{} s"
                msg = msg.format((now - time_last_state_change).total_seconds(), duration_high_s)
                self.logging.info(msg)
            else:
                logging.critical("Unknown state in the state machine")
                raise Exception

            # ------------------------------
            # state machine: transitions
            # ------------------------------
            image_due = False  # flag to indicate if we need to take a picture and measure strain

            new_state = current_state  # if nothing happens in the following conditions, we keep current state and step
            new_step = current_step
            new_target_voltage = current_target_voltage
            dt_state_change = (now - time_last_state_change).total_seconds()  # time since last state change
            if current_state == STATE_STARTUP:
                new_state = STATE_WAITING_LOW
            elif current_state == STATE_WAITING_LOW:
                if dt_state_change > duration_low_s:
                    new_state = STATE_STEPPING
                    new_step = 0  # reset steps
                    while new_target_voltage < min_voltage:
                        new_step += 1  # increase step
                        new_target_voltage = round(new_step * voltage_step)
            elif current_state == STATE_STEPPING:
                if dt_state_change > step_duration_s:
                    new_step = current_step + 1
                    new_target_voltage = round(new_step * voltage_step)
                    if new_step > nsteps:
                        new_state = STATE_WAITING_HIGH
                        new_target_voltage = max_voltage
            elif current_state == STATE_WAITING_HIGH:
                if dt_state_change > duration_high_s:
                    new_state = STATE_WAITING_LOW
                    new_target_voltage = 0
            else:
                logging.critical("Unknown state in the state machine")
                raise Exception

            # check what actions need to happen this cycle
            state_changing = new_state is not current_state or new_step is not current_step
            if state_changing:
                self.logging.debug("State changeing from {} (step {}) to {} (step {})".format(current_state,
                                                                                              current_step,
                                                                                              new_state, new_step))
            dt_image_taken = (now - time_last_image_taken).total_seconds()  # time since last state change
            image_required = dt_image_taken > image_capture_period
            self.logging.debug("Time since last image: {} s,  New image required: {}".format(dt_image_taken,
                                                                                             image_required))

            # ------------------------------
            # Record voltage and DEA state
            # ------------------------------
            measuredVoltage = self.hvpsInst.get_current_voltage()
            self.logging.info("Current voltage: {} V".format(measuredVoltage))

            deaState = self.hvpsInst.get_relay_state()
            self.logging.info("DEA state: {}".format(deaState))

            strain = None
            center_shifts = None
            if state_changing or image_required:

                # -----------------------
                # record, analyze, and store images
                # -----------------------

                # get images
                cap = ImageCapture.SharedInstance
                imgs = cap.read_images()
                # use timestamp of the last image for the whole set so they have a realistic and matching timestamp
                # timestamp = cap.get_timestamps(cap.get_camera_count() - 1)

                # get image name suffix
                suffix = "{}V".format(measuredVoltage)  # store voltage in file name
                if current_state == STATE_STARTUP:
                    suffix += " reference"  # this is the first image we're storing, so it will be the reference

                # save images first so if strain detection fails, we can check the image that caused the failure
                imsaver.save_all(images=imgs, timestamp=now, suffix=suffix)

                # fit ellipses
                strain, center_shifts, res_imgs, outliers = strain_detector.get_dea_strain(imgs, True, True)
                if any(outliers):
                    self.logging.warning("Outlier detected")
                # TODO: handle outliers (don't save data or mark in output file)

                # save result images
                imsaver.save_all(res_images=res_imgs, timestamp=now, suffix=suffix)

                # print average strain for each DEA
                self.logging.info("strain: {}".format(np.reshape(np.mean(strain, 1), (1, -1))))

                # use now instead of t_img to make sure we have consistent timestamp for all data recorded in this cycle
                time_last_image_taken = now

                # -----------------------
                # show images
                # -----------------------

                for i in range(len(res_imgs)):
                    if res_imgs[i] is None:
                        cv.imshow('DEA {}'.format(i), cv.resize(ImageCapture.ImageCapture.IMG_NOT_AVAILABLE, (0, 0),
                                                                fx=0.35, fy=0.35))
                    else:
                        cv.imshow('DEA {}'.format(i), cv.resize(res_imgs[i], (0, 0), fx=0.35, fy=0.35))

            # ------------------------------
            # Save all data voltage and DEA state
            # ------------------------------
            # todo: write elapsed time in seconds to output file
            # todo compute total time at max voltage
            saver.write_data(now, current_target_voltage, measuredVoltage, deaState,
                             current_state, strain, center_shifts)

            # ------------------------------
            # Apply state change and update voltage
            # ------------------------------
            if state_changing:
                # set voltage for new state
                current_target_voltage = new_target_voltage
                ret = self.hvpsInst.set_voltage(current_target_voltage, wait=True)
                if ret is not True:
                    self.logging.debug("Failed to set voltage")

                current_state = new_state
                current_step = new_step
                time_last_state_change = datetime.now()
                time_last_image_taken = time_last_state_change  # also reset image timer to avoid taking too many

            # ------------------------------
            # Keep track of voltage changes
            # ------------------------------
            # if current_target_voltage != previous_target_voltage:
            #     self.callbackVoltageChange(newVoltage=current_target_voltage)

            # ------------------------------
            # keep tack of time
            # ------------------------------
            # self.currentTimeSinceVoltageW.setText("{}s".format(int(time.time() - self.timeStarted)))
            # time.sleep(self.plotUpdatePeriod_s)
            if cv.waitKey(update_period_s * 1000) & 0xFF == ord('q'):
                self.shutdown_flag = True

        self.logging.critical("Exiting...")
        self.hvpsInst.set_output_off()
        self.hvpsInst.set_voltage(0, wait=True)
        del self.hvpsInst
        self.logging.critical("Turned voltage off and disconnected relays")
        saver.close()
        self.image_cap.close_cameras()


if __name__ == '__main__':
    duallog.setup("logs", minlevelConsole=logging.DEBUG, minLevelFile=logging.DEBUG)
    logging.info("Started application")

    # launch Qt
    app = QApplication(sys.argv)

    # hard coded default config
    com_port = "COM3"
    cam_order = [0, 1, 2, 3, 4, 5]
    voltage = 1000

    # run setup
    com_port, cam_order, voltage = _setup(com_port, cam_order, voltage)
    # apply cam order to image capture so we can just address them as cam 0, 1, 2, ...
    ImageCapture.SharedInstance.select_cameras(cam_order)

    nerd = NERD(com_port, cam_order, voltage)
    nerd.run_nogui()

    # ex = App(args=None)
    # retVal = app.exec_()
    retVal = 0

    logging.info('Finished with val: {}'.format(retVal))
    sys.exit(retVal)
