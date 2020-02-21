import logging
import logging.config
import sys
from datetime import datetime, timedelta

import cv2 as cv
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QApplication, QInputDialog

from libs import duallog
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
        _comport, _camorder = read_config(config_file_name)
        logging.info("Read config file")
    except Exception:
        logging.warning("Unable to read config file")

    # Show setup dialog to get COM port and camera order
    setup_dialog = SetupDialog.SetupDialog(com_port_default=_comport, cam_order_default=_camorder)

    if setup_dialog.exec_():
        _comport, _camorder = setup_dialog.get_results()
        logging.info("User selection for HVPS COM port: {}".format(_comport))
        logging.info("User selection for cam order: {}".format(_camorder))

        try:
            write_config(config_file_name, _comport, _camorder)
            logging.info("Wrote config file")
        except Exception:
            logging.warning("Unable to write config file")
    else:
        logging.critical("No COM port or cam order selected")
        raise Exception("No COM port or cam order selected")

    _voltage, okPressed = QInputDialog.getInt(None, "Test voltage", "Voltage:", _voltage, 0, 5000, 1)
    if not okPressed:
        raise Exception("No test voltage specified")

    logging.info("Test voltage set to {} V".format(_voltage))

    return _comport, _camorder, _voltage


class NERD:

    def __init__(self, _comport, _camorder, _voltage):
        self.logging = logging.getLogger("NERD")
        self.comport = _comport
        self.camorder = _camorder
        self.voltage = _voltage
        self.image_cap = ImageCapture.SharedInstance
        self.hvpsInst = NERDHVPS.init_hvps(self.comport)
        self.hvpsInst.set_relay_auto_mode()  # enable auto mode by default
        self.shutdown_flag = False
        self.time_started = None
        self.time_paused = timedelta(0)

    def run_nogui(self):
        self.logging.info("Running NERD protocol in main thread without dedicated GUI "
                          "(using openCV windows for visual output")

        # create an image saver for this session to store the recorded images
        dir_name = "output/NERD test {}".format(datetime.now().strftime("%Y%m%d-%H%M%S"))
        imsaver = ImageSaver(dir_name, 6, save_result_images=True)

        # get settings/values from user
        max_voltage = self.voltage
        nsteps = 10
        voltage_step = max_voltage / nsteps
        # TODO: respect min voltage 100 V (officially 50, but doesn't always seem to work)

        duration_low_s = 5
        duration_high_s = 20  # 1 * 60 * 60  # 1 h = 3600 s
        step_duration_s = 5
        update_period_s = 1

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
        last_state_change_time = t
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
            elif current_state == STATE_WAITING_LOW:
                msg = "waiting low: {:.0f}/{}s".format((now - last_state_change_time).total_seconds(), duration_low_s)
                self.logging.info(msg)
            elif current_state == STATE_STEPPING:
                dt = (now - last_state_change_time).total_seconds()
                msg = "Step {}/{}, current voltage: {} V, next step in {:.0f}/{} s"
                msg = msg.format(current_step, nsteps, current_target_voltage, dt, step_duration_s)
                self.logging.info(msg)
            elif current_state == STATE_WAITING_HIGH:
                msg = "Waiting high: {:.0f}/{} s"
                msg = msg.format((now - last_state_change_time).total_seconds(), duration_high_s)
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
            dt_state_change = (now - last_state_change_time).total_seconds()  # time since last state change
            if current_state == STATE_STARTUP:
                new_state = STATE_WAITING_LOW
            elif current_state == STATE_WAITING_LOW:
                if dt_state_change > duration_low_s:
                    new_state = STATE_STEPPING
                    new_step = 1  # first step
                    current_target_voltage = round(voltage_step)
            elif current_state == STATE_STEPPING:
                if dt_state_change > step_duration_s:
                    new_step = current_step + 1
                    current_target_voltage = round(new_step * voltage_step)
                    if new_step > nsteps:
                        new_state = STATE_WAITING_HIGH
                        current_target_voltage = max_voltage
            elif current_state == STATE_WAITING_HIGH:
                if dt_state_change > duration_high_s:
                    new_state = STATE_WAITING_LOW
                    current_target_voltage = 0
            else:
                logging.critical("Unknown state in the state machine")
                raise Exception

            # apply new state if different
            if new_state is not current_state or new_step is not current_step:

                # -----------------------
                # perform actions on state change
                # -----------------------

                # get current voltage
                measuredVoltage = self.hvpsInst.get_current_voltage()

                # get images and measure strain
                imgs, res_imgs, fits, t_img = self.measure_strain()

                # show images
                for i in range(len(res_imgs)):
                    cv.imshow('DEA {}'.format(i), cv.resize(res_imgs[i], (0, 0), fx=0.25, fy=0.25))

                # save images
                # TODO: fix image naming -> (timestamp, name, suffix)
                # TODO: why is reference at 980V ?!
                suffix = "{}V".format(measuredVoltage)  # store voltage in file name
                if current_state == STATE_STARTUP:
                    suffix += " reference"  # this is the first image we're storing, so it will be the reference
                imsaver.save_all(imgs, t_img, res_imgs, suffix)

                # set voltage for new state
                self.hvpsInst.set_voltage(current_target_voltage)

                current_state = new_state
                current_step = new_step
                last_state_change_time = datetime.now()

            # ------------------------------
            # HVPS feedback and plots
            # ------------------------------
            measuredVoltage = self.hvpsInst.get_current_voltage()
            self.logging.info("Current voltage: {} V".format(measuredVoltage))

            deaState = self.hvpsInst.get_relay_state()
            self.logging.info("DEA state: {}".format(deaState))

            # ------------------------------
            # Save applied voltage
            # ------------------------------
            # TODO: save (current_target_voltage, measuredVoltage)

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

        self.hvpsInst.set_output_off()
        self.hvpsInst.set_voltage(0)
        self.logging.critical("apply voltage quit")

    def measure_strain(self, output_result_image=True):
        cap = ImageCapture.SharedInstance
        imgs = cap.get_images()
        # use timestamp of the last image for the whole set so they have a realistic and matching timestamp
        timestamp = cap.get_timestamp(cap.get_camera_count() - 1)
        ellipses = [StrainDetection.dea_fit_ellipse(img) for img in imgs]  # get fit for each DEA

        res_imgs = None
        if output_result_image:
            res_imgs = [StrainDetection.draw_ellipse(imgs[i], ellipses[i]) for i in range(len(imgs))]

        return imgs, res_imgs, ellipses, timestamp


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
    ImageCapture.SharedInstance.select_cameras(
        cam_order)  # apply cam order to image capture so we can just address them as

    nerd = NERD(com_port, cam_order, voltage)
    nerd.run_nogui()

    # ex = App(args=None)
    # retVal = app.exec_()
    retVal = 0

    logging.info('Finished with val: {}'.format(retVal))
    sys.exit(retVal)
