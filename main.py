import logging
import logging.config
import sys
import time
from datetime import datetime

import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QApplication, QMessageBox

from libs import duallog
from src.fileio.DataSaver import DataSaver
from src.fileio.ImageSaver import ImageSaver
from src.fileio.config import read_config, write_config
from src.gui import SetupDialog, Screen
from src.hvps.Switchboard import SwitchBoard
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


def _setup():
    config_file_name = "config.txt"

    try:
        config = read_config(config_file_name)
        logging.info("Read config file")
    except Exception as ex:
        logging.warning("Unable to read config file: {}".format(ex))
        config = {}

    # Show setup dialog to get COM port and camera order
    setup_dialog = SetupDialog.SetupDialog(config)

    if setup_dialog.exec_():
        config, strain_detector = setup_dialog.get_results()
        logging.info("{}".format(config))

        try:
            write_config(config_file_name, config)
            logging.info("Wrote config file")
        except Exception as ex:
            logging.warning("Unable to write config file: {}".format(ex))
    else:
        logging.critical("Aborted by user")
        raise Exception("Aborted by user")

    return config, strain_detector


class NERD:

    def __init__(self, config, strain_detector=None):
        self.logging = logging.getLogger("NERD")
        self.config = config
        self.image_cap = ImageCapture.SharedInstance
        # TODO: implement robust system for getting number of DEAs and dealing with unexpected number of images etc.
        self.n_dea = self.image_cap.get_camera_count()  # this works if we previously selected the cameras
        self.strain_detector = strain_detector

        # connect to HVPS
        try:
            self.hvpsInst = SwitchBoard()
            self.hvpsInst.open(self.config["com_port"])
        except Exception as ex:
            self.logging.critical("Unable to connect to HVPS: {}".format(ex))
            QMessageBox.critical(None, "Unable to connect to HVPS/Switchboard!",
                                 "Could not connect to HVPS/Switchboard", QMessageBox.Ok)

        # init some internal variable
        self.shutdown_flag = False

    def run_nogui(self):
        self.logging.info("Running NERD protocol with {} DEAs in main thread without dedicated GUI "
                          "(using openCV windows for visual output".format(self.n_dea))

        # set up output folder and image and data saver ################################################

        now_tstamp = datetime.now()
        session_name = "NERD test {}".format(now_tstamp.strftime("%Y%m%d-%H%M%S"))
        dir_name = "output/{}".format(session_name)

        cap = ImageCapture.SharedInstance  # get image capture

        # create an image saver for this session to store the recorded images
        imsaver = ImageSaver(dir_name, self.n_dea, save_result_images=True)

        # TODO: handle varying numbers of DEAs
        save_file_name = "{}/{} data.csv".format(dir_name, session_name)
        saver = DataSaver(self.n_dea, save_file_name)

        # store strain reference ##########################################################

        if self.strain_detector is None:
            self.strain_detector = StrainDetection.StrainDetector()

        if not self.strain_detector.has_reference():
            self.strain_detector.set_reference(self.image_cap.read_images())

        ref_imgs, ref_res_imgs = self.strain_detector.get_reference_images()
        imsaver.save_all(ref_imgs, now_tstamp, ref_res_imgs, suffix="reference")

        time.sleep(1)  # just wait a second so the timestamp for the first image is not the same as the reference

        # calculate image size for GUI #######################################################################

        # preview_image_size = (720, 405)
        img_shape = ref_imgs[0].shape
        preview_image_size = Screen.get_max_size_on_screen((img_shape[1], img_shape[0]), (2, 3), (20, 60))

        # apply user config ######################################################################

        max_voltage = self.config["voltage"]
        min_voltage = 300
        # TODO: check why min voltage is 300 V (officially 50, but doesn't always seem to work)
        nsteps = self.config["steps"]
        voltage_step = max_voltage / nsteps

        duration_low_s = self.config["low_duration_s"]
        duration_high_s = self.config["high_duration_min"] * 60  # convert to seconds
        step_duration_s = self.config["step_duration_s"]
        measurement_period_s = self.config["measurement_period_s"]
        image_capture_period_s = self.config["save_image_period_min"] * 60  # convert to seconds

        ac_mode = self.config["ac_mode"]
        ac_frequency = self.config["ac_frequency"]
        ac_cycle_count = np.ceil(duration_high_s * ac_frequency)
        ac_cycle_count = max(ac_cycle_count, 1)  # make sure it's at least 1, otherwise the HVPS will cycle indefinitely
        cycles_remaining = 0  # used to store remaining cycles when interrupting for a measurement

        # set up HVPS ##############################################################################

        self.hvpsInst.set_switching_mode(1)  # set to DC mode by default to make sure that the HV indicator LED works
        self.hvpsInst.set_voltage(0, block_if_testing=True)  # make sure we start at 0 V
        self.hvpsInst.set_relay_auto_mode()  # enable auto mode by default

        # set up state machine #############################################################

        # possible states for state machine
        STATE_STARTUP = 0
        STATE_RAMP = 1
        STATE_WAITING_HIGH = 2
        STATE_WAITING_LOW = 3

        # initial state
        current_state = STATE_STARTUP  # start with a ramp
        current_step = 0
        current_target_voltage = 0
        prev_V_high = False

        # record start time
        now = time.monotonic()
        time_started = now
        time_last_state_change = now
        time_last_image_saved = now
        time_last_measurement = now
        time_last_voltage_measurement = now
        duration_at_max_V = 0

        while self.shutdown_flag is not True:
            now = time.monotonic()
            now_tstamp = datetime.now()

            # ------------------------------
            # state machine: actions (actually just outputs since all the interesting stuff happens on state change)
            # ------------------------------
            dt_state_change = now - time_last_state_change  # time since last state change

            if ac_mode:
                cycles_completed = self.hvpsInst.get_cycle_number()  # check how many cycles have been completed
            else:
                cycles_completed = None

            if current_state == STATE_STARTUP:
                self.logging.info("Starting up state machine")
            elif current_state == STATE_WAITING_LOW:
                msg = "waiting low: {:.0f}/{}s".format(dt_state_change, duration_low_s)
                self.logging.info(msg)
            elif current_state == STATE_RAMP:
                msg = "Step {}/{}, current voltage: {} V, next step in {:.0f}/{} s"
                msg = msg.format(current_step, nsteps, current_target_voltage, dt_state_change, step_duration_s)
                self.logging.info(msg)
            elif current_state == STATE_WAITING_HIGH:
                if ac_mode:
                    msg = "Cyclic actuation: {}/{} cycles"
                    msg = msg.format(*cycles_completed)
                else:
                    msg = "Waiting high: {:.0f}/{} s"
                    msg = msg.format(dt_state_change, duration_high_s)
                self.logging.info(msg)
            else:
                logging.critical("Unknown state in the state machine")
                raise Exception

            # ------------------------------
            # state machine: transitions
            # ------------------------------
            new_state = current_state  # if nothing happens in the following conditions, we keep current state and step
            new_step = current_step
            new_target_voltage = current_target_voltage
            if current_state == STATE_STARTUP:
                new_state = STATE_RAMP
            elif current_state == STATE_WAITING_LOW:
                if dt_state_change > duration_low_s:
                    new_state = STATE_RAMP
                    new_step = 0  # reset steps
            elif current_state == STATE_RAMP:
                if dt_state_change > step_duration_s:
                    new_step = current_step + 1
                    new_target_voltage = round(new_step * voltage_step)
                    # make sure we
                    while new_target_voltage < min_voltage:
                        new_step += 1  # increase step
                        new_target_voltage = round(new_step * voltage_step)
                    if new_step > nsteps:
                        new_state = STATE_WAITING_HIGH
                        new_target_voltage = max_voltage
            elif current_state == STATE_WAITING_HIGH:
                if ac_mode:
                    finished = cycles_completed[0] == 0  # if n cycles turned back to 0, the cycles have been completed
                else:
                    finished = dt_state_change > duration_high_s  # in DC mode, change state after time has elapsed
                if finished:
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

            dt_measurement = now - time_last_measurement  # time since last state change
            measurement_due = dt_measurement > measurement_period_s or state_changing
            # self.logging.debug("Time since last measurement: {} s,  measurement due: {}".format(dt_measurement,
            #                                                                                     measurement_due))

            dt_image_saved = now - time_last_image_saved  # time since last state change
            image_due = dt_image_saved > image_capture_period_s or state_changing
            # self.logging.debug("Time since last image saved: {} s,  New image required: {}".format(dt_image_saved,
            #                                                                                        image_due))

            # get switchboard state and pause test if it's checking for shorts
            el_state = self.hvpsInst.get_relay_state()
            if el_state[0] == 2:  # switchboard is testing
                self.logging.info("Switchboard is testing for short circuits. NERD test paused...")

                # wait for check to be over
                time_pause_started = time.monotonic()
                while el_state[0] == 2:
                    el_state = self.hvpsInst.get_relay_state()
                    time.sleep(0.01)

                # extend time until next state change/measurement/image by the duration of the test
                pause_duration = time.monotonic() - time_pause_started
                time_last_state_change += pause_duration
                time_last_measurement += pause_duration
                time_last_image_saved += pause_duration
                time_last_voltage_measurement += pause_duration

            # measure voltage and record time spent at high voltage
            measuredVoltage = self.hvpsInst.get_current_voltage()
            V_high = abs(measuredVoltage - max_voltage) < 50  # 50 V margin around max voltage counts as high
            if V_high and prev_V_high:
                if current_state == STATE_WAITING_HIGH and ac_mode:
                    duration_at_max_V += (now - time_last_voltage_measurement) / 2  # if AC, it's only on half the time
                else:
                    duration_at_max_V += now - time_last_voltage_measurement
            time_last_voltage_measurement = now
            prev_V_high = V_high

            if measurement_due:

                # if in AC mode, switch to DC for the duration of the measurement
                if current_state == STATE_WAITING_HIGH and ac_mode:
                    self.logging.debug("Suspended AC mode during measurement")
                    self.hvpsInst.set_switching_mode(1)  # set to DC. number of completed cycles will be remembered
                    time_measurement_started = time.monotonic()  # record how long AC was suspended
                else:
                    time_measurement_started = None

                # ------------------------------
                # Record data (voltage, DEA state, images) TODO: resistance, current?
                # ------------------------------

                # voltage and state are recorded on each cycle anyway.
                self.logging.info("Current voltage: {} V".format(measuredVoltage))
                self.logging.info("DEA state: {}".format(el_state))

                # get images
                imgs = cap.read_images()
                # use timestamp of the last image for the whole set so they have a realistic and matching timestamp
                # timestamp = cap.get_timestamps(cap.get_camera_count() - 1)

                # --- resume cycling if in AC mode -------------------------------------
                if current_state == STATE_WAITING_HIGH and ac_mode:
                    cycles_completed = self.hvpsInst.get_cycle_number()[0]  # cycles back to 0 if completed
                    # don't turn AC back on if all cycles completed (otherwise it'll start over)
                    if cycles_completed > 0:
                        self.logging.debug("Resuming AC mode")
                        self.hvpsInst.set_switching_mode(2)  # set back to AC
                    else:  # should be extremely unlikely to happen but let's make a note if it does
                        self.logging.warning("AC mode note resumed after measurement because no more cycles remained")

                    # measurement time was only counted half in AC mode --> add other half
                    time_measurement_ended = time.monotonic()
                    measurement_duration = time_measurement_ended - time_measurement_started
                    duration_at_max_V += measurement_duration / 2

                # ------------------------------
                # Analyze data (strain) TODO: resistance, current?
                # ------------------------------

                # measure strain
                vs = "{} V".format(measuredVoltage)
                strain, center_shifts, res_imgs, vis_state = self.strain_detector.get_dea_strain(imgs, True, True, vs)
                # print average strain for each DEA
                self.logging.info("strain [%]: {}".format(np.reshape(strain[:, -1], (1, -1))))

                if 0 in vis_state:  # 0 means outlier, 1 means OK
                    self.logging.warning("Outlier detected")

                for img, v, e in zip(res_imgs, vis_state, el_state):
                    StrainDetection.draw_state_visualization(img, v, e)

                # ------------------------------
                # Save all data voltage and DEA state
                # ------------------------------
                saver.write_data(now_tstamp,
                                 now - time_started,
                                 duration_at_max_V,
                                 current_state,
                                 current_target_voltage,
                                 measuredVoltage,
                                 el_state,
                                 vis_state,
                                 strain,
                                 center_shifts,
                                 image_saved=image_due)

                time_last_measurement = now

                # -----------------------
                # save images
                # -----------------------
                if image_due:
                    # get image name suffix
                    suffix = "{}V".format(measuredVoltage)  # store voltage in file name

                    # save result images
                    imsaver.save_all(images=imgs, res_images=res_imgs, timestamp=now_tstamp, suffix=suffix)
                    # use now instead of t_img to make sure we have consistent timestamp for all data in this cycle
                    time_last_image_saved = now

                # -----------------------
                # show images
                # -----------------------
                disp_imgs = [cv.resize(ImageCapture.ImageCapture.IMG_NOT_AVAILABLE, preview_image_size)] * 6
                for i in range(6):
                    if len(res_imgs) > i and res_imgs[i] is not None:
                        disp_imgs[i] = cv.resize(res_imgs[i], preview_image_size)

                sep = np.zeros((preview_image_size[1], 3, 3), dtype=np.uint8)
                row1 = np.concatenate((disp_imgs[0], sep, disp_imgs[1], sep, disp_imgs[2]), axis=1)
                row2 = np.concatenate((disp_imgs[3], sep, disp_imgs[4], sep, disp_imgs[5]), axis=1)
                sep = np.zeros((3, row1.shape[1], 3), dtype=np.uint8)
                disp_img = np.concatenate((row1, sep, row2), axis=0)
                cv.imshow("NERD running... (press [q] to exit)", disp_img)

            # ------------------------------
            # Apply state change and update voltage
            # ------------------------------
            if state_changing:
                # set voltage for new state
                current_target_voltage = new_target_voltage
                ret = self.hvpsInst.set_voltage(current_target_voltage, block_if_testing=True)
                if ret is not True:
                    self.logging.debug("Failed to set voltage")

                if new_state == STATE_WAITING_HIGH and ac_mode:
                    cycles_remaining = ac_cycle_count  # reset remaining cycles to the requested number cycles
                    self.hvpsInst.set_cycle_number(cycles_remaining)
                    self.hvpsInst.set_switching_mode(2)

                current_state = new_state
                current_step = new_step
                time_last_state_change = now
                time_last_image_saved = time_last_state_change  # also reset image timer to avoid taking too many

            # ------------------------------
            # Keep track of voltage changes
            # ------------------------------
            # if current_target_voltage != previous_target_voltage:
            #     self.callbackVoltageChange(newVoltage=current_target_voltage)

            # ------------------------------
            # check for user input to stay responsive
            # ------------------------------

            if cv.waitKey(300) & 0xFF == ord('q'):
                self.shutdown_flag = True

        self.logging.critical("Exiting...")
        self.hvpsInst.set_output_off()
        self.hvpsInst.set_voltage(0, block_if_testing=True)
        self.hvpsInst.set_relays_off()
        del self.hvpsInst
        self.logging.critical("Turned voltage off and disconnected relays")
        saver.close()
        self.image_cap.close_cameras()


if __name__ == '__main__':
    duallog.setup("logs", minlevelConsole=logging.DEBUG, minLevelFile=logging.DEBUG)
    logging.info("Started application")

    # launch Qt
    app = QApplication(sys.argv)

    # run setup
    _config, _strain_detector = _setup()
    # apply cam order to image capture so we can just address them as cam 0, 1, 2, ...
    ImageCapture.SharedInstance.select_cameras(_config["cam_order"])

    nerd = NERD(_config, _strain_detector)
    nerd.run_nogui()

    # ex = App(args=None)
    # retVal = app.exec_()
    retVal = 0

    logging.info('Finished with val: {}'.format(retVal))
    sys.exit(retVal)
