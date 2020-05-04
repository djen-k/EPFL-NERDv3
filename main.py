import logging
import logging.config
import sys
import time
from datetime import datetime

import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import QApplication

from libs import duallog
from src.fileio.DataSaver import DataSaver
from src.fileio.ImageSaver import ImageSaver
from src.fileio.config import read_config, write_config
from src.gui import SetupDialog, Screen
from src.hvps.Switchboard import SwitchBoard
from src.image_processing import ImageCapture, StrainDetection
from src.measurement.keithley import DAQ6510


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
        # TODO: implement robust system for dealing with unexpected number of images etc.
        # to get active DEAs, turn all -1 (=disabled) in cam order to 0, then find non-zeros (returns a tuple of arrays)
        self.active_deas = np.nonzero(np.array(config["cam_order"]) + 1)[0].tolist()
        self.n_dea = len(self.active_deas)
        self.strain_detector = strain_detector

        # connect to HVPS
        try:
            self.hvps = SwitchBoard()
            self.hvps.open(self.config["com_port"])
        except Exception as ex:
            self.logging.error("Unable to connect to HVPS: {}".format(ex))
            raise ex

        # connect to multimeter
        if "daq_id" in self.config:
            daq_id = self.config["daq_id"]
        else:
            daq_id = None  # --> find DAQ automatically

        if daq_id == "None":  # explicitly no multimeter selected by user
            self.logging.info("Running NERD test without multimeter.")
            self.daq = None
        else:
            self.daq = DAQ6510()
            daq_connected = self.daq.connect(daq_id)
            if not daq_connected:
                self.logging.warning("Unable to connect to multimeter! Running test without multimeter.")
                self.daq = None

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
        imsaver = ImageSaver(dir_name, self.active_deas, save_result_images=True)

        # TODO: handle varying numbers of DEAs
        save_file_name = "{}/{} data.csv".format(dir_name, session_name)
        saver = DataSaver(self.active_deas, save_file_name)

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
        voltage_step = max_voltage / nsteps  # can be fractional, will be rounded when voltage is set

        duration_low_s = self.config["low_duration_s"]
        duration_high_s = self.config["high_duration_min"] * 60  # convert to seconds
        step_duration_s = self.config["step_duration_s"]
        measurement_period_s = self.config["measurement_period_s"]
        image_capture_period_s = self.config["save_image_period_min"] * 60  # convert to seconds

        ac_mode = self.config["ac_mode"]
        ac_frequency = self.config["ac_frequency_hz"]
        ac_cycle_count = np.ceil(duration_high_s * ac_frequency)
        ac_cycle_count = max(ac_cycle_count, 1)  # make sure it's at least 1, otherwise the HVPS will cycle indefinitely
        ac_wait_before_measurement = 5  # self.config["ac_wait_before_measurement_s"]

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
        measured_voltage = 0
        prev_V_high = False
        Rdea = None
        leakage_current = None
        leakage_cur_avg = None
        leakage_buf = []
        dea_state_el = None  # previous electrical state - to check if a DEA failed
        breakdown_occurred = False
        failed_deas = []

        # record start time
        now = time.perf_counter()
        time_started = now
        time_last_state_change = now
        time_last_image_saved = now
        time_last_measurement = now
        time_last_voltage_measurement = now
        duration_at_max_V = 0

        # TODO: set switching mode in DC mode
        # TODO: enable only active channels, once this is supported by the switchboard
        self.hvps.set_relay_auto_mode()  # enable relay auto mode for automatic short detection. Opens all channels.
        self.hvps.set_switching_mode(1)  # start in DC mode

        while self.shutdown_flag is not True:
            now = time.perf_counter()
            now_tstamp = datetime.now()

            # ------------------------------
            # state machine: actions (actually just outputs since all the interesting stuff happens on state change)
            # ------------------------------
            dt_state_change = now - time_last_state_change  # time since last state change

            if ac_mode:
                cycles_completed = self.hvps.get_cycle_number()  # check how many cycles have been completed
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
                    # keep increasing step-wise until we are above the min voltage
                    while new_target_voltage < min_voltage and new_step <= nsteps:
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

            ##############################################################
            # check what actions need to happen this cycle
            ##############################################################

            # check if state will change at then end of this cycle
            state_changing = new_state is not current_state or new_step is not current_step
            if state_changing:
                msg = "State changeing from {} (step {}) to {} (step {})"
                self.logging.debug(msg.format(current_state, current_step, new_state, new_step))

            dt_measurement = now - time_last_measurement  # time since last state change
            measurement_due = dt_measurement > measurement_period_s or state_changing or breakdown_occurred
            # if breakdown occurred last cycle, we also want a measurement now so we have before and after
            # self.logging.debug("Time since last measurement: {} s,  measurement due: {}".format(dt_measurement,
            #                                                                                     measurement_due))

            dt_image_saved = now - time_last_image_saved  # time since last state change
            image_due = dt_image_saved > image_capture_period_s or state_changing or breakdown_occurred
            # breakdown_occurred will be true here if the breakdown happened last cycle.
            # Thus, the next image after breakdown is also stored
            # self.logging.debug("Time since last image saved: {} s,  New image required: {}".format(dt_image_saved,
            #                                                                                        image_due))

            ####################################################################
            # pause test if switchboard is checking for shorts, then check if a breakdown occurred
            ####################################################################

            # TODO: make sure switchboard is still working correctly after a reconnect
            dea_state_el_new = self.hvps.get_relay_state()
            if dea_state_el_new[0] == 2:  # switchboard is testing
                self.logging.info("Switchboard is testing for short circuits. NERD test paused...")

                # wait for check to be over
                time_pause_started = time.perf_counter()
                while dea_state_el_new[0] == 2:
                    dea_state_el_new = self.hvps.get_relay_state()
                    time.sleep(0.01)

                # extend time until next state change/measurement/image by the duration of the test
                pause_duration = time.perf_counter() - time_pause_started
                time_last_state_change += pause_duration
                time_last_measurement += pause_duration
                time_last_image_saved += pause_duration
                time_last_voltage_measurement += pause_duration

                self.logging.info("Short circuit detection finished: {} Resuming test.".format(dea_state_el_new))

            # check which samples broke down
            if dea_state_el is not None:
                failed_deas = np.nonzero(np.array(dea_state_el_new) - np.array(dea_state_el))
                failed_deas = failed_deas[0]  # np.nonzero returns tuple - we only want the first value
                breakdown_occurred = len(failed_deas) > 0

            if breakdown_occurred:
                self.logging.info("Breakdown detected! DEAs: {}".format(failed_deas))
                image_due = True
            else:  # no breakdown, so we don't need the prev state anymore. If breakdown we record the prev state first
                dea_state_el = dea_state_el_new

            ###########################################################################
            # Measurements
            ###########################################################################

            # First: fast measurements that are carried out every cycle ###################

            if not breakdown_occurred:  # don't record new data so we can save last data from before the breakdown

                # measure voltage
                measured_voltage = self.hvps.get_current_voltage(from_buffer_if_available=True)

                # measure leakage current
                if self.daq is not None:
                    leakage_current = self.daq.measure_current(nplc=5)  # measure total current for n power line cycles
                    leakage_buf.append(leakage_current)  # append to buffer so we can average when we write the data

                # capture images
                self.image_cap.grab_images()  # tell each camera to grab a frame - will only be retrieved if needed

                # record time spent at high voltage
                V_high = abs(measured_voltage - max_voltage) < 50  # 50 V margin around max voltage counts as high
                if V_high and prev_V_high:
                    if current_state == STATE_WAITING_HIGH and ac_mode:
                        duration_at_max_V += (now - time_last_voltage_measurement) / 2  # if AC, only on half the time
                    else:
                        duration_at_max_V += now - time_last_voltage_measurement
                time_last_voltage_measurement = now
                prev_V_high = V_high
            else:
                leakage_cur_avg = leakage_current  # no averaging, just take last available reading before breakdown

            if measurement_due and not breakdown_occurred:

                # if in AC mode, switch to DC for the duration of the measurement
                if current_state == STATE_WAITING_HIGH and ac_mode and not breakdown_occurred:
                    self.logging.debug("Suspended AC mode during measurement")
                    self.hvps.set_switching_mode(1)  # set to DC. number of completed cycles will be remembered
                    time_measurement_started = time.perf_counter()  # record how long AC was suspended

                    time.sleep(ac_wait_before_measurement)  # wait high for a few seconds before measurement
                else:
                    time_measurement_started = None

                # ------------------------------
                # Record data: resistance, leakage current
                # ------------------------------
                # TODO: make all measurements fail-safe so the test continues running if any instrument fails

                self.logging.info("Current voltage: {} V".format(measured_voltage))
                self.logging.info("DEA state: {}".format(dea_state_el))

                if self.daq is not None:  # perform electrical measurements if possible

                    # measure resistance
                    Rdea = self.daq.measure_DEA_resistance(self.active_deas, n_measurements=1, nplc=3)  # 1-D np array
                    self.logging.info("Resistance [kΩ]: {}".format(Rdea / 1000))

                    # aggregate current measurements
                    if ac_mode or len(leakage_buf) == 0:
                        # can't use buffered measurements in AC mode since they might have been taken while switching
                        self.logging.debug("no leakage measurements in buffer. recording new one...")
                        leakage_current = self.daq.measure_current(nplc=3)  # take new measurement
                        leakage_cur_avg = leakage_current  # nothing to average so we take the newly recorded measurement
                    else:
                        leakage_cur_avg = np.mean(leakage_buf)  # average all current readings since the last time
                    leakage_buf = []  # reset buffer
                    self.logging.info("Leakage current [nA]: {}".format(leakage_cur_avg * 1000000000))

                # --- resume cycling if in AC mode -------------------------------------
                if current_state == STATE_WAITING_HIGH and ac_mode:
                    cycles_completed = self.hvps.get_cycle_number()[0]  # cycles back to 0 if completed
                    # don't turn AC back on if all cycles completed (otherwise it'll start over)
                    if cycles_completed > 0:
                        self.logging.debug("Resuming AC mode")
                        self.hvps.set_switching_mode(2)  # set back to AC
                    else:  # should be extremely unlikely to happen but let's make a note if it does
                        self.logging.warning("AC mode note resumed after measurement because no more cycles remained")

                    # measurement time was only counted half in AC mode --> add other half
                    time_measurement_ended = time.perf_counter()
                    measurement_duration = time_measurement_ended - time_measurement_started
                    duration_at_max_V += measurement_duration / 2

            if measurement_due or breakdown_occurred:  # if breakdown, data from previous cycle is saved
                # ------------------------------
                # Analyze data: images/strain, ... (current and resistance needs no analysis) TODO: partial discharge
                # ------------------------------

                # compile labels to print on the result images (showing voltage and resistance)
                label = "{}V".format(measured_voltage)
                if leakage_cur_avg is not None:
                    label += "  {}nA".format(round(leakage_cur_avg * 10000000000) / 10)  # convert to nA with 1 decimal
                if Rdea is not None:
                    R_kohm = np.round(Rdea / 100) / 10  # convert to kOhm with one decimal place
                    res_img_labels = [label + "  {}kOhm".format(r) for r in R_kohm]
                else:
                    res_img_labels = [label] * self.n_dea

                # retrieve the images that were grabbed earlier
                imgs = cap.retrieve_images()[1]  # get only the images, not the success flags
                # measure strain and get result images
                strain_res = self.strain_detector.get_dea_strain(imgs, True, True, res_img_labels)
                strain, center_shifts, res_imgs, dea_state_vis = strain_res
                # print average strain for each DEA
                self.logging.info("strain [%]: {}".format(np.reshape(strain[:, -1], (1, -1))))

                if 0 in dea_state_vis:  # 0 means outlier, 1 means OK
                    self.logging.warning("Outlier detected")

                for img, v, e in zip(res_imgs, dea_state_vis, dea_state_el):
                    StrainDetection.draw_state_visualization(img, v, e)

                # ------------------------------
                # Save all data voltage and DEA state
                # ------------------------------
                # TODO: record total number of cycles in AC mode
                saver.write_data(now_tstamp,
                                 now - time_started,
                                 duration_at_max_V,
                                 current_state,
                                 current_target_voltage,
                                 measured_voltage,
                                 dea_state_el,
                                 dea_state_vis,
                                 strain,
                                 center_shifts,
                                 Rdea,
                                 leakage_cur_avg,
                                 image_saved=image_due)

                time_last_measurement = now

                # -----------------------
                # save images
                # -----------------------
                if image_due:
                    # get image name suffix
                    suffix = "{}V".format(measured_voltage)  # store voltage in file name

                    # save result images
                    imsaver.save_all(images=imgs, res_images=res_imgs, timestamp=now_tstamp, suffix=suffix)
                    # use now instead of t_img to make sure we have consistent timestamp for all data in this cycle
                    time_last_image_saved = now

                # -----------------------
                # show images
                # -----------------------
                disp_imgs = [cv.resize(ImageCapture.ImageCapture.IMG_NOT_AVAILABLE, preview_image_size,
                                       interpolation=cv.INTER_AREA)] * 6
                for i in range(6):
                    if len(res_imgs) > i and res_imgs[i] is not None:
                        disp_imgs[i] = cv.resize(res_imgs[i], preview_image_size, interpolation=cv.INTER_AREA)

                sep = np.zeros((preview_image_size[1], 3, 3), dtype=np.uint8)
                row1 = np.concatenate((disp_imgs[0], sep, disp_imgs[1], sep, disp_imgs[2]), axis=1)
                row2 = np.concatenate((disp_imgs[3], sep, disp_imgs[4], sep, disp_imgs[5]), axis=1)
                sep = np.zeros((3, row1.shape[1], 3), dtype=np.uint8)
                disp_img = np.concatenate((row1, sep, row2), axis=0)
                cv.imshow("NERD running... (press [q] to exit)", disp_img)

            if breakdown_occurred:  # we still need to update the state with the latest one
                dea_state_el = dea_state_el_new  # because we kept the previous state in memory in order to record it

            # ------------------------------
            # Apply state change and update voltage
            # ------------------------------
            if state_changing:
                # set voltage for new state
                current_target_voltage = new_target_voltage
                ret = self.hvps.set_voltage_no_overshoot(current_target_voltage)
                if ret is not True:
                    self.logging.debug("Failed to set voltage")

                if new_state == STATE_WAITING_HIGH and ac_mode:
                    self.hvps.set_cycle_number(ac_cycle_count)  # set the requested number cycles
                    self.hvps.set_switching_mode(2)
                else:
                    self.hvps.set_switching_mode(1)  # anything but high phase in AC mode requires DC

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
        self.hvps.set_output_off()
        self.hvps.set_voltage(0, block_until_reached=True)
        self.hvps.set_relays_off()
        self.hvps.close()
        self.logging.critical("Turned voltage off and disconnected relays")
        saver.close()
        self.image_cap.close_cameras()
        self.daq.disconnect()


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
