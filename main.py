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
from src.hvps.Switchboard import Switchboard
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

    # set fixed exposure to avoid changes in lighting due to auto exposure
    # TODO: make this an adjustable setting stored in config file
    ImageCapture.SharedInstance.set_fixed_exposure(-9)

    # Show setup dialog to get COM port and camera order
    setup_dialog = SetupDialog.SetupDialog(config)

    ok = setup_dialog.exec_()

    config, strain_detector = setup_dialog.get_results()
    logging.info("{}".format(config))

    try:
        write_config(config_file_name, config)
        logging.info("Wrote config file")
    except Exception as ex:
        logging.warning("Unable to write config file: {}".format(ex))

    if not ok:
        logging.critical("Aborted by user")
        raise Exception("Aborted by user")

    return config, strain_detector


class NERD:

    def __init__(self, config, strain_detector=None):
        self.logging = logging.getLogger("NERD")
        self.config = config

        # TODO: implement robust system for dealing with unexpected number of images etc.
        # to get active DEAs, turn all -1 (=disabled) in cam order to 0, then find non-zeros (returns a tuple of arrays)
        if "active_DEAs" in config:
            active_deas = config["active_DEAs"]
        else:
            active_deas = [1] * 6

        if "cam_order" in config:
            active_deas = np.array(active_deas) == 1  # convert to logical numpy array
            cam_order = np.array(config["cam_order"], dtype=int)
            cam_order[~active_deas] = -1  # de-select cameras of disabled samples
            # find the DEAs that have a camera assigned and are enabled
            active_deas = cam_order > -1  # derive from cam order so any slot that doesn't have a cam is disabled
            # convert to list of indices
            cam_order = cam_order.tolist()  # convert to list, just to be sure the ImageCapture can handle it
        else:
            cam_order = list(range(6))  # no order specified -> use default order

        self.active_deas = active_deas  # boolean numpy array
        self.active_dea_indices = np.nonzero(active_deas)[0].tolist()
        self.n_deas = len(self.active_dea_indices)
        self.cam_order = cam_order

        # apply cam order to image capture so the indices of DEAs and cameras match
        self.image_cap = ImageCapture.SharedInstance
        self.image_cap.select_cameras(cam_order)

        self.strain_detector = strain_detector
        # we don't apply sample selection to strain detector here because we want to save all reference images first

        # connect to HVPS
        try:
            self.hvps = Switchboard()
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
            daq_connected = self.daq.connect(daq_id, reset=True)
            if not daq_connected:
                self.logging.warning("Unable to connect to multimeter! Running test without multimeter.")
                self.daq = None
            self.daq.reset()  # reset to make sure we're in default state at the beginning

        # init some internal variable
        self.shutdown_flag = False

    def run_nogui(self):
        self.logging.info("Running NERD protocol with {} DEAs: {}".format(self.n_deas, self.active_dea_indices))

        # set up output folder and image and data saver ################################################

        now_tstamp = datetime.now()
        session_name = "NERD test {}".format(now_tstamp.strftime("%Y%m%d-%H%M%S"))
        if self.config["title"]:
            session_name += " " + self.config["title"]
        dir_name = "output/{}".format(session_name)

        # create an image saver for this session to store the recorded images
        imsaver = ImageSaver(dir_name, self.active_dea_indices, save_result_images=True)

        # TODO: handle varying numbers of DEAs
        save_file_name = "{}/{} data.csv".format(dir_name, session_name)
        saver = DataSaver(self.active_dea_indices, save_file_name)

        # set up disruption log
        fileHandler = logging.FileHandler("{}/{} disruptions.log".format(dir_name, session_name))
        logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        fileHandler.setFormatter(logFormatter)
        logging.getLogger("Disruption").addHandler(fileHandler)

        # store strain reference ##########################################################

        if self.strain_detector is None:
            self.strain_detector = StrainDetection.StrainDetector()

        if not self.strain_detector.has_reference():
            self.strain_detector.set_reference(self.image_cap.read_images())

        ref_imgs, ref_res_imgs = self.strain_detector.get_reference_images()
        imsaver.save_all(ref_imgs, now_tstamp, ref_res_imgs, suffix="reference")
        ref_selection = self.active_deas[np.array(self.cam_order) > -1]
        ref_selection = np.nonzero(ref_selection)[0].tolist()
        self.strain_detector.select_reference(ref_selection)

        time.sleep(1)  # just wait a second so the timestamp for the first image is not the same as the reference

        # calculate image size for GUI #######################################################################

        # preview_image_size = (720, 405)
        img_shape = ref_imgs[0].shape
        preview_image_size = Screen.get_max_size_on_screen((img_shape[1], img_shape[0]), (2, 3), (20, 60))

        # apply user config ######################################################################

        max_voltage = self.config["voltage"]
        if max_voltage > self.hvps.maximum_voltage:
            msg = "Target voltage is above the maximum voltage of the HVPS. Using the maximum ({} V) instead."
            self.logging.warning(msg.format(self.hvps.maximum_voltage))
            max_voltage = self.hvps.maximum_voltage
        min_voltage = self.hvps.minimum_voltage
        nsteps = self.config["steps"]
        if nsteps == 0:
            voltage_step = 0
        else:
            voltage_step = max_voltage / nsteps  # can be fractional, will be rounded when voltage is set

        duration_low_s = self.config["low_duration_s"]
        duration_high_s = self.config["high_duration_min"] * 60  # convert to seconds
        step_duration_s = self.config["step_duration_s"]
        measurement_period_s = self.config["measurement_period_s"]
        image_capture_period_s = self.config["save_image_period_min"] * 60  # convert to seconds

        ac_mode = self.config["ac_mode"]
        ac_frequency = self.config["ac_frequency_hz"]
        ac_wait_before_measurement = self.config["ac_wait_before_measurement_s"]

        reverse_polarity_mode = self.config["reverse_polarity"]

        # set up state machine #############################################################

        # possible states for state machine
        STATE_STARTUP = 0
        STATE_RAMP = 1
        STATE_WAITING_HIGH = 2
        STATE_WAITING_LOW = 3

        # H-bridge output modes
        hb_current_output_mode = Switchboard.MODE_DC  # let's start with positive DC

        # initial state
        current_state = STATE_STARTUP  # start with a ramp
        current_step = 0
        ac_active = False
        current_target_voltage = 0
        measured_voltage = 0
        prev_V_high = False
        Rdea = None
        V_shunt = None
        V_source = None
        V_DEA = None
        leakage_current = None
        leakage_cur_avg = None
        leakage_buf = []
        dea_state_el = None  # previous electrical state - to check if a DEA failed
        breakdown_occurred = False
        failed_deas = []
        cycles_completed = 0
        ac_paused = False  # indicated if cycling is currently paused (for measurement or breakdown detection)
        imgs = None
        outlier_prev = np.array([False] * self.n_deas)

        # record start time
        now = time.perf_counter()
        time_started = now
        time_last_state_change = now
        time_last_message = now
        time_last_image_saved = now
        time_last_measurement = now
        time_last_voltage_measurement = now
        time_pause_started = -1
        duration_at_max_V = 0

        # TODO: check PID gains
        self.hvps.set_pid_gains((0.15, 1.0, 0.0))  # make sure we're using the correct gains to avoid voltage spikes
        # self.hvps.set_pid_gains((0.2, 1.0, 0.005))  # make sure we're using the correct gains to avoid voltage spikes

        self.hvps.set_HB_mode(1)
        # enable relay auto mode for selected channels
        self.hvps.set_relay_auto_mode(reset_time=0, relays=self.active_dea_indices)
        hvps_log_file = "{}/{} hvps log.csv".format(dir_name, session_name)
        self.hvps.start_continuous_reading(buffer_length=1, reference_time=time_started, log_file=hvps_log_file)
        self.hvps.enable_dead_man_switch(5)  # make sure that the HVPS turns off HV output if connection is lost

        while self.shutdown_flag is not True:
            now = time.perf_counter()
            now_tstamp = datetime.now()

            # ------------------------------
            # state machine: actions (actually just outputs since all the interesting stuff happens on state change)
            # ------------------------------
            dt_state_change = now - time_last_state_change  # time since last state change
            dt_message = now - time_last_message

            if ac_mode:
                # check how many cycles have been completed
                cycles_completed = self.hvps.get_OC_cycles()

            if dt_message > 1:  # write current state to log file every second
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
                    msg = "Waiting high: {:.0f}/{} s".format(dt_state_change, duration_high_s)
                    if ac_mode:
                        msg += " ({:.0f} cycles)".format(cycles_completed)
                    self.logging.info(msg)
                else:
                    logging.critical("Unknown state in the state machine")
                    raise Exception
                time_last_message = now

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
                    new_target_voltage = -1
                    # keep increasing step-wise until we are above the min voltage
                    while new_target_voltage < min_voltage and new_step <= nsteps:
                        new_step += 1  # increase step
                        new_target_voltage = round(new_step * voltage_step)
                    if new_step > nsteps:
                        new_state = STATE_WAITING_HIGH
                        new_target_voltage = max_voltage
            elif current_state == STATE_WAITING_HIGH:
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
            state_changing = new_state is not current_state
            step_changing = new_step is not current_step
            if state_changing and not ac_paused:
                msg = "State changing from {} to {}"
                self.logging.info(msg.format(current_state, new_state))
            elif step_changing:
                msg = "Ramp step changing from {} to {}"
                self.logging.info(msg.format(current_step, new_step))

            # if breakdown happened last loop cycle, let's record another measurement now immediately after
            if breakdown_occurred:  # this means breakdown happened in previous loop
                time_last_measurement -= measurement_period_s  # makes sure that a measurement is due immediately
                time_last_image_saved -= image_capture_period_s  # makes sure that an image is due immediately

            interval_passed = now - time_last_measurement > measurement_period_s  # check if it's time for measurement
            measurement_due = interval_passed or state_changing or step_changing
            # if breakdown occurred last cycle, we also want a measurement now so we have before and after
            # self.logging.debug("Time since last measurement: {} s,  measurement due: {}".format(dt_measurement,
            #                                                                                     measurement_due))

            interval_passed = now - time_last_image_saved > image_capture_period_s  # check if it's time to store image
            image_due = interval_passed or state_changing
            # breakdown_occurred will be true here if the breakdown happened last cycle.
            # Thus, the next image after breakdown is also stored
            # self.logging.debug("Time since last image saved: {} s,  New image required: {}".format(dt_image_saved,
            #                                                                                        image_due))

            ####################################################################
            # pause test if switchboard is checking for shorts, then check if a breakdown occurred
            ####################################################################

            if self.hvps.is_testing():
                self.logging.info("Switchboard is testing for short circuits. NERD test paused...")

                # wait for check to be over
                time_pause_started = time.perf_counter()  # measure how long it took

                if ac_active:
                    self.hvps.set_OC_mode(Switchboard.MODE_DC)  # disable AC while testing
                    ac_paused = True
                    # will be resumed later after we know if breakdown occurred or not

                while self.hvps.is_testing():
                    time.sleep(0.5)  # wait for test to finish

                self.logging.info("Short circuit detection finished. Resuming test.")

                # extend time until next state change/measurement/image by the duration of the test
                pause_duration = time.perf_counter() - time_pause_started
                time_last_state_change += pause_duration
                time_last_measurement += pause_duration
                time_last_image_saved += pause_duration
                time_last_voltage_measurement += pause_duration
                prev_V_high = False  # set to False so the time of the test is not counted as time at high Voltage

            dea_state_el_new = self.hvps.get_relay_state()  # check DEA electrical state (can be None if invalid reply)

            # check if samples broke down
            if dea_state_el is not None and dea_state_el_new is not None:
                failed_deas = np.nonzero(np.array(dea_state_el_new) - np.array(dea_state_el) == -1)
                failed_deas = failed_deas[0]  # np.nonzero returns tuple - we only want the first value
                breakdown_occurred = len(failed_deas) > 0

            if breakdown_occurred:
                # if all relays are off and more than one DEA "failed" at once, it's probably due to a reset
                # TODO: fix this so we don't get stuck here at the end of a test if multiple DEAs failed at once

                if dea_state_el_new.count(0) == 6 and len(failed_deas) > 1:
                    # TODO: re-enable only channels that haven't already failed
                    msg = "All relays are disabled - presumable due to a reset. DEAs will be reconnected: {}"
                    self.logging.info(msg.format(failed_deas))
                    # must have been reset. re-enable auto mode
                    self.hvps.set_relay_auto_mode(reset_time=0, relays=self.active_dea_indices)
                    continue

                self.logging.info("Breakdown detected! DEAs: {}".format(failed_deas))
                image_due = True
            else:  # no breakdown
                if ac_paused and not measurement_due:
                    self.logging.debug("Re-enabling AC mode after pause")
                    self.hvps.set_OC_mode(Switchboard.MODE_AC)  # re-enable AC after testing if no breakdown occurred
                    ac_paused = False

                # we don't need the prev DEA state anymore if there was no breakdown (otherwise we need to record it)
                if dea_state_el_new is not None:
                    dea_state_el = dea_state_el_new

            ###########################################################################
            # Measurements
            ###########################################################################

            # First: fast measurements that are carried out every cycle ###################

            if not breakdown_occurred:  # don't record new data so we can save last data from before the breakdown

                # measure voltage
                measured_voltage = self.hvps.get_current_voltage(True)

                # measure leakage current
                if self.daq is not None and not ac_active:
                    # quick current measurement
                    leakage_current = self.daq.measure_current(nplc=1, front=reverse_polarity_mode)
                    if leakage_current is not None:
                        leakage_buf.append(leakage_current)  # append to buffer so we can average when we write the data

                # capture images
                imgs = self.image_cap.read_images()  # get one set of images
                # check if there are outliers (large image deviation) and try to recapture
                # if it's due to a glitch, we can hopefully get a good image within a few tries
                img_dev = self.strain_detector.get_deviation_from_reference(imgs)
                outlier = np.array(img_dev) > self.strain_detector.image_deviation_threshold
                new_outlier = np.bitwise_and(outlier, np.bitwise_not(outlier_prev))
                if any(new_outlier):
                    new_outlier_idx = np.nonzero(new_outlier)[0]
                    self.logging.info("Outliers in new image set: {}".format(new_outlier_idx))
                    for i in new_outlier_idx:
                        counter = 0
                        idx = int(i)  # to make sure it's a proper int and not some numpy type
                        while outlier[idx]:
                            imgs[idx] = self.image_cap.read_single_image(idx)
                            dev = self.strain_detector.get_deviation_from_reference_single(imgs[idx], idx)
                            outlier[idx] = dev > self.strain_detector.image_deviation_threshold
                            counter += 1
                            if counter > 5:
                                self.logging.info("5 bad images in a row. Probably not a glitch then...")
                                break
                        if counter <= 5:
                            self.logging.info("Glitched image ({}) successfully recaptured".format(idx))
                outlier_prev = outlier

                # record time spent at high voltage
                V_high = abs(measured_voltage - max_voltage) < 50  # 50 V margin around max voltage counts as high
                if V_high and prev_V_high:
                    if ac_active and not ac_paused:
                        duration_at_max_V += (now - time_last_voltage_measurement) / 2  # if AC, only on half the time
                    else:
                        duration_at_max_V += now - time_last_voltage_measurement
                time_last_voltage_measurement = now
                prev_V_high = V_high
            else:
                leakage_cur_avg = leakage_current  # no averaging, just take last available reading before breakdown

            if measurement_due and not breakdown_occurred:

                # if in AC mode, switch to DC for the duration of the measurement
                if ac_active:
                    if time_pause_started == -1:
                        self.logging.debug("Suspending AC mode in preparation for measurement")
                        # when switching from AC to DC, a voltage spike occurs due to the sudden drop in current
                        # turn off so DC voltage can stabilize and we don't overshoot
                        self.hvps.set_OC_mode(Switchboard.MODE_OFF)
                        reduced_V = max(min_voltage, int(current_target_voltage * 0.5))  # don't go below minimum V
                        self.hvps.set_voltage(reduced_V)  # reduce voltage temporarily
                        # time.sleep(1)
                        self.hvps.wait_until_stable()
                        # set to DC. number of completed cycles will be remembered
                        self.hvps.set_OC_mode(Switchboard.MODE_DC)
                        self.hvps.set_voltage(current_target_voltage)  # raise voltage back to the target
                        ac_paused = True
                        time_pause_started = time.perf_counter()  # record how long AC was suspended
                        continue
                    else:
                        time_waited = time.perf_counter() - time_pause_started
                        if time_waited < ac_wait_before_measurement:
                            msg = "Waiting at high voltage (DC) before taking measurement: {:.2f}/{} s "
                            self.logging.debug(msg.format(time_waited, ac_wait_before_measurement))

                            continue  # keep processing the first part of the loop so we keep checking for breakdown
                        else:
                            time_pause_started = -1  # reset wait start time

                # ------------------------------
                # Record data: resistance, leakage current
                # (Images are captured every cycle anyway, so we just need to retrieve them for the analysis)
                # ------------------------------
                # All measurements are fail-safe so the test keeps running even if an instrument fails permanently

                self.logging.info("Taking new measurement:")
                self.logging.info("Current voltage: {} V".format(measured_voltage))
                self.logging.info("DEA state: {}".format(dea_state_el))

                if self.daq is not None:  # perform electrical measurements if possible

                    if not reverse_polarity_mode:
                        # --- measure resistance ----------------------------------------------------
                        res_raw = {}  # empty dict to store raw resistance measurement data
                        Rdea = self.daq.measure_DEA_resistance(self.active_dea_indices,  # 1-D np array
                                                               n_measurements=1, nplc=1,
                                                               out_raw=res_raw)
                        if Rdea is not None:
                            self.logging.info("Resistance [kΩ]: {}".format(Rdea / 1000))
                        else:
                            self.logging.info("Resistance measurement failed (returned None)")

                        # calculate total series resistance of the bottom electrode
                        if len(res_raw) > 0:  # dict has been populated
                            try:
                                V_shunt = res_raw["Vshunt"]
                                V_source = res_raw["Vsource"]
                                V_DEA = res_raw["VDEA"]
                            except Exception as ex:
                                self.logging.warning("Raw resistance values not available. Error: {}".format(ex))
                                V_shunt = None
                                V_source = None
                                V_DEA = None

                    # TODO: fix current measurements in reverse polarity mode
                    # --- aggregate current measurements -------------------------------------------
                    if ac_active or len(leakage_buf) == 0:
                        # can't use buffered measurements in AC mode since they might have been taken while switching
                        self.logging.debug("no leakage measurements in buffer. recording new one...")
                        # take new measurement
                        leakage_current = self.daq.measure_current(nplc=5, front=reverse_polarity_mode)
                        leakage_cur_avg = leakage_current  # nothing to average -> take the newly recorded measurement
                    else:
                        leakage_cur_avg = np.mean(leakage_buf)  # average all current readings since the last time
                    leakage_buf = []  # reset buffer
                    if leakage_cur_avg is not None:
                        self.logging.info("Leakage current [nA]: {}".format(leakage_cur_avg * 1000000000))
                    else:
                        self.logging.info("No leakage current measurement available")

                # --- change polarity if in HIGH phase and reverse polarity is enabled -------------
                if reverse_polarity_mode and current_state == STATE_WAITING_HIGH and not state_changing:
                    if hb_current_output_mode == Switchboard.MODE_DC:
                        hb_current_output_mode = Switchboard.MODE_INVERSE  # switch to negative DC
                        self.logging.info("Setting H-bridge output to negative DC (reverse polarity)")
                    else:
                        hb_current_output_mode = Switchboard.MODE_DC  # set back to positive DC
                        self.logging.info("Setting H-bridge output to positive DC (normal polarity)")
                    # when switching polarity, first turn off to avoid exposing sample to overshoot
                    self.hvps.set_HB_mode(Switchboard.MODE_OFF)
                    self.hvps.wait_until_stable()  # wait until voltage is stable before switching back on in reverse
                    self.hvps.set_HB_mode(hb_current_output_mode)

                # --- resume cycling if in AC mode -------------------------------------
                if ac_paused:
                    ac_paused = False  # we're done with the measurement so pause has ended
                    self.hvps.set_OC_mode(Switchboard.MODE_AC)  # set back to AC
                    self.logging.info("Resuming AC cycling")

            if measurement_due or breakdown_occurred:  # if breakdown, data from previous cycle is saved

                # record time of measurement (before analysis since it takes some time and should not be included in the
                # measurement interval)
                time_last_measurement = time.perf_counter()

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
                    res_img_labels = [label] * self.n_deas

                # retrieve the images that were grabbed earlier
                # imgs = self.image_cap.retrieve_images()[1]  # no longer doing grab and retrieve separately
                # measure strain and get result images
                strain_res = self.strain_detector.get_dea_strain(imgs, True, True, res_img_labels)
                strain, center_shifts, res_imgs, dea_state_vis = strain_res
                # print average strain for each DEA
                self.logging.info("strain [%]: {}".format(np.reshape(strain[:, -1], (1, -1))))

                if 0 in dea_state_vis:  # 0 means outlier, 1 means OK
                    self.logging.warning("Outlier detected")

                dea_state_el_selection = [dea_state_el[i] for i in self.active_dea_indices]
                for img, v, e in zip(res_imgs, dea_state_vis, dea_state_el_selection):
                    StrainDetection.draw_state_visualization(img, v, e)

                # ------------------------------
                # Save all data voltage and DEA state
                # ------------------------------
                saver.write_data(now_tstamp,
                                 now - time_started,
                                 duration_at_max_V,
                                 cycles_completed,
                                 current_state,
                                 current_target_voltage,
                                 measured_voltage,
                                 dea_state_el,
                                 dea_state_vis,
                                 strain,
                                 center_shifts,
                                 Rdea,
                                 V_source,
                                 V_shunt,
                                 V_DEA,
                                 leakage_cur_avg,
                                 image_saved=image_due)

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
                for idx in range(6):
                    if len(res_imgs) > idx and res_imgs[idx] is not None:
                        disp_imgs[idx] = cv.resize(res_imgs[idx], preview_image_size, interpolation=cv.INTER_AREA)

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
            if state_changing or step_changing:
                # set voltage for new state
                current_target_voltage = new_target_voltage
                self.logging.debug("State changing. Setting voltage to {} V".format(current_target_voltage))
                # self.hvps.set_voltage_no_overshoot(current_target_voltage)
                self.hvps.set_voltage(current_target_voltage)
                if new_state == STATE_WAITING_HIGH and ac_mode:
                    self.hvps.set_OC_frequency(ac_frequency)  # this starts AC mode
                    ac_active = True
                    ac_paused = False
                    self.logging.debug("Started AC cyling")
                else:
                    self.hvps.set_OC_mode(Switchboard.MODE_DC)  # anything but high phase in AC mode requires DC
                    ac_active = False

                # whenever we're entering a new state, output mode should be positive DC
                if hb_current_output_mode != Switchboard.MODE_DC:
                    self.hvps.set_HB_mode(Switchboard.MODE_DC)
                    hb_current_output_mode = Switchboard.MODE_DC

                current_state = new_state
                current_step = new_step
                now = time.perf_counter()  # record time when the state actually changed
                time_last_state_change = now
                time_last_measurement = now
                time_last_image_saved = now  # also reset image timer to avoid taking too many

            # ------------------------------
            # Keep track of voltage changes
            # ------------------------------
            # if current_target_voltage != previous_target_voltage:
            #     self.callbackVoltageChange(newVoltage=current_target_voltage)

            # ------------------------------
            # check for user input to stay responsive
            # ------------------------------

            if cv.waitKey(1) & 0xFF == ord('q'):
                self.shutdown_flag = True

        self.logging.critical("Exiting...")
        self.hvps.set_output_off()
        self.hvps.set_voltage(0, block_until_reached=True)
        self.hvps.set_relays_off()
        self.hvps.close()
        self.logging.critical("Turned voltage off and disconnected relays")
        saver.close()
        self.image_cap.close_cameras()


if __name__ == '__main__':
    duallog.setup("logs", minlevelConsole=logging.INFO, minLevelFile=logging.INFO)
    logging.info("Started application")

    # launch Qt
    app = QApplication(sys.argv)

    # run setup
    _config, _strain_detector = _setup()
    nerd = NERD(_config, _strain_detector)
    nerd.run_nogui()

    # ex = App(args=None)
    # retVal = app.exec_()
    retVal = 0

    logging.info('Finished with val: {}'.format(retVal))
    sys.exit(retVal)
