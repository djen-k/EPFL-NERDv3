import csv
import logging
import time
from collections import deque
from datetime import datetime
from sys import stdout
from threading import Thread, Event, Lock, RLock
import numpy as np

import serial
import serial.tools.list_ports

# from src.hvps.daqmx import *

LIB_VER = "2.0"  # Version of this library
FIRM_VER = 20  # Firmware version That this library expects to run on the HVPS


class Switchboard:
    """Class for controlling a NERD Switchboard and HVPS (version 2)"""

    buffer_length = 1000000
    baudrate = 115200

    # Switching mode constants: off-GND=0, on-DC=1, on-inverse polarity=2, off-high Z=3
    MODE_OFF = 0
    MODE_DC = 1
    MODE_AC = 4
    MODE_INVERSE = 2  # only applies to H-bridge
    MODE_HIGHZ = 3

    # TODO: always record current state and restore after reconnect!

    def __init__(self, port=None):
        super().__init__()
        self.logging = logging.getLogger("Switchboard")  # change logger to "Switchboard"
        # self.logging.setLevel(logging.INFO)  # TODO: decide globally who logs at what level

        self.port = port
        self.name = ''
        self.firmware_version = 0
        self.minimum_voltage = 500
        self.maximum_voltage = 4800
        self.vset = 0
        self.vnow = 0
        self.oc_freq = 0
        self.oc_mode = 0
        self.hb_freq = 0
        self.hb_mode = 0
        self.oc_cycles = 0  # number of completed cycles of the optocouplers
        self.t_oc_cycle_counter = 0  # last time the oc cycle count was queried (to calculate no. of cycles since then)
        self.hb_cycles = 0  # number of completed cycles of the H-bridge
        self.t_hb_cycle_counter = 0  # last time the hb cycle count was queried (to calculate no. of cycles since then)
        self.relay_state = [0] * 6
        self.pid_gains = [0, 0, 0]

        self.ser = serial.Serial()
        self.is_open = False
        self.serial_com_lock = RLock()

        self.reconnect_timeout = -1

        self.continuous_voltage_reading_flag = Event()
        self.buffer_lock = Lock()
        self.reading_thread = Thread()

        self.t_0 = None
        self.time_buf = None
        self.voltage_buf = None
        self.voltage_setpoint_buf = None
        self.oc_state_buf = None
        self.hb_state_buf = None
        self.relay_state_buf = None
        self.log_file = None  # will be initialized if needed
        self.log_writer = None  # will be initialized if needed
        self.log_data_fields = ["Time", "Relative time [s]", "Set voltage [V]", "Measured voltage [V]",
                                "OC mode", "OC state", "HB mode", "HB state",
                                "Relay 1", "Relay 2", "Relay 3", "Relay 4", "Relay 5", "Relay 6"]
        self.log_lock = Lock()

    def __del__(self):
        try:
            self.close()
        except Exception as ex:
            self.logging.warning("Unable to switch off HVPS and relays: {}".format(ex))

    ###########################################################################
    # connection and setup ####################################################
    ###########################################################################

    @staticmethod
    def detect(n=0):
        """Detect available HVPS"""
        logger = logging.getLogger("Switchboard")  # change logger to "Switchboard"
        logger.debug("Detecting available switchboards")
        serial_ports = serial.tools.list_ports.comports()
        available_switchboards = []
        for ser in serial_ports:
            if "Arduino" in ser.description:
                sb = Switchboard(ser.device)
                try:  # see if we can open it
                    sb.open(with_continuous_reading=False)
                    logger.info("Device {} found at port {}".format(sb.name, sb.port))
                    available_switchboards.append(sb)
                    sb.close()

                    if 0 < n == len(available_switchboards):
                        break  # already found the desired number of available devices
                except Exception as ex:
                    logger.debug("Unable to connect to device on port {}. Exception: {}".format(sb.port, ex))

        if not available_switchboards:
            logger.warning("No switchboards found!")
        elif len(available_switchboards) < n:
            logger.warning("The requested number of switchboards could not be found!")
        else:
            logger.info("Switchboard detection done. Found {} devices.".format(len(available_switchboards)))
        return available_switchboards

    def _open_connection(self):
        """
        Establishes connection with the HVPS. An exception is raised it the connection could not be established.
        """
        self.logging.debug("Connecting to {}".format(self.port))
        with self.serial_com_lock:
            try:
                self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
                self.ser.reset_input_buffer()
                self.ser.reset_output_buffer()
            except Exception as ex:
                self.logging.info("Impossible to open port {}: {}".format(self.port, ex))
                raise Exception("Unable to connect to port {}!".format(self.port))

            if self.ser.is_open:
                self.logging.debug("Reading device data")
                self.is_open = True

                self.name = self.get_name()
                self.firmware_version = self.get_firmware_version()
                self.maximum_voltage = self.get_maximum_voltage()

                # reset switchboard to make sure we're in a known state
                self.set_voltage(0)
                self.set_OC_mode(Switchboard.MODE_OFF)
                self.set_HB_mode(Switchboard.MODE_OFF)
                self.set_relays_off()

    def open(self, port=None, with_continuous_reading=False, reconnect_timeout=-1):
        """
        Open a connection to a physical switchboard at the given COM port. The port can be specified as a string
        (COM port name), a HvpsInvo object or an index. If specified as an index, the respective port in this
        SwitchBoard's hvps_available_ports list is used. If no port is specified, the first available port is used.
        :param port: The COM port to connect to. Can be a string, HvpsInfo, int or None
        :param with_continuous_reading: Specify if the SwitchBoard should launch a thread to record continuous voltage
        readings
        :param reconnect_timeout: Optional timeout parameter to specify for how long the HVPS should attempt
        reconnecting if the connection is lost. Set -1 to disable (i.e. keep trying to reconnect indefinitely).
        If the timeout expires before the connection can be reastablished, the COM port is closed but no error
        is generated. Check "is_open" regularly to ensure that the device is still connected.
        """
        if self.is_open:
            self.close()

        # figure out which port to connect to

        if port is not None:
            self.port = port

        if self.port is None:  # if not defined, autodetect HVPS
            sb = Switchboard.detect(1)[0]  # get the first switchboard we can find
            port = sb.port
            del sb  # get rid of the switchboard object to release the com port
            self.open(port, with_continuous_reading)
        elif isinstance(self.port, int):  # specified as index  -> pick from list of available ports
            sbs = Switchboard.detect()  # get the first switchboard we can find
            if len(sbs) > self.port:
                self.port = sbs[port].port
            else:
                raise ValueError("Invalid switchboard index: {}! This device does not exist!".format(self.port))
        elif not isinstance(self.port, str):
            raise ValueError("Invalid argument! COM port must be given as a string or index!")

        # connect to HVPS
        self._open_connection()  # will throw an exception if not successful (we just pass it on)

        # ensure firmware version is compatible
        if self.firmware_version < 20:
            logging.critical("{} is running v1 firmware. This software requires a v2 switchboard!".format(self.name))

        self.reconnect_timeout = reconnect_timeout

        if with_continuous_reading:
            self.start_continuous_reading()

    def close(self):
        """Closes connection with the HVPS"""
        if self.is_open:
            self.stop_voltage_reading(wait=True)
            if self.ser.is_open:
                with self.serial_com_lock:
                    self.set_voltage(0, block_while_testing=True)  # set the voltage to 0 as a safety measure
                    self.set_output_off()  # disable OCs and H-bridge
                    self.set_relays_off()
                    self.ser.close()
            # time.sleep(3)  # wait for serial port to properly close
            self.is_open = False
            self.logging.info("Connection to {} closed.".format(self.name))

    def dirty_reconnect(self):

        # update the number of cycles if we're in AC mode (this doesn't use serial communication)
        self.get_OC_cycles()
        self.get_HB_cycles()

        self.ser.close()
        start = time.perf_counter()
        while not self.ser.is_open:
            try:
                self.ser.open()

                # resume operation by restoring the previous state
                self.set_voltage(self.vset)
                self.set_relay_auto_mode(0, self.relay_state)
                if self.get_HB_mode() != self.hb_mode:  # they don't match so must have been a reset
                    if self.hb_mode == Switchboard.MODE_AC:  # need to restart AC mode
                        self.set_HB_frequency(self.hb_freq)  # should resume counting cycles correctly
                    else:
                        self.set_HB_mode(self.hb_mode)
                if self.get_OC_mode() != self.oc_mode:  # they don't match so must have been a reset
                    if self.oc_mode == Switchboard.MODE_AC:  # need to restart AC mode
                        self.set_OC_frequency(self.oc_freq)  # should resume counting cycles correctly
                    else:
                        self.set_OC_mode(self.oc_mode)

            except Exception as ex:
                self.logging.debug("Connection error: {}".format(ex))
                elapsed = time.perf_counter() - start
                if self.reconnect_timeout < 0 or elapsed < self.reconnect_timeout:  # no timeout or not yet expired

                    msg = "Reconnection attempt failed!"
                    if self.reconnect_timeout == 0:
                        msg += " Will keep trying for {} s".format(int(round(self.reconnect_timeout - elapsed)))
                    else:
                        msg += " Will keep trying..."
                    self.logging.critical(msg)
                    self.ser.close()  # close again to make sure it's properly closed before we try again
                    time.sleep(0.5)
                else:
                    self.logging.critical("Unable to reconnect! Timeout expired.")
                    self.close()
                    return

        self.logging.critical("Reconnected!")

    ###########################################################################
    # communication ###########################################################
    ###########################################################################

    def _write_hvps(self, cmd):
        """
        Write a command to the HVPS via serial
        :param cmd: The command to send (string or bytes)
        """

        # convert to bytes if command is given as a string
        if not isinstance(cmd, bytes):
            cmd = bytes(cmd, "ascii")

        # add end of message marker (if not already present)
        if not cmd.endswith(b'\r'):
            cmd += b'\r'

        # send message via serial
        self.logging.debug("Writing message to HVPS: {}".format(cmd))
        try:
            self.ser.write(cmd)
            while self.ser.out_waiting:
                time.sleep(0.01)
        except Exception as ex:
            self.logging.critical("Error writing to HVPS: {}".format(ex))
            self.dirty_reconnect()

    def _read_hvps(self):
        """
        Reads a line on the serial port and converts it to a string (without newline characters at the end)
        :return: The message received from the switchboard (as string) or None, if no line could be read
        from the serial port. If the line contains an auxiliary message, it is written to the log and a new line
        is retrieved until a regular message is received.
        """
        try:
            line = self.ser.readline()
            line = line[:-2]  # removes the last 2 elements of the array because it is \r\n
            res = bytes(line)

            # check if it's an error/warning message from the switchboard or the reply to the query
            if res is not None:
                res = res.decode("ascii")  # directly convert back to string

                if res.startswith("["):  # info/warning/error
                    # TODO: log switchboard messages properly and raise flag in case of warnings
                    self.logging.info("Message from Switchboard: {}".format(res))

                    res = self._read_hvps()  # read another line
                elif res.startswith("Err"):
                    self.logging.warning("Invalid command! Returned error.")
                    res = None

        except Exception as ex:
            self.logging.critical("Error reading from HVPS: {}".format(ex))
            self.dirty_reconnect()
            res = None  # connection was lost so we never got the expected return message

        self.logging.debug("Response from HVPS: {}".format(res))
        return res

    def send_query(self, cmd):
        """
        Send a command to the HVPS and read the response.
        :param cmd: The command to send to the HVPS (string or bytes)
        :return: the response (as string) or None, if no response was received
        """
        if self.is_open:
            with self.serial_com_lock:
                self.logging.debug("Sending query: {}".format(cmd))
                self._write_hvps(cmd)
                res = self._read_hvps()
                self.logging.debug("Response: {}".format(res))
            return res
        else:
            raise Exception("Switchboard is not connected!")

    def _cast_int(self, data):
        try:
            cast_int = int(data)
        except Exception as ex:
            self.logging.warning("Failed to parse int: {}. Error: {}".format(data, ex))
            cast_int = -1
        return cast_int

    def _cast_float(self, data):
        try:
            cast_float = float(data)
        except Exception as ex:
            self.logging.warning("Failed to parse float: {}. Error: {}".format(data, ex))
            cast_float = -1
        return cast_float

    def _parse_relay_state(self, str_state):
        try:
            str_state = str_state.replace(":", ";")  # because v1 and v2 have different formats
            str_state = str_state.split(";")  # separate response for each relay
            relay_mode = self._cast_int(str_state[0])  # is either the relay mode or just "Relay state" (-> -1)
            # TODO: use/store relay mode somehow
            state = str_state[1].split(",")  # split individual states
            state = [self._cast_int(r) for r in state if r]  # convert to int if string is not empty
        except Exception as ex:
            self.logging.warning("Failed to parse relay state: {}. Error: {}".format(str_state, ex))
            return None  # return None to indicate something is wrong

        if len(state) == 6:  # output is only valid if there is a state for each relay
            return state
        else:
            self.logging.warning("Invalid response. Received {} values instead of 6.".format(len(state)))
            return None  # return None to indicate something is wrong

    ###########################################################################
    # getters and setters #####################################################
    ###########################################################################

    def get_maximum_voltage(self):
        """Query the maximum voltage of the switchboard"""
        return self._cast_int(self.send_query("QVmax"))

    def set_maximum_voltage(self, vmax):
        """
        Set the maximum voltage of the switchboard. Will be saved to EEPROM.
        :param vmax: The maximum voltage
        :return: True, if the voltage was set correctly
        """
        self.maximum_voltage = vmax  # update internal model
        res = self._cast_int(self.send_query("SVmax {:.0f}".format(vmax)))
        return res == vmax

    def get_voltage_setpoint(self):
        """Query the voltage setpoint of the switchboard"""
        return self._cast_int(self.send_query("QVset"))

    def set_voltage(self, voltage, block_while_testing=False, block_until_reached=False):  # sets the output voltage
        """
        Sets the output voltage.
        Checks if voltage can be set or if switchboard is currently testing for a short circuit
        :param voltage: The desired output voltage
        :param block_while_testing: Flag to indicate if the function should block until the voltage can be set.
        The voltage set point cannot be changed while the switchboard is testing for short circuits.
        Set this to True, if you want the function to block until the switchboard has finished testing (if it was).
        If false, the function may return without having set the voltage. Check response from switchboard!
        :param block_until_reached: Flag to indicate if the function should block until the measured voltage matches the
        voltage set point (with a 10V margin). If the set point is not reached within 3s, a TimeoutError is raised.
        :return: True if the voltage was set successfuly, false if the switchboard was unable to set the voltage because
        it was busy testing for a short circuit or some other error occurred. If 'block_if_testing' is True,
        a False return value indicates an unexpected error.
        """

        if block_until_reached:
            block_while_testing = True  # it can't reach if it's not set successfully

        # check that specified voltage is within the allowed range #######################################

        if voltage != 0 and voltage < self.minimum_voltage:
            msg = "Specified voltage ({}) is below the allowed minimum ({}). Setting voltage to 0!"
            self.logging.warning(msg.format(voltage, self.minimum_voltage))
            voltage = 0

        if voltage > self.maximum_voltage:
            msg = "Specified voltage ({}) is above the allowed maximum ({}). Setting voltage to {}}!"
            self.logging.warning(msg.format(voltage, self.maximum_voltage, self.maximum_voltage))
            voltage = self.maximum_voltage

        # make sure that the new voltage set point was accepted (it won't be while it's testing) ###############

        if block_while_testing:
            if self.is_testing():
                self.logging.debug("Switchboard is busy testing for shorts. Waiting to set voltage...")
            while self.is_testing():
                time.sleep(0.1)

        self.vset = voltage  # update internal model
        res = self._cast_int(self.send_query("SVset {:.0f}".format(voltage)))

        if not block_until_reached:  # don't wait, return immediately
            return res == voltage

        timeout = 5  # if voltage has not reached its set point in 5 s, something must be wrong!
        start = time.perf_counter()
        elapsed = 0
        while abs(voltage - self.get_current_voltage()) > 50:
            if elapsed > timeout:
                msg = "Voltage has not reached the set point after 5 seconds! Please check the HVPS!"
                raise TimeoutError(msg)
            if elapsed == 0:  # only write message once
                self.logging.debug("Waiting for measured output voltage to reach the set point...")
            time.sleep(0.05)
            while self.is_testing():
                self.logging.debug("Switchboard is testing for shorts...")
                time.sleep(0.5)
                start = time.perf_counter()  # if SB is busy checking for shorts, don't start counting timeout

            elapsed = time.perf_counter() - start

        return True  # if we reached here, it must have been set correctly

    def set_voltage_no_overshoot(self, voltage):
        """
        Sets the output voltage to the specified value, but does so more slowly in several steps to ensure that there
        is no voltage overshoot. This method blocks until the desired voltage has been reached.
        :param voltage: The desired output voltage, in Volts.
        :return: True or False to indicate if the voltage was set correctly.
        """
        prev_v = self.get_current_voltage(True)
        dv = voltage - prev_v
        if dv > 100 and voltage > self.minimum_voltage:  # if increasing (and by more than a few volts), do it slowly
            self.set_voltage(round(prev_v + dv * 0.7), block_until_reached=True)
            self.set_voltage(round(prev_v + dv * 0.9), block_until_reached=True)
        return self.set_voltage(voltage, block_until_reached=True)

    def get_current_voltage(self, from_buffer_if_available=True):
        """
        Read the current voltage from the switchboard
        :param from_buffer_if_available: If true and if continuous voltage reading is on, the most recent value from
        the voltage buffer is returned instead of querying the switchboard.
        :return: The current voltage as measrued by the switchboard voltage feedback.
        """
        if from_buffer_if_available is True and self.reading_thread.is_alive():
            with self.buffer_lock:
                if self.voltage_buf:  # check if there is anything in it
                    v = self.voltage_buf[-1]
                    self.logging.debug("Taking voltage from buffer: {}".format(v))
                    return v

        # if thread not running or nothing in buffer, take normal reading
        return self._cast_int(self.send_query("QVnow"))

    def set_output_on(self):
        """Enable high voltage output by switching on the OCs and H-bridge in DC mode"""
        self.set_HB_mode(Switchboard.MODE_DC)
        self.set_OC_mode(Switchboard.MODE_DC)
        if not self.is_output_enabled():
            self.logging.warning("HV output is currently disabled. Set the output switch to ON to activate HV output!")

    def set_output_off(self):
        """Disable high voltage output by switching the OCs and H-bridge off"""
        self.set_HB_mode(Switchboard.MODE_OFF)
        self.set_OC_mode(Switchboard.MODE_OFF)
        if not self.is_output_enabled():
            self.logging.warning("HV output is currently disabled. Set the output switch to ON to activate HV output!")

    def get_name(self):
        """Queries name of the board"""
        self.logging.debug("Querying device name")
        return self.send_query("QName")

    def set_name(self, name):
        """
        Set the name of the board
        :param name: The new name
        :return: True if the name was set correctly
        """
        self.logging.debug("Setting device name")
        self.name = name  # update internal model
        res = self.send_query("SName {}".format(name))
        return res == name

    def get_firmware_version(self):
        """Queries name of the board"""
        self.logging.debug("Querying firmware version")
        res = self.send_query("QVer")
        ver = self._cast_int(res)
        if ver == -1:  # cast int failed, presumably because it's the older firmware v1
            res = res.split(" ")  # in v1, it's a string in the form of "slave X" need to get the value of X
            ver = self._cast_int(res[1])
        else:  # we're on firmware v2
            ver += 20  # define v2 firmware to be of version 2x
        return ver

    def get_pid_gains(self):
        self.logging.debug("Querying PID gains")
        res = self.send_query("QKp")
        if "," in res:  # v1: always returns all three gains
            res = res.split(",")
        else:  # v2: need to query each gain individually
            res = [res, self.send_query("QKi"), self.send_query("QKd")]
        res = [self._cast_float(s) for s in res]
        return res

    def set_pid_gains(self, gains):
        """
        Set the PID gains of the internal voltage regulator. The new gains will be written to the EEPROM (only if new
        gains are different from the prvious ones).
        :param gains: A list of gain values [P, I, D]
        :return: True if the PID gains have been set correctly
        """
        pid = ['p', 'i', 'd']
        current_gains = self.get_pid_gains()
        self.pid_gains = gains
        for k in range(3):
            if gains[k] != current_gains[k]:  # only update if different to avoid excessive writing to EEPROM
                self.send_query("SK{} {:.4f}".format(pid[k], gains[k]))
        return self.get_pid_gains() == gains

    def set_relays_on(self, relays=None):
        """
        Switch the requested relays on. If not specified, all relays are switched on.
        :param relays: A list of (zero-based) indices specifying which relays to switch on
        :return: The the updated relay state returned by the switchboard
        """
        if relays is None:
            self.logging.debug("Set all relays on")
            self.relay_state = [1] * 6
            res = self.send_query("SRelOn")
            res = self._parse_relay_state(res) == self.relay_state  # check if all are on
        else:
            if self.firmware_version < 20:  # v1
                res = True
                for i in relays:
                    rres = self.set_relay_state(i, 1)
                    if rres is False:  # if any one failed, the operation was not successful
                        res = False
            else:
                self.logging.warning("Switching individual relays is not supported in v2. Use selective auto mode!")
                res = False
        return res

    def set_relays_off(self, relays=None):
        """
        Switch the requested relays off. If not specified, all relays are switched off.
        :param relays: A list of indices specifying which relays to switch off
        :return: True, if the relay state was set successfully
        """
        if relays is None:
            self.logging.debug("Set all relays off")
            self.relay_state = [0] * 6
            res = self.send_query("SRelOff")
            res = self._parse_relay_state(res) == self.relay_state  # check if all are off
        else:
            if self.firmware_version < 20:  # v1
                res = True
                for i in relays:
                    rres = self.set_relay_state(i, 0)
                    if rres is False:  # if any one failed, the operation was not successful
                        res = False
            else:
                self.logging.warning("Switching individual relays is not supported in v2. Use selective auto mode!")
                res = False
        return res

    def set_relay_state(self, relay, state):
        self.relay_state[relay] = state
        if state is 0:
            self.logging.debug("Setting relay {} off".format(relay))
            res = self.send_query("SRelOff {:d}".format(relay))
        else:  # state is 1
            self.logging.debug("Setting relay {} on".format(relay))
            res = self.send_query("SRelOn {:d}".format(relay))
        res = self._parse_relay_state(res)
        return res[relay] == state

    def get_relay_state(self, from_buffer_if_available=True):
        """
        Queries the on/off state of the relais
        :return: A list (int) containing 0 or 1 for each relay, indicating on or off
        """
        if from_buffer_if_available is True and self.reading_thread.is_alive():
            with self.buffer_lock:
                if self.relay_state_buf:  # check if there is anything in it
                    rs = self.relay_state_buf[-1]
                    self.logging.debug("Taking relay state from buffer: {}".format(rs))
                    return rs

        self.logging.debug("Querying relay state")
        res = self._parse_relay_state(self.send_query("QRelState"))
        return res

    def set_relay_auto_mode(self, reset_time=0, relays=None):
        """
        Enable the automatic short circuit detection and isolation function of the switchboard
        :param reset_time: An optional time after which to reconnect and test all relays again (including those
        where a short circuit was detected in a previous test). Set 0 to disable.
        :param relays: List of indices (0-5) of the channels to switch on in auto mode
        :return: True, if the relay state was set successfully
        """
        if relays is None:
            relays = range(6)  # all channels

        rel_bin = [int(i in relays) for i in range(6)]  # get desired state (0/1) for each relay
        # compose string of which relays to switch on
        rel_str = ""
        for rel in rel_bin:
            rel_str += str(rel)
        self.relay_state = rel_bin  # keep state in memory

        self.logging.debug("Enabling auto mode with timeout {} s for channels {}".format(reset_time, rel_str))
        res = self.send_query("SRelAuto {:.0f} {}".format(reset_time, rel_str))
        return self._parse_relay_state(res) == rel_bin

    def is_testing(self):
        """
        Checks if the switchboard is currently testing for a short circuit
        :return: True, if the switchboard is currently busy testing
        """
        self.logging.debug("Querying if switchbaord is testing")
        res = self.send_query("QTestingShort")
        return res == "1"

    def get_OC_state(self, from_buffer_if_available=True):
        """Query the optocoupler state of the switchboard"""

        if from_buffer_if_available is True and self.reading_thread.is_alive():
            with self.buffer_lock:
                if self.oc_state_buf:  # check if there is anything in it
                    state = self.oc_state_buf[-1]
                    self.logging.debug("Taking OC state from buffer: {}".format(state))
                    return state

        self.logging.debug("Querying OC state")
        res = self.send_query("QOC")
        res = res.split(",")
        if len(res) == 2:
            mode = self._cast_int(res[0])
            state = self._cast_int(res[1])
            if mode == 1:  # AC mode
                mode = Switchboard.MODE_AC
            else:
                mode = state  # if in manual (not AC) mode, the current state must reflect the mode
            return [mode, state]
        else:
            return [-1, -1]

    def get_OC_mode(self, from_buffer_if_available=True):
        """Query the optocoupler switching mode of the switchboard: OFF-GND=0, HV-DC=1, HV-AC=4, OFF-HIGHZ=3"""
        ocs = self.get_OC_state(from_buffer_if_available)
        return ocs[0]

    def set_OC_mode(self, oc_mode):
        """
        Set the optocoupler switching mode of the switchboard.
        :param oc_mode: The OC switching mode (GND=0, HV=1, HIGHZ=3)
        :return: True, if the mode was set correctly
        """

        self.get_OC_cycles()  # this updates the cycles one last time if we are currently in AC mode

        if oc_mode == Switchboard.MODE_AC:
            if self.oc_freq == 0:
                raise Exception("No frequency set. To start AC mode, use set_OC_frequency!")
            else:
                return self.set_OC_frequency(self.oc_freq)  # start AC mode with previous frequency
        else:
            self.oc_mode = oc_mode  # update internal model
            res = self._cast_int(self.send_query("SOC {}".format(oc_mode)))
            return res == oc_mode

    def set_OC_frequency(self, oc_freq, reset_cycle_counter=False):
        """
        Set the optocoupler switching frequency of the switchboard.
        :param oc_freq: The OC switching frequency
        :param reset_cycle_counter: Set True to reset the counter of completed cycles to 0 before starting AC mode
        :return: True, if the frequency was set correctly
        """
        if self.oc_mode == Switchboard.MODE_AC:
            self.get_OC_cycles()  # this updates the cycles before changing anything, if we were already in AC mode
        if oc_freq == 0:
            self.set_OC_mode(Switchboard.MODE_OFF)  # freq=0 -> off. Have to do it this way so mode gets set correctly
            return

        self.oc_freq = oc_freq  # update internal model
        self.oc_mode = Switchboard.MODE_AC  # indicate that AC mode is enabled
        res = self._cast_int(self.send_query("SOCF {:.4f}".format(oc_freq)))
        self.t_oc_cycle_counter = time.perf_counter()  # record the time we started AC mode
        if reset_cycle_counter:
            self.oc_cycles = 0.0
        return res == oc_freq

    def get_OC_cycles(self):
        """
        Get the number of cycles the optocouplers have completed in AC mode.
        :return: The number of fractional completed cycles (float)
        """
        # update cycles if AC mode is currently on
        if self.oc_mode == Switchboard.MODE_AC:
            t = time.perf_counter()
            dt = t - self.t_oc_cycle_counter
            self.oc_cycles += dt * self.oc_freq  # update no. of cycles
            self.t_oc_cycle_counter = t  # remember the last time we updated

        return self.oc_cycles

    def get_HB_state(self, from_buffer_if_available=True):
        """Query the optocoupler state of the switchboard"""

        if from_buffer_if_available is True and self.reading_thread.is_alive():
            with self.buffer_lock:
                if self.hb_state_buf:  # check if there is anything in it
                    state = self.hb_state_buf[-1]
                    self.logging.debug("Taking HB state from buffer: {}".format(state))
                    return state

        self.logging.debug("Querying HB state")
        res = self.send_query("QHB")
        res = res.split(",")
        if len(res) == 2:
            mode = self._cast_int(res[0])
            state = self._cast_int(res[1])
            if mode == 1:  # AC mode
                mode = Switchboard.MODE_AC
            else:
                mode = state  # if in manual (not AC) mode, the current state must reflect the mode
            return [mode, state]
        else:
            return [-1, -1]  # didn't get a valid response

    def get_HB_mode(self, from_buffer_if_available=True):
        """Query the optocoupler switching mode of the switchboard: OFF-GND=0, HV-DC=1, HV-AC=4, OFF-HIGHZ=3"""
        hbs = self.get_HB_state(from_buffer_if_available)
        return hbs[0]

    def set_HB_mode(self, hb_mode):
        """
        Set the H-bridge switching mode of the switchboard.
        :param hb_mode: The HB switching mode (GND=0, HVA=1, HVB=2, HIGHZ=3)
        :return: True, if the mode was set correctly
        """
        self.get_HB_cycles()  # this updates the cycles one last time if we are currently in AC mode

        if hb_mode == Switchboard.MODE_AC:
            if self.hb_freq == 0:
                raise Exception("No frequency set. To start AC mode, use set_HB_frequency!")
            else:
                return self.set_HB_frequency(self.hb_freq)  # start AC mode with previous frequency
        else:
            self.hb_mode = hb_mode  # update internal model
            res = self._cast_int(self.send_query("SHB {}".format(hb_mode)))
            return res == hb_mode

    def set_HB_frequency(self, hb_freq, reset_cycle_counter=False):
        """
        Set the H-bridge switching frequency of the switchboard and start AC mode.
        :param hb_freq: The HB switching frequency
        :param reset_cycle_counter: Set True to reset the completed cycle counter to 0 before starting AC mode
        :return: True, if the frequency was set correctly
        """

        if self.oc_mode == Switchboard.MODE_AC:
            self.get_OC_cycles()  # this updates the cycles before changing anything, if we are were already in AC mode
        if hb_freq == 0:
            self.set_HB_mode(Switchboard.MODE_OFF)  # this makes sure the AC cycle number is counted properly
            return

        self.hb_freq = hb_freq  # update internal model
        self.hb_mode = Switchboard.MODE_AC  # indicate that AC mode is enabled
        res = self._cast_int(self.send_query("SHBF {:.4f}".format(hb_freq)))
        self.t_hb_cycle_counter = time.perf_counter()  # record the time we started AC mode
        if reset_cycle_counter:
            self.hb_cycles = 0.0
        return res == hb_freq

    def get_HB_cycles(self):
        """
        Get the number of cycles the H-bridge has completed in AC mode.
        :return: The number of fractional completed cycles (float)
        """
        if self.hb_mode == Switchboard.MODE_AC:
            t = time.perf_counter()
            dt = t - self.t_hb_cycle_counter
            self.hb_cycles += dt * self.hb_freq  # update no. of cycles
            self.t_hb_cycle_counter = t  # remember the last time we updated

        return self.hb_cycles

    def is_output_enabled(self):
        """Query if the mechanical switch to enable/disable HV output is on (1) or off (0) or unknown (-1)"""
        return self._cast_int(self.send_query("QEnable"))

    ###########################################################################
    # continuous reading #####################################################
    ###########################################################################

    def start_continuous_reading(self, buffer_length=None, reference_time=None, log_file=None, append=True):
        """
        Start continuous voltage reading in a separate thread. The data is stored in an internal buffer and can be
        retrieved via 'get_voltage_buffer'.
        :param buffer_length: The length of the buffer in which the voltage data is stored.
        :param reference_time: The time of each voltage measurement will be expressed relative to this time. Useful for
        synchronizing data acquisition from different devices. The reference time must be generated by calling
        'time.perf_counter()'.
        :param log_file: Name of the log file to store voltage data. If None, no data is written to file.
        :param append: Set False to overwrite any existing log file. Otherwise, new data is appended.
        """
        # check if it's already running
        if self.reading_thread is not None and self.reading_thread.is_alive():
            self.logging.info("Voltage reading thread is already running.")
            return

        self.logging.debug("Starting voltage reading thread")

        if log_file is not None:
            self.logging.debug("Setting up log file: {}".format(log_file))
            try:
                if append:
                    fmode = 'a'  # 'append' will append data from this session to the end of the file (if it exists)
                else:
                    fmode = 'w'  # 'write' will overwrite the contents of the existing file (if there is one)
                self.log_file = open(log_file, mode=fmode, newline='')
                self.log_writer = csv.DictWriter(self.log_file, fieldnames=self.log_data_fields)
                self.log_writer.writeheader()
            except OSError as ex:
                self.logging.error("Unable to create log file: {}".format(ex))
                self.log_file = None
                self.log_writer = None

        if buffer_length is not None:
            self.buffer_length = buffer_length
        self.t_0 = reference_time
        self.continuous_voltage_reading_flag.clear()  # reset the flag
        self.voltage_buf = deque(maxlen=self.buffer_length)  # voltage buffer
        self.voltage_setpoint_buf = deque(maxlen=self.buffer_length)  # relay state buffer
        self.oc_state_buf = deque(maxlen=self.buffer_length)  # relay state buffer
        self.hb_state_buf = deque(maxlen=self.buffer_length)  # relay state buffer
        self.relay_state_buf = deque(maxlen=self.buffer_length)  # relay state buffer
        self.time_buf = deque(maxlen=self.buffer_length)  # Times buffer
        self.reading_thread = Thread()  # Thread for continuous reading
        self.reading_thread.run = self._continuous_voltage_reading  # Method associated to thread
        self.reading_thread.daemon = True  # make daemon so it terminates if main thread dies (from some error)
        self.reading_thread.start()  # Starting Thread

    def stop_voltage_reading(self, wait=False, wait_timeout=1.0):
        """Routine for stoping continuous position reading"""
        self.continuous_voltage_reading_flag.set()  # Set Flag to False
        # if requested, wait for thread to finish
        if self.reading_thread.is_alive() and wait:
            self.reading_thread.join(wait_timeout)

    def _continuous_voltage_reading(self):
        """Method for continuous reading"""

        # self.logging.debug("HVPS logger thread started")
        log_to_file = self.log_file is not None  # this won't change so we don't need to check on every loop
        # self.logging.debug("Saving to file: {}".format(log_to_file))

        if self.t_0 is None:
            t_0 = time.perf_counter()  # Initializing reference time
        else:
            t_0 = self.t_0

        while not self.continuous_voltage_reading_flag.is_set() and self.is_open:
            # While Flag is not set and HVPS is connected
            # get data
            c_time = time.perf_counter() - t_0  # Current time (since reference)
            voltage = self.get_current_voltage(False)
            voltage_setpoint = self.get_voltage_setpoint()
            oc_state = self.get_OC_state(False)
            hb_state = self.get_HB_state(False)
            relay_state = self.get_relay_state(False)
            # self.logging.debug("Logger thread: Data received")
            # store data in buffer
            with self.buffer_lock:  # acquire lock for data manipulation
                self.time_buf.append(c_time)
                self.voltage_buf.append(voltage)
                self.voltage_setpoint_buf.append(voltage_setpoint)
                self.oc_state_buf.append(oc_state)
                self.hb_state_buf.append(hb_state)
                self.relay_state_buf.append(relay_state)
                # self.logging.debug("Logger thread: Data stored in buffer (length: {})".format(len(self.voltage_buf)))
            # write data to log file
            if log_to_file:
                with self.log_lock:
                    if relay_state is None:
                        relay_state = [-1] * 6
                        self.logging.debug("Logger thread: Invalid relay state ('None')")
                    data = [datetime.now(), c_time, voltage_setpoint, voltage, *oc_state, *hb_state, *relay_state]
                    row = dict(zip(self.log_data_fields, data))
                    self.log_writer.writerow(row)
                    # self.logging.debug("Logger thread: Data written to file".format(len(self.voltage_buf)))

        # self.logging.debug("Logger thread exiting")
        # close log file when we're done
        if log_to_file:
            with self.log_lock:
                self.log_writer.writerow({})
                self.log_file.flush()
                self.log_file.close()
                self.log_file = None
                self.log_writer = None
                # self.logging.debug("Switchboard log file closed")

    def get_data_buffer(self, clear_buffer=True, initial_t=None):
        """
        Retrieve the voltage buffer. By default, the buffer is cleared after reading
        :param clear_buffer: Set true to clear the buffer after it has been read. (default: True)
        :param initial_t: If not None, all time values in the buffer will be shifted so that the first value in
        the buffer is equal to initialt.
        :return: The voltage buffer (times, voltages, voltage_sps, relay_states, oc_states) as five 1-D numpy arrays
        """
        with self.buffer_lock:  # Get Data lock for multi threading
            times = np.array(self.time_buf)  # convert to array (creates a copy)
            voltages = np.array(self.voltage_buf)  # convert to array (creates a copy)
            voltage_sps = np.array(self.voltage_setpoint_buf)  # convert to array (creates a copy)
            relay_states = np.array(self.relay_state_buf)  # convert to array (creates a copy)
            oc_states = np.array(self.oc_state_buf)  # convert to array (creates a copy)
            hb_states = np.array(self.oc_state_buf)  # convert to array (creates a copy)
            if clear_buffer:
                self.time_buf.clear()
                self.voltage_buf.clear()
                self.voltage_setpoint_buf.clear()
                self.oc_state_buf.clear()
                self.hb_state_buf.clear()
                self.relay_state_buf.clear()

        if initial_t is not None:
            times = times - times[0] + initial_t  # shift starting time to the specified initial time

        return times, voltages, voltage_sps, relay_states, oc_states, hb_states  # Return time and positions


def test_slow_voltage_rise():
    sb = Switchboard()
    sb.open(with_continuous_reading=True)
    sb.set_relays_on()
    v = 300
    # sb.set_voltage(round(v * 0.7), block_until_reached=True)
    # # time.sleep(1)
    # sb.set_voltage(round(v * 0.9), block_until_reached=True)
    # sb.set_voltage(v, block_until_reached=True)
    sb.set_voltage_no_overshoot(v)
    time.sleep(1)
    sb.set_voltage_no_overshoot(0)
    sb.close()
    times, voltages, voltage_sps, relay_states, oc_states, hb_states = sb.get_data_buffer()
    plt.plot(times, voltages)
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.grid(True)
    plt.show()


def test_switchboard():
    sb = Switchboard.detect(1)[0]
    sb.open()
    t_start = time.perf_counter()
    time.sleep(1)
    sb.start_continuous_reading(reference_time=t_start, log_file="voltage log.csv", append=False)
    Vset = 450
    freq = 2
    twait = 5
    cycles_remaining = []
    tc = []
    print(sb.get_name())
    # print("sp: ", sb.get_voltage_setpoint())
    # print("now: ", sb.get_current_voltage())
    # print("set: ", sb.set_voltage(500))
    # print("sp: ", sb.get_voltage_setpoint())
    # time.sleep(2)
    # print("now: ", sb.get_current_voltage())
    # print(sb.set_relays_on())
    # print(sb.get_relay_state())
    # gains = sb.set_pid_gains((0.26, 2.1, 0.005))
    gains = sb.get_pid_gains()
    print("PID gains:", gains)

    # sb.set_output_off()  # = set_switching_mode(0)
    # print(sb.set_relay_auto_mode())
    sb.set_relays_on()
    # sb.set_output_on()
    time.sleep(0.3)
    print(sb.get_relay_state())
    # time.sleep(1)
    # print(sb.get_relay_state())
    # print(sb.set_relay_state(3, 1))
    # time.sleep(1)
    # print(sb.set_relays_off())
    # time.sleep(1)
    sb.set_output_on()
    print(sb.get_current_voltage(), "V")
    sb.set_voltage(1000)
    time.sleep(twait)
    print(sb.get_current_voltage(), "V")
    sb.set_voltage(3000)
    time.sleep(twait)
    print(sb.get_current_voltage(), "V")
    sb.set_voltage(500)
    time.sleep(twait)
    print(sb.get_current_voltage(), "V")
    sb.set_voltage(0)
    # print("sp: ", sb.get_voltage_setpoint())
    time.sleep(twait)
    print(sb.get_current_voltage(), "V")
    # sb.set_relays_on()
    # time.sleep(0.3)
    # print(sb.get_relay_state())
    # print(sb.set_relays_off())
    # time.sleep(0.01)
    # sb.set_frequency(freq)
    # sb.set_cycle_number(25)
    # tc.append(time.perf_counter() - t_start)
    # cn = sb.get_cycle_number()
    # print(cn)
    # cn = np.diff(cn)[0]
    # print(cn)
    # cycles_remaining.append(cn)
    # print("f: ", sb.get_frequency())
    # sb.start_ac_mode()  # = set_switching_mode(2)
    # print("now: ", sb.get_current_voltage())
    # print("sp: ", sb.get_voltage_setpoint())
    # time.sleep(2)
    # sb.set_switching_mode(0)  # DC

    # for i in range(75):
    #     time.sleep(0.2)
    #     tc.append(time.perf_counter() - t_start)
    #     cn = sb.get_cycle_number()
    #     print(cn)
    #     cn = np.diff(cn)[0]
    #     print(cn)
    #     cycles_remaining.append(cn)

    # print("now: ", sb.get_current_voltage())
    # print(sb.get_relay_state())
    # time.sleep(1)
    # sb.set_switching_mode(1)  # DC
    # print(sb.set_relays_off())
    # time.sleep(0.5)
    sb.set_output_off()
    # tc.append(time.perf_counter() - t_start)
    # cn = sb.get_cycle_number()
    # print(cn)
    # cn = np.diff(cn)[0]
    # print(cn)
    # cycles_remaining.append(cn)
    # print("set: ", sb.set_voltage(0))
    # print("sp: ", sb.get_voltage_setpoint())
    # time.sleep(1.5)

    # tc.append(time.perf_counter() - t_start)
    # cn = sb.get_cycle_number()
    # print(cn)
    # cn = np.diff(cn)[0]
    # print(cn)
    # cycles_remaining.append(cn)
    # print("now: ", sb.get_current_voltage())
    # print("sp: ", sb.get_voltage_setpoint())
    sb.close()
    # tV, V = sb.get_voltage_buffer()
    # fig, ax1 = plt.subplots()
    # ax1.plot(tV, np.array(V) * 0 + Vset)
    # ax1.plot(tV, V)
    # ax2 = ax1.twinx()
    # ax2.plot(tc, cycles_remaining)
    # plt.xlabel("Time [s]")
    # ax1.set_ylabel("Voltage [V]")
    # ax2.set_ylabel("Cycles remaining")
    # # title = "Hermione PID {} {}V 3-DEAs".format(str(gains), Vset)
    # title = "Hermione AC mode {}Hz {}V 1-shorted".format(freq, Vset)
    # plt.title(title)
    # plt.savefig("test_data/ac/" + title + ".png")
    # plt.show()

    # ---------- this breaks the switchboard firmware - no longer responds to voltage commands ---------
    # print(sb.get_name())
    # print(sb.get_current_voltage())
    # print(sb.get_relay_state())
    # print(sb.set_relay_auto_mode())
    # time.sleep(1)
    # print(sb.set_voltage(200))
    # time.sleep(5)
    # print(sb.get_relay_state())
    # time.sleep(1)
    # print(sb.get_relay_state())
    # print(sb.set_relay_state(3, 0))
    # time.sleep(2)
    # print(sb.get_relay_state())
    # print(sb.set_relays_off())
    # print(sb.set_voltage(50))
    # time.sleep(2)
    # print(sb.get_current_voltage())
    # print(sb.get_voltage_setpoint())
    # print(sb.set_voltage(0))
    # time.sleep(2)
    # print(sb.get_current_voltage())
    # print(sb.get_voltage_setpoint())
    # print(sb.close())


def test_voltage_sequence():
    sb = Switchboard.detect(1)[0]
    sb.open()
    buf_length = 100000
    sb.start_continuous_reading(buffer_length=buf_length)
    Vset = 450
    freq = 2
    twait = 5
    cycles_remaining = []
    tc = []
    print(sb.get_name())
    # print("sp: ", sb.get_voltage_setpoint())
    # print("now: ", sb.get_current_voltage())
    # print("set: ", sb.set_voltage(500))
    # print("sp: ", sb.get_voltage_setpoint())
    # time.sleep(2)
    # print("now: ", sb.get_current_voltage())
    # print(sb.set_relays_on())
    # print(sb.get_relay_state())
    # gains = sb.set_pid_gains((0.26, 2.1, 0.005))
    gains = sb.get_pid_gains()
    print("PID gains:", gains)

    # sb.set_output_off()  # = set_switching_mode(0)
    # print(sb.set_relay_auto_mode())
    sb.set_relays_on()
    # sb.set_output_on()
    time.sleep(0.3)
    print(sb.get_relay_state())
    # time.sleep(1)
    # print(sb.get_relay_state())
    # print(sb.set_relay_state(3, 1))
    # time.sleep(1)
    # print(sb.set_relays_off())
    # time.sleep(1)
    sb.set_output_on()
    print(sb.get_current_voltage(), "V")
    sb.set_voltage(1000)
    time.sleep(twait)


def tune_pid():
    sb = Switchboard()
    sb.open()
    name = sb.get_name()
    print(name)
    t_start = time.perf_counter()
    sb.start_continuous_reading(reference_time=t_start, buffer_length=100000)
    Vset = 1500
    freq = 50
    samples = [1, 3, 4, 5]
    steps = 3
    wait_period = 3

    time.sleep(0.1)

    sb.set_relays_on(samples)  # switch on only the ones we want

    sb.set_pid_gains((0.2, 1.0, 0.005))
    gains = sb.get_pid_gains()
    print("PID gains:", gains)

    time.sleep(0.1)

    sb.set_output_on()

    # ramp ###########################
    for i in range(1, steps):
        v = i / steps * Vset
        sb.set_voltage(v)
        print(v, "V")
        time.sleep(wait_period)

    # high ##########################
    sb.set_voltage(Vset)
    print(Vset, "V")
    time.sleep(wait_period)

    sb.set_OC_frequency(freq)
    print("Cycling at", freq, "Hz")
    time.sleep(wait_period)

    sb.set_OC_mode(Switchboard.MODE_OFF)
    sb.set_voltage(0.95 * Vset)
    print("Stopped cycling. Switched OC off.")
    time.sleep(1)

    sb.set_OC_mode(Switchboard.MODE_DC)
    sb.set_voltage(Vset)
    print("Switched DC on.")
    time.sleep(wait_period)

    sb.set_voltage(0)
    print("Switched voltage off.")
    time.sleep(3)

    sb.set_voltage(Vset)
    print(Vset, "V")
    time.sleep(2 * wait_period)

    sb.set_voltage(0)
    sb.set_output_off()
    sb.set_relays_off()
    print("Switched everything off.")
    time.sleep(3)

    sb.close()

    # plot #####################

    tV, V, Vsp, rel, oc_mode, hb_mode = sb.get_data_buffer()

    v_on = oc_mode[0] > 0
    v_real = V * v_on
    vmax = np.amax(v_real)
    ivmax = np.argmax(v_real)
    tmax = tV[ivmax]
    str_vmax = "   Vmax: {} V at {:.1f} s".format(vmax, tmax)
    print(str_vmax)

    fig, ax1 = plt.subplots(figsize=(18, 9))
    ax1.plot(tV, Vsp, color="C0")
    ax1.plot(tV, V, color="C1")
    ax2 = ax1.twinx()
    ax2.plot(tV, oc_mode[0], color="C2")
    ax2.set_ylabel("OC on", color="C2")
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(["Off", "DC", "AC {} Hz".format(freq)])
    ax2.tick_params(axis='y', colors='C2')
    plt.xlabel("Time [s]")
    ax1.set_ylabel("Voltage [V]")
    ax2.set_ylabel("Switching mode")
    title = "{} PID{}, {}V, DEA{}, reduceV".format(name, str(gains), Vset, str(samples))
    plt.title(title + str_vmax)
    plt.savefig("test_data/pid/new/" + title + ".png")
    plt.show()


class HVMonitor:

    def __init__(self):
        # load probe parameters
        probe_fit = np.load("src\\hvps\\probe_fit_inverse_2.npy")
        self.p_probe = np.poly1d(probe_fit)

        # init DaqMx
        self.daq = None  # DaqMx()

    def analog_measurement(self, sampling_frequency, sample_number):
        tdata, data = self.daq.set_analog_measurement(['/Dev1/Ai0'], sampling_frequency, typeEch='fini',
                                                      sample_number=sample_number)
        data = self.p_probe(np.mean(data))

        return tdata, data


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    logging.basicConfig(level=logging.INFO, stream=stdout)

    test_switchboard()
    # tune_pid()
    # test_slow_voltage_rise()
