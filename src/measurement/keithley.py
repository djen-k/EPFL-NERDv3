import logging
import time
from datetime import datetime, timedelta
from threading import Lock

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import visa

from src.hvps.Switchboard import SwitchBoard
from src.image_processing import ImageCapture
from src.image_processing.StrainDetection import StrainDetector


def list_visa_instruments(instrument_search_string=None, resource_manager=None):
    """
    Get a list of available VISA instruments.
    :param instrument_search_string: A search string used to filter the results.
    :param resource_manager: If specified, this resource manager is used. If None, a new resource manager is created.
    :return: A dictionary, where the keys represent the device IDs and the values contain a device info object.
    """
    if resource_manager is None:
        resource_manager = visa.ResourceManager()
    resources = resource_manager.list_resources_info(query='?*::INSTR')
    if instrument_search_string is None:
        return resources

    # if a search string is given, find all entries where the id or alias contain the search string
    selection = {}
    for res_id, res_info in resources.items():
        if instrument_search_string in res_id or instrument_search_string in res_info.alias:
            selection[res_id] = res_info  # add to dictionary

    return selection


def get_resistance_channels(deas):
    """
    Get the list of channels on the DAQ6510 that correspond to the given DEAs
    :param deas: A list of indices of the DEAs to measure (in range 0 to 5)
    :return: A list of DAQ channels, as a comma-separated string
    """
    return ",".join([DAQ6510.resistance_channels[dea] for dea in deas])


class DAQ6510:
    resistance_channels = ["103,104", "105,106", "107,108", "109,110", "117,118", "119,120"]

    MODE_DEFAULT = 0  # default mode as after reset - no sensing method or channels selected
    MODE_SENSE_RESISTANCE = 1
    MODE_SENSE_CURRENT = 2
    MODE_UNKNOWN = -1  # state of the DAQ is not known. Probably a good idea to do a reset.

    def __init__(self, auto_connect=False):
        """
        Create an instance of a DAQ6510 digital multimeter and connect to an instrument.
        :param auto_connect: If True, an attempt is made to connect to the first available instrument.
        No error is generated if no instrument is found or the connection fails. Check 'is_connected()'
        to see if an instrument is connected!
        """
        self.logging = logging.getLogger("DAQ6510")

        self.resource_manager = None
        self.instrument = None
        self.instrument_id = None
        self.instrument_name = None
        self.serial_number = None
        self.firmware_version = None
        self.data_format = False
        self.mode = DAQ6510.MODE_UNKNOWN

        if auto_connect is True:
            self.connect()

    def __del__(self):
        self.disconnect()
        del self.logging

    def list_instruments(self):
        """
        Get a list of available DAQ6510 instruments.
        :return: A dictionary of available instruments, containing pairs of device IDs and device info objects.
        """
        try:
            return list_visa_instruments(instrument_search_string="6510", resource_manager=self.resource_manager)
        except Exception as ex:
            self.logging.warning("Unable to get a VISA instrument list: {}".format(ex))
            return {}

    def connect(self, instrument_id=None, reset=True):
        """
        Connect to a physical instrument.
        :param instrument_id: A VISA resource string specifying the desired instrument. If None, it automatically
        connects to the first suitable instrument it finds.
        :param reset: If True (default), the istrument is reset upon connection
        :return: True or False, indicating if connection was successful
        """

        if self.instrument is not None:  # already connected to a different device
            if instrument_id is None or instrument_id == self.instrument_id:  # same device, no need to reconnect
                self.logging.info("Already connected to {}".format(instrument_id))
                return True
            else:  # different device, need to disconnect first
                self.disconnect()

        if instrument_id is not None:  # ID given -> try to connect to to specified device
            try:
                if self.resource_manager is None:
                    self.resource_manager = visa.ResourceManager()

                self.logging.info("Connecting to instrument: {}".format(instrument_id))
                self.instrument = self.resource_manager.open_resource(instrument_id)  # connect to instrument
                self.instrument_id = instrument_id
                if reset:
                    self.reset()
                else:
                    self.mode = DAQ6510.MODE_UNKNOWN

                self.instrument.clear()  # clear input buffer so we don't receive messages from a previous session
                self._get_instrument_properties()
                self.logging.info("Successfully connected to {}".format(self.get_instrument_description()))
                return True  # return True if successful
            except Exception as ex:
                self.logging.warning("Unable to connect: {}".format(instrument_id, ex))
                self.instrument = None
                return False

        else:  # no ID given -> look for available instruments
            if self.instrument is None:
                available_instruments = self.list_instruments()
                for res_id, res_info in available_instruments.items():
                    if self.connect(res_id, reset):  # try connecting to instrument one at a time, stop if successful
                        return True

                if self.instrument is None:  # no instrument was found
                    self.logging.critical("No instrument found! Make sure the DAQ6510 is connected via USB!")
                    return False

    def disconnect(self):
        self.logging.debug("Disconnecting instrument")
        if self.instrument:
            self.instrument.close()
            self.logging.debug("{} disconnected.".format(self.instrument_name))
            # reset all fields
            self.instrument = None
            self.instrument_id = None
            self.instrument_name = None
            self.serial_number = None
            self.firmware_version = None
            self.data_format = False
            self.mode = DAQ6510.MODE_UNKNOWN
        else:
            self.logging.debug("No instrument was connected")

    def is_connected(self):
        return self.instrument is not None

    def reconnect(self, attempts=-1):
        """
        Try to reconnect to the instrument by disconnecting and connecting again to the same device. If unseccessful,
        reconnection will be attempted for the specified number of times before aborting the effort.
        :param attempts: The maximum number of reconnection attempts to perform. Set to -1 (default) to keep trying
        indefinitely. If 0, no attempts are made and the instrument remains disconnected.
        :return: True if the instrument was successfully reconnected, or False if after the specified number of attempts
        reconnection was not successful.
        """
        attempts_performed = 0
        inst_id = self.instrument_id  # remember current instrument ID so we can reconnect to the same one
        while attempts < 0 or attempts_performed < attempts:
            attempts_performed += 1
            self.logging.debug("Trying to reconnect instrument...  (attempt {})".format(attempts_performed))
            try:
                self.disconnect()
                connected = self.connect(inst_id, reset=False)
                if connected:
                    self.logging.debug("Reconnection successful.")
                    return True
                else:
                    self.logging.debug("Reconnection attempt failed.")
            except Exception as ex:
                self.logging.debug("Reconnection attempt failed: {}".format(ex))
            time.sleep(0.1)

        self.logging.critical("Unable to reconnect after {} attempts. Instrument remains disconnected".format(attempts))
        return False

    def send_command(self, command):
        if not self.is_connected():
            raise Exception("Not connected to any instrument")
        self.logging.debug("Sending command: {}".format(command))
        try:
            self.instrument.write(command)
        except Exception as ex:
            self.logging.error("Error receiving data: {}".format(ex))
            if self.reconnect():
                self.send_command(command)

    def send_query(self, command):
        if not self.is_connected():
            raise Exception("Not connected to any instrument")
        self.logging.debug("Sending query: {}".format(command))
        try:
            res = self.instrument.query(command)
            self.logging.debug("Response: {}".format(res))
            return res
        except Exception as ex:
            self.logging.error("Error receiving data: {}".format(ex))
            if isinstance(ex, visa.VisaIOError) and ex.error_code == -1073807339:
                self.logging.debug("Timeout error. Not reconnecting. Returning None")
                return None  # not due to lost connection - just have to try again next time
            else:
                if self.reconnect():
                    return self.send_query(command)
                else:
                    return None  # reconnect failed. nothing else we can do

    def query_data(self, command, data_points=0):
        """
        Sends the given command and receives numeric data in the active data format: ASCII ('s'), float
         ('f') or double ('d') as set by 'set_data_format(...)' (ASCII is default).
        Data is automatically interpreted and converted to a numpy array.
        :param command: The command to send to the instrument.
        :param data_points: The number of data points to read. Set 0 (default) if unknown.
        :return: The data received from the instrument as a numpy array.
        """
        self.logging.debug("Sending data query: {}".format(command))
        try:
            if self.data_format:
                return self.instrument.query_binary_values(command, datatype=self.data_format,
                                                           container=np.array, data_points=data_points)
            else:
                return self.instrument.query_ascii_values(command, container=np.array)
        except Exception as ex:
            self.logging.error("Error receiving data: {}".format(ex))
            if self.reconnect():
                return self.query_data(command, data_points)
            else:
                return None

    def reset(self):
        """
        Reset the instrument to its default state by sending an 'RST' command.
        """
        self.logging.debug("RESET")
        self.send_command("*RST")  # reset instrument to put it in a known state
        self.mode = DAQ6510.MODE_DEFAULT

    def set_timeout(self, timeout_s):
        """
        Set read timeout in seconds. Any communication with the device that is aborted after the specified timeout if
        no response was received.
        :param timeout_s: The timout in seconds after which to abort read operations
        """
        self.logging.debug("Sentting instrument timeout: {} s".format(timeout_s))
        self.instrument.timout = timeout_s * 1000

    def _get_instrument_properties(self):
        idn = self.send_query("*IDN?")
        idn = idn.strip()
        idn = idn.split(",")
        self.instrument_name = idn[1][6:]
        self.serial_number = idn[2]
        self.firmware_version = idn[3]

    def get_instrument_description(self, short=False):
        """Return the name of the opened instrument"""
        if short:
            d = "{} ({})".format(self.instrument_name, self.serial_number)
        else:
            args = (self.instrument_name, self.serial_number, self.firmware_version, self.instrument_id)
            d = "KEITHLEY {}, S/N: {}, Firmware: {} {{{}}}".format(*args)
        return d

    def clear_buffer(self):
        """
        Clear the input buffer and discard any previous messages from the instrument still stored in the buffer.
        """
        self.instrument.clear()

    def set_data_format(self, data_format='d'):
        """
        Set the instrument to use the specified data format:
         - 'f': single-precision floating point numbers (4 bytes per value)
         - 'd': double-precision floating point numbers (8 bytes per value)
         - 's': ASCII string (1 byte per character)
        :param data_format: The data format to use ('f', 'd' or 's')
        """

        if self.data_format == data_format:
            return  # nothing to do here

        if data_format == 'd':
            self.logging.debug("Setting data format to double-precision float")
            self.send_command("FORM REAL")
        elif data_format == 'f':
            self.logging.debug("Setting data format to single-precision float")
            self.send_command("FORM SREAL")
        elif data_format == 's':
            self.logging.debug("Setting data format to ASCII text")
            self.send_command("FORM ASCII")
        else:
            raise Exception("Invalid data format! Format must be 'f', 'd' or 's'!")

        # self.instrument.values_format.use_binary(data_format, False, np.array)
        self.data_format = data_format

    @staticmethod
    def timestamp_to_datetime(tstamp, t_offset=0):
        """
        Convert the given timestamp string to a datetime object and apply the given offset
        :param tstamp: A timestamp string as received from the DAQ6510 ('TST')
        :param t_offset: A offset in fractional seconds to add to the time. Used for matching DAQ time and PC time.
        :return: A datetime object representing the time specified by the given timestamp string, plus the offset.
        """
        nofrag, frag = tstamp.split(".")
        dt = datetime.strptime(nofrag, "%m/%d/%Y %H:%M:%S")
        frag = frag[:6]  # truncate to microseconds
        dt = dt.replace(microsecond=int(frag))
        dt += timedelta(seconds=t_offset)  # add offset to sync device time with computer time
        return dt

    def measure_DEA_resistance(self, deas, n_measurements=1, nplc=1, aperture=None):
        """
        Measure 4-point electrode resistance on the specified channels.
        :param deas: A list of indices of the DEAs to measure (in range 0 to 5)
        :param n_measurements: The number of measurements to take. If more than 1, the results will be averaged.
        :param aperture: The aperture (duration) of each individual measurement in seconds. If defined, nplc takes
        precedence over this parameter.
        :param nplc: The duration of each individual measurement in number of power line cycles (PLCs).
        If None, aperture (in seconds) is used instead. Otherwise nplc takes precedence.
        :return: A 1-D numpy array of resistance values (one for each DEA), or None if the measurement was
        not successful.
        """

        if not self.is_connected():
            return None

        # TODO: Measure shunt resistor to calibrate current measurement
        # TODO: Take two-point resistance measurement of electrode to check quality of contact and warn if bad

        n_deas = len(deas)
        if n_deas == 0:
            self.logging.warning("Resistance measurement requested without any DEAs specified. Nothing is returned.")
            return []

        str_channels = "(@111,{})".format(get_resistance_channels(deas))
        n_channels = len(deas) * 2 + 1

        self.logging.debug("Measuring resistance for DEAs: {} {}".format(deas, str_channels))

        scan_timeout = 1 * n_channels * n_measurements  # should never take more than 1 second per measurement

        # self.send_command("*RST")
        self.send_command("FUNC 'VOLT:DC', {}".format(str_channels))
        self.send_command("VOLT:DC:RANG:AUTO ON, {}".format(str_channels))  # doesn't take long so might as well
        # self.send_command("VOLT:DC:RANGe 10, {}".format(str_channels))  # should give us >10GΩ input impedance
        if aperture is not None:
            self.send_command("VOLT:DC:APERture {}, {}".format(aperture, str_channels))
        else:
            self.send_command("VOLT:DC:NPLC {}, {}".format(nplc, str_channels))
        self.send_command("VOLT:DC:AZER OFF, {}".format(str_channels))
        self.send_command("AZERo:ONCE")  # zero once now before starting the scan
        self.send_command("ROUT:SCAN {}".format(str_channels))
        self.send_command("ROUT:SCAN:COUN:SCAN {}".format(n_measurements))
        self.mode = DAQ6510.MODE_SENSE_RESISTANCE

        dt_start = datetime.now()
        self.send_command("INIT")
        i = 1
        start = time.perf_counter()
        elapsed = 0
        while i < n_channels * n_measurements:
            if elapsed > scan_timeout:
                self.logging.warning("Invalid measurement! "
                                     "Timeout elapsed before the expected amount of data wa received. "
                                     "Resetting instrument and returning 'None' result.")
                self.reset()  # since the scan didn't finish, something went wrong so better reset for a fresh start
                return None
            time.sleep(0.1)
            res = self.send_query("TRACe:ACTual?")
            if res is None:
                continue
            i = int(res)
            elapsed = time.perf_counter() - start

        data = self.query_data("TRACe:DATA? 1, {}, \"defbuffer1\", READ".format(i))
        if data is None:
            return None
        data = np.reshape(data, (n_measurements, n_channels))

        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})  # set print format for arrays

        # check the source voltage
        Vsource = data[:, 0]
        if np.any(Vsource < 9) or np.any(Vsource > 10):
            msg = "Source voltage for resistance measurement is outside the expected range: {}V".format(Vsource)
            self.logging.warning(msg)
        else:
            self.logging.debug("Source voltage for resistance measurement is {}V".format(Vsource))

        data = data[:, 1:]  # don't need the reference voltage anymore

        data = np.reshape(data, (n_measurements, n_deas, 2))
        Vshunt = data[:, :, 0]
        VDEA = data[:, :, 1]
        Rshunt = 100000  # 100kOhm  TODO: read in real calibration data

        Ishunt = Vshunt / Rshunt
        RDEA = VDEA / Ishunt

        self.logging.debug("Applied current: {} µA".format(np.array2string(Ishunt * 1000000, precision=2)))
        self.logging.debug("Voltage across electrode: {} V".format(np.array2string(VDEA, precision=3)))
        self.logging.debug("Electrode resistance: {} kΩ".format(np.array2string(RDEA / 1000, precision=3)))

        RDEA = np.mean(RDEA, axis=0)  # take avg even for single measurement to reduce to 1-D array
        self.logging.debug("Average resistance: {} kΩ".format(np.array2string(RDEA / 1000, precision=3)))

        return RDEA

    def measure_DEA_resistance_with_checks(self, deas, n_measurements=1, nplc=1, aperture=None):
        """
        Measure 4-point electrode resistance on the specified channels. Also performs 2-point measurements to check that
        the contacts are OK.
        :param deas: A list of indices of the DEAs to measure (in range 0 to 5)
        :param n_measurements: The number of measurements to take. If more than 1, the results will be averaged.
        :param aperture: The aperture (duration) of each individual measurement in seconds. If defined, nplc takes
        precedence over this parameter.
        :param nplc: The duration of each individual measurement in number of power line cycles (PLCs).
        If None, aperture (in seconds) is used instead. Otherwise nplc takes precedence.
        :return: A 1-D numpy array of resistance values (one for each DEA), or None if the measurement was
        not successful.
        """

        if not self.is_connected():
            return None

        # TODO: Measure shunt resistor to calibrate current measurement
        # TODO: Take two-point resistance measurement of electrode to check quality of contact and warn if bad

        n_deas = len(deas)
        if n_deas == 0:
            self.logging.warning("Resistance measurement requested without any DEAs specified. Nothing is returned.")
            return []

        # check supply voltage #######################################
        str_channels = "(@111)"
        self.send_command("FUNC 'VOLT:DC', {}".format(str_channels))
        self.send_command("VOLT:DC:RANG:AUTO ON, {}".format(str_channels))
        self.send_command("VOLT:DC:NPLC 1, {}".format(str_channels))
        self.send_command("VOLT:DC:AZER ON, {}".format(str_channels))
        self.send_command("ROUT:CLOSe (@111)")
        time.sleep(1)
        v_source = float(self.send_query("READ?"))
        if np.any(v_source < 9) or np.any(v_source > 10):
            msg = "Source voltage for resistance measurement is outside the expected range: {}V".format(v_source)
            self.logging.warning(msg)
            print(msg)
        else:
            self.logging.debug("Source voltage for resistance measurement is {}V".format(v_source))
            print("Source voltage for resistance measurement is {}V".format(v_source))

        for dea in deas:
            str_channels = "(@106)"  # .format(get_resistance_channels(dea))

            self.logging.debug("Measuring resistance for DEA {} (channels: {})".format(deas, str_channels))

            self.send_command("FUNC 'RES', {}".format(str_channels))
            self.send_command("ROUT:CLOSe {}".format(str_channels))
            time.sleep(1)
            shunt_res_on = float(self.send_query("READ?"))
            print("Shunt resistance ON [kOhm]:", shunt_res_on / 1000)

            self.send_command("ROUT:CLOSe (@123)")  # make sure channels 101-110 are disconnected from 111-120
            self.send_command("ROUT:CLOSe (@112)")  # close relays 112, 113 to connect the relay on the resistance board
            self.send_command("ROUT:CLOSe (@113)")  # this switches off the 9.5V power used for R measurements
            time.sleep(1)  # wait for relay to switch and current to decay

            self.mode = DAQ6510.MODE_SENSE_CURRENT

            shunt_res_off = float(self.send_query("READ?"))
            print("Shunt resistance OFF [kOhm]:", shunt_res_off / 1000)

            time.sleep(1)

            self.send_command("ROUT:OPEN (@112)")  # close relays 112, 113 to connect the relay on the resistance board
            self.send_command("ROUT:OPEN (@113)")  # this switches off the 9.5V power used for R measurements

        # np.set_printoptions(formatter={'float': '{: 0.3f}'.format})  # set print format for arrays
        #
        # self.mode = DAQ6510.MODE_SENSE_RESISTANCE
        #
        # # check the source voltage
        # v_source = data[:, 0]
        # if np.any(v_source < 9) or np.any(v_source > 10):
        #     msg = "Source voltage for resistance measurement is outside the expected range: {}V".format(v_source)
        #     self.logging.warning(msg)
        # else:
        #     self.logging.debug("Source voltage for resistance measurement is {}V".format(v_source))
        #
        # data = data[:, 1:]  # don't need the reference voltage anymore
        #
        # data = np.reshape(data, (n_measurements, n_deas, 2))
        # Vshunt = data[:, :, 0]
        # VDEA = data[:, :, 1]
        # Rshunt = 100000  # 100kOhm  TODO: read in real calibration data
        #
        # Ishunt = Vshunt / Rshunt
        # RDEA = VDEA / Ishunt

        # self.logging.debug("Applied current: {} µA".format(np.array2string(Ishunt * 1000000, precision=2)))
        # self.logging.debug("Voltage across electrode: {} V".format(np.array2string(VDEA, precision=3)))
        # self.logging.debug("Electrode resistance: {} kΩ".format(np.array2string(RDEA / 1000, precision=3)))
        #
        # RDEA = np.mean(RDEA, axis=0)  # take avg even for single measurement to reduce to 1-D array
        # self.logging.debug("Average resistance: {} kΩ".format(np.array2string(RDEA / 1000, precision=3)))

        # return RDEA
        return np.array((shunt_res_on, shunt_res_off))

    def measure_current(self, nplc=1):
        """
        Take a single current measurement on channel 122. 9V5 supply to shunt resistors is switched off before the
        measurement is taken.
        :param nplc: The length of the measurement in number of power line cycles. Can be fractional.
        1-5 PLCs gives the lowest noise according to the DAQ6510 manual.
        :return: A single current reading in A.
        """

        self.logging.debug("Measuring current")

        if self.mode != DAQ6510.MODE_SENSE_CURRENT:  # only do setup if required
            self.logging.debug("Switching to current sensing mode")

            # there is only one current channel used, so ne need to  select different channels for now.
            # self.send_command("*RST")  # make sure we start with default configuration
            self.send_command("SENS:FUNC 'CURR:DC', (@122)")
            # daq.send_command("SENS:CURR:DC:APERture 0.02, (@122)")
            # measurement aperture of 1-2 power line cycles gives the lowest noise (according to DAQ manual)
            self.send_command("SENS:CURR:DC:NPLC {}, (@122)".format(nplc))
            self.send_command("SENS:CURR:DC:RANGe:AUTO ON, (@122)")  # doesn't take long so might as well
            # self.send_command("SENS:CURR:DC:RANGe 100e-6, (@122)")
            # daq.send_query("SENS:CURR:DC:APERture? (@122)")

            self.send_command("ROUT:CLOSe (@122)")  # close relays on current measurement channel
            # must close 122 first because it will open all other channels since it thinks they might interfere
            self.send_command("ROUT:CLOSe (@112)")  # close relays 112, 113 to connect the relay on the resistance board
            self.send_command("ROUT:CLOSe (@113)")  # this switches off the 9.5V power used for R measurements
            time.sleep(1)  # wait for relay to switch and current to decay

            self.mode = DAQ6510.MODE_SENSE_CURRENT

        # perform a measurement
        res = self.send_query("READ?")
        if res is None:
            return None

        cur = float(res)

        self.logging.debug("Measured current: {} A".format(cur))

        # self.send_command("ROUT:OPEN (@112)")  # open relays 112 and 113 again to switch 9.5V back on
        # self.send_command("ROUT:OPEN (@113)")

        return cur

    def calibrate_clock(self):
        """
        Synchronize the clock between the computer and the multimeter by setting the multimeter time to the system time
        of the computer (down to the second) and then measureing the microsecond offset between the two clocks.
        :return: The offset, in fractional seconds, that must be added to the multimeter time in order to make it match
        the computer's system time as determined by datetime.now()
        """

        self.logging.warning("Calibrating instrument clock. This methodis known to be faulty. Do not use!")

        # TODO: don't change system time of the DAQ! It gets applied with some delay and may occur during measurement
        # TODO: get more precise estimate of the offset between the clocks somehow - this doesn't cut it!
        tstart = datetime.now()
        delay = 1 - tstart.microsecond / 1000000
        time.sleep(delay)  # wait until the next full second
        t = datetime.now()
        msg = ":SYSTem:TIME {}, {}, {}, {}, {}, {}".format(t.year, t.month, t.day, t.hour, t.minute, t.second)
        self.send_command(msg)

        n = 50000
        t = []
        res = []
        for i in range(n):
            t.append(datetime.now())
            res.append(self.send_query(":SYSTem:TIME?"))

        res = np.array([int(r) for r in res])
        idx = np.flatnonzero(np.diff(res)) + 1
        tus = np.array([t[i].microsecond for i in idx])
        t_offset = np.round(np.max(tus)) / 1000000 - 1

        return t_offset


def test_resistance_measurement():
    daq = DAQ6510(auto_connect=True)
    # daq.logging.setLevel(logging.DEBUG)
    res = daq.measure_DEA_resistance(range(6), n_measurements=1, nplc=1)
    # res = daq.measure_DEA_resistance_with_checks([0], n_measurements=1, nplc=1)
    print("Result [kOhm]:", np.array2string(res / 1000, precision=2))


def test_current_measurement_simple():
    daq = DAQ6510(auto_connect=True)
    res = daq.measure_current(nplc=1)
    print("Result [mA]:", res * 1000)


def test_current_measurements(nplc=1):
    hvps = SwitchBoard()
    hvps.open()
    print("Connected to HVPS", hvps.get_name())
    print("Relays on:", hvps.set_relays_on([0]))
    # print("Relays on:", hvps.set_relays_on())
    hvps.set_voltage(0, block_until_reached=True)
    hvps.set_switching_mode(0)

    daq = DAQ6510()
    daq.connect(reset=True)
    # daq.set_timeout(0.01)
    daq.get_instrument_description()

    daq.send_command("*RST")
    daq.send_command("SENS:FUNC 'CURR:DC', (@122)")
    # daq.send_command("SENS:CURR:DC:APERture 0.02, (@122)")
    daq.send_command("SENS:CURR:DC:NPLC {}, (@122)".format(nplc))  # measure for n power line cycles
    daq.send_command("SENS:CURR:DC:RANGe 100e-6, (@122)")
    # daq.send_query("SENS:CURR:DC:APERture? (@122)")
    # daq.send_command("SENS:COUNt 10")

    daq.send_command("ROUT:CLOSe (@122)")  # close relays on current measurement channel
    daq.send_command("ROUT:CLOSe (@112)")  # close relays to switch off 9.5V power used for R measurements
    daq.send_command("ROUT:CLOSe (@113)")
    time.sleep(0.5)

    # perform a series of measurements
    Vtest = 1000
    # n = np.array(np.round(np.array([1500, 1500, 3000, 2000])/nplc), dtype=np.int)
    # settling_delay = np.array(np.round(np.array([1000, 1000, 2000, 1500])/nplc), dtype=np.int)
    n = np.array(np.round(np.array([150, 150, 500, 300]) / nplc), dtype=np.int)
    settling_delay = np.array(np.round(np.array([50, 50, 200, 200]) / nplc), dtype=np.int)
    # n = np.array(np.round(np.array([50, 50, 100, 100]) / nplc), dtype=np.int)
    # settling_delay = np.array(np.round(np.array([20, 20, 40, 40]) / nplc), dtype=np.int)

    sw_mode = "OC"
    # sw_mode = "DCDC"

    cur = []
    V = []
    Vset = []
    t = []
    ntotal = np.sum(n)

    hvps.set_voltage(0)  # ramp up voltage but OC remains off
    hvps.set_switching_mode(0)  # open OC but leave voltage 0
    for i in range(n[0]):
        cur.append(float(daq.send_query("READ?")))
        V.append(hvps.get_current_voltage())
        Vset.append(0)
        t.append(time.perf_counter())
        print("{:.1f} %".format(i / ntotal * 100))

    if sw_mode == "OC":
        hvps.set_voltage(Vtest)  # ramp up voltage but OC remains off
    elif sw_mode == "DCDC":
        hvps.set_switching_mode(1)  # open OC but leave voltage 0
    for i in range(n[1]):
        cur.append(float(daq.send_query("READ?")))
        V.append(hvps.get_current_voltage())
        Vset.append(0)
        t.append(time.perf_counter())
        print("{:.1f} %".format((n[0] + i) / ntotal * 100))

    if sw_mode == "OC":
        hvps.set_switching_mode(1)  # open OC to apply Vtest to DEAs
    elif sw_mode == "DCDC":
        hvps.set_voltage(Vtest)  # ramp up voltage to Vtest
    for i in range(n[2]):
        cur.append(float(daq.send_query("READ?")))
        V.append(hvps.get_current_voltage())
        Vset.append(Vtest)
        t.append(time.perf_counter())
        print("{:.1f} %".format((n[0] + n[1] + i) / ntotal * 100))

    if sw_mode == "OC":
        hvps.set_switching_mode(0)  # close OC to discharge DEAs
    elif sw_mode == "DCDC":
        hvps.set_voltage(0)  # reduce voltage to 0
    for i in range(n[3]):
        cur.append(float(daq.send_query("READ?")))
        V.append(hvps.get_current_voltage())
        Vset.append(0)
        t.append(time.perf_counter())
        print("{:.1f} %".format((n[0] + n[1] + n[2] + i) / ntotal * 100))

    daq.send_command("ROUT:OPEN (@112)")  # open relays again to switch 9.5V back on
    daq.send_command("ROUT:OPEN (@113)")

    hvps.set_voltage(0)
    hvps.set_switching_mode(0)
    hvps.set_relays_off()
    hvps.close()

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, linewidth=500)  # set print format for arrays
    cur = np.array(cur)  # .reshape((-1, 1))
    cur = cur * 1000000000  # to nA
    V = np.array(V)  # .reshape((-1, 1))
    Vset = np.array(Vset)  # .reshape((-1, 1))
    t = np.array(t)  # .reshape((-1, 1))
    t = t - t[0]

    excl_avg = np.array([True] * ntotal)
    avg = []
    max_seg = []
    tstart = []
    print("Leakage current [nA]:")
    istart = 0 + settling_delay[0]
    iend = n[0] - 1
    avg.append(np.mean(cur[istart:iend]))
    max_seg.append(np.max(cur[istart:iend]))
    excl_avg[istart:iend] = False
    tstart.append(t[istart])
    print("V off, OC off ({:.2f} to {:.2f}): {:.5f}".format(t[istart], t[iend - 1], avg[-1]))
    istart = n[0] + settling_delay[1]
    iend = n[0] + n[1] - 1
    avg.append(np.mean(cur[istart:iend]))
    max_seg.append(np.max(cur[istart:iend]))
    excl_avg[istart:iend] = False
    tstart.append(t[istart])
    print("V on, OC off ({:.2f} to {:.2f}): {:.5f}".format(t[istart], t[iend - 1], avg[-1]))
    istart = n[0] + n[1] + settling_delay[2]
    iend = n[0] + n[1] + n[2] - 1
    avg.append(np.mean(cur[istart:iend]))
    max_seg.append(np.max(cur[istart:iend]))
    excl_avg[istart:iend] = False
    tstart.append(t[istart])
    print("V on, OC on ({:.2f} to {:.2f}): {:.5f}".format(t[istart], t[iend - 1], avg[-1]))
    istart = n[0] + n[1] + n[2] + settling_delay[3]
    iend = n[0] + n[1] + n[2] + n[3] - 1
    avg.append(np.mean(cur[istart:iend]))
    max_seg.append(np.max(cur[istart:iend]))
    excl_avg[istart:iend] = False
    tstart.append(t[istart])
    print("V on, OC off ({:.2f} to {:.2f}): {:.5f}".format(t[istart], t[iend - 1], avg[-1]))

    ftitle = "Leakage current test 1DEA, {} PLC".format(nplc)

    mat = {"t": t, "I": cur, "V": V, "Iavg": avg, "phase_plc": n, "settling_delay_plc": settling_delay}
    fname = 'test_data/nplc/{} {}'.format(datetime.now().strftime("%Y%m%d-%H%M%S"), ftitle)
    sio.savemat(fname + ".mat", mat)

    plt.close("all")
    plt.rcParams.update({'font.size': 16})
    fig, ax1 = plt.subplots()
    fig.set_size_inches(12, 8)
    axcolor = (231 / 255, 76 / 255, 60 / 255)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Current [nA]', color=axcolor)
    ax1.plot(t, cur, ".-", color=(241 / 255, 148 / 255, 138 / 255))
    cur[excl_avg] = np.nan
    ax1.plot(t, cur, ".-", color=(203 / 255, 67 / 255, 53 / 255), lw=3, ms=10)
    ax1.tick_params(axis='y', labelcolor=axcolor)
    ax1.grid(True)
    for ts, y, c in zip(tstart, max_seg, avg):
        ax1.text(ts, y + 5, "{:.3f} nA".format(c))
    plt.title(ftitle)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Voltage [V]', color=color)  # we already handled the x-label with ax1
    # ax2.plot(t, Vset, color="k")
    ax2.plot(t, V, ".-", color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(fname + '.png')

    plt.show()

    plt.close("all")
    plt.rcParams.update({'font.size': 16})
    fig, ax1 = plt.subplots()
    fig.set_size_inches(12, 8)
    axcolor = (231 / 255, 76 / 255, 60 / 255)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Current [nA]', color=axcolor)
    ax1.plot(t, cur, ".-", color=(203 / 255, 67 / 255, 53 / 255))
    ax1.tick_params(axis='y', labelcolor=axcolor)
    ax1.grid(True)
    for ts, y, c in zip(tstart, max_seg, avg):
        ax1.text(ts, 0, "{:.3f} nA".format(c))
    plt.title(ftitle + "magnified")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(fname + 'mag.png')

    plt.show()


def measure_resistance_sequence():
    #######################################
    #  test parameters
    #######################################

    folder = "test_data/res/"
    fname = "{}_resistance+voltage_DCDC_DEA2".format(datetime.now().strftime("%Y%m%d-%H%M%S"))
    DEAs = [0]

    #######################################
    #  initialize instruments
    #######################################

    # Multimeter #########################

    daq = DAQ6510()
    daq.connect(reset=True)
    # daq.set_timeout(0.01)
    print("Connected to", daq.get_instrument_description())
    R = []

    # Switchboard ########################################

    hvps = SwitchBoard()
    hvps.open()
    print("Connected to HVPS", hvps.get_name())
    print("Relays on:", hvps.set_relays_on())
    # make sure everything is discharged
    hvps.set_voltage(0)
    hvps.set_switching_mode(0)
    V = []
    t = []

    ########################################################
    #  run test
    ########################################################

    hvps.set_switching_mode(1)
    print("Recording data...")

    #  start recording

    for i in range(10):
        t.append(time.perf_counter())
        R.append(daq.measure_DEA_resistance(DEAs)[0])
        V.append(hvps.get_current_voltage())

    # set voltage high
    hvps.set_voltage(1000)

    for i in range(30):
        t.append(time.perf_counter())
        R.append(daq.measure_DEA_resistance(DEAs)[0])
        V.append(hvps.get_current_voltage())

    # set voltage low
    hvps.set_voltage(0)

    for i in range(20):
        t.append(time.perf_counter())
        R.append(daq.measure_DEA_resistance(DEAs)[0])
        V.append(hvps.get_current_voltage())

    ##########################################################
    # retrieve and analyze data
    ##########################################################

    t = np.array(t) - t[0]
    V = np.array(V)
    R = np.array(R)

    ##################################################################
    # save data and visualization
    ##################################################################

    mat = {"t": t, "R": R, "V": V}
    sio.savemat(folder + fname + ".mat", mat)

    plt.rcParams.update({'font.size': 16})
    fig, ax1 = plt.subplots()
    fig.set_size_inches(12, 8)
    axcolor = 'tab:red'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Resistance [kΩ]', color=axcolor)
    ax1.plot(t, R / 1000, ".-", color=axcolor)
    ax1.tick_params(axis='y', labelcolor=axcolor)
    ax1.grid(True)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Voltage [V]', color=color)  # we already handled the x-label with ax1
    ax2.plot(t, V, ".-", color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(folder + fname + '.png')

    plt.show()


def measure_sequence_all():
    #######################################
    #  test parameters
    #######################################

    folder = "test_data/all/"
    fname = "{}_strain+resistance+voltage_OC_DEA1".format(datetime.now().strftime("%Y%m%d-%H%M%S"))
    str_channels = "(@103,104)"
    # str_channels = "(@105,106)"
    n_channels = 2
    n_measurements = 200
    cameras = [0]
    # cameras = [3]  -> DEA2

    #######################################
    #  initialize instruments
    #######################################

    # Multimeter #########################

    daq = DAQ6510()
    # daq.set_timeout(0.01)
    print("Connected to", daq.get_instrument_description())
    daq.send_command("*RST")
    # print("calibrating clock")
    # t_offset = daq.calibrate_clock()
    t_offset = -0.55
    daq.send_command("FUNC 'VOLT:DC', {}".format(str_channels))
    daq.send_command("VOLT:DC:RANGe 10, {}".format(str_channels))  # this should set input impedance to >10 GΩ
    daq.send_command("VOLT:DC:NPLC 1, {}".format(str_channels))
    daq.send_command("VOLT:DC:AZERo OFF, {}".format(str_channels))
    daq.send_command("AZERo:ONCE")  # zero now before starting the scan
    daq.send_command("ROUT:SCAN {}".format(str_channels))
    daq.send_command("ROUT:SCAN:COUN:SCAN {}".format(n_measurements))

    # Camera ################################################

    print("Initializing cameras")
    cap = ImageCapture.SharedInstance
    cap.find_cameras()
    cap.select_cameras(cameras)
    print("cameras found")
    print("setting exposure")
    cap.set_fixed_exposure()
    frame_times = []
    frames = []
    lock = Lock()

    def new_frame(image, timestamp, camera_id):
        with lock:
            frames.append(image)
            frame_times.append(timestamp)

    cap.set_new_image_callback(new_frame)

    # Switchboard ########################################

    hvps = SwitchBoard()
    hvps.open()
    print("Connected to HVPS", hvps.get_name())
    print("Relays on:", hvps.set_relays_on())
    # make sure everything is discharged
    hvps.set_voltage(0)
    hvps.set_switching_mode(0)
    SW = []
    tSW = []

    ########################################################
    #  run test
    ########################################################

    print("Recording data...")

    # record start time
    t_start = time.perf_counter()
    dt_start = datetime.now()

    # start logging data
    hvps.start_voltage_reading(reference_time=t_start)
    cap.start_capture_thread(max_fps=10)
    daq.send_command("INIT")

    tSW.append(t_start)
    SW.append(0)

    time.sleep(0.4)  # record a few ms of initial state before anything is switched on

    # set voltage high
    hvps.set_voltage(1000)

    time.sleep(0.5)  # wait for DCDC output to rise

    # set OC on
    hvps.set_switching_mode(1)
    tnow = time.perf_counter()
    tSW.append(tnow)
    SW.append(0)
    tSW.append(tnow)
    SW.append(1)

    time.sleep(6)  # wait high

    # set OC off
    hvps.set_switching_mode(0)
    tnow = time.perf_counter()
    tSW.append(tnow)
    SW.append(1)
    tSW.append(tnow)
    SW.append(0)

    time.sleep(0.5)  # wait a few ms before turning DCDC off

    # set voltage low
    hvps.set_voltage(0)

    # wait until DAQ is has finished recording
    i = 1
    while i < n_channels * n_measurements:
        time.sleep(0.1)
        i = int(daq.send_query("TRACe:ACTual?"))

    # record end time
    t_stop = time.perf_counter()

    # stop recording data
    hvps.stop_voltage_reading()
    cap.stop_capture_thread()
    tSW.append(t_stop)
    SW.append(0)

    # show elapsed time
    elapsed = t_stop - t_start
    print("Done!")
    print("Measurement duration:", elapsed)

    time.sleep(1)

    ##########################################################
    # retrieve and analyze data
    ##########################################################

    # resistance #####################################################

    Rdata = daq.send_query("TRACe:DATA? 1, {}, \"defbuffer1\", READ".format(i))
    tR = daq.send_query("TRACe:DATA? 1, {}, \"defbuffer1\", TST".format(i))

    Rdata = Rdata.split(",")
    Rdata = [float(d) for d in Rdata]
    Rdata = np.array(Rdata)
    Rdata = np.reshape(Rdata, (n_measurements, n_channels))

    tR = tR.split(",")
    tR = np.array(tR)
    tR = np.reshape(tR, (n_measurements, n_channels))
    tR = tR[:, 0]  # use only the timestamp of the first measurement for each pair
    print(tR)
    tR = [DAQ6510.timestamp_to_datetime(tst, t_offset) for tst in tR]  # convert to datetime and apply offset
    tR = [(t - dt_start).total_seconds() for t in tR]  # get time in seconds since start

    Vshunt = Rdata[:, 0]
    VDEA = Rdata[:, 1]
    Rshunt = 100000  # 100kOhm

    Ishunt = Vshunt / Rshunt
    RDEA = VDEA / Ishunt
    RDEA = np.reshape(RDEA, (-1, 1))

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, linewidth=500)  # set print format for arrays
    print("Voltages [V]:")
    print(Rdata)

    print("Resistance [kΩ]:")
    print(RDEA / 1000)

    # voltage #############################################

    tV, V = hvps.get_data_buffer()  # voltage data
    tSW = [t - t_start for t in tSW]  # switch mode times in seconds since start

    # strain #############################################

    print("measuring strain in {} images".format(len(frames)))
    sdet = StrainDetector()
    strain = [sdet.get_dea_strain([img])[0][0, -1] for img in frames]
    ts = [(t - dt_start).total_seconds() for t in frame_times]  # frame times in seconds since start

    # time
    t_ana = time.perf_counter()
    print("Analysis duration:", t_ana - t_stop)

    ##################################################################
    # save data and visualization
    ##################################################################

    save_and_plot(tR, RDEA, ts, strain, tV, V, tSW, SW, folder, fname)


def test_digitize_current():
    daq = DAQ6510()
    daq.connect(reset=True)
    # daq.set_timeout(10)
    daq.get_instrument_description()
    daq.set_data_format('f')

    hvps = SwitchBoard()
    hvps.open()
    print("Connected to HVPS", hvps.get_name())
    print("Relays on:", hvps.set_relays_on())
    hvps.set_voltage(0)
    hvps.set_switching_mode(0)

    t = 3  # measurement duration in sec
    sample_rate = 1000000  # 1000000
    Vtest = 1000
    sw_mode = "OC"
    # sw_mode = "DCDC"

    count = sample_rate * t  # 1 MHz sample rate

    buffer = "'curDigBuffer'"

    # daq.send_command("TRACe:POINts 10, 'defbuffer1'")  # reduce size of default buffers to 10 (minimum)
    # daq.send_command("TRACe:POINts 10, 'defbuffer2'")  # to make space for current readings (max 7M)
    daq.send_command("TRACe:MAKE {}, {}".format(buffer, count))  # create buffer of correct size to store the data

    daq.send_command("DIG:FUNC 'CURR'")  # select digitize current
    daq.send_command("DIG:CURR:RANG 100e-6")  # set measurement range to 100 µA (smallest range)
    daq.send_command("DIG:CURR:SRAT {}".format(sample_rate))  # set sample rate (max 1MHz)
    daq.send_command("DIG:CURR:APER AUTO")  # set aperture to auto (will be 1µs @ 1MHz)
    daq.send_command("DIG:COUN {}".format(count))  # set number of measurements (theoretical max 7,000,000 at 1 MHz)
    daq.send_command(":TRIGger:BLOCk:MDIGitize 1, {}, AUTO".format(buffer))  # configure a trigger (just single shot)

    srate = float(daq.send_query("DIG:CURR:SRAT?"))
    print("Sample rate:", srate, "Hz")
    aper = daq.send_query("DIG:CURR:APER?")
    print("Aperture:", aper)
    daq_count = float(daq.send_query("DIG:COUN?"))
    if count != daq_count:
        raise Exception("Count is not what it is supposed to be")
    print("Format:", daq.send_query("FORM?"))

    daq.send_command("ROUT:CLOSe (@122)")  # close relays on current measurement channel
    daq.send_command("ROUT:CLOSe (@112)")  # close relays to switch off 9.5V power used for R measurements
    daq.send_command("ROUT:CLOSe (@113)")
    time.sleep(0.5)

    # perform a digitize measurement
    # daq.send_query("READ:DIG? 'curDigBuffer'")
    # time.sleep(count/srate + 1)  # wait for the time it will take to record the data (plus a bit, just to be safe)
    daq.send_command("INIT")
    start = time.perf_counter()

    # measure everything off for a moment
    hvps.set_voltage(0)
    hvps.set_switching_mode(0)
    time.sleep(0.1 * t)

    if sw_mode == "OC":
        hvps.set_voltage(Vtest)  # ramp up voltage but OC remains off
    elif sw_mode == "DCDC":
        hvps.set_switching_mode(1)  # open OC but leave voltage 0
    time.sleep(0.1 * t)

    if sw_mode == "OC":
        hvps.set_switching_mode(1)  # open OC to apply Vtest to DEAs
    elif sw_mode == "DCDC":
        hvps.set_voltage(Vtest)  # ramp up voltage to Vtest
    time.sleep(0.4 * t)

    if sw_mode == "OC":
        hvps.set_switching_mode(0)  # close OC to discharge DEAs
    elif sw_mode == "DCDC":
        hvps.set_voltage(0)  # reduce voltage to 0

    i = 0
    while i < count:
        time.sleep(0.5)
        i = int(daq.send_query("TRACe:ACTual? {}".format(buffer)))
        print(i)
    stop = time.perf_counter()
    print("Time until data is available:", stop - start)
    # retrieve data
    start = time.perf_counter()
    data = daq.query_data("TRACe:DATA? 1, {}, {}, REL, READ".format(count, buffer), 2 * count)
    data = data.reshape((-1, 2)) * [1000, 1000000]  # shape into two columns and convert to ms and µA
    # reltime = daq.query_data("TRACe:DATA? 1, {}, {}, REL".format(count, buffer))
    stop = time.perf_counter()
    print("Time to retrieve data:", stop - start)

    daq.send_command("ROUT:OPEN (@112)")  # open relays again to switch 9.5V back on
    daq.send_command("ROUT:OPEN (@113)")
    hvps.set_relays_off()

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, linewidth=500)  # set print format for arrays
    print(data[[0, 1, 2, -3, -2, -1], :])

    mat = {"t": data[:, 0], "I": data[:, 1]}
    sio.savemat('test_data/leakage_test_{}.mat'.format(datetime.now().strftime("%Y%m%d-%H%M%S")), mat)

    plt.plot(data[:, 0], data[:, 1])
    plt.xlabel("Time [ms]")
    plt.ylabel("Current [µA]")
    plt.show()


def test_digitize_voltage():
    daq = DAQ6510()
    # daq.set_timeout(10)
    daq.get_instrument_description()
    daq.set_data_format('f')

    hvps = SwitchBoard()
    hvps.open()
    print("Connected to HVPS", hvps.get_name())
    print("Relays on:", hvps.set_relays_on([0]))
    hvps.set_voltage(0)
    hvps.set_switching_mode(0)

    t = 3  # measurement duration 1s
    count = 1000000 * t  # 1 MHz sample rate
    buffer = "'curDigBuffer'"

    daq.send_command("TRACe:MAKE {}, {}".format(buffer, count))  # create buffer of correct size to store the data

    daq.send_command("DIG:FUNC 'VOLT'")  # select digitize current
    daq.send_command("DIG:VOLT:RANG 1000")  # set measurement range to 100 µA (smallest range)
    daq.send_command("DIG:VOLT:SRAT MAX")  # set sample rate to max (1MHz)
    daq.send_command("DIG:VOLT:APER AUTO")  # set aperture to auto (will be 1µs @ 1MHz)
    daq.send_command("DIG:COUN {}".format(count))  # set number of measurements to 7,000,000 (max at 1 MHz)
    daq.send_command(":TRIGger:BLOCk:MDIGitize 1, {}, AUTO".format(buffer))  # configure a trigger (just single shot)

    srate = float(daq.send_query("DIG:VOLT:SRAT?"))
    print("Sample rate:", srate, "Hz")
    irange = float(daq.send_query("DIG:VOLT:RANGe?"))
    unit = daq.send_query("DIG:VOLT:UNIT?")
    print("Input range:", irange, unit)
    daq_count = float(daq.send_query("DIG:COUN?"))
    if count != daq_count:
        raise Exception("Count is not what it is supposed to be")
    print("Format:", daq.send_query("FORM?"))

    # start digitize measurement
    daq.send_command("INIT")
    # time.sleep(count/srate + 1)  # wait for the time it will take to record the data (plus a bit, just to be safe)
    start = time.perf_counter()

    time.sleep(0.1)
    hvps.set_voltage(500)
    time.sleep(2.6)
    hvps.set_voltage(0)

    i = 0
    while i < count:
        time.sleep(0.3)
        i = int(daq.send_query("TRACe:ACTual? {}".format(buffer)))
        print(i)
    stop = time.perf_counter()
    print("Time until data is available:", stop - start)
    # retrieve data
    start = time.perf_counter()
    data = daq.query_data("TRACe:DATA? 1, {}, {}, REL, READ".format(count, buffer), 2 * count)
    data = data.reshape((-1, 2)) * [1000, 1]  # shape into two columns and convert to ms and µA
    # reltime = daq.query_data("TRACe:DATA? 1, {}, {}, REL".format(count, buffer))
    stop = time.perf_counter()
    print("Time to retrieve data:", stop - start)

    daq.send_command("ROUT:OPEN (@112)")  # open relays again to switch 9.5V back on
    daq.send_command("ROUT:OPEN (@113)")
    hvps.set_relays_off()

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, linewidth=500)  # set print format for arrays
    print(data[[0, 1, 2, -3, -2, -1], :])

    mat = {"t": data[:, 0], "V": data[:, 1]}
    sio.savemat('test_data/VD_test_{}.mat'.format(datetime.now().strftime("%Y%m%d-%H%M%S")), mat)

    plt.plot(data[:, 0], data[:, 1])
    plt.xlabel("Time [ms]")
    plt.ylabel("Voltage [V]")
    plt.show()


def test_time_synchronisation():
    daq = DAQ6510()
    print("Connected to", daq.get_instrument_description())

    daq.send_command("*RST")
    s_target = 0.0
    tstart = datetime.now()
    delay = s_target - tstart.microsecond / 1000000
    if delay < 0:
        delay += 1
    time.sleep(delay)
    t = datetime.now()
    daq.send_command(":SYSTem:TIME {}, {}, {}, {}, {}, {}".format(t.year, t.month, t.day, t.hour, t.minute, t.second))
    print("time set:", t)
    print()
    time.sleep(0.1)

    n = 30000
    t = []
    res = []
    for i in range(n):
        t.append(datetime.now().microsecond)
        res.append(daq.send_query(":SYSTem:TIME?"))

    res = np.array([int(r) for r in res])
    idx = np.flatnonzero(np.diff(res)) + 1
    tus = np.array([t[i] for i in idx])
    t_offset = np.round(np.max(tus)) / 1000000 - 1
    print(len(tus))
    print(t_offset)


def save_and_plot(tR, R, ts, S, tV, V, tSW, SW, folder, fname):
    mat = {"tR": tR, "R": R, "ts": ts, "strain": S, "tV": tV, "V": V, "tSW": tSW, "SW": SW}
    sio.savemat(folder + fname + ".mat", mat)

    # plt.figure(1, (8, 5))
    fig, ax = plt.subplots(2, 1)
    fig.set_size_inches(12, 9)

    ax1 = ax[0]
    ax2 = ax[1]
    ax12 = ax1.twinx()
    ax22 = ax2.twinx()
    fig.suptitle(fname.replace("_", " "))

    ax1.plot(ts, S, color="C1")
    ax1.set_ylabel("Strain [%]", color="C1")
    ax1.tick_params(axis='y', colors='C1')

    ax12.plot(tR, R / 1000, color="C0")
    ax12.set_ylabel("Resistance [kΩ]", color="C0")
    ax12.tick_params(axis='y', colors='C0')

    # ax1.set_xticklabels([])
    ax1.grid(True)

    ax2.plot(tV, V, color="C2")
    ax2.set_ylabel("Voltage [V]", color="C2")
    ax2.tick_params(axis='y', colors='C2')

    ax22.plot(tSW, SW, color="C3")
    ax22.set_ylabel("OC on", color="C3")
    ax22.set_yticks([0, 1])
    ax22.tick_params(axis='y', colors='C3')

    ax2.set_zorder(ax22.get_zorder() + 1)  # put ax1 in front of ax12
    ax2.patch.set_visible(False)  # hide the 'canvas'

    ax2.set_xlabel("Time [s]")
    ax2.grid(True)

    plt.savefig(folder + fname + ".png")

    plt.show()


if __name__ == '__main__':
    # nplcs = [12, 1, 0.0005]
    # n_meas = [10, 135, 440]
    # for n_m, n_plc in zip(n_meas, nplcs):
    #     test_resistance_measurements(n_m, n_plc)
    logging.basicConfig(level=logging.DEBUG)
    test_resistance_measurement()
    # test_current_measurement_simple()
    # for nplc in [1]:
    #     test_current_measurements(nplc)
    # measure_resistance_sequence()
    # test_digitize_current()
    # test_digitize_voltage()
    # test_time_synchronisation()
    # x = np.arange(100)
    # y = np.random.random((100, 4))
    # save_and_plot(x, y[:, 0], x, y[:, 1], x, y[:, 2], x, y[:, 3], 'test')
