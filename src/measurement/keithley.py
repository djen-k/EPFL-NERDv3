import logging
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import visa

from src.hvps.Switchboard import SwitchBoard


def list_instruments(instrument_search_string=None, resource_manager=None):
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


class DAQ6510:

    def __init__(self, instrument_id=None, instrument_search_string="6510"):
        self.logging = logging.getLogger("DAQ6510")

        self.resource_manager = visa.ResourceManager()
        self.instrument = None
        self.data_format = False

        if instrument_id is None or self._connect(instrument_id):
            # if no id given or unable to connect, search for available instrument
            available_instruments = list_instruments(instrument_search_string, self.resource_manager)
            for res_id, res_info in available_instruments.items():
                success = self._connect(res_id)
                if success:
                    self.logging.info("Successfully connected to {} ({})".format(res_info.alias, res_id))
                    break

        if self.instrument is None:  # no instrument was found
            self.logging.critical("No instrument found! Make sure the DAQ6510 is connected via USB!")

    def __del__(self):
        self._disconnect()
        del self.logging

    def _connect(self, instrument_id, reset=True):
        try:
            self.logging.info("Connecting to instrument: {}".format(instrument_id))
            self.instrument = self.resource_manager.open_resource(instrument_id)  # connect to instrument
            self.instrument_id = instrument_id
            self.instrument.clear()  # clear input buffer to make sure we don't receive messages from a previous session
            if reset:
                self.send_command("*RST")  # reset instrument to put it in a known state
            return True  # return True if successful
        except Exception as ex:
            self.logging.warning("Unable to connect: {}".format(instrument_id, ex))
            return False

    def _disconnect(self):
        if self.instrument:
            self.logging.debug("Disconnecting...")
            self.instrument.close()

    def send_command(self, command):
        self.instrument.write(command)

    def send_query(self, command):
        res = self.instrument.query(command)
        return res

    def set_timeout(self, timeout_s):
        """
        Set read timeout in seconds. Any communication with the device that is aborted after the specified timeout if no
        response was received.
        :param timeout_s: The timout in seconds after which to abort read operations
        """
        self.instrument.timout = timeout_s * 1000

    def get_instrument_name(self):
        """Return the name of the opened instrument"""
        instrument_name = self.send_query("*IDN?")
        return instrument_name

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
            self.send_command("FORM REAL")
        elif data_format == 'f':
            self.send_command("FORM SREAL")
        elif data_format == 's':
            self.send_command("FORM ASCII")
        else:
            raise Exception("Invalid data format! Format must be 'f', 'd' or 's'!")

        # self.instrument.values_format.use_binary(data_format, False, np.array)
        self.data_format = data_format

    def query_data(self, command, data_points=0):
        """
        Sends the given command and receives numeric data in the active data format: ASCII by default,
        float or double if a binary data format was set (using 'use_binary()').
        Data is automatically interpreted and converted to a numpy array.
        :param command: The command to send to the instrument.
        :param data_points: The number of data points to read. Set 0 (default) if unknown.
        :return: The data received from the instrument as a numpy array.
        """
        if self.data_format:
            return self.instrument.query_binary_values(command, datatype=self.data_format,
                                                       container=np.array, data_points=data_points)
        else:
            return self.instrument.query_ascii_values(command, container=np.array)


def test_resistance_measurements(n_measurements=10, nplc=1):
    daq = DAQ6510()
    print("connected to ", daq.get_instrument_name())

    str_channels = "(@111, 103:110, 117:120)"
    n_channels = 13

    daq.send_command("*RST")
    daq.send_command("FUNC 'VOLT:DC', {}".format(str_channels))
    daq.send_command("VOLT:DC:RANG:AUTO ON, {}".format(str_channels))
    daq.send_command("VOLT:DC:NPLC {}, {}".format(nplc, str_channels))
    daq.send_query("VOLT:DC:NPLC? {}".format(str_channels))
    daq.send_command("VOLT:DC:AZER ON, {}".format(str_channels))
    daq.send_command("ROUT:SCAN {}".format(str_channels))
    daq.send_command("ROUT:SCAN:COUN:SCAN {}".format(n_measurements))
    # TODO: disable auto 0 and auto range to see if that helps fixing inconsistencies in resistance measurements
    daq.send_command("INIT")
    i = 1
    start = time.monotonic()
    while i < n_channels * n_measurements:
        time.sleep(0.3)
        i = int(daq.send_query("TRACe:ACTual?"))
        print(i)
    stop = time.monotonic()

    data = daq.send_query("TRACe:DATA? 1, {}, \"defbuffer1\", READ".format(i))
    data = data.split(",")
    data = [float(d) for d in data]
    data = np.array(data)
    data = np.reshape(data, (n_measurements, n_channels))

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})  # set print format for arrays

    Vsource = data[:, 0]
    print("Vsource:", Vsource, "V")

    data = data[:, 1:]  # don't need the reference voltage anymore
    n_channels -= 1
    n_deas = int(n_channels / 2)

    print("")
    print("Voltages: [Vshunt0, VDEA0, Vshunt1, VDEA1, ...]")
    print(data)

    data = np.reshape(data, (n_measurements, n_deas, 2))
    Vshunt = data[:, :, 0]
    VDEA = data[:, :, 1]
    Rshunt = 100000  # 100kOhm

    Ishunt = Vshunt / Rshunt
    RDEA = VDEA / Ishunt

    RDEA_k = RDEA / 1000  # in kΩ
    Ishunt_m = Ishunt * 1000  # in mA

    print("")
    print("Vshunt: [DEA0, DEA1, ...]")
    print(Vshunt)
    print("")
    print("VDEA: [DEA0, DEA1, ...]")
    print(VDEA)
    print("")
    print("Ishunt: [DEA0, DEA1, ...]")
    print(Ishunt_m)

    print("")
    print("")
    print("Resistances: [DEA0, DEA1, ...] in kΩ")
    print(RDEA_k)

    print("")
    print("elapsed time:", stop - start)

    mat = {'R': RDEA_k, 't': stop - start, 'NPLC': nplc}
    sio.savemat('test_data/DAQ_test_{}.mat'.format(datetime.now().strftime("%Y%m%d-%H%M%S")), mat)


def test_current_measurements():
    hvps = SwitchBoard()
    hvps.open()
    print("Connected to HVPS", hvps.get_name())
    print("Relays on:", hvps.set_relays_on([0]))
    hvps.set_voltage(0)
    hvps.set_switching_mode(0)

    daq = DAQ6510()
    # daq.set_timeout(0.01)
    daq.get_instrument_name()

    daq.send_command("*RST")
    daq.send_command("SENS:FUNC 'CURR:DC', (@122)")
    daq.send_command("SENS:CURR:DC:APERture 0.02, (@122)")
    daq.send_command("SENS:CURR:DC:RANGe 100e-6, (@122)")
    daq.send_query("SENS:CURR:DC:APERture? (@122)")
    # daq.send_command("SENS:COUNt 10")

    daq.send_command("ROUT:CLOSe (@122)")  # close relays on current measurement channel
    daq.send_command("ROUT:CLOSe (@112)")  # close relays to switch off 9.5V power used for R measurements
    daq.send_command("ROUT:CLOSe (@113)")
    time.sleep(0.5)

    # perform a series of measurements
    Vtest = 1000
    n = [15, 80, 100]
    settling_delay = 10
    # sw_mode = "OC"
    sw_mode = "DCDC"

    cur = []
    V = []
    Vset = []
    t = []
    ntotal = np.sum(n)

    if sw_mode == "OC":
        hvps.set_voltage(Vtest)  # ramp up voltage but OC remains off
    elif sw_mode == "DCDC":
        hvps.set_switching_mode(1)  # open OC but leave voltage 0
    for i in range(n[0]):
        cur.append(float(daq.send_query("READ?")))
        V.append(hvps.get_current_voltage())
        Vset.append(0)
        t.append(time.monotonic())
        print("{:.1f} %".format(i / ntotal * 100))

    if sw_mode == "OC":
        hvps.set_switching_mode(1)  # open OC to apply Vtest to DEAs
    elif sw_mode == "DCDC":
        hvps.set_voltage(Vtest)  # ramp up voltage to Vtest
    for i in range(n[1]):
        cur.append(float(daq.send_query("READ?")))
        V.append(hvps.get_current_voltage())
        Vset.append(Vtest)
        t.append(time.monotonic())
        print("{:.1f} %".format((n[0] + i) / ntotal * 100))

    if sw_mode == "OC":
        hvps.set_switching_mode(0)  # close OC to discharge DEAs
    elif sw_mode == "DCDC":
        hvps.set_voltage(0)  # reduce voltage to 0
    for i in range(n[2]):
        cur.append(float(daq.send_query("READ?")))
        V.append(hvps.get_current_voltage())
        Vset.append(0)
        t.append(time.monotonic())
        print("{:.1f} %".format((n[0] + n[1] + i) / ntotal * 100))

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

    print("Leakage current [nA]:")
    istart = 0 + settling_delay
    iend = n[0]
    print("V off ({:.2f} to {:.2f}): {:.5f}".format(t[istart], t[iend - 1], np.mean(cur[istart:iend])))
    istart = n[0] + settling_delay
    iend = n[0] + n[1]
    print("V on ({:.2f} to {:.2f}): {:.5f}".format(t[istart], t[iend - 1], np.mean(cur[istart:iend])))
    istart = n[0] + n[1] + settling_delay
    iend = n[0] + n[1] + n[2]
    print("V off ({:.2f} to {:.2f}): {:.5f}".format(t[istart], t[iend - 1], np.mean(cur[istart:iend])))

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Current [nA]', color=color)
    ax1.plot(t, cur, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Voltage [V]', color=color)  # we already handled the x-label with ax1
    # ax2.plot(t, Vset, color="k")
    ax2.plot(t, V, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def measure_resistance_sequence():
    daq = DAQ6510()
    # daq.set_timeout(0.01)
    daq.get_instrument_name()

    str_channels = "(@107:108)"
    n_channels = 2
    n_measurements = 100

    daq.send_command("*RST")
    daq.send_command("FUNC 'VOLT:DC', {}".format(str_channels))
    daq.send_command("VOLT:DC:RANGe 10, {}".format(str_channels))  # this should set input impedance to >10 GΩ
    daq.send_command("VOLT:DC:APERture 0.1, {}".format(str_channels))
    daq.send_query("VOLT:DC:NPLC? {}".format(str_channels))
    daq.send_command("VOLT:DC:AZERo OFF, {}".format(str_channels))
    daq.send_command("AZERo:ONCE")  # zero now before starting the scan
    daq.send_command("ROUT:SCAN {}".format(str_channels))
    daq.send_command("ROUT:SCAN:COUN:SCAN {}".format(n_measurements))

    daq.send_command("INIT")
    i = 1
    start = time.monotonic()
    while i < n_channels * n_measurements:
        time.sleep(5)
        i = int(daq.send_query("TRACe:ACTual?"))
        print(i)
    stop = time.monotonic()

    data = daq.send_query("TRACe:DATA? 1, {}, \"defbuffer1\", READ".format(i))
    data = data.split(",")
    data = [float(d) for d in data]
    data = np.array(data)
    data = np.reshape(data, (n_measurements, n_channels))

    Vshunt = data[:, 0]
    VDEA = data[:, 1]
    Rshunt = 100000  # 100kOhm

    Ishunt = Vshunt / Rshunt
    RDEA = VDEA / Ishunt
    RDEA = np.reshape(RDEA, (-1, 1))

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, linewidth=500)  # set print format for arrays
    print("Voltages [V]:")
    print(data)

    print("Resistance [kΩ]:")
    print(RDEA / 1000)

    print("")
    print("elapsed:", stop - start)

    plt.plot(RDEA / 1000)
    plt.show()


def test_digitize_current():
    daq = DAQ6510()
    # daq.set_timeout(10)
    daq.get_instrument_name()
    daq.set_data_format('f')

    hvps = SwitchBoard()
    hvps.open()
    print("Connected to HVPS", hvps.get_name())
    print("Relays on:", hvps.set_relays_on())
    hvps.set_voltage(0)
    hvps.set_switching_mode(0)

    t = 3  # measurement duration in sec
    sample_rate = 100000  # 1000000
    Vtest = 1000
    # sw_mode = "OC"
    sw_mode = "DCDC"

    count = sample_rate * t  # 1 MHz sample rate

    buffer = "'curDigBuffer'"

    # daq.send_command("TRACe:POINts 10, 'defbuffer1'")  # reduce size of default buffers to 10 (minimum)
    # daq.send_command("TRACe:POINts 10, 'defbuffer2'")  # to make space for current readings (max 7M)
    daq.send_command("TRACe:MAKE {}, {}".format(buffer, count))  # create buffer of correct size to store the data

    daq.send_command("DIG:FUNC 'CURR'")  # select digitize current
    daq.send_command("DIG:CURR:RANG 100e-6")  # set measurement range to 100 µA (smallest range)
    daq.send_command("DIG:CURR:SRAT {}".format(sample_rate))  # set sample rate to max (1MHz)
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
    start = time.monotonic()

    if sw_mode == "OC":
        hvps.set_voltage(Vtest)  # ramp up voltage but OC remains off
    elif sw_mode == "DCDC":
        hvps.set_switching_mode(1)  # open OC but leave voltage 0
    time.sleep(0.1 * t)

    if sw_mode == "OC":
        hvps.set_switching_mode(1)  # open OC to apply Vtest to DEAs
    elif sw_mode == "DCDC":
        hvps.set_voltage(Vtest)  # ramp up voltage to Vtest
    time.sleep(0.5 * t)

    if sw_mode == "OC":
        hvps.set_switching_mode(0)  # close OC to discharge DEAs
    elif sw_mode == "DCDC":
        hvps.set_voltage(0)  # reduce voltage to 0

    i = 0
    while i < count:
        time.sleep(0.5)
        i = int(daq.send_query("TRACe:ACTual? {}".format(buffer)))
        print(i)
    stop = time.monotonic()
    print("Time until data is available:", stop - start)
    # retrieve data
    start = time.monotonic()
    data = daq.query_data("TRACe:DATA? 1, {}, {}, REL, READ".format(count, buffer), 2 * count)
    data = data.reshape((-1, 2)) * [1000, 1000000]  # shape into two columns and convert to ms and µA
    # reltime = daq.query_data("TRACe:DATA? 1, {}, {}, REL".format(count, buffer))
    stop = time.monotonic()
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
    daq.get_instrument_name()
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
    start = time.monotonic()

    time.sleep(0.1)
    hvps.set_voltage(500)
    time.sleep(2.6)
    hvps.set_voltage(0)

    i = 0
    while i < count:
        time.sleep(0.3)
        i = int(daq.send_query("TRACe:ACTual? {}".format(buffer)))
        print(i)
    stop = time.monotonic()
    print("Time until data is available:", stop - start)
    # retrieve data
    start = time.monotonic()
    data = daq.query_data("TRACe:DATA? 1, {}, {}, REL, READ".format(count, buffer), 2 * count)
    data = data.reshape((-1, 2)) * [1000, 1]  # shape into two columns and convert to ms and µA
    # reltime = daq.query_data("TRACe:DATA? 1, {}, {}, REL".format(count, buffer))
    stop = time.monotonic()
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


if __name__ == '__main__':
    # nplcs = [12, 1, 0.0005]
    # n_meas = [10, 135, 440]
    # for n_m, n_plc in zip(n_meas, nplcs):
    #     test_resistance_measurements(n_m, n_plc)

    test_current_measurements()
    # measure_resistance_sequence()
    # test_digitize_current()
    # test_digitize_voltage()
