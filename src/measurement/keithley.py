import logging
import time
from datetime import datetime

import numpy as np
import scipy.io as sio
import visa


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

    def _connect(self, instrument_id):
        try:
            self.logging.info("Connecting to instrument: {}".format(instrument_id))
            self.instrument = self.resource_manager.open_resource(instrument_id)  # connect to instrument
            self.instrument_id = instrument_id
            self.instrument.clear()  # clear input buffer to make sure we don't receive messages from a previous session
            return True  # return True if successful
        except Exception as ex:
            self.logging.warning("Unable to connect: {}".format(instrument_id, ex))
            return False

    def _disconnect(self):
        if self.instrument:
            self.logging.debug("Disconnecting...")
            self.instrument.close()

    def send_command(self, command):
        print("sending command: ", command)
        self.instrument.write(command)

    def send_query(self, command):
        print("sending query: ", command)
        res = self.instrument.query(command)
        print("response: ", res)
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
    daq = DAQ6510()
    # daq.set_timeout(0.01)
    daq.get_instrument_name()

    daq.send_command("*RST")
    daq.send_command("SENS:FUNC 'CURR:DC', (@122)")
    daq.send_command("SENS:CURR:DC:APERture 0.1, (@122)")
    daq.send_query("SENS:CURR:DC:APERture? (@122)")
    # daq.send_command("SENS:COUNt 10")

    daq.send_command("ROUT:CLOSe (@122)")  # close relays on current measurement channel
    daq.send_command("ROUT:CLOSe (@112)")  # close relays to switch off 9.5V power used for R measurements
    daq.send_command("ROUT:CLOSe (@113)")
    time.sleep(0.5)

    # perform a series of measurements
    n_measurements = 100
    cur = [float(daq.send_query("READ?")) for i in range(n_measurements)]

    daq.send_command("ROUT:OPEN (@112)")  # open relays again to switch 9.5V back on
    daq.send_command("ROUT:OPEN (@113)")

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, linewidth=500)  # set print format for arrays
    cur = np.array(cur).reshape((-1, 1))
    print("Current [nA]:")
    print(cur * 1000000000)


if __name__ == '__main__':
    # nplcs = [12, 1, 0.0005]
    # n_meas = [10, 135, 440]
    # for n_m, n_plc in zip(n_meas, nplcs):
    #     test_resistance_measurements(n_m, n_plc)

    test_current_measurements()
