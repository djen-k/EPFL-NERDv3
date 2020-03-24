import logging
import time

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

    def get_instrument_name(self):
        """Return the name of the opened instrument"""
        instrument_name = self.send_query("*IDN?")
        return instrument_name


if __name__ == '__main__':

    daq = DAQ6510()
    print("connected to ", daq.get_instrument_name())

    daq.send_command("*RST")
    daq.send_command("FUNC 'VOLT:DC', (@103:106)")
    daq.send_command("VOLT:DC:RANG:AUTO ON, (@103:106)")
    daq.send_command("VOLT:DC:NPLC 5, (@103:106)")
    daq.send_command("VOLT:DC:AZER ON, (@103:106)")
    daq.send_command("ROUT:SCAN (@103:106)")
    daq.send_command("ROUT:SCAN:COUN:SCAN 10")
    daq.send_command("INIT")
    i = 1
    while i < 40:
        time.sleep(2)

        lastIndex = int(daq.send_query("TRACe:ACTual?"))
        tmpBuff = daq.send_query("TRACe:DATA? {}, {}, \"defbuffer1\", READ".format(i, lastIndex))
        print(tmpBuff)
        i = lastIndex + 1
