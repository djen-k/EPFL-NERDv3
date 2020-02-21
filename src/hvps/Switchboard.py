import logging
import time

from libs.hvps import HVPS, HvpsInfo


class SwitchBoard(HVPS):

    def __init__(self):
        super().__init__()
        self.logging = logging.getLogger("Switchboard")  # change logger to "Switchboard"

    def set_relay_auto_mode(self, reconnect_timeout=0):
        """
        Enable the automatic short circuit detection and isolation function of the switchboard
        :param reconnect_timeout: An optional time after which to reconnect and test all relays again, if the connected
        load has a short circuit (even if a short circuit was detected in a previous test)
        :return: The response from the switchboard
        """
        if reconnect_timeout > 0:
            self.logging.debug("Enabling auto mode with timeout {} s".format(reconnect_timeout))
            self._write_hvps(b'SRelAuto %d\r' % reconnect_timeout)
        else:
            self.logging.debug("Enabling auto mode")
            self._write_hvps(b'SRelAuto\r')
        res = self._read_hvps()
        return res

    def get_relay_state(self):
        """
        Queries the on/off state of the relais
        :return: A list (int) containing 0 or 1 for each relay, indicating on or off
        """
        self.logging.debug("Querying relay state")
        self._write_hvps(b'QRelState\r')
        res = self._parse_relay_state(self._read_hvps())
        return res

    def set_relays_on(self, relays=None):
        """
        Switch the requested relays on. If not specified, all relays are switched on.
        :param relays: A list of indices specifying which relays to switch on
        :return: The the updated relay state returned by the switchboard
        """
        if relays is None:
            self.logging.debug("Set all relays on")
            self._write_hvps(b'SRelOn\r')
            res = self._parse_relay_state(self._read_hvps())
        else:
            res = []
            for i in relays:
                res = self.set_relay_state(i, state=1)
        return res

    def set_relays_off(self, relays=None):
        """
        Switch the requested relays off. If not specified, all relays are switched off.
        :param relays: A list of indices specifying which relays to switch off
        :return: The the updated relay state returned by the switchboard
        """
        if relays is None:
            self.logging.debug("Set all relays off")
            self._write_hvps(b'SRelOff\r')
            res = self._parse_relay_state(self._read_hvps())
        else:
            res = []
            for i in relays:
                res = self.set_relay_state(i, state=0)
        return res

    def set_relay_state(self, relay, state):
        if state is 0:
            self.logging.debug("Setting relay {} off".format(relay))
            self._write_hvps(b'SRelOff %d\r' % relay)
        else:  # state is 1
            self.logging.debug("Setting relay {} on".format(relay))
            self._write_hvps(b'SRelOn %d\r' % relay)
        res = self._parse_relay_state(self._read_hvps())
        return res

    def _parse_relay_state(self, str_state):
        try:
            state = str_state[14:-1].split(b',')  # separate response for each relay
            state = [self._cast_int(r) for r in state]  # convert all to int
        except Exception as ex:
            self.logging.debug("Failed to parse relay state: {}".format(ex))
            return str_state  # return the original message so as not to withhold information
        return state


if __name__ == '__main__':
    sb = SwitchBoard()
    hvpsInfo = HvpsInfo()
    comPort = "COM8"
    hvpsInfo.port = comPort
    sb.open_hvps(hvpsInfo, with_continuous_reading=False)

    print(sb.get_name())
    print("now: ", sb.get_current_voltage())
    print("sp: ", sb.get_voltage_setpoint())
    print("set: ", sb.set_voltage(100))
    time.sleep(1)
    print("now: ", sb.get_current_voltage())
    # print(sb.set_relays_on())
    # print(sb.get_relay_state())
    print(sb.set_relay_auto_mode())
    # time.sleep(1)
    # print(sb.get_relay_state())
    # time.sleep(1)
    # print(sb.get_relay_state())
    # print(sb.set_relay_state(3, 1))
    # time.sleep(1)
    # print(sb.set_relays_off())
    time.sleep(1)
    print("set: ", sb.set_voltage(1000))
    time.sleep(1)
    print("now: ", sb.get_current_voltage())
    print("sp: ", sb.get_voltage_setpoint())
    time.sleep(5)
    print(sb.get_relay_state())
    print(sb.set_relays_off())
    time.sleep(1)
    print("set: ", sb.set_voltage(0))
    time.sleep(2)
    print("now: ", sb.get_current_voltage())
    print("sp: ", sb.get_voltage_setpoint())
    sb.close()

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
