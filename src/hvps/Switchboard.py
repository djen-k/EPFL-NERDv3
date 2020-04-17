import logging
import time

from libs.hvps import HVPS, HvpsInfo


class SwitchBoard(HVPS):

    def __init__(self):
        super().__init__()
        self.logging = logging.getLogger("Switchboard")  # change logger to "Switchboard"

    def __del__(self):
        try:
            self.set_voltage(0, block_if_testing=True)
            self.set_relays_off()
        except Exception as ex:
            self.logging.warning("Unable to switch off the relays: {}".format(ex))

        super().__del__()

    def open(self, com_port=None, with_continuous_reading=False):
        """
        Open a connection to a physical switchboard at the given COM port. The port can be specified as a string
        (COM port name), a HvpsInvo object or an index. If specified as an index, the respective port in this
        SwitchBoard's hvps_available_ports list is used. If no port is specified, the first available port is used.
        :param com_port: The COM port to connect to. Can be a string, HvpsInfo, int or None
        :param with_continuous_reading: Specify if the SwitchBoard should launch a thread to record continuous voltage
        readings
        """
        if self.is_open:
            self.close()

        # connect to HVPS
        if com_port is None:  # if not defined, autodetect HVPS
            self.auto_open(with_continuous_reading)
        elif isinstance(com_port, HvpsInfo):  # specified as HvpsInfo  -> open
            self.open_hvps(com_port, with_continuous_reading)
        elif isinstance(com_port, str):  # specified as port name (string)  -> open by name
            self.open_hvps_from_port(com_port, with_continuous_reading)
        elif isinstance(com_port, int):  # specified as index  -> pick from list of available ports
            self.open_hvps(self.hvps_available_ports[com_port], with_continuous_reading)
        else:
            raise ValueError("Invalid argument! COM port must be given as string, index, or HvpsInfo object!")

        if self.is_open:  # get HVPS name and print success message
            name = self.get_name()
            logging.info("connected successfully to {} on {}".format(name.decode(), self.ser.port))
        else:  # you had one job: giving the correct COM port, and you failed!
            logging.critical("Unable to connect to HVPS, ensure it's connected")
            # raise Exception

        # ensure we have a compatible HVPS, correctly connected and configured
        hvps_type = (self.get_hvps_type() == "slave")  # until when is the term "slave" politically correct ?
        jack = (self.get_jack_status() == 1)  # external power connector, not your friend Jack
        control_mode = (self.get_voltage_control_mode() == 0)
        if not (hvps_type and jack and control_mode):
            logging.critical("either type, power source or control mode is not correctly set, ...")
            # raise Exception

        # ensure firmware version is compatible
        version = self.get_firmware_version()
        if version != 1:
            logging.critical("HVPS firmware version is: {}. Only tested with version 7".format(version))

        # select DC and output OFF
        self.set_switching_source(0)
        self.set_switching_mode(1)
        self.set_output_off()
        self.set_voltage(0)

    def close(self):
        """Closes connection with the HVPS"""
        self.stop_voltage_reading()
        time.sleep(0.1)
        self.serial_com_lock.acquire()
        if self.ser.is_open:
            self.set_voltage(0, block_if_testing=True)  # set the voltage to 0 as a safety measure
            self.set_relays_off()
            self.ser.close()
        self.is_open = False
        self.current_device = None
        self.serial_com_lock.release()

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

    def set_voltage(self, voltage, block_if_testing=False):  # sets the output voltage
        """
        Sets the output voltage.
        Checks if voltage can be set or if switchboard is currently testing for a short circuit
        :param voltage: The desired output voltage
        :param block_if_testing: Flag to indicate if the function should block until the voltage can be set. The voltage
        set point cannot be changed while the switchboard is testing for short circuits. Set this to True, if you want
        the function to block until the switchboard has finished testing (if it was). If false, the function
        may return without having set the voltage. Check response from switchboard!
        :return: True if the voltage was set successfuly, false if the switchboard was unable to set the voltage because
        it was busy testing for a short circuit or some other error occurred. If 'block_if_testing' is True,
        a False return value indicates an unexpected error.
        """

        if block_if_testing:
            while self.is_testing():
                self.logging.debug("Switchboard is busy testing for shorts. Waiting to set voltage...")
                time.sleep(0.1)

        ret = super().set_voltage(voltage)
        return ret == voltage

    def is_testing(self):
        """
        Checks if the switchboard is currently testing for a short circuit
        :return: True, if the switchboard is currently busy testing
        """
        self.logging.debug("Query if switchbaord is testing")
        self._write_hvps(b'QTestingShort\r')
        return self._read_hvps() is b'1'

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

    def dirty_reconnect(self, timeout=10):
        self.ser.close()
        try:
            self.ser.open()
        except:
            self.logging.critical("Reconnextion attempt failed... retry in 500ms")
        start = time.monotonic()
        elapsed = 0
        while not self.ser.is_open and elapsed < timeout:
            time.sleep(0.5)
            self.ser.close()
            try:
                self.ser.open()
            except:
                self.logging.critical("Reconnextion attempt failed... retry in 500ms")

            elapsed = time.monotonic() - start

        if self.ser.is_open:
            self.logging.critical("Reconnected! (?)")
        else:
            self.logging.critical("Failed to reconnect!")

    def get_pid_gains(self):
        """
        Query the PID gains of the internal voltage regulator.
        :return: A list of gain values [P, I, D]
        """
        self._write_hvps(b'QKp\r')
        res = self._read_hvps()  # read response
        res = res.decode("utf-8")
        res = res.split(",")
        res = [float(s) for s in res]
        return res

    def set_pid_gains(self, gains):
        """
        Set the PID gains of the internal voltage regulator. The new gains will be written to the EEPROM.
        :param gains: A list of gain values [P, I, D]
        :return: The updated PID gains [P, I, D]
        """
        self.set_pid_gain('p', gains[0])
        self.set_pid_gain('i', gains[1])
        res = self.set_pid_gain('d', gains[2])
        return res

    def set_pid_gain(self, param, gain):
        """
        Set the specified gain of the internal voltage regulator. The new gains will be written to the EEPROM.
        :param param: The parameter to set (P, I, or D) as a single-character string (case is ignored)
        :param gain: The value to which to set the specified gain.
        :return: The updated PID gains [P, I, D]
        """
        param = str(param)
        param = param.lower()
        if param not in "pid" or len(param) > 1:
            raise ValueError("Parameter must be either 'p', 'i', or 'd'!")
        param = bytes(param, 'utf-8')
        self._write_hvps(b'SK%b %.4f\r' % (param, gain))
        self._read_hvps()  # read response to clear buffer
        res = self.get_pid_gains()  # query again because the return value of the write command has too few digits
        return res


if __name__ == '__main__':
    sb = SwitchBoard()
    sb.open(with_continuous_reading=True)
    Vset = 1000
    print(sb.get_name())
    # print("sp: ", sb.get_voltage_setpoint())
    # print("now: ", sb.get_current_voltage())
    # print("set: ", sb.set_voltage(500))
    # print("sp: ", sb.get_voltage_setpoint())
    # time.sleep(2)
    # print("now: ", sb.get_current_voltage())
    # print(sb.set_relays_on())
    # print(sb.get_relay_state())
    # gains = sb.set_pid_gains((0.32, 1.7, 0.005))
    gains = sb.get_pid_gains()
    print("PID gains:", gains)

    sb.set_output_off()  # = set_switching_mode(0)
    print(sb.set_relay_auto_mode())
    # time.sleep(1)
    # print(sb.get_relay_state())
    # time.sleep(1)
    # print(sb.get_relay_state())
    # print(sb.set_relay_state(3, 1))
    # time.sleep(1)
    # print(sb.set_relays_off())
    # time.sleep(1)
    sb.set_output_on()
    # print("set1: ", sb.set_voltage(700))
    # time.sleep(0.1)
    # print("set2: ", sb.set_voltage(800))
    # time.sleep(0.1)
    # print("set3: ", sb.set_voltage(900))
    # time.sleep(0.1)
    print("set4: ", sb.set_voltage(Vset))
    # print("sp: ", sb.get_voltage_setpoint())
    # sb.set_frequency(1)
    # print("f: ", sb.get_frequency())
    # sb.start_ac_mode()  # = set_switching_mode(2)
    # time.sleep(2)
    # print("now: ", sb.get_current_voltage())
    # print("sp: ", sb.get_voltage_setpoint())
    time.sleep(2)
    # print("now: ", sb.get_current_voltage())
    # print(sb.get_relay_state())
    # time.sleep(2)
    # print(sb.set_relays_off())
    # time.sleep(1)
    print("set: ", sb.set_voltage(0))
    print("sp: ", sb.get_voltage_setpoint())
    time.sleep(1.5)
    # print("now: ", sb.get_current_voltage())
    # print("sp: ", sb.get_voltage_setpoint())
    sb.set_output_off()
    sb.close()

    import matplotlib.pyplot as plt
    import numpy as np

    it = sb.times > 0
    t = sb.times[it]
    V = sb.voltage[it]
    plt.plot(t, np.array(V) * 0 + Vset)
    plt.plot(t, V)
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    title = "Hermione PID {} {}V 3-DEAs".format(str(gains), Vset)
    plt.title(title)
    plt.savefig("test_data/pid/" + title + ".png")
    plt.show()

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
