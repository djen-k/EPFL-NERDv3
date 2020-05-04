import logging
import time
from collections import deque
from threading import Thread, Event

import numpy as np

from libs.hvps import HVPS, HvpsInfo


class SwitchBoard(HVPS):

    # TODO: always record current state and restore after reconnect!

    def __init__(self):
        super().__init__()
        self.logging = logging.getLogger("Switchboard")  # change logger to "Switchboard"
        self.continuous_voltage_reading_flag = Event()
        self.connection_timeout = 0
        self.t_0 = None

    def __del__(self):
        try:
            self.set_voltage(0, block_until_successful=True)
            self.set_relays_off()
        except Exception as ex:
            self.logging.warning("Unable to switch off the relays: {}".format(ex))

        super().__del__()

    def open(self, com_port=None, with_continuous_reading=False, connection_timeout=0):
        """
        Open a connection to a physical switchboard at the given COM port. The port can be specified as a string
        (COM port name), a HvpsInvo object or an index. If specified as an index, the respective port in this
        SwitchBoard's hvps_available_ports list is used. If no port is specified, the first available port is used.
        :param com_port: The COM port to connect to. Can be a string, HvpsInfo, int or None
        :param with_continuous_reading: Specify if the SwitchBoard should launch a thread to record continuous voltage
        readings
        :param connection_timeout: Optional timeout parameter to specify for how long the HVPS should attempt
        reconnecting if the connection is lost. Set -1 to disable (i.e. keep trying to reconnect indefinitely).
        If the timeout expires before the connection can be reastablished, the COM port is closed but no error
        is generated. Check "is_open" regularly to ensure that the device is still connected.
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
        self.set_switching_source(0)  # on-board switching
        self.set_switching_mode(0)  # OFF
        self.set_voltage(0)

        self.connection_timeout = connection_timeout

    def close(self):
        """Closes connection with the HVPS"""
        self.stop_voltage_reading()
        time.sleep(0.1)
        self.serial_com_lock.acquire()
        if self.ser.is_open:
            self.set_voltage(0, block_until_successful=True)  # set the voltage to 0 as a safety measure
            self.set_relays_off()
            self.ser.close()
        self.is_open = False
        self.current_device = None
        self.serial_com_lock.release()

    def set_relay_auto_mode(self, reset_time=0):
        """
        Enable the automatic short circuit detection and isolation function of the switchboard
        :param reset_time: An optional time after which to reconnect and test all relays again (including those
        where a short circuit was detected in a previous test). Set 0 to disable.
        :return: The response from the switchboard
        """
        if reset_time > 0:
            self.logging.debug("Enabling auto mode with timeout {} s".format(reset_time))
            self._write_hvps(b'SRelAuto %d\r' % reset_time)
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

    def set_voltage(self, voltage, block_until_successful=False, block_until_reached=False):  # sets the output voltage
        """
        Sets the output voltage.
        Checks if voltage can be set or if switchboard is currently testing for a short circuit
        :param voltage: The desired output voltage
        :param block_until_successful: Flag to indicate if the function should block until the voltage can be set. The voltage
        set point cannot be changed while the switchboard is testing for short circuits. Set this to True, if you want
        the function to block until the switchboard has finished testing (if it was). If false, the function
        may return without having set the voltage. Check response from switchboard!
        :param block_until_reached: Flag to indicate if the function should block until the measured voltage matches the
        voltage set point (with a 10V margin). If the set point is not reached within 3s, a TimeoutError is raised.
        :return: True if the voltage was set successfuly, false if the switchboard was unable to set the voltage because
        it was busy testing for a short circuit or some other error occurred. If 'block_if_testing' is True,
        a False return value indicates an unexpected error.
        """
        # if we want to wait until reached, also have to wait until V has been set
        if block_until_successful or block_until_reached:
            if self.is_testing():
                self.logging.debug("Switchboard is busy testing for shorts. Waiting to set voltage...")
            while self.is_testing():
                time.sleep(0.1)

        ret = super().set_voltage(voltage)
        if block_until_reached:
            timeout = 3  # if voltage has not reached its set point in 3 s, something must be wrong!
            start = time.perf_counter()
            elapsed = 0
            while abs(voltage - self.get_current_voltage()) > 10:
                if elapsed > timeout:
                    raise TimeoutError("Voltage has not reached the set point after 3 seconds! Please check the HVPS!")
                if elapsed == 0:  # only write message once
                    self.logging.debug("Waiting for measured output voltage to reach the set point...")
                time.sleep(0.01)
                elapsed = time.perf_counter() - start

        return ret == voltage

    def set_voltage_no_overshoot(self, voltage):
        """
        Sets the output voltage to the specified value, but does so more slowly in several steps to ensure that there
        is no voltage overshoot. This method blocks until the desired voltage has been reached.
        :param voltage: The desired output voltage, in Volts.
        :return: True or False to indicate if the voltage was set correctly.
        """
        v = self.get_current_voltage(from_buffer_if_available=True)
        dv = voltage - v
        if dv > 100:  # if increasing voltage (and by more than a few volts), do it slowly
            self.set_voltage(round(voltage * 0.7), block_until_reached=True)
            self.set_voltage(round(voltage * 0.9), block_until_reached=True)
        return self.set_voltage(voltage, block_until_reached=True)

    def get_current_voltage(self, from_buffer_if_available=False):
        """
        Read the current voltage from the switchboard
        :param from_buffer_if_available: If true and if continuous voltage reading is on, the most recent value from
        the voltage buffer is returned instead of querying the switchboard.
        :return: The current voltage as measrued by the switchboard voltage feedback.
        """
        if from_buffer_if_available is True and self.reading_thread.is_alive():
            with self.buffer_lock:
                if len(self.voltage) > 0:
                    return self.voltage[-1]

        return super().get_current_voltage()

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

    def dirty_reconnect(self):
        self.ser.close()
        start = time.perf_counter()
        elapsed = 0
        while not self.ser.is_open:
            try:
                self.ser.open()
            except Exception as ex:
                self.logging.debug("Connection error: {}".format(ex))
                elapsed = time.perf_counter() - start
                if self.connection_timeout < 0 or elapsed < self.connection_timeout:  # no timeout or not yet expired
                    msg = "Reconnection attempt failed... will keep trying for {} s".format(self.connection_timeout)
                    self.logging.critical(msg)
                    self.ser.close()  # close again to make sure it's properly closed before we try again
                    time.sleep(0.5)
                else:
                    self.logging.critical("Unable to reconnect! Timeout expired.")
                    self.close()
                    return

        self.logging.critical("Reconnected!")

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

    def start_voltage_reading(self, buffer_length=None, reference_time=None):
        """
        Start continuous voltage reading in a separate thread. The data is stored in an internal buffer and can be
        retrieved via 'get_voltage_buffer'.
        :param buffer_length: The length of the buffer in which the voltage data is stored.
        :param reference_time: The time of each voltage measurement will be expressed relative to this time. Useful for
        synchronizing data acquisition from different devices. The reference time must be generated by calling
        'time.perf_counter()'.
        """
        # check if it's already running
        if self.reading_thread is not None and self.reading_thread.is_alive():
            self.logging.info("Voltage reading thread is already running.")
            return

        if buffer_length is not None:
            self.buffer_length = buffer_length
        self.t_0 = reference_time
        self.continuous_voltage_reading_flag.clear()  # reset the flag
        # self.voltage = deque(maxlen=self.buffer_length)  # voltage buffer
        # self.times = deque(maxlen=self.buffer_length)  # Times buffer
        self.voltage = deque(maxlen=self.buffer_length)  # voltage buffer
        self.times = deque(maxlen=self.buffer_length)  # Times buffer
        self.reading_thread = Thread()  # Thread for continuous reading
        self.reading_thread.run = self._continuous_voltage_reading  # Method associated to thread
        self.reading_thread.start()  # Starting Thread

    def stop_voltage_reading(self):
        """Routine for stoping continuous position reading"""
        self.continuous_voltage_reading_flag.set()  # Set Flag to False

    def _continuous_voltage_reading(self):
        """Method for continuous reading"""
        if self.t_0 is None:
            t_0 = time.perf_counter()  # Initializing reference time
        else:
            t_0 = self.t_0
        while not self.continuous_voltage_reading_flag.is_set() and self.is_open:
            # While Flag is not set and HVPS is connected
            current_voltage = self.get_current_voltage()  # Read current position
            c_time = time.perf_counter() - t_0  # Current time (since reference)
            with self.buffer_lock:  # acquire lock for data manipulation
                self.voltage.append(current_voltage)
                self.times.append(c_time)

    def get_voltage_buffer(self, clear_buffer=True, initialt=None, copy=False):
        """
        Retrieve the voltage buffer. By default, the buffer is cleared after reading
        :param clear_buffer: Set true to clear the buffer after it has been read. (default: True)
        :param initialt: If not None, all time values in the buffer will be shifted so that the first value in
        the buffer is equal to initialt.
        :param copy: Legacy argument. This has no effect since a copy is always created when reading the buffer.
        :return: The voltage buffer (times, voltages) as two 1-D numpy arrays
        """
        with self.buffer_lock:  # Get Data lock for multi threading
            voltages = np.array(self.voltage)  # convert to array (creates a copy)
            times = np.array(self.times)  # convert to array (creates a copy)
            if clear_buffer:
                self.times.clear()
                self.voltage.clear()

        if initialt is not None:
            times = times - times[0] + initialt  # shift starting time to the specified initial time

        return times, voltages  # Return time and positions


def test_slow_voltage_rise():
    import matplotlib.pyplot as plt

    sb = SwitchBoard()
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
    times, voltages = sb.get_voltage_buffer()
    plt.plot(times, voltages)
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.grid(True)
    plt.show()


def test_switchboard():
    import matplotlib.pyplot as plt

    sb = SwitchBoard()
    sb.open(with_continuous_reading=False)
    t_start = time.perf_counter()
    sb.start_voltage_reading(reference_time=t_start)
    Vset = 900
    freq = 2
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
    # gains = sb.get_pid_gains()
    # print("PID gains:", gains)

    sb.set_output_off()  # = set_switching_mode(0)
    print(sb.set_relay_auto_mode())
    # sb.set_relays_on()
    # time.sleep(1)
    # print(sb.get_relay_state())
    # time.sleep(1)
    # print(sb.get_relay_state())
    # print(sb.set_relay_state(3, 1))
    # time.sleep(1)
    # print(sb.set_relays_off())
    # time.sleep(1)
    # sb.set_output_on()
    # print("set1: ", sb.set_voltage(700))
    # time.sleep(0.1)
    # print("set2: ", sb.set_voltage(800))
    # time.sleep(0.1)
    # print("set3: ", sb.set_voltage(900))
    # time.sleep(0.1)
    print("set4: ", sb.set_voltage(Vset))
    # print("sp: ", sb.get_voltage_setpoint())
    time.sleep(1)
    sb.set_frequency(freq)
    sb.set_cycle_number(25)
    tc.append(time.perf_counter() - t_start)
    cn = sb.get_cycle_number()
    print(cn)
    cn = np.diff(cn)[0]
    print(cn)
    cycles_remaining.append(cn)
    # print("f: ", sb.get_frequency())
    sb.start_ac_mode()  # = set_switching_mode(2)
    # print("now: ", sb.get_current_voltage())
    # print("sp: ", sb.get_voltage_setpoint())
    # time.sleep(2)
    # sb.set_switching_mode(0)  # DC

    for i in range(75):
        time.sleep(0.2)
        tc.append(time.perf_counter() - t_start)
        cn = sb.get_cycle_number()
        print(cn)
        cn = np.diff(cn)[0]
        print(cn)
        cycles_remaining.append(cn)

    # print("now: ", sb.get_current_voltage())
    # print(sb.get_relay_state())
    # time.sleep(1)
    # sb.set_switching_mode(1)  # DC
    # print(sb.set_relays_off())
    # time.sleep(0.5)
    sb.set_output_off()
    tc.append(time.perf_counter() - t_start)
    cn = sb.get_cycle_number()
    print(cn)
    cn = np.diff(cn)[0]
    print(cn)
    cycles_remaining.append(cn)
    print("set: ", sb.set_voltage(0))
    print("sp: ", sb.get_voltage_setpoint())
    time.sleep(1.5)

    tc.append(time.perf_counter() - t_start)
    cn = sb.get_cycle_number()
    print(cn)
    cn = np.diff(cn)[0]
    print(cn)
    cycles_remaining.append(cn)
    # print("now: ", sb.get_current_voltage())
    # print("sp: ", sb.get_voltage_setpoint())
    sb.close()
    tV, V = sb.get_voltage_buffer()
    fig, ax1 = plt.subplots()
    ax1.plot(tV, np.array(V) * 0 + Vset)
    ax1.plot(tV, V)
    ax2 = ax1.twinx()
    ax2.plot(tc, cycles_remaining)
    plt.xlabel("Time [s]")
    ax1.set_ylabel("Voltage [V]")
    ax2.set_ylabel("Cycles remaining")
    # title = "Hermione PID {} {}V 3-DEAs".format(str(gains), Vset)
    title = "Hermione AC mode {}Hz {}V 1-shorted".format(freq, Vset)
    plt.title(title)
    plt.savefig("test_data/ac/" + title + ".png")
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


if __name__ == '__main__':
    # test_switchboard()
    test_slow_voltage_rise()
