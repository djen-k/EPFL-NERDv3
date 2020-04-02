"""
py-hvps-interface GUI of the Project Peta-pico-Voltron
petapicovoltron.com
Copyright 2017-2018 Samuel Rosset
Distributed under the terms of the GNU General Public License GNU GPLv3

This file is part of shvps.

   shvps is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   py-hvps-interface is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with py-hvps-interface.  If not, see  <http://www.gnu.org/licenses/>
"""

# !/usr/bin/python

import logging
import time
from threading import Thread, Lock, RLock

import numpy as np
import pyqtgraph as pg
import serial
import serial.tools.list_ports
from PyQt5 import QtWidgets, QtCore

DEBUG = False
LIB_VER = "2.8"  # Version of this library (matches the Labview Library number)
FIRM_VER = 8  # Firmware version That this library expects to run on the HVPS

# Switching mode constants: off=0, on-DC=1, on-Switching=2, on-waveform=3
SWMODE_OFF = 0
SWMODE_DC = 1
SWMODE_SW = 2
SWMODE_WFRM = 3

# Switching source constants: Timer=0,External=1, Button=2,
SWSRC_TMR = 0
SWSRC_BTTN = 2
SWSRC_EXT = 1

# Voltage control mode constants: Regulated=0, External=1, Open loop=2
VMODE_R = 0
VMODE_EXT = 1
VMODE_O = 2

# Error bits
ERR_FIRM = 0b1  # Incompatibility between python library and Firmware running on HVPS
ERR_TYPE = 0b10  # The connected arduino is not a single channel HVPS
ERR_JACK = 0b100  # The power Jack is not plugged
ERR_COM = 0b1000  # Communication error: The arduino is detected but cannot talk to it.


class HvpsInfo:
    """Structure to store the current parameters of the HVPS"""

    def __init__(self):
        self.port = ''
        self.name = ''
        self.i2c = ''
        self.vmax = 1000
        self.swmode = 0
        self.vset = 0
        self.vnow = 0
        self.freq = 0
        self.cycles = 0
        self.cycle_n = 0  # the current cycle number
        self.swsrc = 0
        self.vmode = 0
        self.latch = 0  # latching behaviour of the button
        self.err = 0
        self.stmode = 0  # Strobe mode
        self.stpos = 0  # Strobe position
        self.stdur = 0  # strobe duration
        self.stsweep = 5000  # Strobe sweep time (default when HVPS starts)
        self.config = 0  # 0=HVPS connected to touch screen on the stand with a USB cable
        # 1=HVPS inegrated with touchscreen in an enclosure with battery
        self.listen = 0  # in touchscreen + battery mode,
        # listen=1 if interface must listen on Pi's serial port for incoming commands


class HVPS:
    """Class for driving an HVPS from the peta-pico-voltron project"""
    buffer_length = 1000000
    baudrate = 115200

    def __init__(self):
        self.logging = logging.getLogger("HVPS")
        self.logging.info("HVPS Class loaded")

        self.ser = serial.Serial()

        self.hvps_available_ports = []
        self.is_open = False
        self.current_device = HvpsInfo()  # Current device

        self.serial_com_lock = RLock()

        self.continuous_voltage_reading_flag = False
        self.buffer_lock = Lock()
        self.voltage = np.zeros((self.buffer_length))  # voltage buffer
        self.times = np.zeros((self.buffer_length))  # Times buffer
        self.index_buffer = 0
        self.reading_thread = Thread()

        self.signal_generation_thread = Thread()

    def __del__(self):
        try:
            """Class destructor"""
            self.stop_voltage_reading()
            self.stop_software_signal()
            time.sleep(0.1)
            self.close()
        except:
            pass

    def detect(self):
        """Detect available HVPS"""
        serial_ports = serial.tools.list_ports.comports()
        self.hvps_available_ports = []
        for ser in serial_ports:
            if "Arduino" in ser.description:
                info = HvpsInfo()
                info.port = ser.device
                try:
                    info = self.open_hvps(info, with_continuous_reading=False)
                    self.close()
                except:
                    self.logging.critical(
                        "Unable to open COM port %s. Unrecognized device or unsupported firmware version." % ser.device)
                else:
                    if info is not None:
                        self.logging.info("Device %s found at port %s" % (info.name, info.port))
                        self.hvps_available_ports.append(info)
        self.logging.info("HVPS Detection done")
        return self.hvps_available_ports

    def check_connection(func):
        """Decorator for checking connection and acquiring multithreading lock"""

        def is_connected_wrapper(*args):
            """Wrapper"""
            if args[0].is_open:
                args[0].serial_com_lock.acquire()
                res = func(*args)
                args[0].serial_com_lock.release()
                return res
            else:
                print("Not Connected")

        return is_connected_wrapper

    def auto_open(self, with_continuous_reading=False):
        """Open first available HVPS"""
        self.detect()
        if self.hvps_available_ports:
            self.current_device = self.open_hvps(self.hvps_available_ports[0], with_continuous_reading)

    def open_hvps(self, info: HvpsInfo, with_continuous_reading=False):
        """Establishes connection with the HVPS."""
        self.logging.debug("Connecting to %s" % info.port)
        self.serial_com_lock.acquire()
        try:
            self.ser = serial.Serial(info.port, self.baudrate, timeout=1)
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
        except:
            self.logging.critical("Impossible to open port %s" % info.port)
            self.current_device = None

        if self.ser.is_open:
            self.logging.debug("Reading device data")
            self.is_open = True
            try:
                info.config = not self.get_power_source()  # get the configuration of the HVPS \
                # (1= touchscreen + battery). If power comes from power Jack, then unit
                # is NOT in touchscreen+battery configuration.
            except ValueError:
                info.err = info.err | ERR_COM
                self.logging.critical("Unrecognized Device")
            else:
                if self.get_hvps_type() != "slave":
                    info.err = info.err | ERR_TYPE
                else:
                    if self.get_firmware_version() != FIRM_VER:
                        info.err = info.err | ERR_FIRM
                    if not self.get_jack_status() and info.config == 0:
                        # We don't check for presence of power jack if HVPS powered via 5V pin
                        # (and not jack)
                        info.err = info.err | ERR_JACK
                    info.name = self.get_name()
                    info.vmax = self.get_maximum_voltage()
                    info.i2c = self.get_i2c_adress()
                    info.freq = self.get_frequency()

                    info.latch = self.get_latch_mode()
                    info.stmode = self.get_strobe_mode()
                    # Strobe mode is not saved in memory. But it is not necessarily 0,
                    # if the interface was closed, but the HVPS not unplugged
                    info.stpos = self.get_strobe_position()
                    info.stdur = self.get_strobe_duration()
                    info.stsweep = self.set_strobe_sweep_period(info.stsweep)

                    if info.config == 0:  # if HVPS connected via USB
                        info.vset = self.set_voltage(0)
                        info.swmode = self.set_switching_mode(SWMODE_DC)
                        info.swsrc = self.set_switching_source(SWSRC_TMR)
                        info.vmode = self.set_voltage_control_mode(VMODE_R)
                        info.cycles = self.set_cycle_number(0)
                    else:  # if in touchscreen housing, then we use the current parameters
                        # (i.e. memory if just started or last params if interface is restarted)
                        info.vset = self.get_voltage_setpoint()  # initialises the info structure
                        # with the current parameters of the HVPS
                        info.swmode = self.get_switching_mode()
                        info.swsrc = self.get_switching_source()
                        info.vmode = self.get_voltage_control_mode()
                        cycles_data = self.get_cycle_number()
                        info.cycles = int(cycles_data[1])
                self.current_device = info
                if with_continuous_reading:
                    self.start_voltage_reading()
            self.serial_com_lock.release()
        return self.current_device

    def open_hvps_from_port(self, port, with_continuous_reading=False):
        """Open an HVPS from a specific COM Port"""
        info = HvpsInfo()
        info.port = port
        self.open_hvps(info, with_continuous_reading)
        return info

    def close(self):
        """Closes connection with the HVPS"""
        self.stop_voltage_reading()
        time.sleep(0.1)
        self.serial_com_lock.acquire()
        if self.ser.is_open:
            self.set_voltage(0)  # set the voltage to 0 as a safety measure
            self.ser.close()
        self.is_open = False
        self.current_device = None
        self.serial_com_lock.release()
        # self.logging.debug("Serial port closed")

    def get_name(self):  # queries the name of the board
        """Queries name of the board"""
        self.logging.debug("Querying device name")
        self._write_hvps(b'QName\r')
        res = self._read_hvps()
        return res

    def get_maximum_voltage(self):  # queries the voltage rating of the board
        """Queries the voltage rating of the board"""
        self.logging.debug("Querying device maximum voltage")
        self._write_hvps(b'QVmax\r')
        res = self._cast_int(self._read_hvps())
        return res

    def set_voltage(self, voltage):  # sets the output voltage
        """Sets the output voltage"""
        self.logging.debug("Set voltage setpoint to %d V" % voltage)
        self._write_hvps(b'SVset %d\r' % voltage)
        res = self._cast_int(self._read_hvps())
        return res

    def get_voltage_setpoint(self):
        """Queries voltage setpoint"""
        self.logging.debug("Querying device voltage set point")
        self._write_hvps(b'QVset\r')
        res = self._cast_int(self._read_hvps())
        return res

    def get_current_voltage(self):
        """Queries voltage output"""
        self.logging.debug("Querying device current voltage")
        self._write_hvps(b'QVnow\r')
        res = self._cast_int(self._read_hvps())
        return res

    def set_pwm(self, pwm_value):  # sets the pwm value
        """Sets the PWM Value.
        Defines the functioning set point of the HV programmable source as a 10-bit (0-1023) raw \
        PWM value.
        Although this command can be used at any time, it mainly useful when the HVPS voltage \
        control mode is 2 (internal openloop; see SVMode command)."""
        self.logging.debug("Setting PWM value to %d" % pwm_value)
        self._write_hvps(b'SPWM %d\r' % pwm_value)
        res = self._cast_int(self._read_hvps())
        return res

    def get_pwm(self):  # queries the pwm setting
        """Queries the current HVPS setpoint as a raw PWM value."""
        self.logging.debug("Querying device current PWM Value")
        self._write_hvps(b'QPWM\r')
        res = self._cast_int(self._read_hvps())
        return res

    def set_frequency(self, frequency):  # sets the frequency
        """Sets the frequency of the signal when the HVPS is in switching mode.
        For example SF 0.5 to set the frequency to 0.5 Hz.
        The value returned is the new frequency, taking quantification into account.
        min freq is 0.001Hz max is 5000Hz"""
        self.logging.debug("Set device frequency to %.3f" % frequency)
        self._write_hvps(b'SF %.3f\r' % frequency)
        res = self._cast_float(self._read_hvps())
        return res

    def get_frequency(self):  # queries the frequency
        """Queries the switching frequency. The returned value is in Hz."""
        self.logging.debug("Querying device current frequency value")
        self._write_hvps(b'QF\r')
        res = self._cast_float(self._read_hvps())
        return res

    def set_cycle_number(self, cycle_number):
        """Sets the number of switching cycles to perform when the HVPS is in switching mode.
        The maximum value is 65535. A value of 0 means continuous switching
        (unlimited number of cycles).
        For any value other than 0, the HVPS will change to switching mode 0 (HVPS off)
        after the desired number of cycles is reached.
        A new series of switching cycles can be can be initiated by placing the HVPS back
        in switching mode (mode 2).
        When you issue a SCycle command, the cycle counter is reset to 0.
        Example: SCycle 1000 to switch 1000 times at the selected frequency and then stop."""
        self.logging.debug("Set device cycle number to %d" % cycle_number)
        self._write_hvps(b'SCycle %d\r' % cycle_number)
        res = self._cast_int(self._read_hvps())
        return res

    def get_cycle_number(self):
        """Queries the number of cycles.
        The returned value is in the form XXX/YYY, with XXX being the current cycle number and YYY
        the total number of cycles to make.
        Once the total number of cycles is reached, the output is turned off,
        and QCycle returns 0/YYY."""
        self.logging.debug("Get device cycle number and current cycle")
        self._write_hvps(b'QCycle\r')
        res = self._read_hvps()
        res = res.decode("utf-8")
        res = res.split("/")
        res = [self._cast_int(i) for i in res]
        self.logging.debug("Cycle %d/%d" % (res[0], res[1]))
        return res

    def set_switching_mode(self, switching_mode):
        """Sets the switching mode of the HVPS.
        Four possible values: 0 HVPS is off (0 V output irrespective of voltage setpoint),
        1 the HVPS is in DC mode with a constant output voltage at the desired setpoint,
        and 2, the HVPS is switching at the desired frequency between 0V and Vset.
        3, the HVPS is in user-defined waveform mode.
        Setting the switching mode is only effective if the switching source
        is 0 (onboard switching).
        Example: SSwMode 1 to set the HVPS in DC mode"""
        self.logging.debug("Set device switching mode to %d" % switching_mode)
        self._write_hvps(b'SSwMode %d\r' % switching_mode)
        res = self._cast_int(self._read_hvps())
        return res

    def set_output_off(self):
        """Set Output Off"""
        return self.set_switching_mode(SWMODE_OFF)

    def set_output_on(self):
        """Set Output Off"""
        return self.set_switching_mode(SWMODE_DC)

    def start_ac_mode(self):
        """Start Switching Mode"""
        self.set_switching_mode(SWMODE_SW)

    def get_switching_mode(self):  # queries the switching mode
        """Queries the switching mode of the HVPS:
            0 box is off (0 V output irrespective of voltage setpoint),
            1 the HVPS is in DC mode with a constant output voltage at the desired setpoint,
            2, the HVPS is switching at the desired frequency between 0V and Vset.
            3, the HVPS is in user-defined waveform mode."""
        self.logging.debug("Get Device switching mode")
        self._write_hvps(b'QSwMode\r')
        res = self._cast_int(self._read_hvps())
        return res

    def set_switching_source(self, switching_source):
        """Sets the source of the switching signal.
        Accepted values are: 0 for onboard switching (from internal clock of the board),
        1 for external switching via pin 6 (Ext.F) on the board main connector,
        and 2 for the push button.
        Example: SSwSrc 1 to use the external signal to switch the source on/off.
        Note that this setting is useful only if the jumper on header H2 is set to
        “onboard control”, else the jumper setting defines the source of the switching signal."""
        self.logging.debug("Set Device switching source to %d" % switching_source)
        self._write_hvps(b'SSwSrc %d\r' % switching_source)
        res = self._cast_int(self._read_hvps())
        return res

    def get_switching_source(self):
        """queries the switching source"""
        self._write_hvps(b'QSwSrc\r')
        res = self._cast_int(self._read_hvps())
        self.logging.debug("Device switching source is %d" % res)
        return res

    def set_voltage_control_mode(self, voltage_control_mode):
        """Sets the voltage control mode (i.e. how is the value of the output voltage controlled):
            0 for internal voltage regulator (regulates the voltage to the value defined with the
            Vset command),
            1 external voltage control (sets the output voltage according to the control voltage
            applied on pin 5 (Ext.V) of the main connector of the board.
            The input voltage range is 0 to 5V.
            2 internal open loop control (on-board regulator disconnected)."""
        self._write_hvps(b'SVMode %d\r' % voltage_control_mode)
        res = self._cast_int(self._read_hvps())
        self.logging.debug("Device voltage control mode set to %d. Result: %d." % (voltage_control_mode, res))
        return res

    def get_voltage_control_mode(self):  # queries the switching source
        """Sets the voltage control mode (i.e. how is the value of the output voltage controlled):
            0 for internal voltage regulator (regulates the voltage to the value defined with the
            Vset command),
            1 external voltage control (sets the output voltage according to the control voltage
            applied on pin 5 (Ext.V) of the main connector of the board.
            The input voltage range is 0 to 5V.
            2 internal open loop control (on-board regulator disconnected)."""
        self._write_hvps(b'QVMode\r')
        res = self._cast_int(self._read_hvps())
        self.logging.debug("Get device voltage control mode : %d." % res)
        return res

    def get_memory(self):
        """queries the content of the memory"""
        mem = HvpsMem()
        self._write_hvps(b'QMem\r')
        res = self._read_hvps()
        raw_memory = res.decode("utf-8")
        res = raw_memory.split(",")
        mem.vset = self._cast_int(res[0])
        mem.frequency = self._cast_float(res[1])
        mem.swsrc = self._cast_int(res[2])
        mem.swmode = self._cast_int(res[3])
        mem.vmode = self._cast_int(res[4])
        mem.k_p = self._cast_float(res[5])
        mem.k_i = self._cast_float(res[6])
        mem.k_d = self._cast_float(res[7])
        mem.c_0 = self._cast_float(res[8])
        mem.c_1 = self._cast_float(res[9])
        mem.c_2 = self._cast_float(res[10])
        mem.cycles = self._cast_int(res[11])
        mem.latch = self._cast_int(res[12])
        self.logging.debug("Get device memory: %s" % raw_memory)
        return mem

    def save_memory(self):
        """save current HVPS paramters into the memory"""
        self._write_hvps(b'Save\r')
        res = self._cast_int(self._read_hvps())
        self.logging.debug("Saving current device status to memory. Result: %d" % res)
        return res

    def set_i2c_adress(self, i2c_adress):
        """sets the I2C address"""
        self._write_hvps(b"SI2C %d\r" % i2c_adress)
        res = self._cast_int(self._read_hvps())
        self.logging.debug("Set I2C adress to %d. Result: %d" % (i2c_adress, res))
        return res

    def get_i2c_adress(self):
        """queries the i2c address of the board"""
        self._write_hvps(b'QI2C\r')
        res = self._cast_int(self._read_hvps())
        self.logging.debug("Get Current I2C adress: %d" % res)
        return res

    def get_firmware_version(self):
        """queries the firmware version"""
        self._write_hvps(b'QVer\r')
        res = self._read_hvps()  # returns a string in the form of "slave X" need to get the value of X
        res = res.decode("utf-8")
        res = res.split(" ")
        res = self._cast_int(res[1])
        self.logging.debug("Get Current Firmware version : %d" % res)
        return res

    def get_hvps_type(self):
        """Queries the type (master (multi-channel), slave (single channel))"""
        self._write_hvps(b'QVer\r')
        res = self._read_hvps()  # returns a string in the form of "slave X" need to get slave
        res = res.decode("utf-8")
        res = res.split(" ")
        res = res[0]
        self.logging.debug("Get HVPS Type: %s" % res)
        return res

    def set_latch_mode(self, latch_mode):
        """sets the latch mode of the push button"""
        self._write_hvps(b'SLatchMode %d\r' % latch_mode)
        res = self._cast_int(self._read_hvps())
        self.logging.debug("Set latch mode %d. Result: %d" % (latch_mode, res))
        return res

    def get_latch_mode(self):
        """queries the latch mode of the push button"""
        self._write_hvps(b'QLatchMode\r')
        res = self._cast_int(self._read_hvps())
        self.logging.debug("Get latch mode: %d" % res)
        return res

    def get_jack_status(self):
        """queries whether power Jack is plugged in."""
        self._write_hvps(b'QJack\r')
        res = self._cast_int(self._read_hvps())
        self.logging.debug("Get jack Status: %d" % res)
        return res

    def get_power_source(self):
        """queries whether power is supposed to come from Power Jack. (0 if in touchscreen \
        + battery configuration)"""
        self._write_hvps(b'QPowJack\r')
        res = self._cast_int(self._read_hvps())
        self.logging.debug("Get power source: %d" % res)
        return res

    def set_strobe_mode(self, strobe_mode):
        """sets the strobe mode"""
        self._write_hvps(b'SStMode %d\r' % strobe_mode)
        res = self._cast_int(self._read_hvps())
        self.logging.debug("Set strobe mode %d. Result: %d" % (strobe_mode, res))
        return res

    def get_strobe_mode(self):
        """queries the strobe mode"""
        self._write_hvps(b'QStMode\r')
        res = self._cast_int(self._read_hvps())
        self.logging.debug("Get strobe mode: %d." % res)
        return res

    def set_strobe_position(self, strobe_position):
        """sets the position of the strobe pulse"""
        self._write_hvps(b'SStPos %d\r' % strobe_position)
        res = self._cast_int(self._read_hvps())
        self.logging.debug("Set strobe position to %d. Result: %d" % (strobe_position, res))
        return res

    def get_strobe_position(self):
        """queries the the position of the strobe pulse"""
        self._write_hvps(b'QStPos\r')
        res = self._cast_int(self._read_hvps())
        self.logging.debug("Get strobe position: %d." % res)
        return res

    def set_strobe_duration(self, strobe_duration):
        """sets the duration of the strobe pulse"""
        self._write_hvps(b'SStDur %d\r' % strobe_duration)
        res = self._cast_int(self._read_hvps())
        self.logging.debug("Set strobe duration to %d. Result: %d" % (strobe_duration, res))
        return res

    def set_strobe_sweep_period(self, sweep_period):
        """sets the sweep period (im ms) when in sweep mode"""
        self._write_hvps(b'SStSw %d\r' % sweep_period)
        res = self._cast_int(self._read_hvps())
        self.logging.debug("Set strobe sweep period to %d. Result: %d" % (sweep_period, res))
        return res

    def get_strobe_duration(self):  #
        """queries the the duration of the strobe pulse"""
        self._write_hvps(b'QStDur\r')
        res = self._cast_int(self._read_hvps())
        self.logging.debug("Get strobe duration: %d." % res)
        return res

    def dirty_reconnect(self):
        self.ser.close()
        try:
            self.ser.open()
        except:
            self.logging.critical("Reconnextion attempt failed... retry in 500ms")
        while not self.ser.is_open:
            time.sleep(0.5)
            self.ser.close()
            try:
                self.ser.open()
            except:
                self.logging.critical("Reconnextion attempt failed... retry in 500ms")
        self.logging.critical("Reconnected! (?)")

    @check_connection
    def _write_hvps(self, cmd):
        try:
            self.ser.write(cmd)
            while self.ser.out_waiting:
                time.sleep(0.01)
        except:
            print("Error writing HVPS")
            self.logging.critical("Error writing HVPS")
            self.dirty_reconnect()

    @check_connection
    def _read_hvps(self):  # reads a line on the serial port and removes the end of line characters
        try:
            line = self.ser.readline()
            line = line[:-2]  # removes the last 2 elements of the array because it is \r\n
            return bytes(line)
        except:
            print("Error reading HVPS")
            self.logging.critical("Error reading HVPS")
            self.dirty_reconnect()

    def _cast_int(self, data):
        try:
            casted_int = int(data)
        except:
            casted_int = -1
        return casted_int

    def _cast_float(self, data):
        try:
            casted_float = float(data)
        except:
            casted_float = -1
        return casted_float

    maxSignalRate = 3

    def start_software_signal(self, type_signal, frequency, amplitude):
        """Generate a voltage waveform driven by the computer"""
        self.stop_software_signal()
        if frequency < self.maxSignalRate:
            self.signal_freq = frequency
            self.signal_type = type_signal
            self.signal_amplitude = np.abs(amplitude)
            self.is_software_signal_running = True
            self.signal_generation_thread = Thread()
            self.signal_generation_thread.run = self._generate_software_signal
            self.signal_generation_thread.start()
            return True
        return False

    def trapezoidal_signal(self, slope=0.1):
        """Generate a trapezoidal waveform"""
        if slope > 0.5:
            slope = 0.5
        number_of_points = 100
        phi = np.linspace(0, 2 * np.pi, number_of_points)
        signal = np.zeros(number_of_points)
        signal[phi < np.pi] = 1
        phase_stop_slope = 2 * np.pi * slope
        signal[phi < phase_stop_slope] = phi[phi < phase_stop_slope] / phase_stop_slope
        down_slope = np.where(np.logical_and(phi > np.pi, phi < np.pi + phase_stop_slope))[0]
        signal[down_slope] = 1 - (phi[down_slope] - np.pi) / phase_stop_slope
        return lambda t: np.interp(t % (2 * np.pi), phi, signal)

    def _generate_software_signal(self):
        """Threaded method for software waveform"""
        t_0 = time.clock()
        if self.signal_type == "Sine":
            signal_function = lambda x: (np.sin(x) + 1) / 2
        elif self.signal_type == "Trapz":
            signal_function = self.trapezoidal_signal(slope=0.2)
        while self.is_software_signal_running:
            c_time = time.clock()
            voltage = self.signal_amplitude * signal_function(2 * np.pi * self.signal_freq * (c_time - t_0))
            if voltage < 0:
                voltage = 0
            self.set_voltage(voltage)
            time.sleep(0.05)

    def stop_software_signal(self):
        """Stop Software waveform generation"""
        self.is_software_signal_running = False
        time.sleep(0.1)

    def start_voltage_reading(self):
        """Routine for starting continuous position reading"""
        self.continuous_voltage_reading_flag = True  # Flag set to true
        self.index_buffer = 0  # Buffer position is 0
        self.voltage = np.zeros(self.buffer_length)  # voltage buffer
        self.times = np.zeros(self.buffer_length)  # Times buffer
        self.reading_thread = Thread()  # Thread for continuous reading
        self.reading_thread.run = self._continuous_voltage_reading  # Method associated to thread
        self.reading_thread.start()  # Starting Thread

    def stop_voltage_reading(self):
        """Routine for stoping continuous position reading"""
        self.continuous_voltage_reading_flag = False  # Set Flag to False

    def _continuous_voltage_reading(self):
        """Method for continuous reading"""
        self.t_0 = time.clock()  # Initialializing reference time
        while (self.continuous_voltage_reading_flag and self.is_open):
            # While Flag is true and HVPS is connected
            current_voltage = self.get_current_voltage()  # Read current position
            self.buffer_lock.acquire()  # Set Lock for data manipulation
            self.c_time = time.clock()  # Current time
            if self.index_buffer < self.buffer_length:  # Buffer not filled
                self.voltage[self.index_buffer] = current_voltage  # Add position data
                self.times[self.index_buffer] = self.c_time  # Add time data
                self.index_buffer += 1  # Increment buffer
            else:  # Buffer filled
                self.voltage[0:-1] = self.voltage[1:]  # Shift buffer of one position
                self.voltage[-1] = current_voltage  # Add position data
                self.times[0:-1] = self.times[1:]  # Shift time buffer
                self.times[-1] = self.c_time  # Add time data
            self.buffer_lock.release()  # Release data lock

    def get_voltage_buffer(self, clear_buffer=True, initialt=None, copy=False):
        """Method for retrieving position buffer. clear_buffer=True causes the buffer to be reset"""
        self.buffer_lock.acquire()  # Get Data lock for multithreading
        if (copy or initialt):
            voltages = self.voltage[:(self.index_buffer)].copy()  # Current stored positions
            times = self.times[:(self.index_buffer)].copy()  # Current stored times
        else:
            voltages = self.voltage[:(self.index_buffer)]
            times = self.times[:(self.index_buffer)]
        if initialt:
            times -= times[0]
            times += initialt
        if clear_buffer:  # If buffer reset
            self.index_buffer = 0  # Reset buffer deaIndex
            self.t_0 = time.clock()  # Reset time reference
        self.buffer_lock.release()  # Release lock
        return (times, voltages)  # Return time and positions


class HvpsMem:
    """Structure to store the memory information of the HVPS"""

    def __init__(self):
        self.vset = 0
        self.frequency = 0
        self.swsrc = 0
        self.swmode = 0
        self.vmode = 0
        self.k_p = 0
        self.k_i = 0
        self.k_d = 0
        self.c_0 = 0
        self.c_1 = 0
        self.c_2 = 0
        self.cycles = 0
        self.latch = 0
        self.vmax = 0  # Vmax, ver and i2c address are not returned by the
        # get_memory function, because they have dedicated query function.
        # They are added to the structure so that all information
        # regarding the configuration of the board can be grouped in a single variable.
        # (Name should be added)
        self.i2c = 0
        self.ver = 0


class HVPSChannelDetectionWidget(QtWidgets.QWidget):
    """Combobox + connect button widget. Autorefresh available HVPS"""

    def __init__(self, hvps_object: HVPS, parent=None):
        """Widget initialization"""
        super(HVPSChannelDetectionWidget, self).__init__(parent)
        self.main_layout = QtWidgets.QHBoxLayout(self)
        self.hvps_object = hvps_object
        self.is_connected = False
        self.connect_lock = Lock()

        self.available_hvps = []
        self.available_hvps_combobox = QtWidgets.QComboBox()
        self.main_layout.addWidget(self.available_hvps_combobox)

        self.connect_hvps_button = QtWidgets.QPushButton("Connect")
        self.connect_hvps_button.clicked.connect(self._connect_hvps_button_callback)
        self.main_layout.addWidget(self.connect_hvps_button)

        self.initialize_layout()

        self.continuous_detection_running = True
        self.updated_list_of_devices = False
        self.update_available_hvps_thread = Thread()
        self.update_available_hvps_thread.run = self._update_available_hvps

        self.update_combobox_timer = QtCore.QTimer()
        self.update_combobox_timer.timeout.connect(self._update_combobox)
        self.update_combobox_timer.setInterval(150)

    def close(self):
        self.continuous_detection_running = False

    def initialize_layout(self):
        """Initialize combo box with current HVPS object status"""
        if self.hvps_object.is_open:
            self.is_connected = True
            self.connect_hvps_button.setText("Disconnect")
            self.available_hvps_combobox.clear()
            self.available_hvps_combobox.addItem(self.hvps_object.current_device.name.decode())

    def _connect_hvps_button_callback(self):
        """Connect button callback"""
        self.connect_lock.acquire()
        if self.is_connected:
            self.hvps_object.close()
            self.is_connected = False
            self.connect_hvps_button.setText("Connect")
        else:
            if self.available_hvps_combobox.count() > 0:
                current_device = self.available_hvps_combobox.currentIndex()
                self.hvps_object.open_hvps(self.hvps_object.hvps_available_ports[current_device])
                if self.hvps_object.is_open:
                    self.is_connected = True
                    self.connect_hvps_button.setText("Disconnect")
        self.connect_lock.release()

    def _update_available_hvps(self):
        """Scan available HVPS and update widget if new devices have been detected"""
        while self.continuous_detection_running:
            if not self.is_connected:
                self.connect_lock.acquire()
                available_ports = self.hvps_object.detect()
                updated = len(available_ports) != len(self.available_hvps)

                for port in available_ports:
                    if port.name not in self.available_hvps:
                        updated = True
                self.available_hvps = available_ports
                self.updated_list_of_devices |= updated
                self.connect_lock.release()
            time.sleep(0.2)

    def _update_combobox(self):
        if self.updated_list_of_devices:
            self.available_hvps_combobox.clear()
            self.available_hvps_combobox.addItems([device.name.decode() for device in self.available_hvps])
            self.updated_list_of_devices = False

    def showEvent(self, event):
        """Start HVPS continuous detection when showing widget"""
        self.continuous_detection_running = True
        self.update_available_hvps_thread.start()
        self.update_combobox_timer.start()
        event.accept()

    def hideEvent(self, event):
        """Stop detecting HVPS when widget is hidden"""
        self.continuous_detection_running = False
        self.update_combobox_timer.stop()
        event.accept()


class HvpsVoltageControlWidget(QtWidgets.QWidget):
    """Widget for controlling voltage/frequency of HVPS"""
    min_frequency = 1e-3
    max_frequency = 5000
    n_step_frequency_slider = 100000
    n_step_voltage_slider = 1000

    def __init__(self, hvps_object: HVPS, parent=None):
        super(HvpsVoltageControlWidget, self).__init__(parent)
        self.hvps_object = hvps_object

        self.main_layout = QtWidgets.QFormLayout(self)

        self.voltage_slider_layout = QtWidgets.QHBoxLayout()
        self.voltage_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.voltage_slider.setRange(0, self.n_step_voltage_slider)
        self.voltage_slider_layout.addWidget(self.voltage_slider)
        self.voltage_set_point_edit = QtWidgets.QLineEdit("0")
        self.voltage_slider_layout.addWidget(self.voltage_set_point_edit)
        self.main_layout.addRow("Voltage set point", self.voltage_slider_layout)
        self.voltage_slider.sliderReleased.connect(self._voltage_slider_released_callback)
        self.voltage_slider.valueChanged.connect(self._voltage_slider_value_changed_callback)

        self.voltage_set_point_edit.returnPressed.connect(self._voltage_set_point_edit_callback)

        self.powering_layout = QtWidgets.QHBoxLayout()
        self.switching_mode_combobox = QtWidgets.QComboBox()
        self.switching_mode_combobox.addItems(["DC", "AC", "Waveform"])
        self.switching_mode_combobox.currentIndexChanged.connect(self._switching_mode_combobox_changed_callback)
        self.powering_layout.addWidget(self.switching_mode_combobox)
        self.power_button = None
        self.main_layout.addRow("Switching Mode", self.switching_mode_combobox)

        self.frequency_slider_layout = QtWidgets.QHBoxLayout()
        self.frequency_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.frequency_slider.setRange(np.log10(self.min_frequency) * self.n_step_frequency_slider,
                                       np.log10(self.max_frequency) * self.n_step_frequency_slider)

        self.frequency_slider_layout.addWidget(self.frequency_slider)
        self.frequency_edit = QtWidgets.QLineEdit("1")
        self.frequency_slider_layout.addWidget(self.frequency_edit)
        self.main_layout.addRow("Frequency", self.frequency_slider_layout)
        self.frequency_slider.sliderReleased.connect(self._frequency_slider_released_callback)
        self.frequency_slider.valueChanged.connect(self._frequency_slider_value_changed_callback)
        self.frequency_edit.returnPressed.connect(self._frequency_edit_callback)

        self.number_of_cycles_layout = QtWidgets.QHBoxLayout()
        self.number_of_cycles_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.number_of_cycles_layout.addWidget(self.number_of_cycles_slider)
        self.number_of_cycles_slider.setMaximum(1000)
        self.number_of_cycles_slider.sliderReleased.connect(self._number_of_cycles_slider_released_callback)
        self.number_of_cycles_slider.valueChanged.connect(self._number_of_cycles_slider_changed_callback)

        self.number_of_cycles_edit = QtWidgets.QLineEdit("")
        self.number_of_cycles_layout.addWidget(self.number_of_cycles_edit)
        self.number_of_cycles_edit.returnPressed.connect(self._number_of_cycles_edit_callback)

        self.number_of_cycles_disable_button = QtWidgets.QPushButton("∞")
        self.number_of_cycles_layout.addWidget(self.number_of_cycles_disable_button)
        self.number_of_cycles_disable_button.setCheckable(True)
        self.number_of_cycles_disable_button.toggled.connect(self._number_of_cycles_disable_button_callback)

        self.main_layout.addRow("Number of cycles", self.number_of_cycles_layout)

        self.power_button = QtWidgets.QPushButton("POWER ON")
        self.power_button.setCheckable(True)
        self.power_button.toggled.connect(self._power_button_callback)
        self.main_layout.addRow(self.power_button)

    def _power_button_callback(self):
        switching_modes = [SWMODE_DC, SWMODE_SW, SWMODE_WFRM]
        selected_mode = switching_modes[self.switching_mode_combobox.currentIndex()]
        if self.power_button.isChecked():
            if self.hvps_object.is_open:
                self.hvps_object.set_switching_mode(selected_mode)
            else:
                self.power_button.setChecked(False)
        else:
            self.hvps_object.set_output_off()

    def _frequency_slider_released_callback(self):
        freq = 10 ** float(self.frequency_slider.value() / self.n_step_frequency_slider)
        self.hvps_object.set_frequency(freq)

    def _frequency_slider_value_changed_callback(self):
        freq = 10 ** float(self.frequency_slider.value() / self.n_step_frequency_slider)
        self.frequency_edit.setText("%.3f" % freq)

    def _frequency_edit_callback(self):
        try:
            freq = float(self.frequency_edit.text())
        except:
            freq = 1
            self.frequency_edit.setText("%.3f" % 1)
        self.frequency_slider.setValue(np.log10(freq) * self.n_step_frequency_slider)
        self.hvps_object.set_frequency(freq)

    def _voltage_slider_released_callback(self):
        voltage = self.voltage_slider.value() * self.hvps_object.current_device.vmax / self.n_step_voltage_slider
        self.hvps_object.set_voltage(voltage)

    def _voltage_slider_value_changed_callback(self):
        voltage = self.voltage_slider.value() * self.hvps_object.current_device.vmax / self.n_step_voltage_slider
        self.voltage_set_point_edit.setText("%.1f" % voltage)

    def _voltage_set_point_edit_callback(self):
        try:
            voltage = float(self.voltage_set_point_edit.text())
        except:
            voltage = 1
            self.voltage_set_point_edit.setText("%.1f" % voltage)
        self.voltage_slider.setValue(voltage)
        self.hvps_object.set_voltage(voltage)

    def _switching_mode_combobox_changed_callback(self):
        switching_modes = [SWMODE_DC, SWMODE_SW, SWMODE_WFRM]
        selected_mode = switching_modes[self.switching_mode_combobox.currentIndex()]
        if self.power_button.isChecked():
            self.hvps_object.set_switching_mode(selected_mode)

    def _number_of_cycles_slider_released_callback(self):
        self.hvps_object.set_cycle_number(self.number_of_cycles_slider.value())

    def _number_of_cycles_slider_changed_callback(self):
        self.number_of_cycles_edit.setText("%d" % self.number_of_cycles_slider.value())

    def _number_of_cycles_edit_callback(self):
        try:
            n_cycles = int(self.number_of_cycles_edit.text())
        except:
            n_cycles = 0
            print("Cast Error in number of cycles edit")
        self.number_of_cycles_slider.setValue(n_cycles)
        self.hvps_object.set_cycle_number(n_cycles)

    def _number_of_cycles_disable_button_callback(self):
        self.number_of_cycles_slider.setValue(0)
        self.number_of_cycles_edit.setText("0")
        self.hvps_object.set_cycle_number(0)


class CurrentHvpsStateWidget(QtWidgets.QWidget):
    """Widget for displaying current HVPS Status"""

    def __init__(self, hvps_object: HVPS, parent=None):
        super(CurrentHvpsStateWidget, self).__init__(parent)
        self.hvps_object = hvps_object

        self.main_layout = QtWidgets.QFormLayout(self)

        self.current_hvps_status_label = QtWidgets.QLabel("")
        self.main_layout.addRow("Current status", self.current_hvps_status_label)
        self.set_point_voltage_label = QtWidgets.QLabel("0")
        self.main_layout.addRow("Set point voltage", self.set_point_voltage_label)
        self.current_voltage_label = QtWidgets.QLabel("0")
        self.main_layout.addRow("Current voltage", self.current_voltage_label)
        self.current_frequency_label = QtWidgets.QLabel("0")
        self.main_layout.addRow("Current frequency", self.current_frequency_label)

    def update_current_status_label(self):
        self.hvps_object.current_device.f
        self.hvps_object.current_device.vnow
        self.hvps_object.current_device.vset
        self.hvps_object.current_device.swmode
        self.hvps_object.current_device.swsrc
        self.hvps_object.current_device.vmode
        self.hvps_object.current_device.latch


class HVPSWidget(QtWidgets.QWidget):
    """Qt Widget for controlling HVPS"""

    def __init__(self, hvps_object: HVPS, auto_connect=True, parent=None):
        super(HVPSWidget, self).__init__(parent)
        self.hvps_object = hvps_object
        self.main_layout = QtWidgets.QHBoxLayout(self)

        self.controls_layout = QtWidgets.QVBoxLayout()
        self.main_layout.addLayout(self.controls_layout)

        self.controls_layout.addWidget(HVPSChannelDetectionWidget(self.hvps_object))
        self.controls_layout.addWidget(HvpsVoltageControlWidget(self.hvps_object))

        self.plot_widget = HVPSPlotWidget(self.hvps_object, self)
        self.main_layout.addWidget(self.plot_widget)

    def hideEvent(self, event):
        """Set HVPS off when closing/hiding window"""
        self.hvps_object.set_output_off()
        event.accept()


class HVPSPlotWidget(pg.PlotWidget):
    """Widget for displaying a graphe with HVPS voltage"""
    buffer_length = 10000

    def __init__(self, hvps_object: HVPS, parent=None):
        super(HVPSPlotWidget, self).__init__(parent)
        self.voltage_plot = self.plot()
        self.setMinimumWidth(400)
        self.setMinimumHeight(200)
        self.setLabel('left', 'Voltage', units='V')
        self.setLabel('bottom', 'Time', units='s')
        self.hvps_object = hvps_object
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.setInterval(100)

    def update_plot(self):
        """Update plot"""
        times, voltages = self.hvps_object.get_voltage_buffer(clear_buffer=False)
        self.voltage_plot.setData(times[-self.buffer_length:], voltages[-self.buffer_length:])

    def hideEvent(self, event):
        """End plot update when window shows"""
        self.timer.stop()
        event.accept()

    def showEvent(self, event):
        """Start plot update when window shows"""
        self.timer.start()
        event.accept()

# #Create window and run app
# if __name__ == '__main__':
#     import sys
#     APP = QtWidgets.QApplication(sys.argv)
#     APP.aboutToQuit.connect(APP.deleteLater)
#     HVPSSO = HVPS()
#     MW = HVPSWidget(HVPSSO)
#     MW.show()
#     sys.exit(APP.exec_())
