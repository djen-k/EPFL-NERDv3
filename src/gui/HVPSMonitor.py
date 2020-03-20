import time

import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import QtSerialPort

from src.hvps import NERDHVPS

if __name__ == '__main__':
    comports = QtSerialPort.QSerialPortInfo().availablePorts()
    portnames = [info.portName() for info in comports]
    port = portnames[0]
    hvps = NERDHVPS.init_hvps(port)
    hvps.set_relay_auto_mode()
    hvps.set_switching_mode(1)

    data = np.ones((1520, 3))
    idata = -1
    start = time.monotonic()

    sp = 0
    length = 20
    hvps.set_voltage(sp)
    for i in range(length):
        idata += 1
        data[idata, 0] = time.monotonic() - start
        data[idata, 1] = hvps.get_current_voltage()
        data[idata, 2] = sp

    sp = 500
    length = 1200
    # hvps.set_frequency(0.7)
    # hvps.set_switching_mode(2)
    hvps.set_voltage(sp)
    for i in range(length):
        idata += 1
        data[idata, 0] = time.monotonic() - start
        data[idata, 1] = hvps.get_current_voltage()
        data[idata, 2] = sp

    sp = 0
    length = 300
    hvps.set_voltage(sp)
    for i in range(length):
        idata += 1
        data[idata, 0] = time.monotonic() - start
        data[idata, 1] = hvps.get_current_voltage()
        data[idata, 2] = sp

    hvps.close()

    plt.plot(data[:, 0], data[:, 1:3])
    plt.xlabel('Time [s]')
    plt.ylabel('Voltage [V]')
    plt.title("Hermione SwMode1 1DEAshort-opto 500V")
    plt.show()
