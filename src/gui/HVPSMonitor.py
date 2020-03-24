import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import QtSerialPort

from src.hvps import NERDHVPS

if __name__ == '__main__':
    comports = QtSerialPort.QSerialPortInfo().availablePorts()
    portnames = [info.portName() for info in comports]
    port = portnames[0]
    hvps = NERDHVPS.init_hvps(port)

    Vset = 1000
    SwMode = 1
    relays = "off"  # "on", "off", or "auto"

    if relays == "on":
        hvps.set_relays_on()
    elif relays == "off":
        hvps.set_relays_off()
    elif relays == "auto":
        hvps.set_relay_auto_mode()

    hvps.set_switching_mode(SwMode)

    data = np.ones((1520, 3))
    idata = -1
    start = time.monotonic()

    V = 0
    length = 20
    hvps.set_voltage(V)
    for i in range(length):
        idata += 1
        data[idata, 0] = time.monotonic() - start
        data[idata, 1] = hvps.get_current_voltage()
        data[idata, 2] = V

    V = Vset
    length = 750
    # hvps.set_frequency(0.7)
    # hvps.set_switching_mode(2)
    hvps.set_voltage(V)
    print("Voltage ON")
    for i in range(length):
        idata += 1
        data[idata, 0] = time.monotonic() - start
        data[idata, 1] = hvps.get_current_voltage()
        data[idata, 2] = V

    V = 0
    length = 750
    hvps.set_voltage(V)
    print("Voltage OFF")
    for i in range(length):
        idata += 1
        data[idata, 0] = time.monotonic() - start
        data[idata, 1] = hvps.get_current_voltage()
        data[idata, 2] = V

    hvps.close()

    plt.plot(data[:, 0], data[:, 1:3])
    plt.xlabel('Time [s]')
    plt.ylabel('Voltage [V]')
    title = "Hermione SwMode{} relays-{} {}V box".format(SwMode, relays, Vset)
    print(title)
    plt.title(title)

    path = "G:/My Drive/NERD Setup/Electronics/Switchboard HVPS Debugging/"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S ")
    extension = ".png"
    fname = path + timestamp + title + extension
    plt.savefig(fname)
    print("Saved plot to ", fname)

    plt.show()
