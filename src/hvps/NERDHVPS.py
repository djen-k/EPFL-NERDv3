import logging

import libs.hvps as hvps
from src.hvps.Switchboard import SwitchBoard


def init_hvps(comPort):
    # instantiate and connect to HVPS
    hvpsInst = SwitchBoard()
    hvpsInfo = hvps.HvpsInfo()
    if comPort is not None:  # use com port given by user
        hvpsInfo.port = comPort
        hvpsInst.open_hvps(hvpsInfo, with_continuous_reading=False)
    else:  # if not defined, autodetect HVPS
        hvpsInst.auto_open()

    if hvpsInst.is_open:  # get HVPS name and print success message
        name = hvpsInst.get_name()
        logging.info("connected successfully to {} on {}".format(name.decode(), hvpsInst.ser.port))
    else:  # you had one job: giving the correct COM port, and you failed!
        logging.critical("Unable to connect to HVPS, ensure it's connected")
        # raise Exception

    # ensure we have a compatible HVPS, correcty connected and configured
    type = (hvpsInst.get_hvps_type() == "slave")  # until when is the term "slave" politically correct ?
    jack = (hvpsInst.get_jack_status() == 1)  # external power connector, not your friend Jack
    control_mode = (hvpsInst.get_voltage_control_mode() == 0)
    if not (type and jack and control_mode):
        logging.critical("either type, power source or control mode is not correctly set, ...")
        # raise Exception

    # ensure firmware version is compatible
    version = hvpsInst.get_firmware_version()
    if (version != 7):
        logging.critical("HVPS firmware version is: {}. Only tested with version 7".format(version))

    # ensure max voltage of the HVPS is 5kV
    max_voltage = hvpsInst.get_maximum_voltage()
    if max_voltage < 2000:
        logging.critical("Max voltage for HVPS is: {}, expected 2000".format(max_voltage))

    # select DC and output OFF
    hvpsInst.set_switching_source(0)
    hvpsInst.set_switching_mode(1)
    hvpsInst.set_output_off()
    hvpsInst.set_voltage(0)

    return hvpsInst
