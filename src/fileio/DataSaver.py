import csv
import logging
import os


class DataSaver:
    def __init__(self, active_deas, outFilename="testing.csv", overwrite=False):
        """
        Constructor for DataSaver class
        :param active_deas: Number of DEAs;  used to generate the right number of fields
        :param outFilename: filename + path of the csv file in which we'll write.
        :param overwrite: Is overwriting a file allowed or not ?
        """

        # Create logger
        self.logging = logging.getLogger("DataSaver")
        self.logging.info("Data saver instantiation: {}".format(outFilename))

        # Save parameters
        self.active_deas = active_deas

        # Open file in which we'll write data, avoid deleting data if overWrite is not true
        if os.path.isfile(outFilename) and not overwrite:
            logging.critical("File alreay exists: {}".format(outFilename))
            raise Exception
        self.file = open(outFilename, mode="w", newline='')

        # Mutex to ensure no concurrent write
        # self.mutex = QMutex()

        # create data fields and header
        self.dataFields = ["timestamp", "elapsed_time", "time_at_max_V", "total_cycles",
                           "test_state", "target_voltage", "measured_voltage",
                           "leakage_current", "partial_discharge_freq", "partial_discharge_mag"]

        for i in active_deas:
            i += 1  # switch to 1-based indexing for user output
            self.dataFields.append("DEA{}_electrical_state".format(i))
            self.dataFields.append("DEA{}_visual_state".format(i))
            self.dataFields.append("DEA{}_resistance".format(i))
            self.dataFields.append("DEA{}_V_source".format(i))
            self.dataFields.append("DEA{}_V_shunt".format(i))
            self.dataFields.append("DEA{}_V_DEA".format(i))
            self.dataFields.append("DEA{}_strain_area".format(i))
            self.dataFields.append("DEA{}_strain_X".format(i))
            self.dataFields.append("DEA{}_strain_Y".format(i))
            self.dataFields.append("DEA{}_strain_average".format(i))
            self.dataFields.append("DEA{}_center_X".format(i))
            self.dataFields.append("DEA{}_center_Y".format(i))

        self.dataFields.append("imagefile")

        # define units

        self.units = ["[datetime]", "[s]", "[s]", "", "[0: Ramp, 1: High, 2: Low]", "[V]", "[V]", "[A]", "[1/s]", "[A]"]
        for i in active_deas:
            self.units.append("[1: OK, 0: Failed]")
            self.units.append("[1: OK, 0: Failed]")
            self.units.append("[Ohm]")
            self.units.append("[V]")
            self.units.append("[V]")
            self.units.append("[V]")
            self.units.append("[%]")
            self.units.append("[%]")
            self.units.append("[%]")
            self.units.append("[%]")
            self.units.append("[pixel]")
            self.units.append("[pixel]")

        self.units.append("[string]")

        # init values with None
        self.dataFieldsEmptyDict = {i: None for i in self.dataFields}

        # write headers and units to file
        self.writer = csv.DictWriter(self.file, fieldnames=self.dataFields)
        self.writer.writeheader()
        units_header = dict(zip(self.dataFields, self.units))
        self.writer.writerow(units_header)
        self.logging.debug("write headers")
        self.file.flush()

    def write_data(self, timestamp, elapsed_time, time_at_max_V, total_cycles, test_state, target_voltage,
                   measured_voltage,
                   electrical_state=None, visual_state=None, strain_AXYa=None, center_shift=None,
                   resistance=None, V_source=None, V_shunt=None, V_DEA=None,
                   leakage_current=None, pd_freq=None, pd_mag=None, image_saved=False):

        data = self.getDictField()

        data["timestamp"] = timestamp.strftime("%d/%m/%Y %H:%M:%S")  # formatted to be interpretable by Excel
        data["elapsed_time"] = elapsed_time
        data["time_at_max_V"] = time_at_max_V
        data["total_cycles"] = total_cycles
        data["test_state"] = test_state
        data["target_voltage"] = target_voltage
        data["measured_voltage"] = measured_voltage

        if leakage_current is not None:
            data["leakage_current"] = leakage_current
        if leakage_current is not None:
            data["partial_discharge_freq"] = pd_freq
        if leakage_current is not None:
            data["partial_discharge_mag"] = pd_mag

        for i in range(len(self.active_deas)):
            disp_id = self.active_deas[i] + 1
            if electrical_state is not None and i < len(electrical_state):
                data["DEA{}_electrical_state".format(disp_id)] = electrical_state[i]
            if visual_state is not None and i < len(visual_state):
                data["DEA{}_visual_state".format(disp_id)] = visual_state[i]
            if resistance is not None and i < len(resistance):
                data["DEA{}_resistance".format(disp_id)] = resistance[i]
            if V_source is not None and i < len(V_source):
                data["DEA{}_V_source".format(disp_id)] = V_source[i]
            if V_shunt is not None and i < len(V_shunt):
                data["DEA{}_V_shunt".format(disp_id)] = V_shunt[i]
            if V_DEA is not None and i < len(V_DEA):
                data["DEA{}_V_DEA".format(disp_id)] = V_DEA[i]
            if strain_AXYa is not None and i < strain_AXYa.shape[0]:
                data["DEA{}_strain_area".format(disp_id)] = strain_AXYa[i, 0]
                data["DEA{}_strain_X".format(disp_id)] = strain_AXYa[i, 1]
                data["DEA{}_strain_Y".format(disp_id)] = strain_AXYa[i, 2]
                data["DEA{}_strain_average".format(disp_id)] = strain_AXYa[i, 3]
            if center_shift is not None and i < center_shift.shape[0]:
                data["DEA{}_center_X".format(disp_id)] = center_shift[i, 0]
                data["DEA{}_center_Y".format(disp_id)] = center_shift[i, 1]

        if image_saved:
            data["imagefile"] = timestamp.strftime("%Y%m%d-%H%M%S")  # same as in image file name, to help find it

        self.writeLine(data)

    def getDictField(self):
        """
        Return an empty dict to be filled with data to write
        :return: empty dict
        """
        return self.dataFieldsEmptyDict.copy()

    def writeLine(self, data: dict):
        """
        Write data into file.
        :param data: dict, with the same fields as returned by self.getDictField()
        :return: nothing
        """

        if None in data.values():
            self.logging.debug("Not all data fields provided!")

        # self.mutex.lock() # acquire mutex before writing, can be blocking
        self.writer.writerow(data)
        self.file.flush()
        self.logging.debug("New line written: {}".format(data))
        # self.mutex.unlock() # release mutex

    def mergeLines(self):
        self.logging.critical("Not implemented!")

    def close(self):
        logging.debug("Closing datasaver")
        self.file.close()
