import csv
import logging
import os


class DataSaver:
    def __init__(self, nbDea, outFilename="testing.csv", overwrite=False):
        """
        Constructor for DataSaver class
        :param nbDea: Number of DEAs ^m used to generate the right number of fields
        :param outFilename: filename + path of the csv file in which we'll write.
        :param overwrite: Is overwriting a file allowed or not ?
        """

        # Create logger
        self.logging = logging.getLogger("DataSaver")
        self.logging.info("Data saver instantiation: {}".format(outFilename))

        # Save parameters
        self.nbDea = nbDea

        # Open file in which we'll write data, avoid deleting data if overWrite is not true
        if os.path.isfile(outFilename) and not overwrite:
            logging.critical("File alreay exists: {}".format(outFilename))
            raise Exception
        self.file = open(outFilename, mode="w", newline='')

        # Mutex to ensure no concurrent write
        # self.mutex = QMutex()

        # create data fields and header
        self.dataFields = ["timestamp", "elapsed_time", "time_at_max_V",
                           "test_state", "target_voltage", "measured_voltage"]

        for i in range(nbDea):
            self.dataFields.append("DEA{}_electrical_state".format(i))
            self.dataFields.append("DEA{}_visual_state".format(i))
            # self.dataFields.append("DEA{}_resistance".format(i))
            self.dataFields.append("DEA{}_strain_X".format(i))
            self.dataFields.append("DEA{}_strain_Y".format(i))
            self.dataFields.append("DEA{}_strain_area".format(i))
            self.dataFields.append("DEA{}_center_X".format(i))
            self.dataFields.append("DEA{}_center_Y".format(i))

        self.dataFields.append("imagefile")

        # define units

        self.units = ["[datetime]", "[s]", "[s]", "[0: Ramp, 1: High, 2: Low]", "[V]", "[V]"]
        for i in range(nbDea):
            self.units.append("[1: OK, 0: Failed]")
            self.units.append("[1: OK, 0: Failed]")
            # self.units.append("[Ohm]")
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

    def write_data(self, timestamp, elapsed_time, time_at_max_V, test_state, target_voltage, measured_voltage,
                   electrical_state=None, visual_state=None, strain_XYa=None, center_shift=None, resistance=None,
                   image_saved=False):

        data = self.getDictField()

        data["timestamp"] = timestamp.strftime("%d/%m/%Y %H:%M:%S")  # formatted to be interpretable by Excel
        data["elapsed_time"] = elapsed_time
        data["time_at_max_V"] = time_at_max_V
        data["test_state"] = test_state
        data["target_voltage"] = target_voltage
        data["measured_voltage"] = measured_voltage

        # TODO: create option to choose channels (e.g. use only channels 1, 2, 5, 6)
        for i in range(self.nbDea):
            if visual_state is not None and i < len(visual_state):
                data["DEA{}_electrical_state".format(i)] = visual_state[i]
            if electrical_state is not None and i < len(electrical_state):
                data["DEA{}_visual_state".format(i)] = electrical_state[i]
            if resistance is not None and i < len(resistance):
                data["DEA{}_resistance".format(i)] = resistance[i]
            if strain_XYa is not None and i < strain_XYa.shape[0]:
                data["DEA{}_strain_X".format(i)] = strain_XYa[i, 0]
                data["DEA{}_strain_Y".format(i)] = strain_XYa[i, 1]
                data["DEA{}_strain_area".format(i)] = strain_XYa[i, 2]
            if center_shift is not None and i < center_shift.shape[0]:
                data["DEA{}_center_X".format(i)] = center_shift[i, 0]
                data["DEA{}_center_Y".format(i)] = center_shift[i, 1]

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
