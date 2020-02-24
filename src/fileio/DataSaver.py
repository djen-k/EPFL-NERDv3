import csv
import os
import time
import logging
from datetime import datetime


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
        # TODO: add units!
        self.dataFields = ["timestamp", "target_voltage", "measured_voltage"]
        for i in range(nbDea):
            self.dataFields.append("DEA{}_state".format(i))
            # self.dataFields.append("DEA{}_resistance".format(i))
            self.dataFields.append("DEA{}_strain_X".format(i))
            self.dataFields.append("DEA{}_strain_Y".format(i))
            self.dataFields.append("DEA{}_center_X".format(i))
            self.dataFields.append("DEA{}_center_Y".format(i))

        self.dataFields.append("imagefile")

        # init values with None
        self.dataFieldsEmptyDict = {i: None for i in self.dataFields}

        # write headers to file
        self.writer = csv.DictWriter(self.file, fieldnames=self.dataFields)
        self.writer.writeheader()
        self.logging.debug("write headers")
        self.file.flush()

    def write_data(self, timestamp, target_voltage, measured_voltage, dea_state,
                   strain=None, center_shift=None, resistance=None):

        data = self.getDictField()

        data["timestamp"] = timestamp.strftime("%d/%m/%Y %H:%M:%S")  # formatted to be interpretable by Excel
        data["target_voltage"] = target_voltage
        data["measured_voltage"] = measured_voltage
        if strain is not None:
            data["imagefile"] = timestamp.strftime("%Y%m%d-%H%M%S")  # same as in image file name, to help find it
        for i in range(len(dea_state)):
            data["DEA{}_state".format(i)] = dea_state[i]
            if resistance is not None:
                data["DEA{}_resistance".format(i)] = resistance[i]
            if strain is not None:
                data["DEA{}_strain_X".format(i)] = strain[i, 0]
                data["DEA{}_strain_Y".format(i)] = strain[i, 1]
            if center_shift is not None:
                data["DEA{}_center_X".format(i)] = center_shift[i, 0]
                data["DEA{}_center_Y".format(i)] = center_shift[i, 1]

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
