import logging
import os
from datetime import datetime

import cv2 as cv


class ImageSaver:
    IMAGE_DIR = "Images"
    RESULT_IMAGE_DIR = "Strain detection results"
    IMAGE_FORMAT_EXTENSION = "png"

    def __init__(self, save_dir, n_deas, dea_labels=None, save_result_images=False):
        self.logging = logging.getLogger("ImageSaver")
        self.save_dir = save_dir
        self.n_deas = n_deas
        self.save_result_images = save_result_images
        if dea_labels is None:
            dea_labels = ["DEA {}".format(i + 1) for i in range(n_deas)]
        assert len(dea_labels) == n_deas
        self.dea_labels = dea_labels
        self._dirs = ["{}/{}/{}".format(save_dir, ImageSaver.IMAGE_DIR, dea_labels[i]) for i in range(n_deas)]
        self.check_dirs()  # create dirs if they don't already exist

    def check_dirs(self):
        for dir_name in self._dirs:
            if not os.path.exists(dir_name):
                try:
                    os.makedirs(dir_name)
                    self.logging.info("Directory {} Created ")
                except Exception as ex:
                    self.logging.critical("Failed to create image directory [{}]: {}".format(dir_name, ex))
                    raise ex
            else:
                self.logging.debug("Directory [{}] already exists".format(dir_name))
                pass

        if self.save_result_images:
            for dir_name in self._dirs:
                dir_name += "/" + ImageSaver.RESULT_IMAGE_DIR  # path of the result image folder
                if not os.path.exists(dir_name):
                    try:
                        os.makedirs(dir_name)
                        self.logging.info("Directory {} Created ".format(dir_name))
                    except Exception as ex:
                        self.logging.critical("Failed to create image directory [{}]: {}".format(dir_name, ex))
                        raise ex
                else:
                    self.logging.debug("Directory [{}] already exists".format(dir_name))

    def save_all(self, images, timestamp=None, res_images=None):
        # make sure we have a reasonable timestamp for each image
        if timestamp is None:
            timestamp = datetime.now()
        if not isinstance(timestamp, list):
            timestamp = [timestamp] * self.n_deas

        # if no res images, make the into a list of Nones so they are easier to pass to the save method
        if res_images is None:
            res_images = [None] * self.n_deas

        for i in range(self.n_deas):
            self.save_image(i, images[i], timestamp[i], res_images[i])

        self.logging.info("Saved images for {} DEAs".format(self.n_deas))
        if res_images[0] is not None:
            self.logging.info("Saved strain detection result images for {} DEAs".format(self.n_deas))

    def save_image(self, index, image, timestamp: datetime, res_image=None):
        t_string = timestamp.strftime("%Y%m%d-%H%M%S")  # get formatted time stamp to put in file name

        # save original image
        fname = "{} {}.{}".format(self.dea_labels[index], t_string, ImageSaver.IMAGE_FORMAT_EXTENSION)
        fpath = "{}/{}".format(self._dirs[index], fname)
        cv.imwrite(fpath, image)

        # save result image
        if self.save_result_images:
            if res_image is not None:
                fname = "{} {} result.{}".format(self.dea_labels[index], t_string, ImageSaver.IMAGE_FORMAT_EXTENSION)
                fpath = "{}/{}/{}".format(self._dirs[index], ImageSaver.RESULT_IMAGE_DIR, fname)
                cv.imwrite(fpath, res_image)
            else:
                self.logging.warning("Result image saving requested but no result image supplied")
        elif res_image is not None:
            self.logging.warning("Result image supplied even though saving of result images was not requested. "
                                 "The image was not saved")
