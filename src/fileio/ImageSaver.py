import logging
import os
from datetime import datetime

import cv2 as cv


class ImageSaver:
    IMAGE_DIR = "Images"
    RESULT_IMAGE_DIR = "Strain detection results"
    IMAGE_FORMAT = "png"

    def __init__(self, save_dir, active_deas, dea_labels=None, save_result_images=False):
        self.logging = logging.getLogger("ImageSaver")
        self.save_dir = save_dir
        self.active_deas = active_deas
        self.n_deas = len(active_deas)
        self.save_result_images = save_result_images
        if dea_labels is None:
            dea_labels = ["DEA {}".format(i + 1) for i in active_deas]
        else:
            assert len(dea_labels) == self.n_deas
        self.dea_labels = dea_labels
        self._dirs = ["{}/{}".format(save_dir, label) for label in dea_labels]
        self.check_dirs()  # create dirs if they don't already exist

    def check_dirs(self):
        for _dir in self._dirs:
            dir_name = "{}/{}".format(_dir, ImageSaver.IMAGE_DIR)  # path of the result image folder
            if not os.path.exists(dir_name):
                try:
                    os.makedirs(dir_name)
                    self.logging.info("Directory [{}] Created ".format(dir_name))
                except Exception as ex:
                    self.logging.critical("Failed to create image directory [{}]: {}".format(dir_name, ex))
                    raise ex
            else:
                self.logging.debug("Directory [{}] already exists".format(dir_name))
                pass

        if self.save_result_images:
            for _dir in self._dirs:
                dir_name = "{}/{}".format(_dir, ImageSaver.RESULT_IMAGE_DIR)  # path of the result image folder
                if not os.path.exists(dir_name):
                    try:
                        os.makedirs(dir_name)
                        self.logging.info("Directory [{}] Created ".format(dir_name))
                    except Exception as ex:
                        self.logging.critical("Failed to create image directory [{}]: {}".format(dir_name, ex))
                        raise ex
                else:
                    self.logging.debug("Directory [{}] already exists".format(dir_name))

    def save_all(self, images=None, timestamp=None, res_images=None, suffix=None):

        if not images:
            return  # nothing to do here

        # make sure we have a reasonable timestamp for each image
        if timestamp is None:
            timestamp = datetime.now()
        if not isinstance(timestamp, list):
            timestamp = [timestamp] * self.n_deas

        # if no images, make them into a list of Nones so they are easier to pass to the save method
        if images is None:
            images = [None] * self.n_deas

        # if no res images, make them into a list of Nones so they are easier to pass to the save method
        if res_images is None:
            res_images = [None] * self.n_deas

        # if suffix is not a list, copy it so we have a suffix for each one
        if not isinstance(suffix, list):
            suffix = [suffix] * self.n_deas

        for i in range(self.n_deas):
            self.save_image(i, images[i], timestamp[i], res_images[i], suffix[i])

        self.logging.info("Saved images for {} DEAs".format(self.n_deas))
        if res_images[0] is not None:
            self.logging.info("Saved strain detection result images for {} DEAs".format(self.n_deas))

    def save_image(self, index, image=None, timestamp=None, res_image=None, suffix=None):
        if image is None and res_image is None:
            self.logging.warning("No images to save for DEA {}".format(index))
            return

        t_string = timestamp.strftime("%Y%m%d-%H%M%S")  # get formatted time stamp to put in file name

        if suffix is not None:
            suffix = " " + suffix  # prepend space, since we only want to add it if a suffix was given

        # save original image
        if image is not None:
            fname = "{} {}{}.{}".format(t_string, self.dea_labels[index], suffix, ImageSaver.IMAGE_FORMAT)
            fpath = "{}/{}/{}".format(self._dirs[index], ImageSaver.IMAGE_DIR, fname)
            cv.imwrite(fpath, image)
        else:
            self.logging.warning("No original image supplied")

        # save result image
        if self.save_result_images:
            if res_image is not None:
                fname = "{} {}{} result.{}".format(self.dea_labels[index], t_string, suffix, ImageSaver.IMAGE_FORMAT)
                fpath = "{}/{}/{}".format(self._dirs[index], ImageSaver.RESULT_IMAGE_DIR, fname)
                cv.imwrite(fpath, res_image)
            else:
                self.logging.warning("Result image saving requested but no result image supplied")
        elif res_image is not None:
            self.logging.warning("Result image supplied even though saving of result images was not requested. "
                                 "The image was not saved")
