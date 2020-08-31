import logging
import time
from concurrent import futures
from datetime import datetime
from threading import Thread, Event, Lock

import cv2 as cv
import numpy as np

from src.image_processing import StrainDetection


class Camera:

    def __init__(self, index, desired_resolution=None, desired_exposure=None, new_image_callback=None):
        self.logging = logging.getLogger("Camera")

        # open image capture
        self.cap = cv.VideoCapture(index + cv.CAP_MSMF)  # use Windows Media Foundation backend. No others work!
        # set properties
        if desired_resolution is not None:
            self.set_resolution(desired_resolution)
        if desired_exposure is not None:
            self.set_exposure(desired_exposure)

        self.access_lock = Lock()  # to prevent accessing the same camera from multiple threads

        # init buffers
        self.name = "Camera {}".format(index + 1)  # use 1-based indexing for natural names
        self.image_buffer = None
        self.grab_timestamp = None  # timestamp for the last grab
        self.image_timestamp = None  # timestamp for image in buffer. May differ from grab_timestamp if retrieve failed
        self.grab_ok = False

        # callback
        self.new_image_callback = new_image_callback

    def set_resolution(self, desired_resolution):
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, desired_resolution[0])
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, desired_resolution[1])

    def set_exposure(self, desired_exposure):
        self.cap.set(cv.CAP_PROP_EXPOSURE, desired_exposure)

    def grab(self):
        """
        Grab a frame from this camera. The frame is not retrieved at this point.
        :return: a flag indicating success or failure
        """
        with self.access_lock:
            if not self.cap.isOpened():  # just to be sure. this should never be the case, theoretically
                self.logging.debug("{} is not open".format(self.name))
                return

            suc = self.cap.grab()  # call grab to quickly obtain a frame, which can be decoded with "retrieve()" later
            if suc:
                self.grab_timestamp = datetime.now()  # record timestamp of the grabbed image
            else:
                self.logging.warning("Unable to grab frame from {}".format(self.name))

            self.grab_ok = suc
        return suc

    def retrieve(self):
        """
        Retrieve a frame from the specified camera and store it in the buffer
        :return: a flag indicating success or failure, and the retrieved frame (or None, if unsuccessful)
        """
        with self.access_lock:
            if not self.cap.isOpened():  # just to be sure. this should never be the case, theoretically
                self.logging.debug("{} is not open".format(self.name))
                return False, None

            if self.grab_ok:  # check if previous grab was successful
                suc, frame = self.cap.retrieve()  # retrieve frame previously captured by "grab()"

                if suc:
                    self.image_buffer = frame.copy()  # store copy in buffer so we cn return this instance
                    self.image_timestamp = self.grab_timestamp  # update the timestamp

                    self.logging.debug("Retrieved frame from {}. Size: {}".format(self.name, frame.shape))

                    # if callback is defined, call callback function
                    if self.new_image_callback is not None:
                        self.logging.debug("Calling new image callback for {}".format(self.name))
                        self.new_image_callback(frame, self.image_timestamp, self.name)

                    return True, frame
                else:
                    self.logging.warning("Unable to retrieve frame from {}".format(self.name))

            else:  # previous grab was not successful
                self.logging.warning("Last grab from {} was unsuccessful. No image was retrieved".format(self.name))

            self.image_buffer = None  # clear buffer to indicate no current image is available
        return False, None

    def read(self):
        """
        Grab and retrieve an image from this camera
        :return: a flag indicating success or failure, and the retrieved image (or None, in case of failure)
        """
        if self.grab():
            return self.retrieve()
        else:
            return False, None

    def get_image_from_buffer(self):
        """
        Return the last image of this camera from the buffer. No new image is captured.
        :return: The last image captured by this camera
        """
        return self.image_buffer.copy()

    def isOpened(self):
        # TODO: maybe take into account grab_ok?
        return self.cap.isOpened()

    def release(self):
        self.cap.release()


class ImageCapture:
    IMG_NOT_AVAILABLE = cv.imread("res/images/no_image.png")
    IMG_WAITING = cv.imread("res/images/waiting.jpeg")

    def __init__(self):
        self.logging = logging.getLogger("ImageCapture")

        # image capture settings
        self.desired_resolution = [1920, 1080]
        self.auto_reconnect = True
        self.max_reconnect_attempts = 50
        self.max_fps = 0  # unlimited
        self.exposure = None

        # internal variables
        self._cameras = []
        self._camera_count = -1  # not yet initialized
        self._camera_names = []
        self._camera_states = []  # to indicate if a camera was lost (no longer able to grab images)
        self._cameras_available = []
        self._camera_selection = None
        self._reconnect_attempt = 0

        # threading
        self._exit_flag = None
        self._capture_thread = None

        # callbacks
        self._new_image_callback = None
        self.new_set_callback = None

    def __del__(self):
        self.stop_capture_thread(block=True)  # wait to make sure the capture thread stopped before closing cams
        self.close_cameras()

    def _reset(self):
        self.stop_capture_thread(block=True)  # wait to make sure the capture thread stopped before closing cams
        self.close_cameras()
        self._cameras = []
        self._camera_count = -1
        self._camera_names = []
        self._camera_states = []
        self._cameras_available = []
        self._camera_selection = None

    def set_new_image_callback(self, new_image_callback):
        """
        Set a function to call every time a new image becomes available.
        :param new_image_callback: Callback function of prototype: func(image, timestamp, camera_id)
        """
        self._new_image_callback = new_image_callback
        for cam in self._cameras:
            cam.new_image_callback = new_image_callback

    def set_new_set_callback(self, new_set_callback):
        """
        Set a function to call every time a new set of images becomes available (i.e. one new image from each camera)
        :param new_set_callback: Callback function of prototype: func(images -> list, timestamps -> list)
        """
        self.new_set_callback = new_set_callback

    def get_camera_count(self):
        """
        Get the number of physical cameras available
        :return: The number of available cameras, or number of selected cameras if a selection has been made,
        """
        if self._camera_count == -1:  # not initialized yet  --> go find cameras
            self.logging.info("ImageCapture not yet initialized. Initializing cameras...")
            self.find_cameras()

        return self._camera_count

    def _get_names_from_cameras(self):
        """
        Assemble a list of the names of the available/selected cameras
        :return: A list of names of the available cameras
        """
        # TODO: deal with startup where names have not yet been assigned
        # if self._camera_count == -1:  # not initialized yet  --> go find cameras
        #     self.logging.info("ImageCapture not yet initialized. Initializing cameras...")
        #     self.find_cameras()

        return [cam.name for cam in self._cameras]

    def get_camera_names(self):
        """
        Get the names of the available/selected cameras
        :return: The names of the available cameras
        """
        # TODO: deal with startup where names have not yet been assigned
        return self._get_names_from_cameras()

    def set_camera_names(self, names):
        """
        Set the names for all cameras.
        :param names: The new name to assign to the camera
        """
        if len(names) != self._camera_count:
            self.logging.warning("Need to supply a name for each camera! No names were changed!")
            return
        else:
            for i in range(self._camera_count):
                cam = self._cameras[i]
                self.logging.info("Changing name '{}' to '{}'".format(cam.name, names[i]))
                cam.name = names[i]
            self._camera_names = names

    def get_camera_property(self, cam_property):
        return [cam.cap.get(cam_property) for cam in self._cameras]

    def set_camera_property(self, cam_property, values):
        """
        Set the given property for all active cameras.
        :param cam_property: The property to set (cv2.CAP_PROP_...)
        :param values: The values to set. Can be a scalar or list of the same length as the number of cameras.
        """
        if np.isscalar(values):
            values = [values] * self._camera_count
        else:
            if len(values) != self._camera_count:
                raise ValueError("Values must be a scalar or list with the same length as the number of cameras!")

        for cam, val in zip(self._cameras, values):
            cam.cap.set(cam_property, val)

    def get_camera(self, cam_id):
        """
        Get camera by name or index
        :param cam_id: The name or index of the camera
        :return: The camera with the given name or index, or None if no camera with that name exists
        """
        if type(cam_id) is str:
            names = self.get_camera_names()
            try:
                cid = names.index(cam_id)
                return self._cameras[cid]
            except ValueError:
                self.logging.warning("No camera with name {} was found".format(cam_id))
                return None
        elif type(cam_id) is int:
            if 0 <= cam_id < self._camera_count:
                return self._cameras[cam_id]
            else:
                self.logging.warning("No camera with index {} available".format(cam_id))
                return None
        else:
            msg = "Not a valid camera id: {} (type {}). Expected string or int!"
            self.logging.warning(msg.format(cam_id, type(cam_id)))
            return None

    def find_cameras(self, parallel=True):
        self._reset()  # release all cams and empty buffers

        n_expected = 13  # in case there are 2 x 6 cams + one built-in web cam or something. Shouldn't ever be more.

        if parallel is True:
            cams_found = {}
            max_index = 0
            # Use executor in with statement to ensure threads are cleaned up promptly
            with futures.ThreadPoolExecutor(max_workers=n_expected) as executor:
                # Start the load operations and mark each future with its index
                future_to_cam = {executor.submit(self._test_camera, index): index for index in range(n_expected)}
                self.logging.debug("Detecting cameras in parallel. Submitted tasks to worker threads.")
                for future in futures.as_completed(future_to_cam):
                    index = future_to_cam[future]
                    try:
                        cam = future.result()
                    except Exception as exc:
                        self.logging.debug('Camera {} generated an exception: {}'.format(index, exc))
                        cam = None

                    if cam is not None:  # either no camera found at this index or some error occurred
                        self.logging.debug('Camera {} is running'.format(index))
                        cams_found[index] = cam
                        max_index = max(max_index, index)

            cams_list = []
            for i in range(max_index+1):
                if i in cams_found.keys():
                    cams_list.append(cams_found[i])
        else:  # don't use threading
            self.logging.debug("Sequential camera detection (no multi-threading)")
            cams_list = []
            for index in range(n_expected):
                cam = self._test_camera(index)
                if cam is not None:
                    cams_list.append(cam)

        self._cameras = cams_list
        self._cameras_available = self._cameras  # store list of all available cameras.

        self._camera_count = len(self._cameras)
        self._camera_names = self._get_names_from_cameras()

        self.logging.info("{} cameras found".format(self._camera_count))

        # discard the first few sets of images. They tend to be rubbish
        for i in range(10):
            time.sleep(0.1)
            self.grab_images()  # no need to retrieve, we just want the cameras to get warmed up (adjust exposure, etc.)

        # call new set callback since the first full set of images is now available
        if self.new_set_callback is not None:
            self.new_set_callback(self.get_images_from_buffer(), self.get_timestamps())

    def _test_camera(self, index):
        """
        Try to open camera and retrieve an image. If successful, return the camera, otherwise None.
        :param index: Index of the camera to try and open
        :return: The camera, if opened successfully.
        """

        cam = Camera(index, self.desired_resolution, self.exposure, self._new_image_callback)

        if not cam.isOpened():  # a camera with this index doesn't exist or it is in use
            self.logging.debug("Could not open {} (index {})".format(cam.name, index))
            return None

        self.logging.debug("{} opened (index {})".format(cam.name, index))
        success, frame = cam.read()  # check if we can read an image
        if success:
            return cam
        else:
            msg = "Unable to retrieve image from {} (index {}). Camera will be closed."
            self.logging.warning(msg.format(cam.name, index))
            cam.release()
            return None

    def reconnect_cameras(self):
        msg = "Cameras lost connection"
        self.logging.info(msg)
        logging.getLogger("Disruption").info(msg)

        if not self.auto_reconnect:
            self.logging.critical("Connection to cameras lost and auto reconnect is disabled. Shutting down...")
            raise Exception("Connection to cameras lost")

        prev_count = self._camera_count
        prev_available = len(self._cameras_available)
        prev_selection = self._camera_selection
        prev_names = self._camera_names
        state = self._camera_states

        while self._reconnect_attempt < self.max_reconnect_attempts:  # don't keep reconnecting forever
            self._reconnect_attempt += 1
            self.logging.info("Reconnecting cameras (attempt {})...".format(self._reconnect_attempt))

            self.close_cameras()
            time.sleep(5)  # wait for cams to close properly before we try to reconnect. Might cause trouble otherwise.

            self.find_cameras()
            new_count = self._camera_count

            cams_lost = prev_available - new_count
            cams_failed = state.count(False)

            if cams_lost is 0:
                msg = "Cameras successfully reconnected after {} attempts".format(self._reconnect_attempt)
                self.logging.info(msg)
                logging.getLogger("Disruption").info(msg)  # also log to separate disruption log file

                if prev_selection is not None:
                    self.select_cameras(prev_selection)
                    self.set_camera_names(prev_names)
                self._reconnect_attempt = 0  # reset attempt count so we start again from 0 next time cameras are lost
                return
            elif cams_lost == cams_failed:  # we can match the failure to the lost camera
                # TODO: check if this even works...
                i_failed = [i for i, x in enumerate(state) if x is False]
                for i in i_failed:
                    self._cameras.insert(i, Camera(100))  # insert dummy camera. safe to assume camera 100 doesn't exist
            else:
                self.logging.info("Not all cameras reconnected. Try again...")
                # TODO: deal with unknown cameras

        # if we reached here, all attempts have failed
        self.logging.critical("Failed to reconnect cameras. Shutting down...")
        raise Exception("Connection to cameras lost")

    def select_cameras(self, camera_ids):

        if self._camera_count == -1:  # not initialized --> find cameras
            self.find_cameras()

        cams = []
        # select cameras in the specified order.
        for cid in camera_ids:
            # Raise error if a requested camera is not available
            if cid >= self._camera_count:
                self.logging.critical("Invalid camera selected: camera {} is not available!".format(cid))
                raise ValueError("Invalid camera selection!")
            elif cid >= 0:  # if -1, the camera is not used so we don't need to do anything
                cams.append(self._cameras_available[cid])

        # release all cameras that are not being used
        for i in range(self._camera_count):
            if i not in camera_ids:
                self._cameras_available[i].release()

        self._cameras = cams
        self._camera_count = len(cams)
        self._camera_names = self._get_names_from_cameras()
        self._camera_selection = camera_ids  # store so we can reconstruct the selection when reconnecting

    def grab_images(self):
        """
        Grab a frame from each selected camera. Only grab is performed and no image is retrieved at this point.
        This allows capturing of images from all cameras as close as possible in time.
        Captured images can be loaded with "retrieve_images" later.
        :return: a list of flags to indicate if each grab succeeded
        """
        self._camera_states = [cam.grab() for cam in self._cameras]
        if False in self._camera_states:
            self.logging.warning("Unable to grab images")
            self.reconnect_cameras()

        return self._camera_states

    def retrieve_images(self):
        """
        Retrieve a frame from each selected camera and store in buffer.
        :return: a list of flags indicating success, and a list of the retrieved frames
        """
        res = [cam.retrieve() for cam in self._cameras]
        # turn list of tuples [(suc, frame), ...] into tuple of lists ([suc, ...], [frame, ...])
        res = [list(o) for o in zip(*res)]
        if len(res) == 0:
            res = ([], [])  # make sure two empty lists are returned if no cameras are found to match the expected type
        return res

    def read_images(self):
        """
        Read an image from each selected camera. First, a frame from each camera is grabbed and later retrieved so as to
        ensure that frames are as close to each other in time as possible. The images are stored in the internal buffer.
        """
        # check if is initialized
        if self._camera_count == -1:  # not initialized yet  --> go find cameras
            self.logging.info("ImageCapture not yet initialized. Initializing cameras...")
            self.find_cameras()

        self.grab_images()
        sucs, frames = self.retrieve_images()

        if self.new_set_callback is not None:
            self.new_set_callback(frames, self.get_timestamps())

        return frames

    def read_single_image(self, cam_id):
        """
        Read a single image from the specified camera
        :param cam_id: A camera identifier which can be either a camera name string or an index
        :return: a newly retrieved image from the specified camera
        """
        # check if is initialized
        if self._camera_count == -1:  # not initialized yet  --> go find cameras
            self.logging.info("ImageCapture not yet initialized. Initializing cameras...")
            self.find_cameras()

        cam = self.get_camera(cam_id)
        if cam is not None:
            self._camera_states[self._cameras.index(cam)] = cam.grab()
            suc, frame = cam.retrieve()
            if suc:
                return frame

        return None  # in case anything failed

    def read_average_images(self, n, deviation_threshold=0.03):
        """
        Read several images and average them (for each camera) to get a set of images with reduced noise.
        If any image in the set of images to be averaged deviates too much from the others, it is discarded and a new image
        is recorded until the desired number of images is available for averaging.
        :param int n: How many images to average
        :param float deviation_threshold: Image deviation threshold above which an image is excluded from the average,
        given as a value between 0 and 1 (e.g. 0.1 -> image is discarded if deviation if more than 10%)
        :return: A set of averaged images
        """
        # get set of n images from each camera
        image_sets = []
        for i in range(n):
            img_set = self.read_images()
            # check for outliers (glitched images)
            if i > 0:  # nothing to compare with just the first image
                # check for new image from each camera
                for c in range(self._camera_count):
                    deviation = StrainDetection.get_image_deviation(image_sets[0][c], img_set[c])
                    new_image_count = 0
                    new_reference_count = 0
                    while deviation > deviation_threshold:
                        if new_image_count > 5:  # make sure we don't get stuck trying indefinitely
                            if new_reference_count > 0:  # make sure we don't get stuck trying indefinitely
                                self.logging.debug("Failed to record matching image for new reference "
                                                   "for camera {}. Aborting.".format(c))
                                img_set[c] = None
                                break
                            self.logging.debug("Failed to record matching images for averaging "
                                               "for camera {}. Using new reference".format(c))
                            image_sets[0][c] = self.read_single_image(c)
                            new_reference_count = + 1
                            new_image_count = 0  # reset count
                        else:
                            self.logging.debug("Image averaging: Outlier detected "
                                               "in image {} from camera {}. Recording new image.".format(i, c))
                        img_set[c] = self.read_single_image(c)
                        deviation = StrainDetection.get_image_deviation(image_sets[0][c], img_set[c])
                        new_image_count += 1
            image_sets.append(img_set)

        # transpose so we have one list of n images for each camera (instead of n lists of one image per camera)
        image_sets = list(map(list, zip(*image_sets)))

        avgs = []
        for i in range(self._camera_count):  # iterate over number of cams. there must be one set for each cam
            img_set = image_sets[i]
            # remove Nones (where not matching image could be recorded even after trying real hard)
            img_set = [img for img in img_set if img is not None]
            if len(img_set) < n:
                msg = "Only {} matching images from {} could be recorded for averaging (instead of {})."
                self.logging.warning(msg.format(len(img_set), self._camera_names[i], n))
            img_set = np.array(img_set)
            avg = np.mean(img_set, axis=0).round().astype(np.uint8)
            avgs.append(avg)

        return avgs

    def get_images_from_buffer(self):
        """
        Get the last image of each selected camera from the image buffer
        :return: A list of image from the buffer
        """
        return [cam.get_image_from_buffer() for cam in self._cameras]

    def get_timestamps(self):
        """
        Return a list containing a timestamps for the most recent image of each selected camera
        :return: A timestamp for each image in the buffer
        """
        return [cam.image_timestamp for cam in self._cameras]  # get timestamp of last image from each camera

    def close_cameras(self):
        """
        Close all cameras
        """
        for cam in self._cameras:
            cam.release()

    def start_capture_thread(self, max_fps=0):
        """
        Starts a thread for continuous image capture. This will grab and retrieve images until stop_capture_thread is
        called. Each time an image was captured, the new image callback is executed.
        :param max_fps: The maximum rate (in frames per second) at which to capture new images. The real framerate may
        be lower than this value but never higher. Set to 0 for maximum possible framerate.
        """
        if self._capture_thread is not None and self._capture_thread.is_alive():
            logging.debug("Capture thread is already running.")
            return

        logging.info("Starting image capture thread")
        if self._new_image_callback is None and self.new_set_callback is None:
            logging.warning("No callback is set!")

        self._exit_flag = Event()
        self.max_fps = max_fps
        self._capture_thread = Thread()  # ImageCaptureThread(self, max_fps, self._exit_flag)
        self._capture_thread.run = self.run_capture_thread
        self._capture_thread.start()

    def stop_capture_thread(self, block=False):
        if self._capture_thread is None:
            logging.debug("Capture thread was not running")
            return
        else:
            logging.debug("Stopping capture thread")
            self._exit_flag.set()  # set flag to indicate capture thread to terminate
            # if required by user, block until the thread has terminated
            if block:
                self._capture_thread.join()

    def is_capture_thread_running(self):
        if self._capture_thread is None:
            return False
        return self._capture_thread.is_alive()

    def run_capture_thread(self):
        if self.max_fps > 0:
            min_delay = 1 / self.max_fps
        else:
            min_delay = 0  # no minimum delay -> capture at maximum fps

        try:
            delay = 0  # initial delay is 0
            while not self._exit_flag.wait(delay):
                t_start = time.monotonic()
                self.read_images()
                t_end = time.monotonic()
                elapsed_time = t_end - t_start
                delay = min_delay - elapsed_time - 0.01  # get remaining time to wait (subtract 10 ms for overhead)
        except Exception as ex:
            self.logging.error("Exception in capture thread: {}".format(ex))

    def set_fixed_exposure(self, exposure):
        """
        Set exposure of all cameras to the specified fixed value
        :param exposure: Exposure value in range [0 -12]
        """
        self.exposure = exposure
        for cam in self._cameras:
            cam.set_exposure(exposure)

    def set_fixed_exposure_auto(self):
        """
        Enable auto exposure on all selected cameras to find the best exposure, then sets these value as the fixed
        exposure for each camera. Because openCV does not return the actual value when auto exposure is enabled,
        the value needs to be found by trying different values and comparing the resulting images against a reference
        image taken with auto exposure.
        :return: The exposure of each camera, as determined by auto exposure.
        """
        # check if initialized
        if self._camera_count == -1:  # not initialized yet  --> go find cameras
            self.logging.info("ImageCapture not yet initialized. Initializing cameras...")
            self.find_cameras()

        self.logging.debug("Determining best exposure level using autp exposure...")

        self.set_camera_property(cv.CAP_PROP_AUTO_EXPOSURE, 1)  # turn auto exposure on
        time.sleep(3)  # wait for auto exposure to settle (at least 2.2 s!)
        refs = self.read_images()  # record set of images with auto exposure
        # for i in range(len(refs)):
        #     cv.imshow('Camera {} - press [q] to exit'.format(i), cv.resize(refs[i], (0, 0), fx=0.25, fy=0.25))
        # res = cv.waitKey(0)

        self.set_camera_property(cv.CAP_PROP_AUTO_EXPOSURE, 0)  # turn auto exposure off
        exposures = np.ones(self._camera_count) * -13
        increment = np.ones(self._camera_count)
        prev_deviation = np.ones(self._camera_count) * np.inf
        while np.any(increment != 0):
            exposures += increment
            self.set_camera_property(cv.CAP_PROP_EXPOSURE, exposures)  # set fixed exposure
            time.sleep(0.3)  # wait for cameras to apply settings (at least 200 ms!)
            imgs = self.read_images()
            new_deviation = np.array([StrainDetection.get_image_deviation(img, ref) for img, ref in zip(imgs, refs)])
            self.logging.debug("Deviation: {}".format(new_deviation))
            i_dev_inc = new_deviation > prev_deviation  # get cam indices where the deviation has started increasing
            exposures[i_dev_inc] -= increment[i_dev_inc]  # undo increment to return to exposure with lowest deviation
            increment[i_dev_inc] = 0  # stop incrementing if deviation has started increasing
            increment[exposures == 0] = 0  # stop incrementing if the maximum exposure (0) is reached

            prev_deviation = new_deviation

        self.set_camera_property(cv.CAP_PROP_EXPOSURE, exposures)  # set best exposure
        time.sleep(0.5)  # wait a little while so the cameras can adjust their exposure before the next image is grabbed
        self.logging.debug("Exposure levels fixed at {}".format(exposures))
        return exposures

    def run_test(self):
        res = 0
        # todo: implement proper resource management so files can be accessed consistently from anywhere
        no_img = cv.imread("../../res/images/no_image.png")
        waiting_img = cv.imread("../../res/images/waiting.jpeg")
        cv.imshow('Camera 0', cv.resize(waiting_img, (0, 0), fx=1, fy=1))

        self.find_cameras()
        self.select_cameras([0])
        origs = self.read_images()
        cv.imshow('Camera 0', cv.resize(origs[0], (0, 0), fx=0.5, fy=0.5))
        res = cv.waitKey(500)

        cap0 = self.get_camera(0).cap
        cap0.set(cv.CAP_PROP_AUTO_EXPOSURE, 1)
        start = time.monotonic()
        # time.sleep(2.5)  # wait for auto exposure to settle (at least 2.2 s!)
        for i in range(50):
            imgs = self.read_images()
            print("elapsed:", time.monotonic() - start)
            print("dev:", StrainDetection.get_image_deviation(imgs[0], origs[0]))
            cv.imshow('Camera 0', cv.resize(imgs[0], (0, 0), fx=0.5, fy=0.5))
            res = cv.waitKey(10)
        res = cv.waitKey(0)
        origs = self.read_images()
        cap0.set(cv.CAP_PROP_AUTO_EXPOSURE, 0)
        # time.sleep(2.5)  # wait for auto exposure to settle (at least 2.2 s!)
        for i in range(50):
            imgs = self.read_images()
            print("elapsed:", time.monotonic() - start)
            print("dev:", StrainDetection.get_image_deviation(imgs[0], origs[0]))
            cv.imshow('Camera 0', cv.resize(imgs[0], (0, 0), fx=0.5, fy=0.5))
            res = cv.waitKey(10)
        res = cv.waitKey(0)
        exposure = 1
        mindev = np.inf
        minexp = 0
        while exposure > -13:
            exposure -= 1
            print("set exposure to", exposure)
            cap0.set(cv.CAP_PROP_EXPOSURE, exposure)
            print("exposure:", cap0.get(cv.CAP_PROP_EXPOSURE))
            print("auto wb:", cap0.get(cv.CAP_PROP_AUTO_WB))
            print("wb temp:", cap0.get(cv.CAP_PROP_WB_TEMPERATURE))
            print("brightness:", cap0.get(cv.CAP_PROP_BRIGHTNESS))
            time.sleep(0.3)  # wait for camera to apply settings (at least 200 ms!)
            dev = 0
            for i in range(5):
                imgs = self.read_images()
                dev = StrainDetection.get_image_deviation(origs[0], imgs[0])
                print(dev)
                cv.imshow('Camera 0', cv.resize(imgs[0], (0, 0), fx=0.5, fy=0.5))
                res = cv.waitKey(1)

            print("Deviation:", dev)
            if dev < mindev:
                mindev = dev
                minexp = exposure

        print("Done!")
        print("set exposure to", minexp)
        cap0.set(cv.CAP_PROP_EXPOSURE, minexp)
        imgs = self.read_images()

        for i in range(5):  # len(imgs)):
            imgs = self.read_images()
            cv.imshow('Camera 0', cv.resize(imgs[0], (0, 0), fx=0.5, fy=0.5))
            res = cv.waitKey(500)

        cap0.set(cv.CAP_PROP_EXPOSURE, 0)
        res = cv.waitKey(0)


SharedInstance = ImageCapture()

if __name__ == '__main__':
    # duallog.setup("camera test logs", minlevelConsole=logging.DEBUG, minLevelFile=logging.DEBUG)
    logging.basicConfig(level=logging.DEBUG)
    logging.info("Testing image capture")

    # cap = SharedInstance
    # cap.find_cameras()
    ImageCapture().run_test()
    # _avg = cap.read_average_images(10)
    # for _img in _avg:
    #     cv.imshow("Image", _img)
    #     cv.waitKey()
    # cap.select_cameras([0])
    #
    # # print("exposure:", cap.set_fixed_exposure())
    #
    # times = []
    # frames = []
    # lock = Lock()
    # timestamp_start = datetime.now()  # record reference time
    #
    #
    # def new_frame(image, timestamp, camera_id):
    #     with lock:
    #         frames.append(image)
    #         times.append((timestamp - timestamp_start).total_seconds())
    #         print("received frame {}".format(len(frames)))
    #
    #
    # cap.set_new_image_callback(new_frame)
    # cap.start_capture_thread(max_fps=0)
    # timer = Timer(10, cap.stop_capture_thread)
    # timer.start()
    #
    # frame_index = 0
    # time.sleep(11)
    # # while cap.is_capture_thread_running():
    # #     with lock:
    # #         if len(frames) > frame_index:
    # #             cv.imshow("Current Frame", frames[-1])
    # #             cv.waitKey(1)  # refresh window and check for user input
    # #             frame_index = len(frames)
    #
    # elapsed_time_s = (datetime.now() - timestamp_start).total_seconds()
    # cap.close_cameras()
    #
    # print("elapsed time: {} s".format(elapsed_time_s))
    # np.set_printoptions(formatter={'float': '{: 0.2f}'.format}, linewidth=500)  # set print format for arrays
    # print(np.array(times).reshape((-1, 1)))
    # time.sleep(1)
    # logging.info("done")
