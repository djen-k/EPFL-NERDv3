import logging
import time
from datetime import datetime
from threading import Thread, Event

import cv2 as cv

from libs.duallog import duallog


class Camera:

    def __init__(self, index, desired_resolution=None, new_image_callback=None):
        self.logging = logging.getLogger("Camera")

        # open image capture
        self.cap = cv.VideoCapture(index + cv.CAP_MSMF)  # use Windows Media Foundation backend. No others work!
        # set properties
        if desired_resolution is not None:
            self.set_resolution(desired_resolution)

        # init buffers
        self.name = "Camera {}".format(index)
        self.image_buffer = None
        self.grab_timestamp = None  # timestamp for the last grab
        self.image_timestamp = None  # timestamp for image in buffer. May differ from grab_timestamp if retrieve failed
        self.grab_ok = False

        # callback
        self.new_image_callback = new_image_callback

    def set_resolution(self, desired_resolution):
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, desired_resolution[0])
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, desired_resolution[1])

    def grab(self):
        """
        Grab a frame from this camera. The frame is not retrieved at this point.
        :return: a flag indicating success or failure
        """
        cam = self.cap
        if not cam.isOpened():  # just to be sure. this should never be the case, theoretically
            self.logging.debug("{} is not open".format(self.name))
            return

        suc = cam.grab()  # call grab to quickly obtain a frame, which can be decoded with "retrieve()" later
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
        cam = self.cap

        if not cam.isOpened():  # just to be sure. this should never be the case, theoretically
            self.logging.debug("{} is not open".format(self.name))
            return False, None

        if self.grab_ok:  # check if previous grab was successful
            suc, frame = cam.retrieve()  # retrieve frame previously captured by "grab()"

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

        # internal variables
        self._cameras = []
        self._camera_count = -1  # not yet initialized
        self._camera_names = []
        self._camera_states = []  # to indicate if a camera was lost (no longer able to grab images)
        self._cameras_available = []
        self._camera_selection = None

        # threading
        self._exit_flag = None
        self._capture_thread = None

        # callbacks
        self._new_image_callback = None
        self.new_set_callback = None

    def _reset(self):
        # todo: end capture thread if it's running
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
        :param new_set_callback: Callback function of prototype: func(images[], timestamps[])
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
            self.logging.warning("Not a valid camera id: {}. Expected string or int!".format(cam_id))
            return None

    def find_cameras(self):
        """
        Opens all available image capture devices and stores a handle to each device as well as a list of device names
        (currently just "Camera [index]"). One initial image from each camera is retrieved and stored in the buffer.
        Can be used to re-initialize the cameras.
        """

        self._reset()  # release all cams and empty buffers

        i = -1
        while True:
            # open all cameras starting at index 0
            i += 1  # increase index to try next camera
            # if camera was already open, it will be closed and re-opened internally
            cam = Camera(i, self.desired_resolution, self._new_image_callback)

            if not cam.isOpened():
                break  # if unable to open, a camera with this index doesn't exist. Stop search.
            else:
                self.logging.info("{} opened".format(cam.name))
                success, frame = cam.read()  # check if we can read an image
                if success:
                    # store camera handle, name, and captured image
                    self._cameras.append(cam)

                else:
                    self.logging.warning("Unable to retrieve image from {}. Camera will be closed.".format(cam.name))
                    cam.release()

        self._cameras_available = self._cameras  # store list of all available cameras.

        self._camera_count = len(self._cameras)
        self._camera_names = self._get_names_from_cameras()

        self.logging.info("{} cameras found".format(self._camera_count))

        # call new set callback since the first full set of images is now available
        if self.new_set_callback is not None:
            self.new_set_callback(self.get_images_from_buffer(), self.get_timestamps())

    def reconnect_cameras(self):
        self.logging.info("attempting to reconnect cameras")
        # TODO: handle camera selection

        prev_count = self._camera_count
        prev_available = len(self._cameras_available)
        prev_selection = self._camera_selection
        prev_names = self._camera_names
        state = self._camera_states

        self.close_cameras()
        time.sleep(5)

        self.find_cameras()
        new_count = self._camera_count

        cams_lost = prev_available - new_count
        cams_failed = state.count(False)

        if cams_lost is 0:
            if prev_selection is not None:
                self.select_cameras(prev_selection)
                self.set_camera_names(prev_names)
            self.logging.info("Cameras successfully reconnected")
        elif cams_lost is cams_failed:  # we can match the failure to the lost camera
            i_failed = [i for i, x in enumerate(state) if x is False]
            for i in i_failed:
                self._cameras.insert(i, Camera(100))  # insert dummy camera. safe to assume camera 100 doesn't exist
        else:
            self.logging.info("Don't know what to do here. Going to try and carry on as if nothing happened...")
            if prev_selection is not None:
                self.select_cameras(prev_selection)
                self.set_camera_names(prev_names)
            # TODO: deal with unknown cameras

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
        return self._camera_states

    def retrieve_images(self):
        """
        Retrieve a frame from each selected camera and store in buffer.
        :return: a list of flags indicating success, and a list of the retrieved frames
        """
        res = [cam.retrieve() for cam in self._cameras]
        # turn list of tuples [(suc, frame), ...] into two lists [suc, ...], [frame, ...]
        return [list(o) for o in zip(*res)]

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

        if False in self._camera_states:
            if self.auto_reconnect:
                self.reconnect_cameras()
                self.auto_reconnect = False  # turn off so we don't keep reconnecting forever
            else:
                self.logging.critical("Failed to reconnect. Shutting down")
                raise Exception("Could not reconnect cameras")
        else:
            # all well, reconnect (if there was one) has succeeded, so if it happens again, we can try again
            self.auto_reconnect = True

        sucs, frames = self.retrieve_images()

        if self.new_set_callback is not None:
            self.new_set_callback(frames, self.get_timestamps())

        return frames

    def get_images_from_buffer(self):
        """
        Get the last image of each selected camera from the image buffer
        :return: A copy of the image buffer
        """
        # create a copy of the image buffer to ensure we always keep an unadulterated copy
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
        self._capture_thread = ImageCaptureThread(self, max_fps, self._exit_flag)
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

    def run_test(self):
        res = 0
        # todo: implement proper resource management so files can be accessed consistently from anywhere
        no_img = cv.imread("../../res/images/no_image.png")
        waiting_img = cv.imread("../../res/images/waiting.jpeg")
        cv.imshow('Camera 0', cv.resize(waiting_img, (0, 0), fx=0.5, fy=0.5))

        while res & 0xFF != ord('q'):

            imgs = self.read_images()

            for i in range(len(imgs)):
                if imgs[i] is None:
                    cv.imshow('Camera {}'.format(i), cv.resize(no_img, (0, 0), fx=0.5, fy=0.5))
                else:
                    cv.imshow('Camera {}'.format(i), cv.resize(imgs[i], (0, 0), fx=0.25, fy=0.25))

            res = cv.waitKey(500)


class ImageCaptureThread(Thread):

    # TODO add mutex

    def __init__(self, image_capture: ImageCapture, max_fps, exit_flag: Event):
        Thread.__init__(self)
        self._image_capture = image_capture
        self._exit_flag = exit_flag
        if max_fps > 0:
            self._min_delay = 1.0 / max_fps
        else:
            self._min_delay = 0  # no minimum delay -> capture at maximum fps

    def run(self):
        try:
            delay = 0  # initial delay is 0
            while not self._exit_flag.wait(delay):
                start_time = datetime.now()
                self._image_capture.read_images()
                end_time = datetime.now()
                elapsed_time = end_time - start_time
                delay = self._min_delay - elapsed_time.total_seconds()  # calculate remaining time to wait
                # self._image_capture.logging.debug("loop complete. delay: {}".format(delay))
        except Exception as ex:
            self._image_capture.logging.error("Exception in CemeraSelection.updateImage: {}".format(ex))


SharedInstance = ImageCapture()

if __name__ == '__main__':
    duallog.setup("camera test logs", minlevelConsole=logging.DEBUG, minLevelFile=logging.DEBUG)
    logging.info("Testing image capture")
    SharedInstance.run_test()
