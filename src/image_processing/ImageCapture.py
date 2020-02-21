import copy
import logging
from datetime import datetime
from threading import Thread, Event

import cv2 as cv


class ImageCapture:
    IMG_NOT_AVAILABLE = cv.imread("res/images/no_image.png")
    IMG_WAITING = cv.imread("res/images/waiting.jpeg")

    def __init__(self):
        self.logging = logging.getLogger("ImageCapture")
        self._camera_count = -1  # not yet initialized
        self._cameras = []
        self._camera_names = []
        self._image_buffer = []
        self._grab_timestamp = []  # timestamp for the last grab
        self._image_timestamp = []  # timestamps for images in buffer. May differ from grab_timestamp if retrieve failed
        self._desired_resolution = [1920, 1080]
        self._new_image_callback = None
        self._new_set_callback = None
        self._exit_flag = None
        self._capture_thread = None

    def _reset(self):
        self.close_cameras()
        self._camera_count = -1  # not yet initialized
        self._cameras = []
        self._camera_names = []
        self._image_buffer = []
        self._grab_timestamp = []  # timestamp for the last grab
        self._image_timestamp = []  # timestamps for images in buffer. May differ from grab_timestamp if retrieve failed

    def set_new_image_callback(self, new_image_callback):
        """
        Set a function to call every time a new image becomes available.
        :param new_image_callback: Callback function of prototype: func(image, timestamp, camera_id)
        """
        self._new_image_callback = new_image_callback

    def set_new_set_callback(self, new_set_callback):
        """
        Set a function to call every time a new set of images becomes available (i.e. one new image from each camera)
        :param new_set_callback: Callback function of prototype: func(images[], timestamps[])
        """
        self._new_set_callback = new_set_callback

    def get_camera_count(self):
        """
        Get the number of physical cameras available
        :return: The number of available cameras, or number of selected cameras if a selection has been made,
        """
        if self._camera_count == -1:  # not initialized yet  --> go find cameras
            self.logging.info("ImageCapture not yet initialized. Initializing cameras...")
            self.find_cameras()

        return self._camera_count

    def get_camera_names(self):
        """
        Get the names of the available/selected cameras
        :return: The names of the available cameras
        """
        if self._camera_count == -1:  # not initialized yet  --> go find cameras
            self.logging.info("ImageCapture not yet initialized. Initializing cameras...")
            self.find_cameras()

        return copy.copy(self._camera_names)  # copy to prevent write access. probably unnecessary...

    def get_camera_name(self, cam_id):
        """
        Get the names of the available/selected cameras
        :return: The names of the available cameras
        """
        assert cam_id < self._camera_count
        return self._camera_names[cam_id]

    def set_camera_name(self, cam_id, name):
        """
        Set the name of the specified camera.
        :param cam_id: The index of the camera (if a selection has been made, this may be different from the index used
        by the OpenCV backend)
        :param name: The new name to assign to the camera
        """
        if cam_id < self._camera_count:
            self.logging.info("Changing name '{}' to '{}'".format(self._camera_names[cam_id], name))
            self._camera_names[cam_id] = name

    def find_cameras(self):
        """
        Opens all available image capture devices and stores a handle to each device as well as a list of device names
        (currently just "Camera [index]"). One initial image from each camera is retrieved and stored in the buffer.
        Can be used to re-initialize the cameras.
        """

        self._reset()  # release all cams and empty buffers

        i = 0
        while True:
            # open all cameras starting at index 0
            # is camera was already open, it will be closed and re-opened internally
            cap = cv.VideoCapture(i + cv.CAP_MSMF)  # be sure to use Windows Media Foundation backend. No others work!
            cap.set(cv.CAP_PROP_FRAME_WIDTH, self._desired_resolution[0])
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, self._desired_resolution[1])

            if not cap.isOpened():
                break  # if unable to open, a camera with this index doesn't exist. Stop search.
            else:
                i += 1  # increase index each time we found a camera. Do this first so numbering starts at 1
                self.logging.info("Camera {} opened".format(i))
                success, frame = cap.read()  # check if we can read an image
                if success:
                    # store camera handle, name, and captured image
                    self._cameras.append(cap)
                    self._camera_names.append("Camera {}".format(i))
                    self._image_buffer.append(frame)
                    t = datetime.now()
                    self._grab_timestamp.append(t)
                    self._image_timestamp.append(t)
                    self._camera_count = i

                    # if callback is defined, call callback function
                    if self._new_image_callback is not None:
                        self._new_image_callback(frame.copy(), t, i - 1)  # -1 to use 0-based indexing
                else:
                    self.logging.warning("Unable to retrieve image from camera {}. Camera will be closed.".format(i))
                    cap.release()

        self.logging.info("{} cameras found".format(i))
        # call new set callback since the first full set of images is now available
        if self._new_set_callback is not None:
            self._new_set_callback(self.get_images_from_buffer(), self.get_timestamps())

    def select_cameras(self, camera_ids):
        if self._camera_count == -1:  # not initialized --> find cameras
            self.find_cameras()

        cams = []
        names = []
        images = []

        # select cameras in the specified order.
        for cid in camera_ids:
            # Raise error if a requested camera is not available
            if cid >= self._camera_count:
                self.logging.critical("Invalid camera selected: camera {} is not available!".format(cid))
                raise ValueError("Invalid camera selection!")
            else:
                cams.append(self._cameras[cid])
                names.append(self._camera_names[cid])
                images.append(self._image_buffer[cid])

        # release all cameras that are not being used
        for i in range(self._camera_count):
            if i not in camera_ids:
                self._cameras[i].release()

        self._cameras = cams
        self._camera_names = names
        self._camera_count = len(cams)
        # reset buffer and timestamps
        self._image_buffer = [None] * self._camera_count
        self._grab_timestamp = [None] * self._camera_count
        self._image_timestamp = [None] * self._camera_count

        # grab a new image from each cam to update the buffer
        self.grab_images()
        self.retrieve_images()

    def grab_single_image(self, cam_id):
        """
        Grab a frame from the specified camera. The frame is not retrieved at this point.
        :param cam_id: The camera index
        """
        assert cam_id < self._camera_count

        cam = self._cameras[cam_id]

        if not cam.isOpened():  # just to be sure. this should never be the case, theoretically
            self.logging.debug("Camera {} is not open".format(self._cameras.index(cam)))
            return

        suc = cam.grab()  # call grab to quickly obtain a frame, which can be decoded with "retrieve()" later
        if suc:
            self._grab_timestamp[cam_id] = datetime.now()  # record timestamp of the grabbed image
        else:
            self.logging.warning("Unable to grab frame from camera {}".format(self._cameras.index(cam)))

    def grab_images(self):
        """
        Grab a frame from each selected camera. Only grab is performed and no image is retrieved at this point.
        This allows capturing of images from all cameras as close as possible in time.
        Captured images can be loaded with "retrieve_images" later.
        """

        for i in range(self._camera_count):
            self.grab_single_image(i)

    def retrieve_single_image(self, cam_id):
        """
        Retrieve a frame from the specified camera and store it in the buffer
        :param cam_id: The camera index
        """
        assert cam_id < self._camera_count

        cam = self._cameras[cam_id]

        if not cam.isOpened():  # just to be sure. this should never be the case, theoretically
            self.logging.debug("Camera {} is not open".format(self._cameras.index(cam)))
            return

        suc, frame = cam.retrieve()  # retrieve frame previously captured by "grab()"
        if suc:
            self._image_buffer[cam_id] = frame
            self._image_timestamp[cam_id] = self._grab_timestamp[cam_id]  # update the timestamp

            self.logging.debug("Retrieved frame from camera {}. Size: {}".format(cam_id, frame.shape))

            # if callback is defined, call callback function
            if self._new_image_callback is not None:
                self._new_image_callback(frame.copy(), self._image_timestamp[cam_id], cam_id)
        else:
            self._image_buffer[cam_id] = ImageCapture.IMG_NOT_AVAILABLE
            self.logging.warning("Unable to retrieve frame from camera {}".format(cam_id))

    def retrieve_images(self):
        """
        Retrieve a frame from each selected camera and store in buffer.
        """
        for i in range(self._camera_count):
            self.retrieve_single_image(i)

    def read_single_image(self, cam_id):
        self.grab_single_image(cam_id)
        self.retrieve_single_image(cam_id)

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
        self.retrieve_images()

        if self._new_set_callback is not None:
            self._new_set_callback(self.get_images_from_buffer(), self.get_timestamps())

    def get_images(self):
        """
        Read an image from each selected camera. First, a frame from each camera is grabbed and later retrieved so as to
        ensure that frames are as close to each other in time as possible. The images are stored in the internal buffer
        and a copy of the buffer is returned.
        :return: A copy of the image buffer, after it was updated with the latest frame from each selected camera
        """
        self.read_images()
        # return a copy of the image buffer (which was just updated)
        return self.get_images_from_buffer()

    def get_single_image(self, cam_id):
        """
        Read an image from the specified camera. The image is stored in the internal buffer
        and a copy of the image is returned.
        :return: A newly captured image from the specified camera
        """
        self.read_single_image(cam_id)
        # return a copy of the requested image from the buffer (which was just updated)
        return self.get_single_image_from_buffer(cam_id)

    def get_single_image_from_buffer(self, cam_id):
        """
        Return the last image of the specified camera from the buffer. No new image is captured.
        :param cam_id: The camera index
        :return: The last image captured by the specified camera
        """
        assert cam_id < self._camera_count
        return self._image_buffer[cam_id].copy()

    def get_images_from_buffer(self):
        """
        Get the last image of each selected camera from the image buffer
        :return: A copy of the image buffer
        """
        # create a copy of the image buffer to ensure we always keep an unadulterated copy
        return copy.deepcopy(self._image_buffer)

    def get_timestamp(self, cam_id):
        """
        Returns the timestamp of the frame from the specified camera that is currently stored in the buffer
        :param cam_id: The camera index
        :return: The timestamp of the current frame from that camera
        """
        assert cam_id < self._camera_count
        return self._image_timestamp[cam_id]

    def get_timestamps(self):
        """
        Return a list containing a timestamps for the most recent image of each selected camera
        :return: A timestamp for each image in the buffer
        """
        return copy.deepcopy(self._image_timestamp)  # copy to prevent any outside access. probably unnecessary

    def close_camera(self, cam_id):
        """
        Close the specified camera
        :param cam_id: The index of the camera to close
        """
        assert cam_id < self._camera_count
        self._cameras[cam_id].release()

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
        if self._new_image_callback is None and self._new_set_callback is None:
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
