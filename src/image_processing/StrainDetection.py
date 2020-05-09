import logging

import cv2 as cv
import numpy as np

logger = logging.getLogger("Strain Detection")


class StrainDetector:

    def __init__(self, query_angles=(0, 90)):

        self.logging = logging.getLogger("StrainDetector")

        self._reference_images = None
        self._reference_result_images = None  # store so they can be saved later
        self._reference_ellipses = None
        self._exclude_masks = None  # masks defining areas to exclude (for faster electrode outline detection)
        self._reference_radii = None
        self._reference_centers = None
        self._reference_pseudo_areas = None
        self.setting_reference = False

        self.query_angles = np.array(query_angles)
        self._strain_threshold = 50  # above 50 % strain can be considered an outlier
        self._neg_strain_threshold = -10  # below -10 % strain can be considered an outlier
        self._shift_threshold = 100  # above 100 px shift along either axis can be considered an outlier
        self._image_deviation_threshold = 0.15  # if the average deviation is more than 15%, something bad happened

    def set_reference(self, reference_images):
        self._reference_images = None
        self._reference_ellipses = None
        self._exclude_masks = None
        self._reference_radii = None
        self._reference_centers = None
        self.setting_reference = True
        res = self.get_dea_strain(reference_images, True)
        self._reference_result_images = res[2]

    def get_reference_images(self):
        return self._reference_images, self._reference_result_images

    def has_reference(self):
        return self._reference_images is not None

    def get_deviation_from_reference(self, images):
        if self._reference_images is None:
            self.logging.debug("No reference set, therefore deviation is 0.")
            return [0] * len(images)
        else:
            return [get_image_deviation(img, ref) for img, ref in zip(images, self._reference_images)]

    def get_dea_strain(self, imgs, output_result_image=False, check_visual_state=False, title=""):
        """
        Detect strain for the given set of images.
        :param imgs: A set of images (1 image per DEA)
        :param output_result_image: if True, the function returns a visualisation of the measured strain
        :param check_visual_state: Checks if any image has deviated too much from its reference
        :param title: A title to print on the result image. Can be a list of titles, one for each result image.
        :return: four object containing the strain, center shift, result images, and visual_state for each of the
        n input images. 'strain' is a n-by-4 array with the following columns:
        [area strain, radial strain X, radial strain Y, average radial strain (sqrt of area strain)]
        'center shift' is a n-by-2 array containing the shift in X and Y direction, in pixels
        (reference: top left corner).
        'result images' is a n-element list of images with a visualization of the measured strain.
        'visual' state is a n-element list of ints indicating 1 if an image is OK, or 0 if an image shows great
        deviation from its reference image.
        """

        n_img = len(imgs)

        # TODO: make sure the algorithm is robust to Nones
        if self._reference_images is not None and n_img != len(self._reference_images):
            raise Exception("Need to specify one image for each reference image! Set elements to None where no image is"
                            "available for a reference")

        # indication if any of the samples must be considered an outlier
        outlier = np.array([False] * len(imgs))

        # check if image deviation is too large
        if check_visual_state:
            if self._reference_images is not None:
                img_dev = self.get_deviation_from_reference(imgs)
                self.logging.debug("Image deviations: {}".format(img_dev))
                outlier = np.bitwise_or(outlier, np.array(img_dev) > self._image_deviation_threshold)

        # initialize lists to store results
        ellipses = [None] * n_img
        if self._exclude_masks is None:
            self._exclude_masks = [None] * n_img
        masks = self._exclude_masks

        # get fit for each DEA
        for i in range(n_img):
            ellipse, mask = dea_fit_ellipse(imgs[i], self._exclude_masks[i])
            ellipses[i] = ellipse
            masks[i] = mask

            # mark as outlier if no ellipse was found
            if ellipse is None:
                outlier[i] = True

        # calculate strain and center shift
        ellipses_np = ellipses_to_numpy_array(ellipses)  # get as numpy array
        xy_radii = ellipse_radius_at_angle(ellipses_np, self.query_angles)
        centers = ellipses_np[:, 0:2]
        # pseudo because we don't bother multiplying by pi; divide by 4 because ellipse uses diameter not radius
        pseudo_areas = np.prod(ellipses_np[:, 2:4], axis=1) / 4

        # if there is not yet any reference for strain measurements, set these ellipses as the reference
        if self._reference_ellipses is None:
            if np.any(ellipses_np == np.nan) or any(outlier):
                self.logging.critical("Invalid reference! Cannot proceed without a valid reference!")
                raise Exception("The given reference is not valid")
            # if not explicitly calling set_reference, print a warning that a new reference is being set
            if self.setting_reference:
                self.logging.info("Setting new strain reference.")
            else:
                self.logging.warning("No strain reference has been set yet. The given images will be set as reference.")
            self._reference_images = imgs
            self._reference_ellipses = ellipses
            self._reference_radii = xy_radii
            self._reference_centers = centers
            self._reference_pseudo_areas = pseudo_areas
            title = "Reference"  # to be displayed on the result image

        # calculate strain and shift
        strain = xy_radii / self._reference_radii
        strain_area = pseudo_areas / self._reference_pseudo_areas
        strain_avg = np.sqrt(strain_area)
        strain_all = np.concatenate((np.reshape(strain_area, (-1, 1)), strain, np.reshape(strain_avg, (-1, 1))), axis=1)
        strain_all = strain_all * 100 - 100  # convert to engineering strain
        center_shift = centers - self._reference_centers

        # check if strain is too large
        if check_visual_state:
            strain_out = np.bitwise_or(strain > self._strain_threshold, strain < self._neg_strain_threshold)
            strain_out = np.any(strain_out, axis=1)
            outlier = np.bitwise_or(outlier, strain_out)
            shift_out = np.abs(center_shift) > self._shift_threshold
            shift_out = np.any(shift_out, axis=1)
            outlier = np.bitwise_or(outlier, shift_out)

            visual_state = list(np.invert(outlier).astype(np.uint8))
        else:
            visual_state = [None] * n_img

        # update exclude masks
        for i in range(n_img):
            if check_visual_state and outlier[i]:
                masks[i] = None  # keeps it from saving this mask if it belongs to an outlier
            if masks[i] is not None:
                self._exclude_masks[i] = masks[i]

        if isinstance(title, str):
            title = [title] * n_img  # make into a list with the same title for each image
        elif len(title) == 1 and n_img > 1:
            title = title * n_img  # if single string in list, duplicate to match number of images

        res_imgs = None
        if output_result_image:
            # create a list of flags indicating the DEA state (as determined visually) or non to disable the indicator
            res_imgs = [visualize_result(imgs[i], ellipses[i], tuple(xy_radii[i, :]), tuple(self.query_angles),
                                         self._reference_ellipses[i], strain_all[i, -1], title=title[i])
                        for i in range(n_img)]

        return strain_all, center_shift, res_imgs, visual_state


def draw_ellipse(img, ellipse, color, line_width, draw_axes=False, axis_line_width=1):
    # draw the ellipse
    cv.ellipse(img, ellipse, color, line_width)
    if draw_axes:
        center, diameters, angle = ellipse
        # draw both axes of the ellipse
        draw_diameter(img, center, diameters[0] / 2, angle, color, axis_line_width)
        draw_diameter(img, center, diameters[1] / 2, angle + 90, color, axis_line_width)


def draw_diameter(img, center, radius, angle, color, line_width, also_draw_perpendicular=False):
    """
    Draw the diameter of a circle.
    :param img: The image to draw on
    :param center: The center of the circle
    :param radius: The radius of the circle
    :param angle: the angle at which to draw the diameter line
    :param color: The line color
    :param line_width: The line thickness
    :param also_draw_perpendicular: Flag to indicate if the perpendicular diameter should also be drawn (making a cross)
    """
    # calculate start and end
    x, y = center
    a = np.deg2rad(angle)
    rx = np.cos(a) * radius
    ry = np.sin(a) * radius
    p1 = (x - rx, y - ry)
    p2 = (x + rx, y + ry)
    # convert to int tuple for drawing method
    p1_int = tuple(np.round(p1).astype(np.int))
    p2_int = tuple(np.round(p2).astype(np.int))
    cv.line(img, p1_int, p2_int, color, line_width)  # first line

    if also_draw_perpendicular:
        rx, ry = (-ry, rx)  # rotate 90°
        p1_int = tuple(np.round((x - rx, y - ry)).astype(np.int))
        p2_int = tuple(np.round((x + rx, y + ry)).astype(np.int))
        cv.line(img, p1_int, p2_int, color, line_width)  # second line

    return p1, p2


def draw_cross(img, center, radius, angle, color, line_width):
    # the draw diameter function already does this. but calling draw_cross reads more clearly
    draw_diameter(img, center, radius, angle, color, line_width, True)


def get_binary_image(img, closing_radius):
    """
    Turns the given color image into a binary image using Otsu's method to determine the threshold value.
    Morphological closing is applied to the output.
    :param img: The image to convert to binary
    :param closing_radius: the radius of the kernel used for morphological closing (must be non-negative)
    :return: A binary representation of the given image
    """

    # convert to grayscale
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    img = cv.GaussianBlur(img, (5, 5), 0)  # little bit of filtering to reduce noise

    # threshold using Otsu. Seems to work very well if there is enough contrast and sharp edges in the image
    thrsh, bw = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # do some morphological closing to get rid of holes
    if closing_radius > 0:
        d = 2 * closing_radius + 1
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (d, d))
        bw = cv.morphologyEx(bw, cv.MORPH_CLOSE, kernel)
        bw = cv.morphologyEx(bw, cv.MORPH_OPEN, kernel)
    return bw


def find_contour(bw, center=None):
    """
    Find the outline (contour) of the object at a given point of a binary image
    :param bw: The binary image
    :param center: the point at which the desired object is expected. If None, the center of the image is used
    :return: the contour of the object at the given point, or the largest object if no object contains the given point
    """

    # find contours (only top level - we don't want to detect any nested regions)
    contrs, chierarchy = cv.findContours(bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # if no center defined, use center of image
    if center is None:
        center = (bw.shape[1] // 2 + 1, bw.shape[0] // 2 + 1)

    # find the contour that contains the center
    for cnt in contrs:
        if cv.pointPolygonTest(cnt, center, False) == 1:
            return cnt

    # if no contour contained the center, return largest one
    best_cont = None
    max_area = 0
    for cnt in contrs:
        a = cv.contourArea(cnt)
        if a > max_area:
            best_cont = cnt
            max_area = a

    if best_cont is None:
        logger.warning("No contour found.")
    else:
        logger.warning("Did not find any object at the center of the image. The largest contour is returned instead.")

    return best_cont


def get_ellipse_mask(ellipse, img_shape, offset=0):
    """
    Create an ellipse shaped mask of the given size. The ellipse is dilated by the specified offset
    :param ellipse: the ellipse to draw on the mask: ((center_x, center_y), (r_x, r_y), angle)
    :param img_shape: the desired size of the mask: (height, width)
    :param offset: the amoun by which to dilate the mask (can be negative for erosion)
    :return: A binary image with the shape of the specified ellipse
    """
    # create image
    mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)

    try:
        # fill ellipse
        draw_ellipse(mask, ellipse, (255, 255, 255), -1)
    except Exception as ex:
        logging.getLogger("StrainDetection").warning("Unable to create ellipse mask: {}".format(ex))
        mask += 255  # make mask white to include everything
        return mask

    # dilate/erode by given offset, if necessary
    if offset != 0:
        operation = cv.MORPH_DILATE
        if offset < 0:
            operation = cv.MORPH_ERODE  # if offset is negative --> erode

        # create kernel
        n = 2 * abs(offset) + 1
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (n, n))

        # perform morphological operation
        mask = cv.morphologyEx(mask, operation, kernel)

    return mask


def dea_fit_ellipse(img, mask=None, closing_radius=10, algorithm=1):
    if img is None:
        return None, None

    # binarize
    img_bw = get_binary_image(img, closing_radius)

    if algorithm is 0:
        mask = approximate_electrode_outline(img_bw)
        ellipse = refine_electrode_outline(img_bw, mask)
    else:
        ellipse, mask = find_electrode_outline(img_bw, mask)
    return ellipse, mask


def find_electrode_outline(img_bw, exclude_mask=None, max_iterations=20, center=None, convergence_threshold=0.000001):
    if img_bw is None:
        logger.warning("No binary image specified!")
        return None, None

    # If no mask is specified, create one with no exclusions
    if exclude_mask is None:
        exclude_mask = np.zeros(img_bw.shape, dtype=np.uint8)

    # get contours
    cont = find_contour(img_bw, center)
    ellipse = None
    old_area = np.nan
    new_area = np.nan
    converged = False
    i = 0
    while not converged:  # continue until area change per iteration is less than 1 %
        # exclude regions of contour
        cont_in = exclude_contour_points(cont, exclude_mask)
        # fit ellipse
        if cont_in is None or len(cont_in) < 5:
            logger.warning("Unable to fit ellipse because no outline was found.")
            if np.any(exclude_mask > 0):  # if any exclusions were applied
                exclude_mask = np.zeros(img_bw.shape, dtype=np.uint8)  # reset mask and try again
                logger.warning("Trying again without any exclusions.")
                continue
            return None, None  # if we can't fit an ellipse and there was no mask applied, we have to abort

        ellipse = cv.fitEllipseDirect(cont_in)
        # calculate area to determine convergence
        old_area = new_area
        new_area = ellipse[1][0] * ellipse[1][1]  # pseudo area (no need to bother multiplying by pi)

        # create ellipse mask and apply inverse to image to create exclude mask
        ellipse_mask = get_ellipse_mask(ellipse, img_bw.shape, 25)
        exclude_mask = cv.bitwise_and(img_bw, cv.bitwise_not(ellipse_mask))

        # dilate exclude mask to overlap ellipse contour
        operation = cv.MORPH_DILATE
        n = 61
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (n, n))
        exclude_mask = cv.morphologyEx(exclude_mask, operation, kernel)

        # count iterations to abort after a while if the result does not converge for some reason
        i += 1
        if i >= max_iterations:
            break

        if old_area == 0:
            converged = False
        else:
            converged = (old_area - new_area) / old_area < convergence_threshold

    if i < max_iterations:
        logger.debug("Electrode outline detection converged after {} iterations".format(i))
    else:
        logger.warning("Electrode outline detection did not converge. Aborted after {} iterations".format(i))

    return ellipse, exclude_mask


def exclude_contour_points(contour, exclude_mask):
    if contour is None or exclude_mask is None:
        return contour

    # draw white contour 1 pixel wide on black background
    img = np.zeros(exclude_mask.shape, dtype=np.uint8)
    cv.drawContours(img, [contour], 0, (255, 255, 255), thickness=1)  # draw contour in white
    # make excluded areas black (erase contour in those areas)
    img = np.bitwise_and(img, np.bitwise_not(exclude_mask))
    # find all remaining white pixels
    points = np.nonzero(img)
    x = points[1].reshape((-1, 1))
    y = points[0].reshape((-1, 1))
    points = np.concatenate((x, y), axis=1)

    # exclude any points on the image border
    width = exclude_mask.shape[1]
    height = exclude_mask.shape[0]
    on_border = np.any(np.concatenate((x == 0, x == width - 1, y == 0, y == height - 1), axis=1), axis=1)
    points = points[np.bitwise_not(on_border), :]

    # put in the right contour shape
    points = np.reshape(points, (-1, 1, 2))

    return points


def approximate_electrode_outline(img_bw, iterations=20, center=None):
    """
    Iteratively approximate the electrode area in the given binary image.
    :param img_bw: The binary image
    :param iterations: The number of iterations
    :param center: The approximate center of the electrode (if None, center of the image will be assumed)
    :return: an ellipse mask outlining the approximate shape of the electrode,
             the ellipse description ((cx, cy), (ra, rb), angle)
             the outline of the electrodes masked by the returned mask
    """
    # TODO: check if copy is redundant
    img_bw_orig = np.copy(img_bw)
    mask = None
    area = np.nan
    new_area = np.nan
    i = 0
    while not (area - new_area) / area < 0.01:  # continue until area change per iteration is less than 1 %
        # get contours
        cont = find_contour(img_bw, center)
        # fit ellipse
        ellipse = cv.fitEllipseDirect(cont)
        # calculate area to determine convergence
        area = new_area
        new_area = ellipse[1][0] * ellipse[1][1]  # pseudo area (no need to bother multiplying by pi)
        # update center based on fit
        center = ellipse[0]
        # create ellipse mask to use in next iteration
        mask = get_ellipse_mask(ellipse, img_bw.shape, 25)
        img_bw = cv.bitwise_and(img_bw_orig, mask)
        i += 1
        if i >= iterations:
            break

    return mask


def refine_electrode_outline(img_bw, mask):
    # get contour of the ellipse mask
    ell_cont, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    ell_cont = ell_cont[0]  # there can be only one contour
    # get centroid of the contour
    m = cv.moments(ell_cont)
    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])
    center = (cx, cy)

    # get apply mask to input image and get contour
    img_bw = cv.bitwise_and(img_bw, mask)
    cont = find_contour(img_bw, center)

    # visualize result
    # img_col = cv.cvtColor(img_bw, cv.COLOR_GRAY2BGR)
    # cv.drawContours(img_col, cont, -1, (255, 0, 0), 3)
    # cv.drawContours(img_col, ell_cont, -1, (0, 255, 0), 3)
    # cv.imshow("Image", img_col)
    # cv.waitKey()

    # check contour against ellipse contour
    contour = np.squeeze(cont)
    exclude_contour = np.squeeze(ell_cont)
    is_on_ellipse = np.zeros((contour.shape[0]), dtype=bool)
    for i1 in range(0, contour.shape[0]):
        if (exclude_contour == contour[i1, :]).all(axis=1).any():
            is_on_ellipse[i1] = True
    is_included = np.logical_not(is_on_ellipse)
    # post-process excluded region (convert to BW image and use morphological operations)
    is_included = np.array(is_included * 255, dtype=np.uint8)  # to bw image
    is_included = np.reshape(is_included, (-1, 1))
    # close small gaps in the contour (random peaks that got excluded)
    strel = cv.getStructuringElement(cv.MORPH_RECT, (1, 51))
    is_included = cv.morphologyEx(is_included, cv.MORPH_CLOSE, strel)
    strel = cv.getStructuringElement(cv.MORPH_RECT, (1, 51))
    is_included = cv.morphologyEx(is_included, cv.MORPH_OPEN, strel)
    # extend excluded regions
    strel = cv.getStructuringElement(cv.MORPH_RECT, (1, 201))
    is_included = cv.morphologyEx(is_included, cv.MORPH_ERODE, strel)
    is_included = is_included > 0
    is_included = np.reshape(is_included, (-1))
    # find islands
    incl_diff = np.diff(np.array(is_included, dtype=np.int8))
    i_start = np.nonzero(incl_diff > 0)[0]
    i_end = np.nonzero(incl_diff < 0)[0]
    # fix length mismatch (in case a transition is at the beginning/end of the data and no caught by diff)
    if len(i_start) < len(i_end):  # fewer starts than ends
        i_start = np.insert(i_start, 0, 0)  # first block starts at index 0
    elif len(i_start) > len(i_end):  # fewer ends than starts
        i_end = np.append(i_end, len(is_included) - 1)  # last block ends at last index
    # if lengths still don't match, we have a problem!
    if len(i_start) != len(i_end):
        raise ValueError("can't match start and end of included regions")
    if len(i_start) != 3:
        logger.warning("Expected 3 contour regions to include. Found: {}".format(len(i_start)))
    # TODO: solve this problem properly! Not just catch the error
    if i_start[0] > i_end[0]:  # start and end don't match -> shift order to match them
        i_end = np.roll(i_end, -1)
    # include_regions = np.concatenate(i_start, i_end, axis=1)  # combine matched start and end indices
    contour_split = []
    for i in range(len(i_start)):
        if i_start[i] < i_end[i]:  # the normal case, where start and end are in order
            section = cont[i_start[i] + 1:i_end[i] + 1, :]
        else:
            s1 = cont[i_start[i] + 1:, :]
            s2 = cont[:i_end[i] + 1, :]
            section = np.append(s1, s2, axis=1)

        contour_split.append(np.reshape(section, (-1, 1, 2)))
    # stitch together to get ellipse fit
    cont_stitched = np.concatenate(contour_split)
    ellipse = cv.fitEllipseDirect(cont_stitched)
    return ellipse


def get_image_deviation(img1, img2):
    """
    Calculate the per-pixel average difference between the two given images.
    :param img1: The first image to compare
    :param img2: The second image to compare
    :return: the difference between the images in range 0 to 1, where 0 inidcates that the images are identical.
    """
    if img1 is None or img2 is None:
        logger.warning("Image if None, therefore deviation is 0.")
        return 0

    dev_img = np.abs(img1.astype(np.int16) - img2.astype(np.int16)).astype(np.uint8)
    dev_per_pixel = np.mean(dev_img) / 255
    return dev_per_pixel


def ellipses_to_numpy_array(ellipses):
    """
    Convert a list or tuple of ellipses (with diameter)
    [((cx, cy), (d1, d2), angle), ...]
    to a numpy array (with radii):
    [ cx, cy, r1, r2, angle;
      ...                    ]
    :param ellipses: List or tuple of OpenCV ellipses
    :return: The parameters of all ellipses arranged in a numpy array
    """
    # TODO: optimize ellipse decomposition maybe...?
    centers = []
    diameters = []
    angles = []
    for el in ellipses:
        if el is None:
            centers.append((np.nan, np.nan))
            diameters.append((np.nan, np.nan))
            angles.append(np.nan)
        else:
            centers.append(el[0])
            diameters.append(el[1])
            angles.append(el[2])

    # convert diameters to radii
    return np.concatenate((np.array(centers), np.array(diameters) / 2, np.reshape(np.array(angles), (-1, 1))), axis=1)


def ellipse_radius_at_angle(ellipses_np, query_angles):
    """
    Calculate the radius of a given ellipse at the specified angle.
    :param ellipses_np: A n-by-5 numpy array describing a set of ellipses [cx, cy, rx, ry, a; ...]
    :param query_angles: 1-by-m array of angles (in deg, in world coordinates - not relative to the ellipse)
    for which to calculate the radii.
    :return: n-by-m array of radii for the given ellipses at the specified angles
    """

    # ensure that all arrays have the right dimensions
    radii = np.reshape(ellipses_np[:, 2:4], (-1, 2))  # make sure it's an n-by-2 matrix
    ellipse_angle = np.reshape(ellipses_np[:, 4], (-1, 1))  # make sure it's a column vector
    query_angles = np.reshape(query_angles, (1, -1))  # make sure it's a row vector

    # calculate radii
    a = np.deg2rad(query_angles - ellipse_angle)
    prod = np.prod(radii, 1, keepdims=True)
    sqr_sum = np.sqrt((radii[:, None, 0] * np.sin(a)) ** 2 + (radii[:, None, 1] * np.cos(a)) ** 2)  # None to keep dims

    return prod / sqr_sum


def visualize_result(img, ellipse, radii=None, angles=None, reference_ellipse=None, strain=None, marker_size=8,
                     title=""):
    """
    Visualizes the strain detection result by drawing the detected ellipse on the image. Can also show specific radial
    strain values at certain angles.
    :param title: text to be displayed on the image
    :param strain: The detected strain (single value!)
    :param img: The image to draw the visualization on
    :param ellipse: The detected ellipse to draw on the image. In OpenCV ellipse format: ((cx, cy), (r1, r2), angle)
    :param reference_ellipse: the reference ellipse
    :param radii: A list of calculated radii at different angles to be visualised with a dotted line and cross
    :param angles: A list of the corresponding angles for the given radii
    to either side of the ellipse center. Expecting one value per query angle
    :param marker_size: The size of the markers indicating the centers and radial strain
    :return: A copy of the given image with a visualization of the strain detection result drawn on top.
    """
    if img is None or ellipse is None:
        return img

    img = np.copy(img)  # copy so we don't alter the original

    blue = (255, 0, 0)
    green = (0, 200, 0)
    red = (0, 0, 255)
    black = (0, 0, 0)
    strain_colors = ((0, 127, 255), (127, 0, 255))

    # draw the reference ellipse
    if reference_ellipse is not None:
        draw_ellipse(img, reference_ellipse, green, 1, True, 1)

    # draw the fitted ellipse
    draw_ellipse(img, ellipse, blue, 2, True, 1)

    # draw radial strain
    if radii is not None and angles is not None:
        assert len(radii) == len(angles)
        # duplicate colors to make sure we have enough for each radius
        strain_colors = strain_colors * int(np.ceil(len(radii) / len(strain_colors)))
        center, diam, angle = ellipse
        for r, a, c in zip(radii, angles, strain_colors):
            p1, p2 = draw_diameter(img, center, r, a, c, 1)
            draw_cross(img, p1, marker_size, a, c, 2)
            draw_cross(img, p2, marker_size, a, c, 2)

    # draw text
    font = cv.FONT_HERSHEY_SIMPLEX
    scale = 1.5
    margin = 20
    cv.putText(img, title, (margin, 60), font, scale, black, 3)
    if reference_ellipse is not None:
        cv.putText(img, "Detected ellipse", (margin, 120), font, scale, blue, 2)
        cv.putText(img, "Reference ellipse", (margin, 180), font, scale, green, 2)

    if strain is not None:
        cv.putText(img, "Strain: {:.1f} %".format(strain), (margin, 240), font, scale, black, 2)

    return img


def draw_state_visualization(img, visual, electrical):
    r = 20
    margin = 20
    font = cv.FONT_HERSHEY_SIMPLEX
    scale = 1.5
    black = (0, 0, 0)
    green = (0, 200, 0)
    red = (0, 0, 255)
    amber = (0, 200, 255)

    state_colors = (red, green, amber)

    combined = None
    if visual is not None:
        cv.putText(img, "Visual state", (2 * (margin + r), img.shape[0] - margin - 5), font, scale, black, 2)
        c = state_colors[visual]
        cv.circle(img, (margin + r, img.shape[0] - margin - r), r, c, -1)
        combined = visual

    if electrical is not None:
        cv.putText(img, "Electrical state", (2 * (margin + r), img.shape[0] - margin - 65), font, scale, black, 2)
        c = state_colors[electrical]
        cv.circle(img, (margin + r, img.shape[0] - margin - r - 60), r, c, -1)
        combined = electrical

    if visual is not None and electrical is not None:  # both state is given
        if visual == electrical == 1:
            combined = 1
        elif visual == 1 and electrical == 2:
            combined = 2
        else:
            combined = 0

    if combined is not None:
        c = state_colors[combined]
        r = 160
        cv.circle(img, (margin + r, img.shape[0] - margin - r - 2 * 60), r, c, -1)


if __name__ == '__main__':
    # frame = np.ones((401, 401, 3), dtype=np.uint8)
    # cx, cy = 180, 210
    # ax1, ax2 = 100, 180
    # angle = 20
    # center = (cx, cy)
    # axes = (ax1, ax2)
    # cv.ellipse(frame, center, axes, angle, 0, 360, (255, 0, 0), -1)
    # draw_diameter(frame, center, ax1, angle, (0, 255, 0), 1)
    # draw_diameter(frame, center, ax2, angle+90, (0, 255, 0), 1)
    # cv.ellipse(frame, (center, axes, angle), (0, 0, 255), 1)
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # thrsh, bw = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # cnt, hier = cv.findContours(bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # ell = cv.fitEllipse(cnt[0])
    # print(ell)
    # cv.imshow("Test", frame)
    # cv.waitKey()

    import os
    import time
    from libs import duallog

    duallog.setup("logs", minlevelConsole=logging.DEBUG, minLevelFile=logging.DEBUG)

    # test_image_folder = "D:/NERD output/NERD test 20200225-182457/DEA 1/Images/"
    test_image_folder = "../../output/Test7/"
    files = os.listdir(test_image_folder)

    _strain_detector = StrainDetector((0, 90))

    tic = time.perf_counter()
    ref_img = None
    for file in files:
        fpath = os.path.join(test_image_folder, file)
        if not os.path.isfile(fpath):
            continue
        _img = cv.imread(fpath)
        if ref_img is None:
            ref_img = _img

        print("File:", file)
        _result_strain, _result_center_shift, \
        _result_images, _outlier = _strain_detector.get_dea_strain([_img], True, True)
        print("Strain:", _result_strain)
        print("Area strain:", _result_strain[0, 0])
        print("Average strain:", _result_strain[0, -1])
        print("Average approx:", np.mean(_result_strain[0, 1:-1]))

        cv.imshow("Difference", _result_images[0])
        cv.waitKey()

        # print("Outlier: ", _outlier)
        #
        # print("Shift:", _result_center_shift[0])
        # _ellipse, _mask = dea_fit_ellipse(_img)
        # _res_img = visualize_result(_img, _ellipse)
        # _result_images = [_res_img]

        # _img = _result_images[0]
        # draw_state_visualization(_img, _outlier[0], None)
        # if ref_img is not None and ref_img.shape == _img.shape:
        #     _dev_img = np.abs(_img.astype(np.int16) - ref_img.astype(np.int16))
        #     dev = np.mean(_dev_img) / 255
        #     print("Deviation: ", dev)
        #     _dev_img = np.abs(_img.astype(np.int16) - ref_img.astype(np.int16)).astype(np.uint8)
        #     dev = np.mean(_dev_img) / 255
        #     print("Deviation: ", dev)
        #
        #     cv.imshow("Image1", _img)
        #     cv.imshow("Image2", ref_img)
        #     cv.imshow("Difference", _dev_img)
        #     cv.waitKey()

        # cv.imwrite(os.path.join(test_image_folder + "Results", file), _result_images[0])

    cv.destroyAllWindows()
    # toc = time.perf_counter()
    # print("elapsed time: ", toc - tic)
