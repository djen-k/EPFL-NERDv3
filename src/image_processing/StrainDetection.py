import logging

import cv2 as cv
import numpy as np


class StrainDetector:

    def __init__(self):

        self.logging = logging.getLogger("StrainDetector")

        self._reference_ellipses = None
        self._reference_radii = None
        self._reference_centers = None
        self._flag_setting_reference = False
        self._query_angles = np.array([0, 90])

    def set_reference(self, reference_images):
        self._reference_ellipses = None
        self._reference_radii = None
        self._reference_centers = None
        self._flag_setting_reference = True
        self.get_dea_strain(reference_images)

    def get_dea_strain(self, imgs, output_result_image=True):

        # TODO: deal with mismatch in n of images and n of references

        ellipses = [dea_fit_ellipse(img) for img in imgs]  # get fit for each DEA
        # calculate strain and center shift
        ellipses_np = ellipses_to_numpy_array(ellipses)  # get as numpy array
        xy_radii = ellipse_radius_at_angle(ellipses_np, self._query_angles)
        centers = ellipses_np[:, 0:2]

        # if there is not yet any reference for strain measurements, set these ellipses as the reference
        if self._reference_ellipses is None:
            if not self._flag_setting_reference:
                self.logging.warning("No strain reference has been set yet. The given images will be set as reference.")
            else:
                self.logging.info("A new strain reference was set")
            self._reference_ellipses = ellipses
            self._reference_radii = xy_radii
            self._reference_centers = centers

        # calculate strain and shift
        strain = xy_radii / self._reference_radii
        strain_area = np.prod(xy_radii, axis=1) / np.prod(self._reference_radii, axis=1)
        strain_all = np.concatenate((strain, np.reshape(strain_area, (-1, 1))), axis=1)
        center_shift = centers - self._reference_centers

        res_imgs = None
        if output_result_image:
            res_imgs = [visualize_result(imgs[i], ellipses[i], self._reference_ellipses[i], tuple(xy_radii[i, :]),
                                         tuple(self._query_angles), strain_all[i, :]) for i in range(len(imgs))]

        return strain, center_shift, res_imgs


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

    logging.warning("did not find any object at the center of the image")

    # if no contour contained the center, return largest one
    best_cont = None
    max_area = 0
    for cnt in contrs:
        a = cv.contourArea(cnt)
        if a > max_area:
            best_cont = cnt
            max_area = a
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
    # draw ellipse
    cv.ellipse(mask, ellipse, (255, 255, 255), -1)  # use this version of the function which takes the ellipse as
    # returned by fitEllipse (as floats and diameter instead of radius

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


def approximate_electrode_area(img_bw, iterations=10, center=None):
    """
    Iteratively approximate the electrode area in the given binary image.
    :param img_bw: The binary image
    :param iterations: The number of iterations
    :param center: The approximate center of the electrode (if None, center of the image will be assumed)
    :return: an ellipse mask outlining the approximate shape of the electrode,
             the ellipse description ((cx, cy), (ra, rb), angle)
             the outline of the electrodes masked by the returned mask
    """
    img_bw_orig = np.copy(img_bw)
    mask = None
    ellipse = None
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

    return mask, ellipse


def dea_fit_ellipse(img, closing_radius=5):
    if img is None:
        return None

    try:
        # binarize
        img_bw_orig = get_binary_image(img, closing_radius)

        # make copy to keep original2
        img_bw = np.copy(img_bw_orig)

        mask, ellipse = approximate_electrode_area(img_bw)
        img_bw = cv.bitwise_and(img_bw_orig, mask)
        cont = find_contour(img_bw, ellipse[0])
        ell_cont, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        ell_cont = ell_cont[0]  # there can be only one contour

        # check contour against ellipse contour
        cont = np.squeeze(cont)
        ell_cont = np.squeeze(ell_cont)
        is_on_ellipse = np.zeros((cont.shape[0]), dtype=bool)
        for i1 in range(0, cont.shape[0]):
            if (ell_cont == cont[i1, :]).all(axis=1).any():
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
            logging.warning("Expected 3 contour regions to include. Found: {}".format(len(i_start)))

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

    except Exception as ex:
        logging.warning("unable to fit ellipse {}".format(ex))
        return None


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


def visualize_result(img, ellipse, reference_ellipse, radii, angles, strain, marker_size=8):
    """
    Visualizes the strain detection result by drawing the detected ellipse on the image. Can also show specific radial
    strain values at certain angles.
    :param strain: The detected strain values
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

    # draw the reference ellipse
    ref_color = (0, 255, 0)
    draw_ellipse(img, reference_ellipse, ref_color, 1, True, 1)
    # draw the fitted ellipse
    fit_color = (255, 0, 0)
    draw_ellipse(img, ellipse, fit_color, 2, True, 1)

    # draw radial strain
    strain_color = ((0, 127, 255), (127, 0, 255))
    center, diam, angle = ellipse
    for r, a, c in zip(radii, angles, strain_color):
        p1, p2 = draw_diameter(img, center, r, a, c, 1)
        draw_cross(img, p1, marker_size, a, c, 2)
        draw_cross(img, p2, marker_size, a, c, 2)

    # draw text
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(img, "Detected ellipse", (10, 40), font, 1, fit_color, 2)
    cv.putText(img, "Reference ellipse", (10, 80), font, 1, ref_color, 2)
    x_strain, y_strain, a_strain = strain * 100 - 100  # convert to engineering strain and split
    cv.putText(img, "X strain: {:.2f}".format(x_strain), (10, 120), font, 1, strain_color[0], 2)
    cv.putText(img, "Y strain: {:.2f}".format(y_strain), (10, 160), font, 1, strain_color[1], 2)
    cv.putText(img, "Area strain: {:.2f}".format(a_strain), (10, 200), font, 1, (0, 0, 0), 2)

    return img


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
    # # cv.ellipse(frame, (center, axes, angle), (0, 0, 255), 1)
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # thrsh, bw = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # cnt, hier = cv.findContours(bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # ell = cv.fitEllipse(cnt[0])
    # print(ell)
    # cv.imshow("Test", frame)
    # cv.waitKey()

    import os

    # test_image_folder = "D:/NERD output/NERD test 20200225-182457/DEA 1/Images/"
    test_image_folder = "../../output/Test4/"
    files = os.listdir(test_image_folder)

    _strain_detector = StrainDetector()

    for file in files:
        fpath = os.path.join(test_image_folder, file)
        if not os.path.isfile(fpath):
            continue
        _img = cv.imread(fpath)
        print("File:", file)
        _result_strain, _result_center_shift, _result_images = _strain_detector.get_dea_strain([_img])
        print("Strain:", _result_strain[0, 0] * 100 - 100, " %    , ", _result_strain[0, 1] * 100 - 100, " %")
        print("Shift:", _result_center_shift[0])

        cv.imshow("Image", _result_images[0])
        cv.waitKey(100)
        cv.imwrite(os.path.join(test_image_folder + "Results", file), _result_images[0])

    cv.destroyAllWindows()
