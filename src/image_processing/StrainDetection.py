from logging import warning

import cv2 as cv
import numpy as np


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

    warning("did not find any object at the center of the image")

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
    cv.ellipse(mask, ellipse, (255, 255, 255), -1)

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

    print("stopped approximation after", i, "iterations")

    return mask, ellipse


def dea_fit_ellipse(img, closing_radius=5):
    if img is None:
        return None

    img_orig = np.copy(img)  # keep copy of original image

    # binarize
    img_bw_orig = get_binary_image(img, closing_radius)

    # make copy to keep original
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
        warning("Expected 3 contour regions to include. Found: {}".format(len(i_start)))

    if i_start[0] > i_end[0]:  # start and end don't match -> shift order to match them
        i_end = np.roll(i_end, -1)
    # include_regions = np.concatenate(i_start, i_end, axis=1)  # combine matched start and end indices

    contour_split = []
    for i in range(len(i_start)):
        section = None
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


def draw_ellipse(img, ellipse):
    if img is None:
        return None
    img = np.copy(img)  # copy so we don't alter the original
    cv.ellipse(img, ellipse, (255, 0, 0), 2)
    return img
