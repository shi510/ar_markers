from __future__ import print_function
from __future__ import division
import numpy as np

try:
    import cv2
except ImportError:
    raise Exception('Error: OpenCv is not installed')

from numpy import array, rot90

from ar_markers.coding import decode, extract_hamming_code
from ar_markers.marker import MARKER_SIZE, HammingMarker

BORDER_COORDINATES = [
    [0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [1, 0], [1, 6], [2, 0], [2, 6], [3, 0],
    [3, 6], [4, 0], [4, 6], [5, 0], [5, 6], [6, 0], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [6, 6],
]

ORIENTATION_MARKER_COORDINATES = [[1, 1], [1, 5], [5, 1], [5, 5]]


def validate_and_turn(marker):
    # first, lets make sure that the border contains only zeros
    for crd in BORDER_COORDINATES:
        if marker[crd[0], crd[1]] != 0.0:
            raise ValueError('Border contians not entirely black parts.')
    # search for the corner marker for orientation and make sure, there is only 1
    orientation_marker = None
    for crd in ORIENTATION_MARKER_COORDINATES:
        marker_found = False
        if marker[crd[0], crd[1]] == 1.0:
            marker_found = True
        if marker_found and orientation_marker:
            raise ValueError('More than 1 orientation_marker found.')
        elif marker_found:
            orientation_marker = crd
    if not orientation_marker:
        raise ValueError('No orientation marker found.')
    rotation = 0
    if orientation_marker == [1, 5]:
        rotation = 1
    elif orientation_marker == [5, 5]:
        rotation = 2
    elif orientation_marker == [5, 1]:
        rotation = 3
    marker = rot90(marker, k=rotation)
    return marker

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def detect_markers(img):
    """
    This is the main function for detecting markers in an image.

    Input:
      img: a color or grayscale image that may or may not contain a marker.

    Output:
      a list of found markers. If no markers are found, then it is an empty list.
    """
    if len(img.shape) > 2:
        width, height, _ = img.shape
        gray_ori = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        width, height = img.shape
        gray_ori = img
    scale = [1, 1.5, 2]
    # scale = [2]
    markers_list = []
    for s in scale:
        size = (int(gray_ori.shape[1] // s), int(gray_ori.shape[0] // s))
        gray = cv2.resize(gray_ori, dsize=size)
        g1 = cv2.GaussianBlur(gray, (5, 5), 1.2)
        g2 = cv2.GaussianBlur(gray, (5, 5), 1.3)
        # diff = cv2.GaussianBlur(gray, (5, 5), 1.2)
        diff = np.abs(g2-g1)
        # diff = g2-g1
        # _, diff = cv2.threshold(diff, 127, 255, cv2.THRESH_BINARY)
        # diff = g1 - g2

        edges = cv2.Canny(diff, 10, 100)
        # cv2.imshow('edges', edges)
        edges = cv2.dilate(edges, None, iterations=4)
        edges = cv2.erode(edges, None, iterations=4)
        # cv2.imshow('dilate', edges)

        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]

        # We only keep the long enough contours
        min_contour_length = min(width // s, height // s) / 50
        contours = [contour for contour in contours if len(contour) > min_contour_length]
        warped_size = 49
        canonical_marker_coords = array(
            (
                (0, 0),
                (warped_size - 1, 0),
                (warped_size - 1, warped_size - 1),
                (0, warped_size - 1)
            ),
            dtype='float32')


        for contour in contours:
            cnt_len = cv2.arcLength(contour, True)
            approx_curve = cv2.approxPolyDP(contour, cnt_len * 0.02, True)
            if not (len(approx_curve) == 4 and cv2.isContourConvex(approx_curve) and cv2.contourArea(approx_curve) > (500 / s)):
                continue
            cnt = approx_curve.reshape(-1, 2)
            max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
            if max_cos >= 0.15:
                continue

            sorted_curve = array(
                cv2.convexHull(approx_curve, clockwise=False),
                dtype='float32'
            )
            persp_transf = cv2.getPerspectiveTransform(sorted_curve, canonical_marker_coords)
            warped_img = cv2.warpPerspective(gray, persp_transf, (warped_size, warped_size))
            warped_gray = warped_img

            _, warped_bin = cv2.threshold(warped_gray, 127, 255, cv2.THRESH_BINARY)

            marker = warped_bin.reshape(
                [MARKER_SIZE, warped_size // MARKER_SIZE, MARKER_SIZE, warped_size // MARKER_SIZE]
            )
            marker = marker.mean(axis=3).mean(axis=1)
            marker[marker < 127] = 0
            marker[marker >= 127] = 1

            try:
                marker = validate_and_turn(marker)
                hamming_code = extract_hamming_code(marker)
                marker_id = int(decode(hamming_code), 2)
                approx_curve = approx_curve * s
                markers_list.append(HammingMarker(id=marker_id, contours=approx_curve.astype('int')))

            except ValueError as e:
                # print(e)
                continue
    return markers_list
