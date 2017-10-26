import numpy as np
import cv2


def __ensure3_channels(image):
    '''
    Ensures the the image has 3 channels. If not, it will stack it 3 times.
    :param image: the image to check
    :return: the original image if it had 3 channels, os the one converted to 3 channels
    '''
    return np.dstack((image, image, image)) if len(image.shape) == 2 else image


def overlay_lane_lines(image, left_coors, right_coors, line_width=2, overlay_weight=1):
    '''
    Overlays the lane lines on the given image.
    :param image: the image to overlay the lines on
    :param left_coors: the left line coordinates
    :param right_coors: the right line coordinates
    :param line_width: the line width
    :param overlay_weight: the weight to use for overlaying
    :return: the new image with the overlays applied
    '''
    out_image = __ensure3_channels(image)
    overlay_image = np.zeros_like(out_image)

    cv2.polylines(overlay_image, [left_coors], False, (0, 255, 255),
                  thickness=line_width, lineType=cv2.LINE_AA)
    cv2.polylines(overlay_image, [right_coors], False, (0, 255, 255),
                  thickness=line_width, lineType=cv2.LINE_AA)

    return cv2.addWeighted(out_image, .7, overlay_image, overlay_weight, 0)


def overlay_lane(image, left_coors, right_coors, camera, line_width=60, overlay_weight=0.3):
    '''
    Overlayes the lane on the unwarped image. This includes the middle area as well as the left and right lines.
    :param image: the image to overlay to
    :param left_coors: the left line coordinates
    :param right_coors: the right line coordinates
    :param camera: the camera to use for unwarping the overlay
    :param line_width: the line width to use
    :param overlay_weight: the weight to use when overlaying
    :return: the image with the lane overlayed.
    '''
    out_image = __ensure3_channels(image)
    if left_coors is None or right_coors is None:
        return out_image

    overlay_image = np.zeros_like(out_image)

    lane_overlay_coors = np.hstack((left_coors, np.array([np.flipud(right_coors[0])])))

    cv2.fillPoly(overlay_image, [lane_overlay_coors], (0, 255, 0))

    cv2.polylines(overlay_image, [left_coors], False, (255, 255, 0),
                  thickness=line_width, lineType=cv2.LINE_AA)
    cv2.polylines(overlay_image, [right_coors], False, (255, 0, 255),
                  thickness=line_width, lineType=cv2.LINE_AA)

    return cv2.addWeighted(out_image, 1, camera.unwarp(overlay_image), overlay_weight, 0)


def overlay_text(image, text, x, y):
    '''
    Adds text to the image.
    :return: the image with the text added. This might modify the original image if it already has 3 channels.
    '''
    out_image = __ensure3_channels(image)
    cv2.putText(out_image, text, org=(x, y), fontScale=1, thickness=2, color=(255, 255, 255),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, lineType=cv2.LINE_AA)

    return out_image


def overlay_windows(image, window_pos, overlay_weight=0.5):
    '''
    Overlayes the windows on the image.
    '''
    out_image = __ensure3_channels(image)
    overlay_image = np.zeros_like(out_image)
    for top_left, bottom_right in window_pos:
        overlay_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = (0, 255, 0)

    return cv2.addWeighted(out_image, .7, overlay_image, overlay_weight, 0)


def overlay_perspective_lines(image, perpective_coordinates):
    '''
    Draws the polygon to use for perspective transformation on the given image.
    '''
    overlay_image = np.zeros_like(image)
    cv2.polylines(overlay_image, [perpective_coordinates], True, [0, 0, 255], thickness=3)
    return cv2.addWeighted(image, .7 , overlay_image, 1, 0)