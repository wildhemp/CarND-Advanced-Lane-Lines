import numpy as np
import cv2


def __ensure3_channels(image):
    return np.dstack((image, image, image)) if len(image.shape) == 2 else image


def overlay_lane_lines(image, left_coors, right_coors, line_width=60, overlay_weight=0.3):
    out_image = __ensure3_channels(image)
    overlay_image = np.zeros_like(out_image)

    cv2.polylines(overlay_image, [left_coors], False, (255, 255, 0),
                  thickness=line_width, lineType=cv2.LINE_AA)
    cv2.polylines(overlay_image, [right_coors], False, (255, 0, 255),
                  thickness=line_width, lineType=cv2.LINE_AA)

    return cv2.addWeighted(out_image, 1, overlay_image, overlay_weight, 0)


def overlay_lane(image, left_coors, right_coors, camera, line_width=60, overlay_weight=0.3):
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
    out_image = __ensure3_channels(image)
    cv2.putText(out_image, text, org=(x, y), fontScale=1, thickness=2, color=(255, 255, 255),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, lineType=cv2.LINE_AA)

    return out_image


def overlay_windows(image, window_pos, camera, overlay_weight=0.3):
    out_image = __ensure3_channels(image)
    overlay_image = np.zeros_like(out_image)
    for top_left, bottom_right in window_pos:
        overlay_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = (0, 255, 0)

    return cv2.addWeighted(out_image, 1, overlay_image, overlay_weight, 0)