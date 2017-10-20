import numpy as np
import matplotlib.pyplot as plt
from collections import deque


class LaneFinder:

    def __init__(self, image_size=(720, 1280), debug_windows=False):
        self.__image_size = image_size
        self.__debug_windows = debug_windows
        self.__num_windows = 9
        self.__margin = 100
        self.__min_points = 50
        self.__ym_per_pix = 30 / image_size[0]  # meters per pixel in y dimension
        self.__xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        self.__debug_window_pos = None
        self.__left_lane_x = None
        self.__left_lane_y = None
        self.__right_lane_x = None
        self.__right_lane_y = None
        self.__left_fit = None
        self.__right_fit = None
        self.__recent_left_fits = deque()
        self.__recent_right_fits = deque()

    def __histogram(self, binary_warped):
        return np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

    def show_histogram(self, binary_warped):
        histogram = self.__histogram(binary_warped)
        plt.plot(histogram)
        plt.show()

    def __update_recent_fits(self):
        if self.__left_fit is not None:
            self.__recent_left_fits.append(self.__left_fit)
        if self.__right_fit is not None:
            self.__recent_right_fits.append(self.__right_fit)

        if len(self.__recent_left_fits) > 5:
            self.__recent_left_fits.popleft()
        if len(self.__recent_right_fits) > 5:
            self.__recent_right_fits.popleft()

    def __polyfit(self, y, x):
        if len(y) > 0 and len(x) > 0:
            return np.polyfit(y, x, 2)
        return None

    def __validate_lines(self):
        middle = self.__image_size[1] / 2
        if np.mean(self.__left_lane_x) > middle:
            self.__left_fit = None

        if np.mean(self.__right_lane_x) < middle:
            self.__right_fit = None
        # if left_fit is not None and right_fit is not None:
        #     if not (3 < (np.mean(self.__right_lane_x) - np.mean(self.__left_lane_x)) * self.__xm_per_pix < 5):
        #         return False

    def find_lanes(self, binary_warped):
        if self.__left_fit is None or self.__right_fit is None:
            self.__find_lane_first(binary_warped)
        else:
            self.__update_lane(binary_warped)

    def __find_lane_first(self, binary_warped):
        histogram = self.__histogram(binary_warped)

        middle = histogram.shape[0] // 2

        left_search_base = np.argmax(histogram[:middle])
        right_search_base = np.argmax(histogram[middle:]) + middle

        nonzeros = binary_warped.nonzero()
        nonzeros_y = np.array(nonzeros[0])
        nonzeros_x = np.array(nonzeros[1])

        image_height = binary_warped.shape[0]
        window_height = image_height // self.__num_windows

        left_lane_indices = []
        right_lane_indices = []
        self.__debug_window_pos = []

        for window in range(self.__num_windows):
            w_top = image_height - (window + 1) * window_height
            w_bottom = image_height - window * window_height
            left_w_left = left_search_base - self.__margin
            left_w_right = left_search_base + self.__margin
            right_w_left = right_search_base - self.__margin
            right_w_right = right_search_base + self.__margin

            if self.__debug_windows:
                self.__debug_window_pos.append(((left_w_left, w_top), (left_w_right, w_bottom)))
                self.__debug_window_pos.append(((right_w_left, w_top), (right_w_right, w_bottom)))

            left_window_points = ((nonzeros_x >= left_w_left) & (nonzeros_x < left_w_right) &
                                  (nonzeros_y <= w_bottom) & (nonzeros_y > w_top)).nonzero()[0]

            right_window_points = ((nonzeros_x >= right_w_left) & (nonzeros_x < right_w_right) &
                                   (nonzeros_y <= w_bottom) & (nonzeros_y > w_top)).nonzero()[0]

            left_lane_indices.append(left_window_points)
            right_lane_indices.append(right_window_points)

            if len(left_window_points) > self.__min_points:
                left_search_base = np.int32(np.mean(nonzeros_x[left_window_points]))

            if len(right_window_points) > self.__min_points:
                right_search_base = np.int32(np.mean(nonzeros_x[right_window_points]))

        left_lane_indices = np.concatenate(left_lane_indices)
        right_lane_indices = np.concatenate(right_lane_indices)

        self.__left_lane_x = nonzeros_x[left_lane_indices]
        self.__left_lane_y = nonzeros_y[left_lane_indices]
        self.__right_lane_x = nonzeros_x[right_lane_indices]
        self.__right_lane_y = nonzeros_y[right_lane_indices]
        self.__left_fit = self.__polyfit(self.__left_lane_y, self.__left_lane_x)
        self.__right_fit = self.__polyfit(self.__right_lane_y, self.__right_lane_x)

        self.__validate_lines()

        self.__update_recent_fits()

    def __update_lane(self, binary_warped):
        self.__debug_window_pos = []
        nonzero = binary_warped.nonzero()
        nonzeros_y = np.array(nonzero[0])
        nonzeros_x = np.array(nonzero[1])
        left_lane_indices = ((nonzeros_x > (self.__left_fit[0] * (nonzeros_y ** 2) +
                                            self.__left_fit[1] * nonzeros_y +
                                            self.__left_fit[2] - self.__margin)) &
                             (nonzeros_x < (self.__left_fit[0] * (nonzeros_y ** 2) +
                                            self.__left_fit[1] * nonzeros_y +
                                            self.__left_fit[2] + self.__margin)))

        right_lane_indices = ((nonzeros_x > (self.__right_fit[0] * (nonzeros_y ** 2) +
                                             self.__right_fit[1] * nonzeros_y +
                                             self.__right_fit[2] - self.__margin)) &
                              (nonzeros_x < (self.__right_fit[0] * (nonzeros_y ** 2) +
                                             self.__right_fit[1] * nonzeros_y +
                                             self.__right_fit[2] + self.__margin)))

        self.__left_lane_x = nonzeros_x[left_lane_indices]
        self.__left_lane_y = nonzeros_y[left_lane_indices]
        self.__right_lane_x = nonzeros_x[right_lane_indices]
        self.__right_lane_y = nonzeros_y[right_lane_indices]
        self.__left_fit = self.__polyfit(self.__left_lane_y, self.__left_lane_x)
        self.__right_fit = self.__polyfit(self.__right_lane_y, self.__right_lane_x)

        self.__validate_lines()

        self.__update_recent_fits()

    def line_overlay_coordinates(self):
        left_fit = np.mean(self.__recent_left_fits, axis=0)
        right_fit = np.mean(self.__recent_right_fits, axis=0)
        lane_y_coors = np.linspace(0, self.__image_size[0] - 1, self.__image_size[0])
        left_lane_x_coors = (left_fit[0] * lane_y_coors ** 2 +
                             left_fit[1] * lane_y_coors + left_fit[2])
        right_lane_x_coors = (right_fit[0] * lane_y_coors ** 2 +
                              right_fit[1] * lane_y_coors + right_fit[2])

        left_lane_coors = np.int32(np.array([np.transpose(np.vstack([left_lane_x_coors, lane_y_coors]))]))
        right_lane_coors = np.int32(np.array([np.flipud(np.transpose(np.vstack([right_lane_x_coors, lane_y_coors])))]))

        return left_lane_coors, right_lane_coors

    def calculate_curvature(self):
        # Fit new polynomials to x,y in world space
        left_curverad = 0
        right_curverad = 0
        left_fit_cr = self.__polyfit(np.array(self.__left_lane_y) * self.__ym_per_pix,
                                     np.array(self.__left_lane_x) * self.__xm_per_pix)
        if left_fit_cr is not None:
            left_curverad = ((1 + (2 * left_fit_cr[0] * (self.__image_size[0] - 1) * self.__ym_per_pix +
                                   left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])

        right_fit_cr = self.__polyfit(np.array(self.__right_lane_y) * self.__ym_per_pix,
                                      np.array(self.__right_lane_x) * self.__xm_per_pix)
        if right_fit_cr is not None:
            right_curverad = ((1 + (2 * right_fit_cr[0] * (self.__image_size[0] - 1) * self.__ym_per_pix +
                                   right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
        # left_fit_cr = np.mean(self.__recent_left_fits, axis=0)
        # right_fit_cr = np.mean(self.__recent_right_fits, axis=0)

        # Calculate the new radii of curvature

        return left_curverad, right_curverad

    def calculate_position(self):
        left_fit = np.mean(self.__recent_left_fits, axis=0)
        right_fit = np.mean(self.__recent_right_fits, axis=0)

        left = (left_fit[0] * (self.__image_size[0] - 1) ** 2 +
                             left_fit[1] * (self.__image_size[0] - 1) + left_fit[2])
        right = (right_fit[0] * (self.__image_size[0] - 1) ** 2 +
                              right_fit[1] * (self.__image_size[0] - 1) + right_fit[2])
        # lane_middle_pos = (right_line_pos_x - left_line_pos_x) / 2
        return np.interp(self.__image_size[1] / 2, [left, right], [0, 1]) * 3.7
        # return (640 - lane_middle_pos) * self.__xm_per_pix

    def debug_window_coordinates(self):
        return self.__debug_window_pos
