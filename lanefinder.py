import numpy as np
import matplotlib.pyplot as plt
from line import Line
from window import Window


class LaneFinder:

    def __init__(self, image_size=(720, 1280)):
        self.__image_size = image_size
        self.__num_windows = 9
        self.__margin = 100
        self.__min_points = 50

        windows = []
        window_height = image_size[0] // self.__num_windows
        for level in range(0, self.__num_windows):
            windows.append([Window(level, image_size[0], image_size[1], h=window_height),
                            Window(level, image_size[0], image_size[1], h=window_height)])
        windows = np.array(windows)
        self.__left_line = Line(image_size, windows[:,0])
        self.__right_line = Line(image_size, windows[:,1])

    def __histogram(self, binary_warped):
        return np.sum(binary_warped[binary_warped.shape[0] // 3:, :], axis=0)

    def show_histogram(self, binary_warped):
        histogram = self.__histogram(binary_warped)
        plt.plot(histogram)
        plt.show()

    def find_lanes(self, binary_warped):
        histogram = self.__histogram(binary_warped)

        middle = histogram.shape[0] // 2

        left_search_center = np.argmax(histogram[:middle])
        right_search_center = np.argmax(histogram[middle:]) + middle

        if left_search_center == 0:
            left_search_center = middle // 2

        if right_search_center == middle:
            right_search_center = middle // 2 + middle

        self.__left_line.update(binary_warped, left_search_center, self.__margin)
        self.__right_line.update(binary_warped, right_search_center, self.__margin)

        self.__validate_lines()

        return self.__left_line, self.__right_line

    def __validate_lines(self):
        left_fit, right_fit = self.__left_line.fit(), self.__right_line.fit()
        if abs((left_fit[0] * self.__image_size[0] ** 2 + left_fit[1] * self.__image_size[0]) -
                (right_fit[0] * self.__image_size[0] ** 2 + right_fit[1] * self.__image_size[0])) > 120:
            self.__left_line.set_valid(False)
            self.__right_line.set_valid(False)

    def calculate_position(self):
        left_fit, right_fit = self.__left_line.fit(), self.__right_line.fit()
        if left_fit is None or right_fit is None:
            return 0

        left = (left_fit[0] * (self.__image_size[0] - 1) ** 2 +
                             left_fit[1] * (self.__image_size[0] - 1) + left_fit[2])
        right = (right_fit[0] * (self.__image_size[0] - 1) ** 2 +
                              right_fit[1] * (self.__image_size[0] - 1) + right_fit[2])
        return np.interp(self.__image_size[1] / 2, [left, right], [0, 1]) * 3.7
