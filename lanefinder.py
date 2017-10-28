import numpy as np
from line import Line
from window import Window


class LaneFinder:
    '''
    Encapsulates the lane finding logic.
    '''

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

    def histogram(self, binary_warped):
        '''
        Returns a simple histogram of the bottom 3rd of the given binary image.
        '''
        return np.sum(binary_warped[binary_warped.shape[0] // 3:, :], axis=0)

    def find_lanes(self, binary_warped):
        '''
        Finds the lane on the given binary warped image.
         - This first determines the search center for both the left and right lines.
         - Then passes it to line.update for both lines, together with a default margin used when finding lines mostly
           from scratch.
         - When the lines updated, it check them so that they are approximately parallel and invalidates them if they
           are not.
        :return: the left and right lines
        '''
        histogram = self.histogram(binary_warped)

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
        '''
        Validates lines against each other by checking if they are roughly parallel. The algorithm simply makes sure,
        that the difference at the middle of the lines is within threshold with difference on top and bottom.
        :return:
        '''
        left_fit, right_fit = self.__left_line.fit(), self.__right_line.fit()
        if left_fit is None or right_fit is None:
            return

        # diff_bottom = abs((left_fit[0] * self.__image_size[0] ** 2 + left_fit[1] * self.__image_size[0] + left_fit[2]) -
        #                   (right_fit[0] * self.__image_size[0] ** 2 + right_fit[1] * self.__image_size[0] + right_fit[2]))
        middle = self.__image_size[0] // 2
        # diff_middle = abs((left_fit[0] * middle ** 2 + left_fit[1] * middle + left_fit[2]) -
        #                   (right_fit[0] * middle ** 2 + right_fit[1] * middle + right_fit[2]))
        # diff_top = abs(left_fit[2] - right_fit[2])
        left_x_middle = left_fit[0] * middle ** 2 + left_fit[1] * middle + left_fit[2]
        right_x_middle = right_fit[0] * middle ** 2 + right_fit[1] * middle + right_fit[2]
        # if abs(diff_top - diff_middle > 120) or (diff_bottom - diff_middle > 120):
        left_slope = abs(-middle / (left_x_middle - left_fit[2]))
        right_slope = abs(-middle / (right_x_middle - right_fit[2]))
        if  .9 < (left_slope / right_slope) < 1.1:
            self.__left_line.set_valid(False)
            self.__right_line.set_valid(False)

    def calculate_position(self):
        '''
        Calculates the position of the car in the road. This is basically the difference from the left lane.
        '''
        left_fit, right_fit = self.__left_line.fit(), self.__right_line.fit()
        if left_fit is None or right_fit is None:
            return 0

        left = (left_fit[0] * (self.__image_size[0] - 1) ** 2 +
                             left_fit[1] * (self.__image_size[0] - 1) + left_fit[2])
        right = (right_fit[0] * (self.__image_size[0] - 1) ** 2 +
                              right_fit[1] * (self.__image_size[0] - 1) + right_fit[2])
        return np.interp(self.__image_size[1] / 2, [left, right], [0, 1]) * 3.7 - 1.85
