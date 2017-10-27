import numpy as np
from scipy.ndimage.filters import gaussian_filter


class Window:
    '''
    Class to detect a small piece of a lane line. The y position of the window is fixed, but it can move horizontally,
    tracking the line on subsequent frames.
    '''

    def __init__(self, level, image_height, image_width, w=60, h=80):
        self.level = level
        self.__image_height = image_height
        self.__image_width = image_width

        self.y_start = max(0, image_height - (level + 1) * h)
        self.height = min(h, image_height - self.y_start)
        self.__x_start = None
        self.width = w

        self.__min_valid_window_points = 50
        self.valid = False
        self.skip = True
        self.__max_num_invalid_frames = 15
        self.__num_invalid_frames = 0
        self.__window_layer = np.ones(self.width)
        self.__window_points = None
        self.__empty_window_points = (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        self.__last_valid_window_points = self.__empty_window_points
        self.__last_valid_x_start = None
        self.__search_center = None
        self.__last_search_center = None

    def update(self, binary, nonzeros, x_search_center_when_invalid, search_margin_when_invalid):
        '''
        Updates the window position for the current frame. The algorithm to find the piece of the line on window's
        vertical space is as follows:
        1. Figure out where to search horizontally:
            a. If the previous position was valid, use the center of that, with a margin of the window's width on both
               side
            b. else, use the search center and margin provided
        2. Sum up the binary pixels vertically in the search space
        3. Convolve with mas the size of the window
        4. Get the maximum density area and center the window on that.
        5. Gather all the points and validate them by checking the ratio of the noise in the search area and requiring
           a minimum number of points in the window.

        :param binary: the binary image to find the line on
        :param nonzeros: all the nonzero points on the binary image. This is only an optimization so that it's not
            queried on all the windows
        :param x_search_center_when_invalid: the suggested horizontal center of the search area.
        :param search_margin_when_invalid: the suggested margin for the search area.
        '''
        if self.valid:
            self.__last_valid_x_start = self.__x_start
            self.__last_valid_window_points = self.__window_points
            self.__last_search_center = self.__search_center

        if self.valid:
            x_start = max(0, self.__x_start + self.width // 2 - self.width)
            x_end = min(self.__image_width, self.__x_start + self.width // 2 + self.width)
        else:
            x_start = max(0, x_search_center_when_invalid - search_margin_when_invalid)
            x_end = min(self.__image_width, x_search_center_when_invalid + search_margin_when_invalid)

        search_layer = np.sum(binary[self.y_start:self.y_start + self.height, x_start:x_end], axis=0)
        conv = np.convolve(self.__window_layer, search_layer)

        self.__search_center = np.argmax(conv) + x_start
        self.__x_start = max(0, self.__search_center - self.width)

        nonzeros_y = np.array(nonzeros[0])
        nonzeros_x = np.array(nonzeros[1])

        point_indices = ((nonzeros_x >= self.__x_start) & (nonzeros_x < self.__x_start + self.width) &
                         (nonzeros_y >= self.y_start) & (nonzeros_y < self.y_start + self.height)).nonzero()[0]

        self.__window_points = (nonzeros_x[point_indices], nonzeros_y[point_indices])
        self.__validate()

    def set_valid(self, is_valid):
        '''
        Sets whether the current window position is valid.
        '''
        self.valid = is_valid

        if self.valid:
            self.__num_invalid_frames = 0
        else:
            self.__num_invalid_frames += 1

        self.skip = ((not self.valid and len(self.__last_valid_window_points[0]) == 0) or
                     self.__num_invalid_frames >= self.__max_num_invalid_frames)

    def force_skip(self):
        self.__last_valid_window_points = self.__empty_window_points
        self.skip = True

    def __validate(self):
        '''
        Validates the window points and sets the windows validity based on that.
        '''
        self.set_valid(self.__valid_window_points())

    def __valid_window_points(self):
        '''
        Helper method to do the validation of the points. Right now only checks if it has the required minimum number of
        points.
        :return: True if the points are valid, False otherwise.
        '''
        return self.__window_points is not None and \
               len(self.__window_points[0]) > self.__min_valid_window_points

    def points(self):
        '''
        :return: the points available for this window. This can either be the points determined on the last update, or
        if those were invalid, the last valid points, or if there are none or the number of updates since having
        determined valid points pass the maximum number (i.e. 10), it return and empty array.
        '''
        if self.valid:
            return self.__window_points
        elif self.__num_invalid_frames < self.__max_num_invalid_frames:
            return self.__last_valid_window_points
        else:
            return self.__empty_window_points

    def weights(self):
        '''
        :return: Weights for the points. This is 3 times the max number of invalid frames for valid points on the
            current frame and decreases every time the points are used, since there's no valid ones on the current
            frame.
        '''
        if self.valid:
            return np.full_like(self.__window_points[0], self.__max_num_invalid_frames * 3)
        elif self.__num_invalid_frames < self.__max_num_invalid_frames:
            return np.full_like(self.__last_valid_window_points[0],
                                (self.__max_num_invalid_frames - self.__num_invalid_frames) * 3)
        else:
            return np.array([], dtype=np.int64)

    def search_center(self):
        if self.valid:
            return self.__search_center
        elif not self.skip:
            return self.__last_search_center
        else:
            return None

    def __valid_x_start(self):
        return self.__x_start if self.valid else self.__last_valid_x_start

    def coordinates(self):
        return (None if self.skip
                else ((self.__valid_x_start(), self.y_start),
                      (self.__valid_x_start() + self.width, self.y_start + self.height)))
