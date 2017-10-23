import numpy as np
from scipy.ndimage.filters import gaussian_filter


class Window:

    def __init__(self, level, image_height, image_width, w=60, h=80, margin=100):
        self.__level = level
        self.__image_height = image_height
        self.__image_width = image_width
        self.__margin = margin

        self.__y_start = max(0, image_height - (level + 1) * h)
        self.__h = min(h, image_height - self.__y_start)
        self.__x_start = None
        self.__w = w

        self.__min_valid_window_points = 50
        self.valid = False
        self.skip = True
        self.__max_num_invalid_frames = 15
        self.__num_invalid_frames = 0
        self.__window_layer = np.ones(self.__w)
        self.__window_points = None
        self.__last_valid_window_points = (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        self.__last_valid_x_start = None
        self.__search_center = None
        self.__last_search_center = None

    def update(self, binary, nonzeros, x_search_center_when_invalid):
        if self.valid:
            self.__last_valid_x_start = self.__x_start
            self.__last_valid_window_points = self.__window_points
            self.__last_search_center = self.__search_center

        if self.valid:
            x_start = max(0, self.__x_start + self.__w // 2 - self.__margin)
            x_end = min(self.__image_width, self.__x_start + self.__w // 2 + self.__margin)
        else:
            x_start = max(0, x_search_center_when_invalid - self.__margin)
            x_end = min(self.__image_width, x_search_center_when_invalid + self.__margin)

        search_layer = np.sum(binary[self.__y_start:self.__y_start + self.__h, x_start:x_end], axis=0)
        filtered_layer = gaussian_filter(search_layer, sigma=self.__w / 3)
        conv = np.convolve(self.__window_layer, filtered_layer)

        self.__search_center = np.argmax(conv) + x_start
        self.__x_start = min(max(0, self.__search_center - self.__w), self.__image_width)

        nonzeros_y = np.array(nonzeros[0])
        nonzeros_x = np.array(nonzeros[1])

        point_indices = ((nonzeros_x >= self.__x_start) & (nonzeros_x < self.__x_start + self.__w) &
                         (nonzeros_y > self.__y_start) & (nonzeros_y < self.__y_start + self.__h)).nonzero()[0]
        self.__window_points = (nonzeros_x[point_indices], nonzeros_y[point_indices])

        self.__validate()

    def __validate(self):
        self.valid = self.__valid_window_points()

        if self.valid:
            self.__num_invalid_frames = 0
        else:
            self.__num_invalid_frames += 1

        self.skip = ((not self.valid and len(self.__last_valid_window_points[0]) == 0) or
                     self.__num_invalid_frames >= self.__max_num_invalid_frames)

    def __valid_window_points(self):
        return self.__window_points is not None and \
               len(self.__window_points[0]) > self.__min_valid_window_points

    def points(self):
        if self.valid:
            return self.__window_points
        elif self.__num_invalid_frames < self.__max_num_invalid_frames:
            return self.__last_valid_window_points
        else:
            return (np.array([], dtype=np.int64), np.array([], dtype=np.int64))

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
                else ((self.__valid_x_start(), self.__y_start),
                      (self.__valid_x_start() + self.__w, self.__y_start + self.__h)))
