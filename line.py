import numpy as np
from scipy.ndimage.filters import gaussian_filter


class Line:

    def __init__(self, image_size, windows):
        self.__image_size = image_size
        self.__windows = windows
        self.__x_coors = None
        self.__y_coors = None
        self.__ym_per_pix = 30 / image_size[0]  # meters per pixel in y dimension
        self.__xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        self.curr_fit = None
        self.last_valid_fit = None

    def update(self, binary_warped, search_center):
        if self.curr_fit is not None:
            self.last_valid_fit = self.curr_fit

        if not self.__windows[0].skip:
            search_center = self.__windows[0].search_center()

        nonzeros = binary_warped.nonzero()
        line_points = []
        valid_line_points = []
        num_valid_windows = 0

        for window in self.__windows:
            window.update(binary_warped, nonzeros, search_center)
            line_points.append(window.points())
            if window.valid:
                num_valid_windows += 1
                valid_line_points.append(window.points)

            if window.search_center() is not None:
                search_center = window.search_center()

        line_points = np.concatenate(line_points, axis=1)
        if num_valid_windows > 3 and len(valid_line_points) > 0:
            valid_line_points = np.concatenate(valid_line_points, axis=1)

        self.__x_coors = line_points[0]
        self.__y_coors = line_points[1]
        if len(self.__y_coors) > 0 and len(self.__x_coors) > 0:
            self.curr_fit = np.polyfit(self.__y_coors, self.__x_coors, 2)
            if num_valid_windows > 3:
                valid_fit = np.polyfit(valid_line_points[1], valid_line_points[0], 2)
                self.curr_fit = np.mean(self.curr_fit, valid_fit)
        else:
            self.curr_fit = None

        if not self.__is_valid_poly() or not self.__is_valid_line():
            self.curr_fit = None

    def __is_valid_poly(self):
        prev_fit = self.last_valid_fit
        current_fit = self.curr_fit
        # if prev_fit is not None and current_fit is not None:
        #     prev_square = prev_fit[0] ** 2 + prev_fit[1] ** 2
        #     current_square = current_fit[0] ** 2 + current_fit[1] ** 2
        #     diff = prev_square / current_square
        #     # print('Diff between mean and current is: %f' % diff)
        #     if not (.25 < diff < 4):
        #         return False

        if current_fit is not None:
            square = current_fit[0] ** 2 + current_fit[1] ** 2
            # print(square)
            if square > 2.:
                return False

        return True

    def __is_valid_line(self):
        usable_windows = 0
        for window in self.__windows:
            if not window.skip: usable_windows += 1

        return usable_windows > 3
        # middle = self.__image_size[0] // 2
        # return len(self.__y_coors[self.__y_coors >= middle]) > 100 and \
        #        len(self.__y_coors[self.__y_coors < middle]) > 100

    def __line_coordinates(self, fit):
        y_coors = np.linspace(0, self.__image_size[0] - 1, self.__image_size[0])
        x_coors = None
        if fit is not None:
            x_coors = (fit[0] * y_coors ** 2 +
                       fit[1] * y_coors + fit[2])

        return y_coors, x_coors

    def line_overlay_coordinates(self):
        y_coors, x_coors = self.__line_coordinates(self.curr_fit if self.curr_fit is not None else self.last_valid_fit)

        return None if x_coors is None else \
            np.int32(np.array([np.transpose(np.vstack([x_coors, y_coors]))]))

    def curvature(self):
        left_curverad = 0
        y_coors, x_coors = self.__line_coordinates(self.curr_fit)

        left_fit_cr = np.polyfit(y_coors * self.__ym_per_pix, x_coors * self.__xm_per_pix, 2) \
            if y_coors is not None and x_coors is not None \
            else None

        if left_fit_cr is not None:
            left_curverad = ((1 + (2 * left_fit_cr[0] * (self.__image_size[0] - 1) * self.__ym_per_pix +
                                   left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])

        return left_curverad

    def window_coordinates(self):
        window_positions = []
        for window in self.__windows:
            if not window.skip:
                window_positions.append(window.coordinates())

        return window_positions

    def debug_text(self):
        current_fit = self.curr_fit
        prev_fit = self.last_valid_fit
        text = 'Prev: {:.4f}, Current: {:>.4f}, Diff: {:>.4f}'
        prev_square = 0
        square = .001
        if prev_fit is not None:
            prev_square = prev_fit[0] ** 2 + prev_fit[1] ** 2

        if current_fit is not None:
            square = current_fit[0] ** 2 + current_fit[1] ** 2

        return text.format(prev_square, square, prev_square / square)
