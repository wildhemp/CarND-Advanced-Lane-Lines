import numpy as np


class Line:
    '''
    Class to handle line related concepts. It can find the line on the image provided and provides tracking on
    subsequent image frames. The tracking/calculation is based on pre-initialized windows provided at instantiation.
    '''

    def __init__(self, image_size, windows):
        self.__image_size = image_size
        self.__windows = windows
        self.__x_coors = None
        self.__y_coors = None
        self.__ym_per_pix = 30 / image_size[0]  # meters per pixel in y dimension
        self.__xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        self.curr_fit = None
        self.last_valid_fit = None
        self.__debug_text = ''

    def update(self, binary_warped, search_center, search_margin):
        '''
        Updates the line using the provided image. It does it by going through all the windows and updating them for the
        current image. The previous windows center is provided as an input to the next window, to help defining the area
        to search for. If the previous window does not have a valid area, the previous line is used to calculate a
        possible search area. Te window might decide to not use this information.
        After updating all the windows, it post validates them.
        Following this it gather all the points covered by the window and calculates a second order polynomial, to fit
        the points. This is the line fit. It validates this fit and if not valid, deletes it.

        :param binary_warped: the birds eye view thresholded image
        :param search_center: the x coordinate to start the search at, if no other means are available to determine one
            (i.e. no previous valid window or good fit)
        :param search_margin: the search margin when search_center provided here is used
        '''
        if self.curr_fit is not None:
            self.last_valid_fit = self.curr_fit

        nonzeros = binary_warped.nonzero()

        # Update/find all the windows on the current image
        for window in self.__windows:
            window.update(binary_warped, nonzeros, search_center, search_margin)

            if window.valid:
                search_center = window.search_center()
                search_margin = window.width
            elif self.last_valid_fit is not None:
                search_center = int(self.last_valid_fit[0] * (window.y_start ** 2) + \
                                    self.last_valid_fit[1] * window.y_start + self.last_valid_fit[2])
                search_margin = window.width
            else:
                search_margin = int(search_margin * 1.1)

        self.__debug_text = ''
        self.__post_validate_windows()

        # Get all the points covered by the windows and store them
        line_points = []
        point_weights = []
        for window in self.__windows:
            line_points.append(window.points())
            point_weights.append(window.weights())

        line_points = np.concatenate(line_points, axis=1)
        point_weights = np.concatenate(point_weights)

        self.__x_coors = line_points[0]
        self.__y_coors = line_points[1]

        # Fit a second order polynomial
        if len(self.__y_coors) > 0 and len(self.__x_coors) > 0:
            self.curr_fit = np.polyfit(self.__y_coors, self.__x_coors, 2, w=point_weights)
        else:
            self.curr_fit = None

        # ... and if it's not a valid fit, then discard it.
        if not self.__is_valid_line() or not self.__is_valid_poly():
            self.curr_fit = None

    def __post_validate_windows(self):
        '''
        Post validates the windows. Windows validate themselves, but that is limited, since they don't have all the
        useful information (e.g. other windows, previous fit, etc).
        '''
        prev_fit = self.last_valid_fit
        if prev_fit is not None:
            # Invalidates windows which are too far from the previous good fit.
            for i, window in enumerate(self.__windows):
                if window.valid:
                    points = window.points()
                    y_center = window.y_start + window.height // 2
                    prev_line = prev_fit[0] * (y_center**2) + prev_fit[1] * y_center + prev_fit[2]
                    margins = np.int32([prev_line - window.width, prev_line + window.width])
                    inliers = points[0][((points[0] >= margins[0]) & (points[0] < margins[1]))]
                    if i == 0:
                        self.__debug_text += 'm:{},{},{},{},{},{}'.format(
                            margins[0], margins[1],len(inliers), len(points[0]), np.min(points[0]), np.max(points[0]))
                    if len(inliers) / len(points[0]) < .6:
                        self.__debug_text += 'iw: %d '% i
                        window.set_valid(False)

        # points = []#[[first_search_center, self.__image_size[0]]]
        # not_skipped_windows = []
        # for window in self.__windows:
        #     if not window.skip:
        #         not_skipped_windows.append(window)
        #         points.append(np.mean(window.points(), axis=1))
        #
        # if len(points) == 0:
        #     return
        #
        # points = np.transpose(points)
        #
        # not_skipped_windows = np.array(not_skipped_windows)
        # points = np.transpose(points)
        # x = points[0]
        # while True:
        #     ratio_bottom_up = x[:-1]/x[1:]
        #     ratio_bottom_up = np.concatenate(([1.], ratio_bottom_up))
        #     good = np.abs(1 - ratio_bottom_up/np.median(ratio_bottom_up)) < .1
        #     not_good = np.logical_not(good)
        #     x_new = x[good]
        #     for window in not_skipped_windows[not_good]:
        #         window.set_valid(False)
        #     not_skipped_windows = not_skipped_windows[good]
        #
        #     if len(x_new) == len(x) or len(x_new) == 0:
        #         break
        #     x = x_new
        pass

    def __is_valid_poly(self):
        '''
        Checks if the polynomial is valid or not.
        :return: False if it is invalid, True otherwise.
        '''
        prev_fit = self.last_valid_fit
        current_fit = self.curr_fit
        if prev_fit is not None and current_fit is not None:
            self.__debug_text += 'Fit Diff: {:>.6f}'.format(
                np.sqrt(np.sum((prev_fit[0:1] - current_fit[0:1]) ** 2)))
            if np.sqrt(np.sum((prev_fit[0:1] - current_fit[0:1]) ** 2)) > .0005:
                self.__debug_text += ' Skipping frame'
                return False
        #
        # if current_fit is not None:
        #     square = current_fit[0] ** 2 + current_fit[1] ** 2
        #     # print(square)
        #     if square > 2.:
        #         return False

        return True

    def __is_valid_line(self):
        '''
        Checks if the current line is possibly valid.
        :return: False if the line is probably not valid, True otherwise.
        '''
        usable_windows = 0
        for window in self.__windows:
            if not window.skip:
                usable_windows += 1

        return usable_windows > 3

    def __line_coordinates(self, fit):
        '''
        :param fit: the fit to use to determine the line coordinates
        :return: the line coordinates. The y-s are linearly generated and the x-s are calculated using the fit provided.
        '''
        y_coors = np.linspace(0, self.__image_size[0] - 1, self.__image_size[0])
        x_coors = None
        if fit is not None:
            x_coors = (fit[0] * y_coors ** 2 +
                       fit[1] * y_coors + fit[2])

        return y_coors, x_coors

    def line_overlay_coordinates(self):
        '''
        :return: the line coordinates which can be used for overlaying, if there's a valid fit.
        '''
        y_coors, x_coors = self.__line_coordinates(self.curr_fit if self.curr_fit is not None else self.last_valid_fit)

        return None if x_coors is None else \
            np.int32(np.array([np.transpose(np.vstack([x_coors, y_coors]))]))

    def curvature(self):
        '''
        :return: the curvature of the line.
        '''
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
        '''
        :return: the window coordinates which were used to generate the current valid fit.
        '''
        window_positions = []
        for window in self.__windows:
            if not window.skip:
                window_positions.append(window.coordinates())

        return window_positions

    def debug_text(self):
        return self.__debug_text