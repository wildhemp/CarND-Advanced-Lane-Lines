import numpy as np
import os
import cv2


class Camera:
    '''
    Camera class to capture the logic needed to calibrate a camera and to do warping/unwapring of an image.
    '''

    def __init__(self, nx, ny,
                 perspective_source=((584, 458), (701, 458), (1022, 665), (295, 665)),
                 # perspective_source=((584, 474), (701, 474), (1022, 665), (295, 665)),
                 path='./camera_cal'):
        '''
        Initializes the camera.
        :param nx: the number of corners horizontally on the checkerboard images used to calibrate the camera.
        :param ny: the number of corners vertically on the checkerboard images used to calibrate the camera.
        :param perspective_source: the polygon to use for perspective transforms on the source image
        :param path: the path to read the camera calibration images from.
        '''
        self.nx = nx
        self.ny = ny
        self.path = path

        self.perspective_source = np.float32(((perspective_source[0]), (perspective_source[1]),
                                              (perspective_source[3]), (perspective_source[2])))

        self.__camera_matrix = None
        self.__distortion_coefficients = None

    def calibrate(self):
        '''
        Calibrates the camera by reading the images from the path and finding chessboard corners and calculating camera
        matrix and distortion coefficients.
        :return:
        '''
        print('Calibrating camera...')
        images = []
        for f in os.listdir(self.path):
            images.append(cv2.imread(os.path.join(self.path, f)))

        objp = np.zeros((self.nx * self.ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1, 2)

        image_points = []
        object_points = []
        gray = None

        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)

            if ret:
                image_points.append(corners)
                object_points.append(objp)

        _, self.__camera_matrix, self.__distortion_coefficients, _, _ = cv2.calibrateCamera(
            object_points, image_points, gray.shape[::-1], None, None)

    def undistort(self, image):
        '''
        Undistorst the image using the matrix and distortion coefficients calculated using calibrate.
        :return: the undistorted image.
        '''
        return cv2.undistort(image, self.__camera_matrix, self.__distortion_coefficients) \
            if self.__camera_matrix is not None \
            else image

    def warp(self, image):
        '''
        Warps (to birds eye view) the image using the perspective transform coordinates give at initialization times.
        :return: the warped image
        '''
        _, _, bottom_left, bottom_right = self.perspective_source
        image_height, image_width = image.shape[0:2]
        offset = 0
        lane_dest_coordinates = np.float32([(bottom_left[0] + offset, 0), (bottom_right[0] - offset, 0),
                                            (bottom_left[0] + offset, image_height - 10),
                                            (bottom_right[0] - offset, image_height - 10)])
        M = cv2.getPerspectiveTransform(self.perspective_source, lane_dest_coordinates)
        return cv2.warpPerspective(image, M, (image_width, image_height))

    def unwarp(self, image):
        '''
        Un-warps (to camera view) the image using the perspective transform coordinates give at initialization times.
        :return: the un-warped image
        '''
        _, _, bottom_left, bottom_right = self.perspective_source
        image_height, image_width = image.shape[0:2]
        offset = 0
        lane_dest_coordinates = np.float32([(bottom_left[0] + offset, 0), (bottom_right[0] - offset, 0),
                                            (bottom_left[0] + offset, image_height - 10),
                                            (bottom_right[0] - offset, image_height - 10)])
        M = cv2.getPerspectiveTransform(lane_dest_coordinates, self.perspective_source)
        return cv2.warpPerspective(image, M, (image_width, image_height))