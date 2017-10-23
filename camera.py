import numpy as np
import os
import cv2


class Camera:

    def __init__(self, nx, ny,
                 # lane_coordinates=((584, 458), (701, 458), (295, 665), (1022, 665)),
                 lane_coordinates=((584, 474), (701, 474), (295, 665), (1022, 665)),
                 path='./camera_cal'):
        self.nx = nx
        self.ny = ny
        self.path = path

        self.lane_source_coordinates = np.float32(lane_coordinates)

        self.camera_matrix = None
        self.distortion_coefficient = None

    def calibrate(self):
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

        _, self.camera_matrix, self.distortion_coefficient, _, _ = cv2.calibrateCamera(
            object_points, image_points, gray.shape[::-1], None, None)

    def undistort(self, image):
        return cv2.undistort(image, self.camera_matrix, self.distortion_coefficient)

    def warp(self, image):
        _, _, bottom_left, bottom_right = self.lane_source_coordinates
        image_height, image_width = image.shape[0:2]
        lane_dest_coordinates = np.float32([(bottom_left[0], 0), (bottom_right[0], 0),
                                            (bottom_left[0], image_height - 20), (bottom_right[0], image_height - 20)])
        M = cv2.getPerspectiveTransform(self.lane_source_coordinates, lane_dest_coordinates)
        return cv2.warpPerspective(image, M, (image_width, image_height))

    def unwarp(self, image):
        _, _, bottom_left, bottom_right = self.lane_source_coordinates
        image_height, image_width = image.shape[0:2]
        lane_dest_coordinates = np.float32([(bottom_left[0], 0), (bottom_right[0], 0),
                                            (bottom_left[0], image_height - 20), (bottom_right[0], image_height - 20)])
        M = cv2.getPerspectiveTransform(lane_dest_coordinates, self.lane_source_coordinates)
        return cv2.warpPerspective(image, M, (image_width, image_height))