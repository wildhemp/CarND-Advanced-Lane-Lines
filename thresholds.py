import numpy as np
import cv2
from itertools import product


class Threshold:

    def __abs_sobel(self, sobel, sobel_thresh):
        abs_sobel = np.absolute(sobel)  # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        s_binary = np.zeros_like(scaled_sobel)
        s_binary[(scaled_sobel >= sobel_thresh[0]) & (scaled_sobel <= sobel_thresh[1])] = 1
        return s_binary

    def __color(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        luv = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)

        clahe = cv2.createCLAHE(2, tileGridSize=(8, 8))
        binary = None

        steps = (2, 2, 1, 7)
        for phase, step in product(range(2), range(11)):
            binary = np.zeros_like(hsv[:, :, 0])
            binary[((clahe.apply(hsv[:, :, 2]) > 200 + step * steps[0]) &
                    (hls[:, :, 1] > 180 + step * steps[0]))] = 1
            binary[((hsv[:, :, 1] < 30) & (hls[:, :, 1] > 190 + phase * step * steps[1]))] = 1
            binary[(clahe.apply(luv[:, :, 2]) > 160 + step * steps[2])] = 1
            binary[(clahe.apply(hls[:, :, 2]) > 50) & (hls[:, :, 1] > 120 + step * steps[3])] = 1

            if np.average(np.sum(binary[binary.shape[0] // 2:, :], axis=0)) < 23:
                break

        # print(np.average(np.sum(binary[binary.shape[0]//2:,:], axis=0)))
        return binary

    def threshold(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=9)

        grady = self.__abs_sobel(sobel_y, (85, 100))
        color_binary = self.__color(image)

        combined = np.copy(color_binary)
        combined[(grady == 1)] += 1

        return np.uint8(combined * 255)


def show_color_channels(path):
    test_image = cv2.resize(cv2.imread(path), (0, 0), fx=1. / 3., fy=1. / 3.)
    hls = cv2.cvtColor(test_image, cv2.COLOR_BGR2HLS)
    hls_img = np.concatenate((hls[:,:,0], hls[:, :, 1], hls[:, :, 2]), axis=1)

    hsv = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)
    hsv_img = np.concatenate((hsv[:,:,0], hsv[:, :, 1], hsv[:, :, 2]), axis=1)

    luv = cv2.cvtColor(test_image, cv2.COLOR_BGR2LUV)
    luv_img = np.concatenate((luv[:,:,0], luv[:, :, 1], luv[:, :, 2]), axis=1)

    lab = cv2.cvtColor(test_image, cv2.COLOR_BGR2LAB)
    lab_img = np.concatenate((lab[:,:,0], lab[:, :, 1], lab[:, :, 2]), axis=1)

    img = np.concatenate((hls_img, hsv_img, luv_img, lab_img), axis=0)

    cv2.imshow('HLS + HSV + LUV + LAB', np.dstack((img, img, img)))


def test_threshold_values(path):
    test_image = cv2.resize(cv2.imread(path), (0, 0), fx=1., fy=1.)

    binary = Threshold().threshold(test_image)
    histogram = np.sum(binary, axis=0)
    print(np.average(histogram))

    cv2.imshow('Color threshold', np.dstack((binary, binary, binary)))


if __name__ == '__main__':
    path = './debug_images/frame350.png'
    # path = './test_images/straight_lines2.jpg'
    show_color_channels(path)
    test_threshold_values(path)
    cv2.waitKey(0)

