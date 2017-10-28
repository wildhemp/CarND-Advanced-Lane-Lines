import numpy as np
import cv2


class Threshold:
    '''
    Class to threshold the image and convert it to binary in a way, that lane lines remain visible and become easily
    recognizable.
    '''

    def __abs_sobel(self, sobel, sobel_thresh):
        '''
        Gets the absolute value of the threshold, scales it and returns a binary image after applying the threshold.
        '''
        abs_sobel = np.absolute(sobel)  # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        s_binary = np.zeros_like(scaled_sobel)
        s_binary[(scaled_sobel >= sobel_thresh[0]) & (scaled_sobel <= sobel_thresh[1])] = 1
        return s_binary

    def __color(self, image):
        '''
        Applies an adaptive color threshold on the image. This takes into account various channels of the image in
        various color spaces, to best be able to detect lines.
        The adaptivity is needed in order to make it work with different light conditions, ie. shadows, more/less sunny
        days, etc.
        This is quite slow, because it tries different threshold values on different channels and compares the average
        number of points remaining after thresholding against a chose value. It tries to push the binary image below
        that value but as close to it as possible.
        :return: The thresholded binary image.
        '''
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        luv = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)

        clahe = cv2.createCLAHE(2, tileGridSize=(8, 8))
        binary = None

        steps = (2, 2, 1, 7)
        best_binary = None
        best_score = 0
        for phase in range(2):
            for step in range(15):
                binary = np.zeros_like(hsv[:, :, 0])
                binary[((clahe.apply(hsv[:, :, 2]) > 200 + step * steps[0]) &
                        (hls[:, :, 1] > 180 + step * steps[0]))] = 1
                binary[((hsv[:, :, 1] < 30) & (hls[:, :, 1] > 190 + phase * step * steps[1]))] = 1
                binary[(clahe.apply(luv[:, :, 2]) > 160 + step * steps[2])] = 1
                binary[(clahe.apply(hls[:, :, 2]) > 50) & (hls[:, :, 1] > 120 + step * steps[3])] = 1

                score_1 = np.average(np.sum(binary[binary.shape[0] // 2:binary.shape[0] // 4 * 3, :], axis=0))
                score_2 = np.average(np.sum(binary[binary.shape[0] // 4 * 3:, :], axis=0))
                if 13 > score_1 and 13 > score_2 and score_1 + score_2 > best_score:
                    best_score = score_1 + score_2
                    best_binary = binary
                    break

        return binary if best_binary is None else best_binary

    def threshold(self, image):
        '''
        Combines a color and a sobel threshold to convert the image to binary.
        '''
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=9)

        grady = self.__abs_sobel(sobel_y, (85, 100))
        color_binary = self.__color(image)

        combined = np.copy(color_binary)
        combined[(grady == 1)] = 1

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


def test_threshold_values(path, title):
    image = cv2.imread(path)
    test_image = cv2.resize(image, (0, 0), fx=1., fy=1.)

    binary = Threshold().threshold(test_image)
    histogram = np.sum(binary, axis=0)
    print(np.average(histogram))

    cv2.imshow(title,
               np.concatenate((cv2.resize(image, (0, 0), fx=1. / 3., fy=1. / 3.),
                               cv2.resize(np.dstack((binary, binary, binary)), (0, 0), fx = 1./3., fy=1./3.)), axis=1))


if __name__ == '__main__':
    path = './test_images/test6.jpg'
    test_threshold_values(path, 'Less sunlight')
    path = './debug_images/frame53.png'
    test_threshold_values(path, "More sunlight")
    path = './debug_images/frame46.png'
    test_threshold_values(path, "Partial shade ")
    # show_color_channels(path)
    cv2.waitKey(0)

