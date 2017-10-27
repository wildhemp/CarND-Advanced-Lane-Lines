from camera import Camera
from thresholds import Threshold
from lanefinder import LaneFinder
import numpy as np
import cv2
import argparse
import os
import vizutils
from moviepy.editor import VideoFileClip
import functools
import matplotlib.pyplot as plt



def test_calibrated_camera(camera):
    '''
    Tests the camera calibration and displays undistorted image.

    :param camera: the calibrated camera
    '''
    img = cv2.imread('./camera_cal/calibration1.jpg')
    cv2.imshow('Original image', cv2.resize(img, (0, 0), fx=1./3., fy=1./3.))
    cv2.imshow('Undistorted image', cv2.resize(camera.undistort(img), (0, 0), fx=1./3., fy=1./3.))
    cv2.waitKey(0)


def test_perspective_coordinates():
    '''
    Tests perspective coordinates displaying the polygon on the image then warping it.
    '''
    perspective_source = ((584, 458), (701, 458), (1022, 665), (295, 665))
    camera = Camera(9, 6, perspective_source=perspective_source)
    camera.calibrate()
    test_images = ['test_images/straight_lines2.jpg', 'test_images/test5.jpg']
    captions = ['Perspective straight lines', 'Perspective curved lines']
    for i, test_image in enumerate(test_images):
        image = camera.undistort(cv2.imread(test_image))
        image = vizutils.overlay_perspective_lines(image, np.array(perspective_source, dtype=np.int32))

        warped = camera.warp(image)

        image = cv2.resize(image, (0, 0), fx=.5, fy=.5)
        warped = cv2.resize(warped, (0, 0), fx=.5, fy=.5)
        final = np.concatenate((image, warped), axis=0)
        cv2.imshow(captions[i], final)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_image(camera, thresholder, lane_finder, image, plot_histogram=False, debug_text=False,
                  display_binary_overlayed=False):
    '''
    Finds the lane lines and the lane and produces an image which has the original image overalyed with the lane and the
    lane lines and also having the bords eye view and threshold images added as a pip.

    :param camera: the calibrated camera
    :param thresholder: the thresholder to use for creating the binary image
    :param lane_finder: the lane finder
    :param image: the image to find the lane on
    :return: the resulting image with overlays and pip images
    '''
    undistorted = camera.undistort(image)
    warped = camera.warp(undistorted)

    binary = thresholder.threshold(undistorted)
    binary_warped = camera.warp(binary)

    if plot_histogram:
        histogram = lane_finder.histogram(binary_warped)
        plt.plot(histogram)
        plt.show()

    left_line, right_line = lane_finder.find_lanes(binary_warped)
    left_coors, right_coors = left_line.line_overlay_coordinates(), right_line.line_overlay_coordinates()
    position = lane_finder.calculate_position()
    window_pos = np.concatenate((left_line.window_coordinates(), right_line.window_coordinates()))

    warped_overlayed = warped
    binary_warped_overlayed = vizutils.overlay_windows(binary_warped, window_pos)
    binary_warped_overlayed = vizutils.overlay_lane_lines(binary_warped_overlayed, left_coors, right_coors)
    if display_binary_overlayed:
        cv2.imshow('Fit lines', cv2.resize(binary_warped_overlayed, (0, 0), fx=.5, fy=.5))

    text_start_y = image.shape[0] // 3

    img_overlayed = vizutils.overlay_lane(image, left_coors, right_coors, camera)
    img_overlayed = vizutils.overlay_text(img_overlayed, '{:>6}m'.format(int(left_line.curvature())),
                                          10, text_start_y + 30)
    img_overlayed = vizutils.overlay_text(img_overlayed, '{:>6}m'.format(int( int(right_line.curvature()))),
                                          image.shape[1] - 200, text_start_y + 30)
    img_overlayed = vizutils.overlay_text(img_overlayed, '{:>1.2f}m'.format(position),
                                          image.shape[1] // 2 - 50, image.shape[0] - 20)

    if debug_text:
        img_overlayed = vizutils.overlay_text(img_overlayed, left_line.debug_text(),
                                              10, text_start_y + 70)
        img_overlayed = vizutils.overlay_text(img_overlayed, right_line.debug_text(),
                                              10, text_start_y + 110)

    warped_overlayed = cv2.resize(warped_overlayed, (0, 0), fx=.3, fy=.3)
    binary_resized = cv2.resize(np.dstack((binary, binary, binary)), (0, 0), fx=.3, fy=.3)
    binary_warped_overlayed = cv2.resize(binary_warped_overlayed, (0, 0), fx=.3, fy=.3)
    offs = 20
    third = img_overlayed.shape[1] // 3
    img_overlayed[offs:warped_overlayed.shape[0] + offs, offs:warped_overlayed.shape[1] + offs, :] = warped_overlayed
    img_overlayed[offs:binary_resized.shape[0] + offs, third + offs:binary_resized.shape[1] + third + offs, :] = \
        binary_resized
    img_overlayed[
        offs:binary_warped_overlayed.shape[0] + offs,
        2 * third + offs:binary_warped_overlayed.shape[1] + 2 * third + offs, :] = binary_warped_overlayed

    return img_overlayed


def process_test_images(camera, threshold, path='./test_images', show_images=False):
    '''
    Runs the whole pipeline on test images found on a given path (be it an image or a directory).
    '''
    images = {}
    if os.path.isdir(path):
        for f in os.listdir(path):
            images[f] = cv2.imread(os.path.join(path, f))
    else:
        images[os.path.basename(path)] = cv2.imread(path)

    for name, img in images.items():
        print('Processing image: ', name)
        final = process_image(camera, threshold, LaneFinder(), img)

        if show_images:
            cv2.imshow(name, final)

    if show_images:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video_image(camera, threshold, lane_finder, image):
    '''
    Processes an image from the video. First it converts the image to BGR from RGB, because that's what the pipeline
    uses.
    :return: The processed image.
    '''
    return cv2.cvtColor(process_image(
        camera,
        threshold,
        lane_finder,
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR)),
        cv2.COLOR_BGR2RGB)


def process_video(input_path, output_path, start, end):
    '''
    Runs the pipeline on every frame of a video and then combines it back into a video.
    '''
    camera = Camera(9, 6)
    camera.calibrate()
    threshold = Threshold()
    lane_finder = LaneFinder()

    clip1 = VideoFileClip(input_path).subclip(start, end)
    white_clip = clip1.fl_image(functools.partial(process_video_image, camera, threshold, lane_finder))
    white_clip.write_videofile(output_path, audio=False)


def save_debug_video_frames(input_path, output_path, start, end, fps):
    '''
    Saves the video frame for debugging purposes.
    :param input_path: the input video path
    :param output_path: the output directory to save the frames to
    :param start: the start position in the video, can be float
    :param end: the end position in the video, can be float
    :param fps: the fps to use, i.e. save 10 images per second, etc.
    '''
    clip1 = VideoFileClip(input_path)
    if end is None: end = clip1.end
    for frame in range(int(start * fps), int(end * fps)):
        clip1.save_frame(os.path.join(output_path, 'frame%d.png'%frame), frame / fps)


def main(args):
    if args.test_warp:
        test_perspective_coordinates()

    if args.test_camera:
        camera = Camera(9, 6)
        camera.calibrate()
        test_calibrated_camera(camera)

    if bool(args.images):
        camera = Camera(9, 6)
        camera.calibrate()
        threshold = Threshold()
        process_test_images(camera=camera, threshold=threshold, path=args.images, show_images=True)

    if bool(args.video):
        if args.debug:
            save_debug_video_frames(
                args.video[0], args.video[1], args.start if args.start is not None else 0, args.end, args.fps)
        else:
            process_video(args.video[0], args.video[1], args.start if args.start is not None else 0, args.end)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_camera', action='store_true', help='Tests camera calibration')
    parser.add_argument('--test_warp', action='store_true', help='Tests warping of the image')
    parser.add_argument('--images', type=str, help='Runs the pipeline on the images found at the given path')
    parser.add_argument('--video', type=str, nargs=2,
                        help='Runs the pipeline on the given video and saves it to the output path. If debug is'
                                      ' set, instead of running the pipeline, it saves the frames of the video.')
    parser.add_argument('--start', type=float, help='Start position in the video.')
    parser.add_argument('--end', type=float, help='End position in the video.')
    parser.add_argument('--fps', type=int, default=10, help='The fps to save frames from the video in.')
    parser.add_argument('--debug', action='store_true', help='Whether to debug the current process or not.')
    args = parser.parse_args()

    main(args)