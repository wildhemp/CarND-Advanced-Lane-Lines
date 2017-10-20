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


def test_calibrated_camera(camera):
    img = cv2.imread('./camera_cal/calibration1.jpg')
    cv2.imshow('img', camera.undistort(img))
    cv2.waitKey(0)


def process_image(camera, threshold, lane_finder, image):
    undistorted = camera.undistort(image)
    warped = camera.warp(undistorted)

    binary = threshold.threshold(undistorted)

    binary_warped = camera.warp(binary)

    lane_finder.find_lanes(binary_warped)
    left_coors, right_coors = lane_finder.line_overlay_coordinates()
    left_curv, right_curv = lane_finder.calculate_curvature()
    position = lane_finder.calculate_position()
    debug_window_pos = lane_finder.debug_window_coordinates()

    warped_overlayed = warped #vizutils.overlay_lane_lines(warped, left_coors, right_coors)
    binary_warped_overlayed = vizutils.overlay_lane_lines(binary_warped, left_coors, right_coors)
    if len(debug_window_pos) > 0:
        binary_warped_overlayed = vizutils.overlay_debug_windows(binary_warped_overlayed, debug_window_pos)

    text_start_y = image.shape[0] // 3

    img_overlayed = vizutils.overlay_lane(image, left_coors, right_coors, camera)
    img_overlayed = vizutils.overlay_text(img_overlayed, 'Curvature = left: {:>4.0f}m right: {:>4.0f}m'
                                          .format(int(left_curv), int(right_curv)),
                                          10, text_start_y + 30)
    img_overlayed = vizutils.overlay_text(img_overlayed, 'Position = {:>1.2f}m'.format(position),
                                          10, text_start_y + 70)

    top = img_overlayed[img_overlayed.shape[0] // 3:, :, :]
    bottom = cv2.resize(
        np.concatenate((warped_overlayed,
                        np.dstack((binary, binary, binary)),
                        binary_warped_overlayed), axis=1),
        (0, 0), fx=1. / 3., fy=1. / 3.)

    return np.concatenate((top, bottom), axis=0)


def process_test_images(camera, threshold, path='./test_images', show_images=False):
    images = {}
    if os.path.isdir(path):
        for f in os.listdir(path):
            images[f] = cv2.imread(os.path.join(path, f))
    else:
        images[os.path.basename(path)] = cv2.imread(path)

    for name, img in images.items():
        print('Processing image: ', name)
        final = process_image(camera, threshold, LaneFinder(debug_windows=True), img)

        if show_images:
            cv2.imshow(name, final)

    if show_images:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video_image(camera, threshold, lane_finder, image):
    return cv2.cvtColor(process_image(
        camera,
        threshold,
        lane_finder,
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR)),
        cv2.COLOR_BGR2RGB)


def process_video(input_path, output_path, start, end):
    camera = Camera(9, 6)
    camera.calibrate()
    threshold = Threshold()
    lane_finder = LaneFinder(debug_windows=True)

    clip1 = VideoFileClip(input_path).subclip(start, end)
    white_clip = clip1.fl_image(functools.partial(process_video_image, camera, threshold, lane_finder))
    white_clip.write_videofile(output_path, audio=False)


def save_debug_video_frames(input_path, output_path, start, end):
    clip1 = VideoFileClip(input_path)
    if end is None: end = clip1.end
    for frame in range(start * 10, end * 10):
        clip1.save_frame(os.path.join(output_path, 'frame%d.png'%frame), frame / 10.)


def main(args):
    camera = Camera(9, 6)
    camera.calibrate()

    threshold = Threshold()

    if args.test_camera:
        test_calibrated_camera(camera)

    if bool(args.images):
        process_test_images(camera=camera, threshold=threshold, path=args.images, show_images=True)

    if bool(args.video):
        if args.debug:
            save_debug_video_frames(
                args.video[0], args.video[1], args.start if args.start is not None else 0, args.end)
        else:
            process_video(args.video[0], args.video[1], args.start if args.start is not None else 0, args.end)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_camera', action='store_true')
    parser.add_argument('--images', type=str)
    parser.add_argument('--video', type=str, nargs=2)
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    main(args)