#**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # "Image References"

[calibration1]: ./examples/calibration_undistorted.png "Undistorted"
[calibration2]: ./examples/calibration_original.png "Original"
[undistorted1]: ./examples/undistorted.png "Undistorted Image"
[threshold1]: ./examples/threshold_less_sunlight.png "Binary with less sunlight"
[threshold2]: ./examples/threshold_more_sunlight.png "Binary with more sunlights"
[threshold3]: ./examples/threshold_partial_shade.png "Binary with partial shadow"
[warp1]: ./examples/warp_straight.png "Warp Example"
[fit1]: ./examples/fit_lines_warped.png "Fit Visual"
[output1]: ./examples/final_output.png "Output"
## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Computing the camera matrix and distortion coefficients aka calibrating the camera

The code for this step is contained in  `camera.py` between lines #32 and #59.   

I start by loading all the image files from the path (here `camera_cal/`). Then I prepare "object points" (`objp`, lines #42 and #43), which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `object_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `image_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `object_points` and `image_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][calibration2]

![alt text][calibration1]

### Pipeline (single images)

#### 1. Distortion correcting images

The hard part of distortion correcting an image is to calibrate the camera and compute the camera matrix and the distortion coefficients described above. After this is done, undistorting an image becomes as simple as calling `cv2.undistort` and passing in the already calculated camera matrix and coefficients.

The final result looks like this: 
![alt text][undistorted1]

#### 2. Thresholding the image

To generate the thresholded image I used a combination of color threshold from different color spaces and channels. Though gradient thresholding works mostly well on the project video, it introduces way too much noise on the challenge video. To make sure this doesn't happen the threshold values were tuned in a way, that made it mostly not contribute much to the final threshold image. Still, it would be useful in some harder frames on the project video (`threshold.py` between #61 and #74 ).

The color spaces and channels used, are the following

* HSV - S and V channels, both limited to lightness on the same pixel via HLS L channel
* HLS - L and S channels. L is only playing a limiting factor, so that not too much noise is introduced by other color spaces and channels
* LUV - V channel, which is especially good for yellow lines

Here are some examples of the resulting images. Note, that although in the last image the lines are barely visible, warping it will still give useful lines for the algorithm to find in a video (on a single image it won't find them because we are looking for the histogram spikes on the bottom half of the image).

![alt text][threshold1]

![alt text][threshold2]

![alt text][threshold3]

#### 3. Perspective transforming (aka warping) the image

My warping code resides in the Camera class in `camera.py` between lines #70 and #82. It takes the image as a parameter. The source points can be passed into the `__init__` method and they also have the default values set to my chosen values. The source points are basically hardcoded and the destination points are being calculated from those and the image size.

The resulting source and destination points are as follows:

|  Source   | Destination |
| :-------: | :---------: |
| 584, 458  |   295, 0    |
| 701, 458  |   1022, 0   |
| 295, 665  |  295, 710   |
| 1022, 665 |  1022, 710  |

I verified that my perspective transform was working as expected by drawing the `src` points onto a test image and verified that the lines appear parallel in the warped image.

![alt text][warp1]

#### 4. Finding lane lines and fitting a polynomial on them

The process to identify and fit lane lines are split into different stages and classes:

`window.py`: Finds the pixels for part of the lane line, and does basic validation (lines #34 to #79)

`line.py`: Manages the windows the line consists of. Fits a polynomial on the pixels gathered by the windows. Does validation across the windows and related to the previous polynomial fit. Also calculates the curvature of the line (lines #24 to #82)

`lanefinder.py`: Manages the left and right lines. Validates them against each other (lines #32 to #60)

Together they look something like this:

1. `lanefinder.find_lane()`:
   1. Find spikes in the histogram for left and right part of the image
   2. Call `left_line.update()` and `right_line.update()`
   3. Cross-validate the lines
2. `line.py`:
   1. Update all the windows calling `window.update()`
   2. Validate the windows against the previous fit
   3. Put together the pixels from all the window
   4. Validate the information
   5. Fit a polynomial to the pixels
3. `window.py`:
   1. If previous window was valid, search around that, if not, search around the center provided by the caller (`line.py`), which is based on the previous fit or the spike in the histogram.
   2. Do a convolution of the window and the search area of [-window.width, +window.width]
   3. Find the nonzero pixels centering the window at the maximum density area
   4. Validate the pixels (check if there are enough)

The resulting image looks like this:

![alt text][fit1]

#### 5. Calculating the curvature of the lines

I calculate the radius of the lines in  `line.py` (between lines #200 and #215). To calculate the radius I needed conversion from pixel space to real world space, which I did by hardcoding values for  `ym_per_pixel` and `xm_per_pixel` in `line.py`. Then I rescaled all the coordinates for the line pixels using these values and fit a line with those coordinates. With the now real world fit I calculated the curvature.

#### 6. Putting it altogether on an output image

The whole pipeline is kicked off from the `lane.py` (line #50 to #112). Processing the image is consist of the following steps:

1. Undistort the image using the already calibrated `Camera` (`camera.py`)
2. Threshold the undistorted image
3. Warp the thresholded (`Threshold` in `thresholder.py`) and the original image as well to birds eye view (both will be added to the final image to help understanding what's happening)
4. Find both lane lines using `Lanefinder` (`lanefinder.py`)
5. Overlay the lane lines to both the original and the warped binary image. This is done calling the helper methods in `vizutils.py` (lines #14 to #64). The window positions and the fit line is also overlayed on the binary image.
6. Add the curvature and position to the original image
7. Resized color warped, binary and binary warped images at the top of the final image, doing kind of a pip display.

The final image looks like this:

![alt text][output1]

---

### Pipeline (video)

#### 1. Now let's see them on some videos!

Here's a [link to my video result](./project_video_out.mp4)

Heres' another [link to the challenge video result](./challenge_video_out.mp4)

---

### Discussion

#### 1. Thresholding in different and changing light conditions

One problem to overcome in when trying to generate the binary image was the different light conditions. The color threshold values have to be tuned differently for images where there's a lot of light (hence there's not such a big difference between the road and the white/yellow lines) from where there's not so much light or where there's too much shade.

In order to not have to tune this for the different videos separately, I chose an adaptive algorithm (`threshold.py` between #22 and #59), which has it's maximum values set for very light conditions on the second video, and it adjust this gradually until it gets below but remain very close to a predefined threshold value (basically the horizontal average of column-wise sums of the number of white pixels on the bottom half of the image). While this already worked quite well, it had a problem with frames, where the car was coming out of a shade, where it would chose values which introduced too much noise on the sunny part of the image. To compensate for this, the algorithm splits the lower half of the image in two horizontally and makes sure there's not too much noise on any of them.

This works very well in all but 1-2 frames on bot the project and challenge videos. (It also works somewhat well on the harder challenge video, but it would need adjusting. I believe it would be possible to adjust it so that it works well on all 3 videos.)

#### 2. Where to look for the lane lines

Normally when the light conditions are good for the thresholding algorithm, the lane lines are visible from top to bottom on the image. However, even with the adaptive algorithm I have, there are cases, when the lane lines are only partially visible. In my case it's especially bad, when there are no lane lines visible on the bottom half of the image, as that's the part of the image the histogram is used to find possible places for the lines.

Fortunately it's not really necessary to overcome this problem on single images, but for the video it' s a must. However, in case of the video we have priori information from the previous frames.

Further, another problem might be properly finding dashed lines and in some cases, when there are not enough dashed lines available (e.g. maybe due to light conditions), deciding where to draw the line.

To overcome these problems, the basis of finding lane lines in my algorithm are the windows. Each window is responsible for finding the line at it's position, if it's present. If not, the window still can provide its pixels from a previous frame. Normally this information is good enough to be useful.

To find the lane lines windows use one of two informations:

1. If the previous window position was deemed valid, the new detection will use the previous windows center position as a base and a margin of the windows width (so basically the search area becomes twice as wide as the window itself)
2. If the previous window position was invalid, it uses the search center provided by the line.

The line also uses one of three informations to provide useful search center information for the window. This happen while iterating over the windows

1. If the just calculated position for the previous window is valid, use it's center
2. If it's not valid, calculate the x position of the previous line for the current windows y position and use that
3. Otherwise use the last used search position * 1.2, so that we search in a wider area. For the very first window, the position from the lanefinder is used, which is based on the histogram.

After updating all the windows, the pixels are collected. for windows where the current position was deemed invalid, or where there was no line to center it on, the previous position is used, as many as 10 times. However, in order to gradually decrease the influence of these windows when fitting the line, weights are being added in a way, that the more the fames a specific window was invalid the smaller the weight, when calculating the polynomial fit.

#### 3. Finding invalid windows, lines, line pairs

This was probably the hardest problem. One part of this is to eliminate as many sources of confusion as possible. Using adaptive thresholding, even if it's limited, and using windows which can look at a smaller area based on the information they have from the window below and their previous position goes a long way.

However, uneven road or uneven road color or maybe a car passing close enough to one of the lane lines provide a lot of noise which will confuse the algorithm.

I used several steps to figure out if a window or a line is invalid

* Need to have a minimum number of pixels in every window
* Every line have to have more than three windows (these can be currently or previously valid ones though)
* The pixels in each valid window has to be not further from the previous valid line than the window's width
* The lines shouldn't change too much compared to the previous fit
* The lines should be close to parallel

Besides these I tried a couple other things which didn't turn out to be too useful

* Finding windows which were more than 10% farther from their neighbor, than the median. This unfortunately more often weeded out valid windows than invalid ones.
* Removing noise from the image, by finding cluster of pixels not further then 2-3 pixels apart from each other and removing those with small number of pixels. This worked somewhat well, but was very slow, and usually the problematic noises are quite big cluster of pixels. Also tried DBSCAN here, but that didn't work well, it has a slightly different use case I guess.
* Removing lines which are too curved. I guess it might be helpful when we know the curves should be very big, but it would have been a problem on the harder challenge, and also seems to be too specific

#### 4. Ways to improve robustness

One of the biggest shortcomings of my algorithm in my opinion is, that it's too tailored to the conditions in the videos. Although it might be possible  to cover all the scenarios, there should be better ways to handle the problem. To have a more robust algorithm a couple ideas I can think of are:

* Better adaptive thresholding, which can work in a wider range of light conditions
* Stabilizing the video - shaking video cause the lines and their angles to move quite suddenly from frame to frame, which results in the need to allow a wider range of differences between frames, which in turn allows some of the invalid lines to creep in. I actually meant to add this, but run out of time...
* Changing the window size/behavior so that sharper turns are easier to find - one of the biggest problems I faced for the challenge video was, that on sharp turn the way the windows were stacked up vertically wasn't able to fully cover the line. This would need to change.
* Better filtering and predicting window positions - this should be possible using a Kalman filter. One of the most used applications for the Kalman filter are for gps navigation, as far as I know, so it should be relatively simple to use it in this case for both filtering and predicting.
* Beside lines, it would probably be better if the algorithm could follow pavements or road sides as well. Those could help predicting where the line should be.

