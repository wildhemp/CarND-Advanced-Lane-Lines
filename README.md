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
[threshold1]: ./examples/threshold_less_sunlight.png "Binary with less sunlight"
[threshold2]: ./examples/threshold_more_sunlight.png "Binary with more sunlights"
[threshold3]: ./examples/threshold_partial_shade.png "Binary with partial shadow"
[warp1]: ./examples/warp_straight.png "Warp Example"
[fit1]: ./examples/fit_lines_warped.png "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in  `camera.py` between lines #32 and #59.   

I start by loading all the image files from the path (here `camera_cal/`). Then I prepare "object points" (`objp`, lines #42 and #43), which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `object_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `image_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `object_points` and `image_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][calibration2]

![alt text][calibration1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

To generate the thresholded image I used a combination of color threshold from different color spaces and channels. Though gradient thresholding works mostly well on the project video, it introduces way too much noise on the challenge video. To make sure this doesn't happen the threshold values were tuned in a way, that made it mostly not contribute much to the final threshold image. Still, it would be useful in some harder frames on the project video (`threshold.py` between #61 and #74 ).

The color spaces and channels used, are the following

* HSV - S and V channels, both limited to lightness on the same pixel via HLS L channel
* HLS - L and S channels. L is only playing a limiting factor, so that not too much noise is introduced by other color spaces and channels
* LUV - V channel, which is especially good for yellow lines

Another problem to overcome was, the different light conditions. The color threshold values have to be tuned differently for images where there's a lot of light (hence there's not such a big difference between the road and the white/yellow lines) from where there's not so much light or where there's too much shade.

In order to not have to tune this for the different videos separately, I chose an adaptive algorithm (`threshold.py` between #22 and #59), which has it's maximum values set for very light conditions on the second video, and it adjust this gradually until it gets below but remain very close to a predefined threshold value (basically the horizontal average of column-wise sums of the number of white pixels on the bottom half of the image). While this already worked quite well, it had a problem with frames, where the car was coming out of a shade, where it would chose values which introduced too much noise on the sunny part of the image. To compensate for this, the algorithm splits the lower half of the image in two horizontally and makes sure there's not too much noise on any of them.

This works very well in all but 1-2 frames on bot the project and challenge videos. (It also works somewhat well on the harder challenge video, but it would need adjusting. I believe it would be possible to adjust it so that it works well on all 3 videos.)

![alt text][threshold1]

![alt text][threshold2]

![alt text][threshold3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

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

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The pipeline to identify and fit lane lines are split into different stages and classes:

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

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculate the radius of the lines in  `line.py` (between lines #200 and #215). To calculate the radius I needed conversion from pixel space to real world space, which I did by hardcoding values for  `ym_per_pixel` and `xm_per_pixel` in `line.py`. Then I rescaled all the coordinates for the line pixels using these values and fit a line with those coordinates. With the now real world fit I calculated the curvature.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
