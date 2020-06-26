## Advanced lane detection and tracking project

---

**Main goals**

The goal is to detect and track the lane lines in the project video from the front-facing camera on the ego vehicle.

The steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

All the code are presented in Advanced Lane Tracking.ipynb.

[//]: # (Image References)

[image1]: output_images/undistorted5.png "Undistorted"
[image2]: camera_cal/calibration5.jpg "Original image"
[image3]: test_images/test3.jpg "test3"
[image4]: output_images/pers1.png "Original image"
[image5]: output_images/pers2.png "Top-down view"
[image6]: output_images/only_s.png "Only S ch."
[image7]: output_images/test_roi2.png 
[image8]: output_images/test_un2.jpg 

[image9]: output_images/color_RGB1.png "R"
[image10]: output_images/color_RGB2.png "G"
[image11]: output_images/color_RGB3.png "B"
[image12]: output_images/color_HLS1.png "H"
[image13]: output_images/color_HLS2.png "L"
[image14]: output_images/color_HLS3.png "S"

[image15]: output_images/histo.png "Histogram"
[image16]: output_images/window.png "Sliding window"
[image17]: output_images/targeted.png "Targeted search"
[image18]: output_images/histo.png "Histogram"
[image19]: output_images/histo.png "Histogram"
[image20]: output_images/histo.png "Histogram"
[image21]: output_images/histo.png "Histogram"

<!-- [image22]: ./examples/example_output.jpg "Output" -->

[video1]: ./project_video.mp4 "Video"

---

### Camera Calibration

The transformation from 3D object to 2D image plane isn't perfect, due to the distortion from lenses and different camera configurations. Therefore, camera calibration is necessary in order to obtain actual object data. Most common target for camera calibration is chessboard.

1. Load all the pictures of the chessboard taken by the same camera(preferably from different angles). 
2. Define two lists for image points(2D) and object points(3D). The object points are the (x, y, z) coordinates of the chessboard corners in the world, which should be the same for all images. Here we assume that the all the points on the chessboard lie in x, y plane hence z = 0.
3. For each image, turn into grayscale and run `cv2.findChessboardCorners()` to get the corners. 
4. If the corners are found, store them in the list of image points. 
5. Pass these two lists into `cv2.calibrateCamera()` and get the intended coefficients. 

Note that this calibration only needs to be run once and the parameters remain the same throughout the project. `C_mtx` is the intrinsic camera matrix. Distortion coefficients contains all the coefficients for radial and tangential distortion.

The code is shown in the function `calibration()`. 


Original image             |  Undistorted image
:-------------------------:|:-------------------------:
![alt text][image2] |  ![alt text][image1]

### Evaluation and testing

In order to select better combination, a set of functions are created and comparison between different methods or approaches is discussed in this section. The function `plot_compare()` is intended to plot the result and original image side by side, thereby easily visualizign the difference. This helps to determine which type of color and gradient thresholding give better result.

#### Sobel operator 

Sobel operator is used for edge detection(the input of Canny detection is simply the output of Sobel operator). First `cv2.Sobel()` is used to calculate the gradient in x and y direction. Then the magnitude and orientation of the gradient can be determined by simple trigonometry. Put these outputs into thresholding function and we can acquire the binary images.

The code is shown in the functions `sobel_magnitude()`, `sobel_direction()`, and `sobel_xy()`.
The following images are generated from test4.jpg.

Sobel in x direction            |  Sobel in y direction
:-------------------------:|:-------------------------:
![alt text](output_images/sobel_x.png) |  ![alt text](output_images/sobel_y.png)

Magnitude of Sobel         |  Orientation of Sobel 
:-------------------------:|:-------------------------:
![alt text](output_images/sobel_mag.png) |  ![alt text](output_images/sobel_dir.png)

In order to find out the best case for gradient thresholding, I plot the combined binary image of few cases. Y direction contain horizontal lines, which would be seen as noisy when fitting the lane. As for x direction, it is less noisy than the case of magnitude and orientation. Besides, the detected lines are more consistent in this case, which is reasonable since lane lines are close to vertical lines while viewing from the driver's position. Therefore, I decided to use Sobel operator in only x direction. 

Comparison:

x direction           |  x + y direction
:-------------------------:|:-------------------------:
![alt text](output_images/sobel_x.png) |  ![alt text](output_images/xy.png)

Magnitude + Orientation         |  All four combined
:-------------------------:|:-------------------------:
![alt text](output_images/mag_ori.png) |  ![alt text](output_images/all.png)

#### Color space

When dealing with edge detection, we usually convert the original images to grayscale images. This results in a loss of color information, e.g. the lane color. Thus, we can also extract useful information from color space in order to obtain better binary images. A binary image is just a representation of the original image with only 0 and 1 values.

Here are the types of color space that I've tested: RGB, HSV, HLS, Lab, Luv, YUV, YCrCb. They use the same function from OpenCV `cv2.cvtColor(image, cv2.COLOR_RGB2{color_space})`. Then single channel of each color space is separated and passed into a thresholding function to get the binary image.

The result shows that not every channel is able to detect lanes due to the properties of different color spaces. In addition, some of the channels are sensitive to the color change of the road(the brightness of the road color), e.g. B of RGB and L of Lab. This will result in unstable lane tracking when the vehicle passes through this road segment. Hence these two cases have good detection in some images but perform poorly on others. Note that RGB color space works best on white lane pixels. Therefore, S channel from HLS is more stable among all the test images. 

The following images are genereated from test4.jpg and the threshold is equal to (170, 255).

![alt text](test_images/test4.jpg "Original image")

R  |  G |  B
:-------------------------:|:-------------------------:|:-------------------------:
![alt text](output_images/color_RGB1.png) | ![alt text](output_images/color_RGB2.png "G") | ![alt text](output_images/color_RGB3.png "B") 
H  |  S |  V
:-------------------------:|:-------------------------:|:-------------------------:
![alt text](output_images/color_HSV1.png "H") |![alt text](output_images/color_HSV2.png "S") |![alt text](output_images/color_HSV3.png "V")
![alt text](output_images/color_HLS1.png "H") ![alt text](output_images/color_HLS2.png "L") ![alt text](output_images/color_HLS3.png "S")

![alt text](output_images/color_Lab1.png "L") ![alt text](output_images/color_Lab2.png "a") ![alt text](output_images/color_Lab3.png "b")

![alt text](output_images/color_Luv1.png "L") ![alt text](output_images/color_Luv2.png "u") ![alt text](output_images/color_Luv3.png "v")

![alt text](output_images/color_YUV1.png "Y") ![alt text](output_images/color_YUV2.png "U") ![alt text](output_images/color_YUV3.png "V")

![alt text](output_images/color_YCrCb1.png "Y") ![alt text](output_images/color_YCrCb2.png "Cr") ![alt text](output_images/color_YCrCb3.png "Cb")



#### Regoin of interest

The region of interest(ROI) is set to the region around the current staying lane. This is also related to the coordinate of source point for the perspective transform. 


![alt text]

Original image with ROI            |  Undistorted image
:-------------------------:|:-------------------------:
![alt text](output_images/test_roi2.png) |  ![alt text](output_images/test_un2.png)
Original image with ROI            |  Undistorted image
:-------------------------:|:-------------------------:
![alt text](output_images/test_roi3.png) |  ![alt text](output_images/test_un3.png)
Original image with ROI            |  Undistorted image
:-------------------------:|:-------------------------:
![alt text](output_images/test_roi4.png) |  ![alt text](output_images/test_un4.png)
Original image with ROI            |  Undistorted image
:-------------------------:|:-------------------------:
![alt text](output_images/test_roi5.png) |  ![alt text](output_images/test_un5.png)
Original image with ROI            |  Undistorted image
:-------------------------:|:-------------------------:
![alt text](output_images/test_roi6.png) |  ![alt text](output_images/test_un6.png)
Original image with ROI            |  Undistorted image
:-------------------------:|:-------------------------:
![alt text](output_images/test_roi7.png) |  ![alt text](output_images/test_un7.png)



### Pipeline (single image or frame)

The code is presented in the function `pipeline()` and here we use test.jpg as an example.

![alt text] [image3]

#### 1. Obtain undistorted image

To demonstrate this step, the camera matrix and distortion coefficients are required. The undistorted image can be obtained by using `cv2.undistort()`.

The code is shown in the function `undistorted()`.

Original image             |  Undistorted image
:-------------------------:|:-------------------------:
![alt text][image7] |  ![alt text][image8]

#### 2. Color transform and gradient thresholding

A combination of color and gradient thresholds is used to generate a binary image. From the plots, it is evident that the S channel from HLS is relatively stable. However, the tree shaddow in test5.jpg also affect the performance. This can be addressed by including other color spaces. Therefore, the selected color transform are S channel from HLS and B channel from RGB with a threshold of (170, 255).

According to the evaluation, the Sobel gradient in x direction is used, along with a threshold of (25, 150). The lower bound(minimum of threshold) should not be too large, otherwise we won't acuiqre enough data points to detect the lanes. The combined binary image will be the input of the next step. 

The code is shown in the function `color_thresholding()`.

![alt text][image9] ![alt text][image10] ![alt text][image11]
![alt text][image12] ![alt text][image13] ![alt text][image14]

#### 3. Perspective transform

Perspective transform allows us to transform the camera view point into bird's-eye view. First obtain the grayscale image. Here I made the function more general so that it can accept both colored and binary images as input. Then use cv2.getPerspectiveTransform() to get the transform matrix. The inverse of transform matrix can be determined by switching the order of source and destination points. Finally, use `cv2.warpPerspective()` to generate a warped image.

The code is shown in the function `perspective_transform()` and the source and destination points are shown in the following table. Note that we can decide the points for these two input. This will affect the performance of lane detection. 


| Source        | Destination   | 
|:-------------:|:-------------:| 
| 560, 460      | 160, 0        | 
| 160, 720      | 160, 720      |
| 1200, 720     | 1200, 720     |
| 760, 460      | 1200, 0       |


![alt text][image4] 
![alt text][image5] 
![alt text][image6] 

#### 4. Define a class for lanes

The class `Line()` is created to store all the information of the lanes. Here I decide to optimize the performance by using `collections.deque()`. It's a list-like container which is designated to efficiently add or remove elements from both end.

Such as: 
- buffer_size: the size of the buffer that stores previous Nth detections
- yp: y coordinates of each frame(height of the image)
- current_fit: the coefficients of current best-fit polynomial
- current_fitx: x coordinates of current best-fit polynomial
- check: check if the sanity checks are passed or not.



#### 5. Identified lane-line pixels and polynomial fitting

1. Calculate the histogram of the warped binary image and find out the maximum of each side. Here the bottom half of the original image is used to identify the peaks. (`histogram()`)
![alt text][image15]
2. Use this as the starting points of left and right lane.
3. Set the parameters for sliding windows, namely the window size, margin from the starting points, minimum number of pixels found to recenter window.
4. Create empty lists to receive left and right lane pixel indices.
5. Loop through all the window, define and draw the boundaries of each window. (`sliding_window()`)
6. Search for those activated pixels(white points) that fall into the window.
7. Store the indices of these points in the lists. If the number of discovered points is larger than minpix, recenter the starting points. 
8. Return the pixel coordinate of all these discovered points.
9. Use `np.polyfit()` to get the coefficients of the best-fit second-order polynomial. (`poly_fitting()`)
10. Execute `ring_buffer()` to perform sanity check, add current detection to the averaging list and calculate the average.
11. Return the unwarped image and visualize the result.(`visualize()`)
12. A subwindow in the top-right corner is created for the purpose of debugging and visualization. It shows the bird's-eye view of current lane finding process.

![alt text][image16]

A moving averaging method `ring_buffer()` is used to stablize and smooth the result. This method takes a fixed amount `buffer_size` of buffer of previous lane detection and obtain the average of these values. If the size of current list exceeds the buffer size, then the earliest information will be discarded.

Note that it is inefficient to go through all the windows for every singel frame. An better alternative is to reuse the previous polynomial and search inside a particular range(margin) around the previous line. If no points has been found, then we go back to sliding window search. (`search_around_previous()` and `lane_finding()`) 

![alt text][image17]

As for the function `sanity_check()`, few examinations are carried out in this part. This ensures that erroneous detections are properly processed and will not affect the overall estimation.

1. If the distance between right and left lane line is too small or even negative-valued, then go back to sliding window search.
2. If the radius of curvature is smaller than the minimum or larger than the maximum, this detection is viewed as noisy and thus is discarded. 
3. If the change between current and previous detection is large, 



#### 6. Radius of curvature and the position of the vehicle with respect to center

The code for calculation of the radius of curvature is shown in `curvature()` and it's based on the equation below.  

<img src="https://latex.codecogs.com/gif.latex?R_c = \frac{[1+(\frac{dx}{dy})^2]^{3/2}}{\left | \frac{d^2x}{dy^2} \right |} " /> 

As for the position of the center of ego car, first calculate the x coordinate of the lane center using the last point of both fitting lines. Then the center position can be determined by the difference between camera center and the lane center and turning it into length in meter.(`get_offset()`)



#### 7. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

1. Sharp change of curvature

2. Rapid change of brightness

3. Time complexity

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. 

When it comes to roads with large curves, the sliding window search is unable to change or recenter the window, due to the lack of discovered points(< minpix). This will result in continuous stacking of window, and thus incorrect fitting. 

