# 3D-Laser-Scanner
Exam of the Geometric and 3D Computer Vision, prof. Filippo Bergamasco, Ca' Foscari University of Venice, A.Y. 2019-2020

## Instructions
The requirements can be found in the `docs` folder. In there you will also find my afternotes of the course.

## Before running the code
Download the videos from [here](https://drive.google.com/drive/folders/16plqQTJDrx6IOw-fuAVIqDZnMRrcrwZx?usp=sharing) and put them in the `videos/` folder.

## Calibrating the camera
`python src/CameraCalibrator.py`. Add the `--debug` flag for debug images. However notice that, since we are working on 50 images, only one image out of 10 is displayed with debug information for brevity. This means that you will see 5 triples of debug images, wait for few seconds for the camera calibration to actually happen, and finally see 3 undistorted images to check the quality of the undistortion process, and hence verify that the camera has successfully been calibrated.

## Running the code for 3D reconstruction
`python main.py`. Add the `--debug` flag to see debug images, and the `--filename <cup1.mp4|cup2.mp4|soap.mp4|puppet.mp4>` flag to specify which video to use.

## Dependencies
In order to install the required dependencies, running `conda env create --file environment.yml` should be enough. In case it is not, manually install the following packages via Pip:

* `numpy`
* `opencv-python`
* `open3d`
* `sklearn`

Finally, the following tools have been used while developing:

* `black`
* `isort`
* `mypy`
* `flake8`

## Processing times for 3D reconstruction
Computed on an MSI PS63 laptop: i7-8565U, 16GB DDR4 RAM, 500GB SSD. Tests performed on Pop_OS! 20.04.

| File name  | Video length | Processing Time |
|------------|--------------|-----------------|
| soap.mp4   | 1min 12s     | 38s             |
| cup1.mp4   | 1min 02s     | 32s             |
| cup2.mp4   | 45s          | 26s             |
| puppet.mp4 | 53s          | 28s             |

The videos are all 15FPS ones.

Given this data, the average processing time per frame is 0.035 seconds, with a standard deviation of 0.0016 seconds.

According to PyCharm's profiler, most of the processing time is required by:
* **unavoidable OpenCV functions**: reading a frame, undistorting it, displaying it, etcetera. In particular, undistorting an image requires ~42% of the total runtime, reading from the VideoCapture object adds an 8% of the total runtime to that, and the `imshow/waitKey` operation requires an additional 14%.

* **finding the laser points**, which takes ~13% of the total time. Here most of the time is required by running DBSCAN in order to remove outliers: this could be avoided, for instance by enforcing a single "laser point" per line and finetuning thresholds even more. However, I chose to get as many points as possible and filter them with DBSCAN to try and retrieve as much information from the video as possible. 

* **computing plane-ray intersections**, which requires ~9% of the total time. This is also quite unavoidable, and only depends on how many exiting rays are being considered. The only solution for reducing the time required would be to consider fewer points.

All in all, the code seems to be sufficiently efficient. It could definitely work faster by using a lower level language such as C++ and performing some additional optimizations, but the current performance already seem quite reasonable.

![](https://i.imgur.com/1xuOQL8.png)


## Acknowledgments
* Prof. Filippo Bergamasco for the help in developing the project, the course material and the code snippets.