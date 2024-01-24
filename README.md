# Structured light algorithm using multiple cameras

Implementation of the structured light algorithm in python using two monochrome cameras and one projector. it is expected that the  camera and projector intrinsic and extrinsic parameters are known.


# Algorithm
- decode gray code pattern.
- clean up invalid pixels in decoded images.
- estimate the parameters of the projected planes.
- calculate the direction of each camera rays originating from a camera center and passing through each pixel.
- find the intersection of camera rays with the projected plane; this intersection yields the coordinates of the object's 3D point.


# Requirements
- intrinsic and extrinsic parameters of the camera
- intrinsic and extrinsic parameters of the projector
- OpenCV
- Open3D

# Examples
## Yeti dataset
### pointclouds
![](docs/yeti_anim1.png)

### left and right pointclouds
<img src="docs/yeti_left_00340.png" alt="drawing" width="33%"/><img src="docs/yeti_right_00340.png" alt="drawing" width="33%"/>

### joint pointclouds without texture
![](docs/yeti_anim2.png)

## Steve (from minecraft) dataset
### pointclouds
![](docs/steve_anim1.png)

### left and right pointclouds
<img src="docs/steve_left_00410.png" alt="drawing" width="33%"/>
<img src="docs/steve_right_00340.png" alt="drawing" width="33%"/>

### joint pointclouds without texture
![](docs/steve_anim2.png)
