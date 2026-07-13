# Air Brightness Controller
A real-time, gesture-controlled brightness adjustment tool built with OpenCV. Move your hand up or down in front of your webcam to control video brightness live — no colored markers, no ML models, just classic computer vision.

 -------
### How it works
Your hand is detected using skin-tone HSV color segmentation. Its vertical position in the frame is mapped to a brightness value, which is applied to the live video feed in real time, along with an on-screen percentage bar.
Features

Real-time hand detection via HSV thresholding
Noise-resistant mask cleanup using morphological operations
Contour-based hand tracking
Live brightness mapping and visual feedback bar
No external ML models or dependencies beyond OpenCV/NumPy

Show your hand to the webcam. Move it up = brighter, down = dimmer. Press q to quit.

### Tech Stack
Python, OpenCV, NumPy

### Author
Divisha Tapadiya

