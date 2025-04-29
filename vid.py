import cv2
import numpy as np
from scipy.signal import convolve2d


cap = cv2.VideoCapture(0)


filter = np.array([
   [0, -1, 0],
   [-1, 4, -1],
   [0, -1, 0]
]).astype(float)

# blur = np.array([
#    [1/9, 1/9, 1/9],
#    [1/9, 1/9, 1/9],
#    [1/9, 1/9, 1/9]
# ]).astype(float)


success, last = cap.read()
while cap.isOpened():
    success, img = cap.read()
    if success:
        gray = cv2.cvtColor(img - last, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("Result", convolve2d(gray, filter / 255))
        cv2.imshow("Result", last and gray)
        # last = last + img
        # last = last / 2
        last = img
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
