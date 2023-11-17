import cv2
import numpy as np
import time

"""
Warp with the calculated deformation.
"""

distorted_path = "fisheye.png"  # distorted image path
fisheye = cv2.imread(distorted_path)
mapping = np.load("mapping.npz")
x_flow, y_flow = mapping["x_flow"], mapping["y_flow"]
output = cv2.remap(fisheye, x_flow.astype(np.float32), y_flow.astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
cv2.imwrite("output1.png", output)