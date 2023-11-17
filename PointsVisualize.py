import cv2
import numpy as np

"""
Visualize the control points and the final warped image.
"""
distorted_path = "fisheye.png"  # distorted image path
undistorted_path = "flat.png"  # undistorted image path
distorted_points_path = "points_fisheye.txt"  # distorted points path
undistorted_points_path = "points_flat.txt"  # undistorted points path

fisheye = cv2.imread(distorted_path)
flat = cv2.imread(distorted_path)
out = cv2.imread("output.png")
h, w = flat.shape[:2]
# read points from txt file
def read_points(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        points = []
        for line in lines:
            x, y = line.split()
            points.append((int(x), int(y)))
        return points

fisheye_points = np.array(read_points(distorted_points_path), dtype=float)
flat_points = np.array(read_points(undistorted_points_path), dtype=float)

# show point of both images with a number by the side
for i, point in enumerate(fisheye_points):
    cv2.circle(fisheye, tuple(point.astype(int)), 3, (0, 0, 255), -1)
    cv2.putText(fisheye, str(i), tuple(point.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
for i, point in enumerate(flat_points):
    cv2.circle(flat, tuple(point.astype(int)), 3, (255, 0, 0), -1)
    cv2.putText(flat, str(i), tuple(point.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.circle(out, tuple(point.astype(int)), 3, (255, 0, 0), -1)
    cv2.putText(out, str(i), tuple(point.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
cv2.imshow("fisheye", fisheye)
cv2.imshow("flat", flat)
cv2.imshow("out", out)
cv2.imwrite("fisheye_points.png", fisheye)
cv2.imwrite("flat_points.png", flat)
cv2.imwrite("out_points.png", out)
cv2.waitKey(0)
cv2.destroyAllWindows()