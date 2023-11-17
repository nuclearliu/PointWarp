import cv2
import numpy as np
from scipy.optimize import curve_fit

"""
Use multinomial functions to interpolate the deformation between control points.
"""

distorted_path = "fisheye.png"  # distorted image path
undistorted_path = "flat.png"  # undistorted image path
distorted_points_path = "points_fisheye.txt"  # distorted points path
undistorted_points_path = "points_flat.txt"  # undistorted points path

fisheye = cv2.imread(distorted_path)
flat = cv2.imread(undistorted_path)
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

def func1(x,a,b,c,d,e,f):
    y=a*x[0]*x[0]+b*x[1]*x[1]+c*x[0]*x[1]+d*x[0]+e*x[1]+f
    return y

# another multinomial function with much more parameters to fit
def func2(x, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, y, z, aa, ab, ac, ad, ae, af, ag, ah, ai):
    x1, x2 = x
    y = a*x1*x1 + b*x2*x2 + c*x1*x2 + d*x1 + e*x2 + f + g*x1*x1*x1 + h*x2*x2*x2 + i*x1*x1*x2 + j*x1*x2*x2 + k*x1*x1*x1*x1 + l*x2*x2*x2*x2 + m*x1*x2*x1*x2 + n*x1*x1*x1*x2 + o*x1*x2*x2*x2\
        + p*x1*x1*x1*x1*x1 + q*x2*x2*x2*x2*x2 + r*x1*x1*x1*x1*x2 + s*x1*x2*x2*x2*x2 + t*x1*x1*x1*x1*x1*x1 + u*x2*x2*x2*x2*x2*x2 + v*x1*x1*x1*x1*x1*x2 + w*x1*x2*x2*x2*x2*x2 + y*x1*x1*x1*x1*x1*x1*x1 + z*x2*x2*x2*x2*x2*x2*x2\
        + aa*x1*x1*x1*x1*x1*x1*x2 + ab*x1*x1*x1*x1*x1*x2*x2 + ac*x1*x1*x1*x1*x2*x2*x2 + ad*x1*x1*x2*x2*x2*x2*x2 + ae*x1*x1*x1*x1*x1*x1*x1*x1 + af*x2*x2*x2*x2*x2*x2*x2*x2 + ag*x1*x1*x1*x1*x1*x1*x1*x2 + ah*x1*x1*x1*x1*x1*x1*x2*x2\
        + ai*x1*x1*x1*x1*x1*x2*x2*x2
    return y

def eighth_order_polynomial(x, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a50):
    x1, x2 = x
    return (a0 + a1 * x1 + a2 * x2 + a3 * x1 ** 2 + a4 * x1 * x2 + a5 * x2 ** 2 + a6 * x1 ** 3 + a7 * x1 ** 2 * x2 + a8 * x1 * x2 ** 2 + a9 * x2 ** 3 + a10 * x1 ** 4 + a11 * x1 ** 3 * x2 + a12 * x1 ** 2 * x2 ** 2 + a13 * x1 * x2 ** 3 + a14 * x2 ** 4 + a15 * x1 ** 5 + a16 * x1 ** 4 * x2 + a17 * x1 ** 3 * x2 ** 2 + a18 * x1 ** 2 * x2 ** 3 + a19 * x1 * x2 ** 4 + a20 * x2 ** 5 + a21 * x1 ** 6 + a22 * x1 ** 5 * x2 + a23 * x1 ** 4 * x2 ** 2 + a24 * x1 ** 3 * x2 ** 3 + a25 * x1 ** 2 * x2 ** 4 + a26 * x1 * x2 ** 5 + a27 * x2 ** 6 + a28 * x1 ** 7 + a29 * x1 ** 6 * x2 + a30 * x1 ** 5 * x2 ** 2 + a31 * x1 ** 4 * x2 ** 3 + a32 * x1 ** 3 * x2 ** 4 + a33 * x1 ** 2 * x2 ** 5 + a34 * x1 * x2 ** 6 + a35 * x2 ** 7 + a36 * x1 ** 8 + a37 * x1 ** 7 * x2 + a38 * x1 ** 6 * x2 ** 2 + a39 * x1 ** 5 * x2 ** 3 + a40 * x1 ** 4 * x2 ** 4 + a41 * x1 ** 3 * x2 ** 5 + a42 * x1 ** 2 * x2 ** 6 + a43 * x1 * x2 ** 7 + a44 * x2 ** 8)



f = func2  # for 130 points, func2 should suffice
# fit the function with the control points
popt, pcov = curve_fit(f, flat_points.T, fisheye_points[:,0])
X = np.linspace(0, w-1, w)
Y = np.linspace(0, h-1, h)
X, Y = np.meshgrid(X, Y)
# calculate the deformation of each pixel
coords = np.append(np.expand_dims(X, axis=0), np.expand_dims(Y, axis=0), axis=0)
x_flow = f(coords, *popt)
popt, pcov = curve_fit(f, flat_points.T, fisheye_points[:,1])
y_flow = f(coords, *popt)
# save the deformation
np.savez("mapping.npz", x_flow=x_flow, y_flow=y_flow)
# apply the deformation to the image
output = cv2.remap(fisheye, x_flow.astype(np.float32), y_flow.astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
cv2.imwrite("output.png", output)