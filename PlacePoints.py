import cv2

"""
Allow users to interactively place control points on the image.
These control points will be the anchor points for deformation.
"""
distorted_path = "fisheye.png"  # distorted image path
undistorted_path = "flat.png"  # undistorted image path
distorted_points_path = "points_fisheye.txt"  # distorted points path
undistorted_points_path = "points_flat.txt"  # undistorted points path

if __name__ == "__main__":
    fisheye = cv2.imread(distorted_path)
    flat = cv2.imread(undistorted_path)
    # show fisheye image
    cv2.imshow("fisheye", fisheye)
    # record the pixel user clicks
    points_fisheye = []

    def record(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points_fisheye.append((x, y))
            # mark the point user clicks
            cv2.circle(fisheye, (x, y), 3, (0, 0, 255), -1)
            cv2.imshow("fisheye", fisheye)
    cv2.setMouseCallback("fisheye", record)

    # show flat image and also record the pixel user clicks in another list
    cv2.imshow("flat", flat)
    points_flat = []
    def record_flat(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points_flat.append((x, y))
            cv2.circle(flat, (x, y), 3, (255, 0, 0), -1)
            cv2.imshow("flat", flat)
    cv2.setMouseCallback("flat", record_flat)
    cv2.waitKey(0)

    # define a function that append points to a txt file
    def store_points(points, filename):
        with open(filename, "a") as f:
            for point in points:
                f.write(f"{point[0]} {point[1]}\n")
    # store points to txt file
    store_points(points_fisheye, distorted_points_path)
    store_points(points_flat, undistorted_points_path)
    cv2.destroyAllWindows()
