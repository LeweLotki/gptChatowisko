import numpy as np
import cv2 as cv
import glob
from os import mkdir
from os.path import isdir, join

# Parameters
chessboardSize = (7, 7)
frameSize = (672, 376)
size_of_chessboard_squares_mm = 20
calibration_images_dir_left = 'images/stereoLeft/'
calibration_images_dir_right = 'images/stereoRight/'
output_dir = 'Calibration_Files'

# Prepare object points
objp = np.zeros((chessboardSize[0]*chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2) * size_of_chessboard_squares_mm

# Arrays to store object points and image points
objpoints = []  # 3d points in real world space
imgpointsL = []  # 2d points in left image plane
imgpointsR = []  # 2d points in right image plane

# Criteria for subpixel corner detection
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def find_chessboard_corners(images_dir, chessboardSize, criteria):
    """
    Find and refine chessboard corners in images from a given directory.
    """
    images = sorted(glob.glob(join(images_dir, '*.png')))
    objpoints_temp = []  # Temporary list to hold objpoints for valid images
    imgpoints_temp = []  # Temporary list to store image points for valid images
    for img_path in images:
        img = cv.imread(img_path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
        if ret:
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints_temp.append(corners2)
            objpoints_temp.append(objp)
    return objpoints_temp, imgpoints_temp

def calibrate_camera(objpoints, imgpoints, frameSize):
    """
    Calibrate the camera given object points and image points.
    """
    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
    return cameraMatrix, dist

def stereo_calibrate(objpoints, imgpointsL, imgpointsR, cameraMatrixL, distL, cameraMatrixR, distR, imageSize):
    """
    Perform stereo calibration to find the rotation and translation between two cameras.
    """
    flags = cv.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ret, cameraMatrixL, distL, cameraMatrixR, distR, R, T, E, F = cv.stereoCalibrate(
        objpoints, imgpointsL, imgpointsR, cameraMatrixL, distL, cameraMatrixR, distR, imageSize, criteria=criteria_stereo, flags=flags)
    return R, T, E, F

def save_calibration_files(output_dir, cameraMatrixL, distL, cameraMatrixR, distR, R, T):
    """
    Save the calibration parameters to files.
    """
    if not isdir(output_dir):
        mkdir(output_dir)
    np.savetxt(join(output_dir, 'cameraMatrixL.txt'), cameraMatrixL, fmt='%.5e')
    np.savetxt(join(output_dir, 'distL.txt'), distL, fmt='%.5e')
    np.savetxt(join(output_dir, 'cameraMatrixR.txt'), cameraMatrixR, fmt='%.5e')
    np.savetxt(join(output_dir, 'distR.txt'), distR, fmt='%.5e')
    np.savetxt(join(output_dir, 'R.txt'), R, fmt='%.5e')
    np.savetxt(join(output_dir, 'T.txt'), T, fmt='%.5e')

# Find chessboard corners for left and right images
objpointsL, imgpointsL = find_chessboard_corners(calibration_images_dir_left, chessboardSize, criteria)
objpointsR, imgpointsR = find_chessboard_corners(calibration_images_dir_right, chessboardSize, criteria)

# Ensure we only keep pairs where corners were found in both images
min_pairs = min(len(imgpointsL), len(imgpointsR))
objpoints = objpointsL[:min_pairs]
imgpointsL = imgpointsL[:min_pairs]
imgpointsR = imgpointsR[:min_pairs]

# Calibrate left and right cameras individually
cameraMatrixL, distL = calibrate_camera(objpoints, imgpointsL, frameSize)
cameraMatrixR, distR = calibrate_camera(objpoints, imgpointsR, frameSize)

# Perform stereo calibration
R, T, E, F = stereo_calibrate(objpoints, imgpointsL, imgpointsR, cameraMatrixL, distL, cameraMatrixR, distR, frameSize)

# Save calibration parameters
save_calibration_files(output_dir, cameraMatrixL, distL, cameraMatrixR, distR, R, T)

print("Calibration files saved.")

