import numpy as np
import cv2 as cv
import glob
from os import mkdir
from os.path import isdir, join
from stream.stream import Stream

class Calibration:

    def __init__(self, output_dir="calibration_parameteres"):
            
        # Parameters
        self.chessboardSize = (7, 7)
        self.frameSize = (672, 376)
        self.size_of_chessboard_squares_mm = 20
        # self.calibration_images_dir_left = 'images/stereoLeft/'
        # self.calibration_images_dir_right = 'images/stereoRight/'
        self.output_dir = output_dir
        self.stream = Stream()
        # Prepare object points
        self.objp = np.zeros((self.chessboardSize[0]*self.chessboardSize[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.chessboardSize[0], 0:self.chessboardSize[1]].T.reshape(-1, 2) * self.size_of_chessboard_squares_mm

        # Arrays to store object points and image points
        self.objpoints = []  # 3d points in real world space
        self.objpointsL = []  # 3d points in real world space
        self.objpointsR = []  # 3d points in real world space
        self.imgpointsL = []  # 2d points in left image plane
        self.imgpointsR = []  # 2d points in right image plan

        # Criteria for subpixel corner detection
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def find_chessboard_corners(self):
        """
        Find and refine chessboard corners in images from a given directory.
        """
        # images = sorted(glob.glob(join(images_dir, '*.png')))
        # objpoints_temp = []  # Temporary list to hold objpoints for valid images
        # imgpoints_temp = []  # Temporary list to store image points for valid images
        
        for i in range(100):

            # frames = self.stream.get_single_frame()
            (imgL, imgR) = self.stream.get_single_frame()
            objpointsL_temp, imgpointsL_temp  = self.get_imgpoints(imgL)            
            objpointsR_temp, imgpointsR_temp = self.get_imgpoints(imgR)

            self.objpointsL.extend(objpointsL_temp)
            self.imgpointsL.extend(imgpointsL_temp)
            self.objpointsR.extend(objpointsR_temp)
            self.imgpointsR.extend(imgpointsR_temp)
            
            #cv.imshow("img",imgL)
            #cv.waitKey(1)
            #print(objpointsL_temp)

    def get_imgpoints(self, frame):
        
        objpoints_temp = []  # Temporary list to hold objpoints for valid images
        imgpoints_temp = []  # Temporary list to store image points for valid images
                
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, self.chessboardSize, None)
        if ret:
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
            imgpoints_temp.append(corners2)
            objpoints_temp.append(self.objp)
            # print(corners)
            fnl = cv.drawChessboardCorners(frame,self.chessboardSize, corners, ret)
            cv.imshow("fnl", fnl)
            cv.waitKey(1)


        return objpoints_temp, imgpoints_temp

    def calibrate_camera(self, objpoints, imgpoints, frameSize):
        """
        Calibrate the camera given object points and image points.
        """
        ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
        return cameraMatrix, dist

    def stereo_calibrate(self,objpoints, imgpointsL, imgpointsR, cameraMatrixL, distL, cameraMatrixR, distR, imageSize):
        """
        Perform stereo calibration to find the rotation and translation between two cameras.
        """
        flags = cv.CALIB_FIX_INTRINSIC
        criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        ret, cameraMatrixL, distL, cameraMatrixR, distR, R, T, E, F = cv.stereoCalibrate(
            objpoints, imgpointsL, imgpointsR, cameraMatrixL, distL, cameraMatrixR, distR, imageSize, criteria=criteria_stereo, flags=flags)
        return R, T, E, F

    def save_calibration_files(self, cameraMatrixL, distL, cameraMatrixR, distR, R, T):
        """
        Save the calibration parameters to files.
        """
        if not isdir(self.output_dir):
            mkdir(self.output_dir)
        np.savetxt(join(self.output_dir, 'cameraMatrixL.txt'), cameraMatrixL, fmt='%.5e')
        np.savetxt(join(self.output_dir, 'distL.txt'), distL, fmt='%.5e')
        np.savetxt(join(self.output_dir, 'cameraMatrixR.txt'), cameraMatrixR, fmt='%.5e')
        np.savetxt(join(self.output_dir, 'distR.txt'), distR, fmt='%.5e')
        np.savetxt(join(self.output_dir, 'R.txt'), R, fmt='%.5e')
        np.savetxt(join(self.output_dir, 'T.txt'), T, fmt='%.5e')

   
    def run(self):

        # Find chessboard corners for left and right images
        self.find_chessboard_corners()

        # Ensure we only keep pairs where corners were found in both images
        min_pairs = min(len(self.imgpointsL), len(self.imgpointsR))
        self.objpoints = self.objpointsL[:min_pairs]
        self.imgpointsL = self.imgpointsL[:min_pairs]
        self.imgpointsR = self.imgpointsR[:min_pairs]

        # Calibrate left and right cameras individually
        cameraMatrixL, distL = self.calibrate_camera(self.objpoints, self.imgpointsL, self.frameSize)
        cameraMatrixR, distR = self.calibrate_camera(self.objpoints, self.imgpointsR, self.frameSize)

        # Perform stereo calibration
        R, T, E, F = self.stereo_calibrate(self.objpoints, self.imgpointsL, self.imgpointsR, cameraMatrixL, distL, cameraMatrixR, distR, self.frameSize)

        # Save calibration parameters
        self.save_calibration_files(cameraMatrixL, distL, cameraMatrixR, distR, R, T)

        print("Calibration files saved.")



