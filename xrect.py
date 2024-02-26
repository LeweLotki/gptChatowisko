import numpy as np
import cv2 as cv
from os.path import join

def load_calibration_parameters(calib_dir):
    """
    Load saved calibration parameters from the specified directory and ensure they are of type np.float64 (double precision).
    """
    cameraMatrixL = np.loadtxt(join(calib_dir, 'cameraMatrixL.txt'), dtype=np.float64)
    distL = np.loadtxt(join(calib_dir, 'distL.txt'), dtype=np.float64)
    cameraMatrixR = np.loadtxt(join(calib_dir, 'cameraMatrixR.txt'), dtype=np.float64)
    distR = np.loadtxt(join(calib_dir, 'distR.txt'), dtype=np.float64)
    R = np.loadtxt(join(calib_dir, 'R.txt'), dtype=np.float64)
    T = np.loadtxt(join(calib_dir, 'T.txt'), dtype=np.float64)
    
    return cameraMatrixL, distL, cameraMatrixR, distR, R, T

def save_rectification_and_projection_matrices(calib_dir, rectL, rectR, projMatrixL, projMatrixR, Q):
    """
    Save the computed stereo rectification and projection matrices.
    """
    np.savetxt(join(calib_dir, 'RectifL.txt'), rectL, fmt='%.6e')
    np.savetxt(join(calib_dir, 'RectifR.txt'), rectR, fmt='%.6e')
    np.savetxt(join(calib_dir, 'ProjL.txt'), projMatrixL, fmt='%.6e')
    np.savetxt(join(calib_dir, 'ProjR.txt'), projMatrixR, fmt='%.6e')
    np.savetxt(join(calib_dir, 'Q.txt'), Q, fmt='%.6e')

def compute_and_save_stereo_rectification(calib_dir):
    """
    Compute stereo rectification and projection matrices in double precision and save them.
    """
    cameraMatrixL, distL, cameraMatrixR, distR, R, T = load_calibration_parameters(calib_dir)
    
    # Ensure the imageSize parameter matches the actual size of your images.
    imageSize = (672, 376)  # Adjust as necessary
    
    # Now using np.float64 for all matrices to meet OpenCV's expectation for cv.stereoRectify
    RL, RR, PL, PR, Q, validPixROI1, validPixROI2 = cv.stereoRectify(
        cameraMatrixL.astype(np.float64), distL.astype(np.float64), 
        cameraMatrixR.astype(np.float64), distR.astype(np.float64), 
        imageSize, R.astype(np.float64), T.astype(np.float64), 
        None, None, None, None, None, 
        cv.CALIB_ZERO_DISPARITY, 1)
    
    save_rectification_and_projection_matrices(calib_dir, RL, RR, PL, PR, Q)
    print("Stereo rectification and projection matrices have been saved.")

if __name__ == "__main__":
    calib_dir = 'Calibration_Files'
    compute_and_save_stereo_rectification(calib_dir)

