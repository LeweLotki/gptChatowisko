import numpy as np
import cv2 as cv
import glob
from os.path import join, basename
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re

def load_calibration_parameters(calib_dir):
    """
    Load saved calibration parameters from the specified directory.
    """
    cameraMatrixL = np.loadtxt(join(calib_dir, 'cameraMatrixL.txt'), dtype=np.float64)
    distL = np.loadtxt(join(calib_dir, 'distL.txt'), dtype=np.float64)
    cameraMatrixR = np.loadtxt(join(calib_dir, 'cameraMatrixR.txt'), dtype=np.float64)
    distR = np.loadtxt(join(calib_dir, 'distR.txt'), dtype=np.float64)
    R = np.loadtxt(join(calib_dir, 'R.txt'), dtype=np.float64)
    T = np.loadtxt(join(calib_dir, 'T.txt'), dtype=np.float64)
    Q = np.loadtxt(join(calib_dir, 'Q.txt'), dtype=np.float64)

    return cameraMatrixL, distL, cameraMatrixR, distR, R, T, Q

def process_images(left_img_path, right_img_path, calibration_data):
    """
    Process a pair of images to compute the disparity and depth maps.
    """
    cameraMatrixL, distL, cameraMatrixR, distR, _, _, Q = calibration_data

    # Load images
    imgL = cv.imread(left_img_path, cv.IMREAD_GRAYSCALE)
    imgR = cv.imread(right_img_path, cv.IMREAD_GRAYSCALE)

    # Compute the maps for remapping the perspective
    h, w = imgL.shape[:2]
    left_map_x, left_map_y = cv.initUndistortRectifyMap(cameraMatrixL, distL, None, cameraMatrixL, (w, h), cv.CV_32FC1)
    right_map_x, right_map_y = cv.initUndistortRectifyMap(cameraMatrixR, distR, None, cameraMatrixR, (w, h), cv.CV_32FC1)

    # Remap the images
    imgL_remapped = cv.remap(imgL, left_map_x, left_map_y, cv.INTER_LINEAR)
    imgR_remapped = cv.remap(imgR, right_map_x, right_map_y, cv.INTER_LINEAR)

    # Initialize the stereo block matcher
    stereo = cv.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16*6,
        blockSize=5,
        P1=8 * 3 * 5**2,
        P2=32 * 3 * 5**2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # Compute the disparity map
    disparity = stereo.compute(imgL_remapped, imgR_remapped).astype(np.float32) / 16.0

    return disparity

def normalize_and_reverse_depth(depth_map):
    """Normalize and reverse depth values so that closer objects have higher values."""
    depth_valid = depth_map[depth_map > 0]  # Exclude non-positive values
    depth_normalized = (depth_map - np.min(depth_valid)) / (np.max(depth_valid) - np.min(depth_valid))
    depth_normalized = 1 - depth_normalized  # Reverse depth
    depth_normalized[depth_map <= 0] = 0  # Keep invalid depth as 0
    return depth_normalized


def sort_numerically(file_paths):
    """Sort file paths based on the numerical value in their names."""
    return sorted(file_paths, key=lambda x: int(re.search(r'\d+', basename(x)).group()))

def display_3d_depth_map(left_images_dir, right_images_dir, calib_data):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    left_images = sort_numerically(glob.glob(join(left_images_dir, '*.png')))
    right_images = sort_numerically(glob.glob(join(right_images_dir, '*.png')))

    for left_img_path, right_img_path in zip(left_images, right_images):
        disparity = process_images(left_img_path, right_img_path, calib_data)
        depth = normalize_and_reverse_depth(disparity)  # Use the adjusted normalization
        
        h, w = depth.shape
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        X, Y, depth = X.flatten(), Y.flatten(), depth.flatten()

        mask = depth > 0  # Filtering out zero depth
        ax.clear()  # Clear previous data
        ax.scatter(X[mask], Y[mask], depth[mask], c=depth[mask], cmap='viridis', marker='.')
        
        ax.set_title('Normalized and Corrected 3D Depth Map')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Depth (Normalized and Reversed)')
        ax.view_init(elev=287, azim=270)  # Adjusted view angles
        
        plt.pause(0.1)  # Adjust the pause if necessary to achieve the desired FPS
    plt.close()

def main():
    calib_dir = 'Calibration_Files'
    left_images_dir = 'output/L'
    right_images_dir = 'output/R'
    calib_data = load_calibration_parameters(calib_dir)
    display_3d_depth_map(left_images_dir, right_images_dir, calib_data)

if __name__ == '__main__':
    main()