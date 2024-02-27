import numpy as np
import cv2 as cv
import glob
from os.path import join
import matplotlib.pyplot as plt

def load_calibration_parameters(calib_dir):
    """
    Load saved calibration parameters from the specified directory, ensuring double precision format.
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
    cameraMatrixL, distL, cameraMatrixR, distR, R, T, Q = calibration_data

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

    # Compute the depth map
    depth_map = cv.reprojectImageTo3D(disparity, Q)

    return disparity, depth_map

def display_depth_and_disparity(left_images_dir, right_images_dir, calib_data, delay=0.1):
    """
    Automatically display depth and disparity maps side by side for each pair of stereo images.
    """
    left_images = sorted(glob.glob(join(left_images_dir, '*.png')))
    right_images = sorted(glob.glob(join(right_images_dir, '*.png')))

    plt.figure(figsize=(10, 5))

    for left_img_path, right_img_path in zip(left_images, right_images):
        disparity, depth_map = process_images(left_img_path, right_img_path, calib_data)

        plt.subplot(1, 2, 1)
        plt.imshow(disparity, cmap='magma')
        plt.title('Disparity Map')
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.imshow(depth_map[:, :, 2], cmap='jet')  # Displaying depth; consider only one channel
        plt.title('Depth Map')
        plt.colorbar()

        plt.pause(delay)
        plt.clf()

    plt.close()

def main():
    calib_dir = 'Calibration_Files'
    calib_data = load_calibration_parameters(calib_dir)
    
    left_images_dir = 'output/L'
    right_images_dir = 'output/R'
    
    display_depth_and_disparity(left_images_dir, right_images_dir, calib_data, delay=0.1)

if __name__ == '__main__':
    main()