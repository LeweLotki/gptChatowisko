import numpy as np
import cv2 as cv
import glob
from os.path import isdir, join, exists
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_calibration_parameters(calib_dir):
    cameraMatrixL = np.loadtxt(join(calib_dir, 'cameraMatrixL.txt'), dtype=np.float64)
    distL = np.loadtxt(join(calib_dir, 'distL.txt'), dtype=np.float64)
    cameraMatrixR = np.loadtxt(join(calib_dir, 'cameraMatrixR.txt'), dtype=np.float64)
    distR = np.loadtxt(join(calib_dir, 'distR.txt'), dtype=np.float64)
    R = np.loadtxt(join(calib_dir, 'R.txt'), dtype=np.float64)
    T = np.loadtxt(join(calib_dir, 'T.txt'), dtype=np.float64)
    Q = np.loadtxt(join(calib_dir, 'Q.txt'), dtype=np.float64)
    R1 = np.loadtxt(join(calib_dir, 'RectifL.txt'), dtype=np.float64)
    R2 = np.loadtxt(join(calib_dir, 'RectifR.txt'), dtype=np.float64)
    P1 = np.loadtxt(join(calib_dir, 'ProjL.txt'), dtype=np.float64)
    P2 = np.loadtxt(join(calib_dir, 'ProjR.txt'), dtype=np.float64)
    return cameraMatrixL, distL, cameraMatrixR, distR, R, T, Q, R1, R2, P1, P2

def rectify_images(imgL, imgR, calibration_data):
    cameraMatrixL, distL, cameraMatrixR, distR, _, _, _, R1, R2, P1, P2 = calibration_data
    h, w = imgL.shape[:2]
    mapLx, mapLy = cv.initUndistortRectifyMap(cameraMatrixL, distL, R1, P1, (w, h), cv.CV_32FC1)
    mapRx, mapRy = cv.initUndistortRectifyMap(cameraMatrixR, distR, R2, P2, (w, h), cv.CV_32FC1)
    imgL_rect = cv.remap(imgL, mapLx, mapLy, cv.INTER_LINEAR)
    imgR_rect = cv.remap(imgR, mapRx, mapRy, cv.INTER_LINEAR)
    return imgL_rect, imgR_rect

def compute_disparity(imgL_rect, imgR_rect):
    numDisparities = int(1e2)
    blockSize = 5
    stereo = cv.StereoSGBM_create(
        minDisparity=0,
        numDisparities=numDisparities,
        blockSize=blockSize,
        P1=8 * 3 * blockSize**2,
        P2=32 * 3 * blockSize**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv.StereoSGBM_MODE_SGBM_3WAY
    )
    disparity = stereo.compute(imgL_rect, imgR_rect).astype(np.float32) / 16.0
    return disparity

def compute_depth_map(disparity, Q):
    points_3D = cv.reprojectImageTo3D(disparity, Q)
    return points_3D

def plot_3d_depth_map(points_3D):
    Z = points_3D[:, :, 2]
    mask = np.isfinite(Z)
    Z = np.where(mask, Z, np.nan)
    min_z = np.nanmin(Z)
    max_z = np.nanmax(Z)
    
    X, Y = np.meshgrid(range(Z.shape[1]), range(Z.shape[0]))
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', rstride=1, cstride=1, linewidth=0, antialiased=False)
    ax.set_title('Depth Map')
    ax.set_zlabel('Depth')
    ax.set_xlim(0, Z.shape[1])
    ax.set_ylim(0, Z.shape[0])
    ax.set_zlim(min_z, max_z)
    ax.view_init(elev=0, azim=-90)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Depth')
    plt.pause(100)
    plt.show()

def main():
    calib_dir = 'Calibration_Files'
    if not isdir(calib_dir) or not exists(join(calib_dir, 'cameraMatrixL.txt')):
        print("Calibration data not found.")
        return
    
    calibration_data = load_calibration_parameters(calib_dir)
    
    left_images_dir = 'output/L'
    right_images_dir = 'output/R'
    left_images = sorted(glob.glob(join(left_images_dir, '*.png')))
    right_images = sorted(glob.glob(join(right_images_dir, '*.png')))
    
    for left_img_path, right_img_path in zip(left_images, right_images):
        imgL = cv.imread(left_img_path, cv.IMREAD_GRAYSCALE)
        imgR = cv.imread(right_img_path, cv.IMREAD_GRAYSCALE)
        
        imgL_rect, imgR_rect = rectify_images(imgL, imgR, calibration_data)
        disparity = compute_disparity(imgL_rect, imgR_rect)
        points_3D = compute_depth_map(disparity, calibration_data[6])  # Q matrix is at index 6
        
        plot_3d_depth_map(points_3D)
        break
if __name__ == '__main__':
    main()

