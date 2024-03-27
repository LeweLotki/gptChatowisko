import numpy as np
import cv2 as cv
import glob
from os.path import join, basename
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
from stream import Stream

class Depth:
    def __init__(self, calib_dir, left_images_dir, right_images_dir):
        self.calib_dir = calib_dir
        self.left_images_dir = left_images_dir
        self.right_images_dir = right_images_dir
        self.calib_data = self.load_calibration_parameters()
        self.stream = Stream()

    def load_calibration_parameters(self):
        cameraMatrixL = np.loadtxt(join(self.calib_dir, 'cameraMatrixL.txt'), dtype=np.float64)
        distL = np.loadtxt(join(self.calib_dir, 'distL.txt'), dtype=np.float64)
        cameraMatrixR = np.loadtxt(join(self.calib_dir, 'cameraMatrixR.txt'), dtype=np.float64)
        distR = np.loadtxt(join(self.calib_dir, 'distR.txt'), dtype=np.float64)
        R = np.loadtxt(join(self.calib_dir, 'R.txt'), dtype=np.float64)
        T = np.loadtxt(join(self.calib_dir, 'T.txt'), dtype=np.float64)
        Q = np.loadtxt(join(self.calib_dir, 'Q.txt'), dtype=np.float64)

        return cameraMatrixL, distL, cameraMatrixR, distR, R, T, Q

    def process_images(self, left_img_path, right_img_path):
        # TODO
        cameraMatrixL, distL, cameraMatrixR, distR, _, _, Q = self.calib_data
        # imgL = cv.imread(left_img_path, cv.IMREAD_GRAYSCALE)
        # imgR = cv.imread(right_img_path, cv.IMREAD_GRAYSCALE)
        (imgL, imgR) = self.stream.get_single_frame()
        if imgL is None:
            print('Fatal error. Empty frame.')
        imgL = cv.cvtColor(imgL, cv.IMREAD_GRAYSCALE)
        imgR = cv.cvtColor(imgR, cv.IMREAD_GRAYSCALE)
        h, w = imgL.shape[:2]
        left_map_x, left_map_y = cv.initUndistortRectifyMap(cameraMatrixL, distL, None, cameraMatrixL, (w, h), cv.CV_32FC1)
        right_map_x, right_map_y = cv.initUndistortRectifyMap(cameraMatrixR, distR, None, cameraMatrixR, (w, h), cv.CV_32FC1)
        imgL_remapped = cv.remap(imgL, left_map_x, left_map_y, cv.INTER_LINEAR)
        imgR_remapped = cv.remap(imgR, right_map_x, right_map_y, cv.INTER_LINEAR)
        stereo = cv.StereoSGBM_create(minDisparity=0, numDisparities=16*6, blockSize=5, P1=8 * 3 * 5**2, P2=32 * 3 * 5**2, disp12MaxDiff=1, uniquenessRatio=15, speckleWindowSize=0, speckleRange=2, preFilterCap=63, mode=cv.STEREO_SGBM_MODE_SGBM_3WAY)
        disparity = stereo.compute(imgL_remapped, imgR_remapped).astype(np.float32) / 16.0
        return disparity

    def normalize_and_reverse_depth(self, depth_map):
        depth_valid = depth_map[depth_map > 0]
        depth_normalized = (depth_map - np.min(depth_valid)) / (np.max(depth_valid) - np.min(depth_valid))
        depth_normalized = 1 - depth_normalized
        depth_normalized[depth_map <= 0] = 0
        return depth_normalized

    @staticmethod
    def sort_numerically(file_paths):
        return sorted(file_paths, key=lambda x: int(re.search(r'\d+', basename(x)).group()))

    def display_3d_depth_map(self):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        left_images = self.sort_numerically(glob.glob(join(self.left_images_dir, '*.png')))
        right_images = self.sort_numerically(glob.glob(join(self.right_images_dir, '*.png')))

        for left_img_path, right_img_path in zip(left_images, right_images):
            disparity = self.process_images(left_img_path, right_img_path)
            depth = self.normalize_and_reverse_depth(disparity)
            h, w = depth.shape
            X, Y = np.meshgrid(np.arange(w), np.arange(h))
            X, Y, depth = X.flatten(), Y.flatten(), depth.flatten()
            mask = depth > 0
            ax.clear()
            ax.scatter(X[mask], Y[mask], depth[mask], c=depth[mask], cmap='viridis', marker='.')
            ax.set_title('Normalized and Corrected 3D Depth Map')
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Depth (Normalized and Reversed)')
            ax.view_init(elev=287, azim=270)
            plt.pause(0.1)
        plt.close()

if __name__ == '__main__':
    
    def main():
        
        calib_dir = 'Calibration_Files'
        left_images_dir = 'output/L'
        right_images_dir = 'output/R'
        depth_calculator = Depth(calib_dir, left_images_dir, right_images_dir)
        depth_calculator.display_3d_depth_map()
    
    main()
