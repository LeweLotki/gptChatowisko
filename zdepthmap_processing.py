from depth import Depth
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import glob
from os.path import join
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

class DepthPostProcessor(Depth):
    def __init__(self, calib_dir, left_images_dir, right_images_dir):
        super().__init__(calib_dir, left_images_dir, right_images_dir)

    def filter_isolated_points(self, depth_map):
        
        h, w = depth_map.shape
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        points = np.stack((X.flatten(), Y.flatten(), depth_map.flatten()), axis=-1)
        
        # Filter out points with zero depth
        points_filtered = points[points[:, 2] > 0]
        
        # Randomly select 10% of the points to reduce computational load
        num_points = points_filtered.shape[0]
        indices = np.random.choice(num_points, size=int(num_points * 0.15), replace=False)
        points_sampled = points_filtered[indices]
        
        # Normalize X, Y, and depth values
        scaler = StandardScaler()
        points_normalized = scaler.fit_transform(points_sampled)
        
        # Apply DBSCAN clustering on normalized points
        clustering = DBSCAN(eps=0.05, min_samples=18).fit(points_normalized)
        labels = clustering.labels_
        
        # Reconstruct depth map with points that belong to clusters (excluding noise)
        new_depth_map = np.zeros(depth_map.shape)
        for point, label in zip(points_sampled, labels):
            if label != -1:  # If the point is not labeled as noise
                x, y, depth = point
                new_depth_map[int(y), int(x)] = depth
        
        return new_depth_map

    def filter_floor_points(self, depth_map):
        h, w = depth_map.shape
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        points = np.stack((X.flatten(), Y.flatten(), depth_map.flatten()), axis=-1)
        
        # Filter out points with zero depth
        points_filtered = points[points[:, 2] > 0]

        # Keep points that are in the bottom 20% of the image height (floor area)
        floor_threshold_min = 0.5 * h  # Minimum Y value for floor points
        floor_points = points_filtered[(points_filtered[:, 1] >= floor_threshold_min)]

        # Reconstruct depth map with floor points only
        new_depth_map = np.zeros(depth_map.shape)
        for x, y, depth in floor_points:
            if 0 <= int(x) < w and 0 <= int(y) < h:
                new_depth_map[int(y), int(x)] = depth

        return new_depth_map

    def display_filtered_3d_depth_map(self):
        
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        left_images = super().sort_numerically(glob.glob(join(self.left_images_dir, '*.png')))
        right_images = super().sort_numerically(glob.glob(join(self.right_images_dir, '*.png')))

        for left_img_path, right_img_path in zip(left_images, right_images):
            disparity = super().process_images(left_img_path, right_img_path)
            depth = super().normalize_and_reverse_depth(disparity)
            filtered_depth = self.filter_isolated_points(depth)
            floor_filtered_depth = self.filter_floor_points(filtered_depth)  # Apply floor level filtering

            h, w = floor_filtered_depth.shape
            X, Y = np.meshgrid(np.arange(w), np.arange(h))
            X, Y, depth = X.flatten(), Y.flatten(), floor_filtered_depth.flatten()

            mask = depth > 0
            ax.clear()
            ax.scatter(X[mask], Y[mask], depth[mask], c=depth[mask], cmap='viridis', marker='.')

            # Set titles and labels
            ax.set_title('Filtered Normalized and Corrected 3D Depth Map')
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Depth (Normalized and Reversed)')

            # Adjust the view angle for better visualization
            ax.view_init(elev=287, azim=270)

            # Fix the axes ranges
            ax.set_xlim([0, w])
            ax.set_ylim([0, h])
            max_depth = depth[mask].max() if mask.any() else 1  # Use the maximum depth if any, otherwise 1
            ax.set_zlim([0, max_depth])

            plt.pause(1)  # Short pause to allow the plot to update
        plt.close()

    def project_to_xz_plane(self, depth_map):
        """
        Project the depth map to the XZ plane.
        """
        z_resolution = depth_map.shape[0]
        # Flatten the depth map and get non-zero depth values
        h, w = depth_map.shape
        X, Z = np.meshgrid(np.arange(w), np.arange(h))
        non_zero_mask = depth_map > 0

        # Project onto XZ plane
        X = X[non_zero_mask]
        Z = depth_map[non_zero_mask]

        # Map depth values to Z indices
        Z_indices = np.round(Z * (z_resolution - 1)).astype(int)

        # Initialize the new matrix with dimensions X size by Z(depth) size
        xz_projection = np.zeros((w, z_resolution))

        # Fill the projection matrix
        for i in range(len(X)):
            xz_projection[X[i], Z_indices[i]] = 1

        # Return the transposed matrix to match the orientation
        return (xz_projection.T)[::-1] != 1

    def fill_unknown_space(self, occupancy_map):
        
        filled_map = np.ones(occupancy_map.shape)
        for col in range(occupancy_map.shape[1]):  # Iterate through each column
            for row in reversed(range(occupancy_map.shape[0])):  # Start from the bottom of the column
                if occupancy_map[row, col] == 0:  # If a 1 is found
                    filled_map[:row, col] = 0  # Set all rows above this row in the column to 1
                    break  # Once the first 1 is found and action is taken, stop checking this column

        return filled_map
        
    def display_xz_projection(self):
        
        plt.figure(figsize=(10, 7))

        left_images = super().sort_numerically(glob.glob(join(self.left_images_dir, '*.png')))
        right_images = super().sort_numerically(glob.glob(join(self.right_images_dir, '*.png')))

        for left_img_path, right_img_path in zip(left_images, right_images):
            disparity = super().process_images(left_img_path, right_img_path)
            depth = super().normalize_and_reverse_depth(disparity)
            filtered_depth = self.filter_isolated_points(depth)
            limited_depth = self.filter_floor_points(filtered_depth)
            xz_projection = self.project_to_xz_plane(limited_depth)
            occupancy_map = self.fill_unknown_space(xz_projection)

            plt.imshow(occupancy_map, cmap='gray')
            plt.title('XZ Plane Projection')
            plt.xlabel('X axis')
            plt.ylabel('Z axis (Depth)')
            plt.pause(1)  # Display each frame for 0.1 seconds

        plt.close()


if __name__ == '__main__':
    
    def main():
        calib_dir = 'Calibration_Files'
        left_images_dir = 'output/L'
        right_images_dir = 'output/R'
        depth_post_processor = DepthPostProcessor(calib_dir, left_images_dir, right_images_dir)
        depth_post_processor.display_xz_projection()
        # depth_post_processor.display_filtered_3d_depth_map()

    
    main()
