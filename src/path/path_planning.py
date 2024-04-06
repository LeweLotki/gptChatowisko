import numpy as np
import matplotlib.pyplot as plt
from skimage.graph import route_through_array
from zdepthmap_processing import DepthPostProcessor

calib_dir = 'Calibration_Files'
left_images_dir = 'output/L'
right_images_dir = 'output/R'
depth_post_processor = DepthPostProcessor(calib_dir, left_images_dir, right_images_dir)
# Assuming depth_map_single_binary is the occupancy map (1s as free space, 0s as occupied)
# Inverting the map since route_through_array considers non-zero values as obstacles

plt.figure(figsize=(10, 7))

for occupancy_map in depth_post_processor.generate_occupancy_grid():
    
    plt.cla()
    
    occupancy_map = 1 - occupancy_map

    # Start point is the center column at the highest row, and end point is the same column at the lowest row
    rows, cols = occupancy_map.shape
    start = (0, cols // 2)  # Start at the top center
    end = (rows - 1, cols // 2)  # End at the bottom center

    # Compute the path using route_through_array from skimage.graph
    indices, weight = route_through_array(occupancy_map, start, end, fully_connected=True)

    # Convert indices to an array for plotting
    path = np.array(indices).T

    # Plotting the path on the occupancy map
    plt.imshow(occupancy_map, cmap='gray', interpolation='nearest')
    plt.plot(path[1], path[0], color='red')  # path[1] is x, path[0] is y
    plt.scatter([start[1], end[1]], [start[0], end[0]], color='blue')  # Start and end points
    plt.pause(0.1)
    
plt.close()
