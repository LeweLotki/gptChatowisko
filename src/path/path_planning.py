import numpy as np
import matplotlib.pyplot as plt
from skimage.graph import route_through_array
from depth.depth_postprocessing import DepthPostProcessor


class PathPlanner:

    def __init__(self, calib_dir):
        
        self.depth_post_processor = DepthPostProcessor(calib_dir)
        self.__get_path()

    def __get_path(self):

        plt.figure(figsize=(10, 7))

        for occupancy_map in self.depth_post_processor.generate_occupancy_grid():
            
            plt.cla()
            
            occupancy_map = 1 - occupancy_map

            rows, cols = occupancy_map.shape
            start = (0, cols // 2)  # Start at the top center
            end = (rows - 1, cols // 2)  # End at the bottom center

            indices, weight = route_through_array(occupancy_map, start, end, fully_connected=True)

            path = np.array(indices).T

            plt.imshow(occupancy_map, cmap='gray', interpolation='nearest')
            plt.plot(path[1], path[0], color='red')  # path[1] is x, path[0] is y
            plt.scatter([start[1], end[1]], [start[0], end[0]], color='blue')  # Start and end points
            plt.pause(0.01)
            
        plt.close()
