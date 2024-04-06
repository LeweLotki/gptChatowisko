import numpy as np
import os
import pandas as pd
from zdepthmap_processing import DepthPostProcessor

# Define the ranges with floating-point values using numpy's arange
eps_array = np.arange(0.01, 1, 0.01)
taken_points_per_array = np.arange(0.1, 1, 0.05)
min_samples_array = np.arange(15, 20, 1)

# Round the arrays if needed
eps_array = np.round(eps_array, 2)
taken_points_per_array = np.round(taken_points_per_array, 2)
min_samples_array = np.round(min_samples_array)

# Function to save data to CSV
def save_to_csv(data, filename='data.csv'):
    if os.path.exists(filename):
        os.remove(filename)
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

data = []

for eps in eps_array:
    for taken_points_per in taken_points_per_array:
        for min_samples in min_samples_array:
            try:
                calib_dir = 'Calibration_Files'
                left_images_dir = 'output/L'
                right_images_dir = 'output/R'
                depth_post_processor = DepthPostProcessor(calib_dir, left_images_dir[20:40], right_images_dir[20:40])
                depth_post_processor.display_photo()
                
                data.append({
                    'eps': eps,
                    'taken_points_per': taken_points_per,
                    'min_samples': min_samples
                })
                
            except Exception as e:
                print(e)
                continue

# Save the collected data to CSV
save_to_csv(data)
