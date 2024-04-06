import numpy as np
import os
import pandas as pd
from zdepthmap_processing import DepthPostProcessor

# Define the ranges with floating-point values using numpy's arange
eps_array = np.arange(0.01, 1, 0.01)
taken_points_per_array = np.arange(0.1, 1,0.05)
min_samples_array = np.arange(15, 20, 1)

# Round the arrays if needed
eps_array = np.round(eps_array, 2)
taken_points_per_array = np.round(taken_points_per_array, 2)


# Function to save data to CSV

if os.path.exists("data.csv"):
        os.remove("data.csv")

#data = []

for eps in eps_array:
    for taken_points_per in taken_points_per_array:
        for min_samples in min_samples_array:
            try:
                #print(min_samples)
                calib_dir = 'Calibration_Files'
                left_images_dir = 'output/L'
                right_images_dir = 'output/R'
                depth_post_processor = DepthPostProcessor(calib_dir, left_images_dir, right_images_dir)
                
                depth_post_processor.display_photo(taken_points_per=taken_points_per, eps=eps, min_samples=min_samples)
                
                # data.append({
                #     'eps': eps,
                #     'taken_points_per': taken_points_per,
                #     'min_samples': min_samples,
                #     'count':depth_post_processor.count_nonzero,
                     
                # })
                
            except Exception as e:
                print(e)
                continue


# Read the CSV file into a DataFrame
#df = pd.read_csv('data.csv')

# Get unique combinations of 'eps', 'taken_points_per', and 'min_samples'
#unique_combinations = df[['eps', 'taken_points_per', 'min_samples']].drop_duplicates()

# Count the number of unique combinations
#num_unique_combinations = len(unique_combinations)

#print("Number of unique combinations:", num_unique_combinations)



