import cv2
import mediapipe as mp
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
#mp_drawing = mp.solutions.drawing_utils

""" # Define text offset values
x_offset = -4  # Adjust this value for horizontal offset
y_offset = 2   # Adjust this value for vertical offset """

# Input and output folder paths
input_folder = r'D:\Term 3 2024\COMP9444\COMP9444_project\Dataset\Yoga-82\yoga_dataset_links'
output_csv = r'D:\Term 3 2024\COMP9444\COMP9444_project\Dataset\Yoga-82\pose_landmarks.csv'

# Dictionary to map class labels to integer values
index = 0
class_labels = {}
for folder_name in os.listdir(input_folder):
    if os.path.isdir(os.path.join(input_folder, folder_name)):
        class_labels[folder_name] = index
        index += 1


print("Class Label Mapping:", class_labels)

# Prepare a list to store data for each image
data_rows = []




# Process each subfolder in the input directory
for subfolder in os.listdir(input_folder):
    subfolder_path = os.path.join(input_folder, subfolder)
    
    # Only process if it's a directory
    if os.path.isdir(subfolder_path):
        class_label = class_labels[subfolder]  # Get the integer label for this class
        
        # Process each image in the input folder
        for filename in os.listdir(subfolder_path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                # Read the image
                image_path = os.path.join(subfolder_path, filename)
                image = cv2.imread(image_path)

                if image is None:
                    continue  # Skip if image can't be read

                image_height, image_width, _ = image.shape

                # Convert the image to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Process the image and find the pose
                result = pose.process(image_rgb)

                # Create a blank numpy array with zeros
                keypoints_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

                # Draw the keypoints on the blank image
                if result.pose_landmarks:
                    # Flatten x, y, z coordinates of all landmarks into a single row
                    row = []
                    for landmark in result.pose_landmarks.landmark:
                        row.extend([landmark.x, landmark.y, landmark.z])
                    
                    # Append the class label to the row
                    row.append(class_label)
                    
                    # Add this row to our data list
                    data_rows.append(row)
                    

# Define column names
num_landmarks = 33
columns = [f'{coord}_{i}' for i in range(num_landmarks) for coord in ['x', 'y', 'z']]
columns.append('class_label')  # Add column for class label

# Create a DataFrame and save it as a CSV file
df = pd.DataFrame(data_rows, columns=columns)
df.to_csv(output_csv, index=False)

print("CSV file created:", output_csv)
