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
output_folder = r'D:\Term 3 2024\COMP9444\COMP9444_project\Dataset\Yoga-82\slices_output'
os.makedirs(output_folder, exist_ok=True)

# Dictionary to map class labels to integer values
index = 0
class_labels = {}
for folder_name in os.listdir(input_folder):
    if os.path.isdir(os.path.join(input_folder, folder_name)):
        class_labels[folder_name] = index
        index += 1


print("Class Label Mapping:", class_labels)


# Define rotation angles
angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]  # Different angles in degrees


# Process each subfolder in the input directory
for subfolder in os.listdir(input_folder):
    subfolder_path = os.path.join(input_folder, subfolder)
    
    # Only process if it's a directory
    if os.path.isdir(subfolder_path):
        class_label = class_labels[subfolder]  # Get the integer label for this class
        class_output_folder = os.path.join(output_folder, subfolder)
        os.makedirs(class_output_folder, exist_ok=True)

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

                """ # Create a blank numpy array with zeros
                keypoints_image = np.zeros((image_height, image_width, 3), dtype=np.uint8) """


                if result.pose_landmarks:
                    
                    # Extract the x, y, z coordinates of each landmark
                    x_coords = [landmark.x for landmark in result.pose_landmarks.landmark]
                    y_coords = [landmark.y for landmark in result.pose_landmarks.landmark]
                    z_coords = [landmark.z for landmark in result.pose_landmarks.landmark] 

                    # Plot the normalized coordinates in 3D
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(x_coords, y_coords, z_coords, c='r', marker='o')

                    # Draw lines between connected landmarks
                    for connection in mp_pose.POSE_CONNECTIONS:
                        start_idx, end_idx = connection
                        ax.plot(
                            [x_coords[start_idx], x_coords[end_idx]],
                            [y_coords[start_idx], y_coords[end_idx]],
                            [z_coords[start_idx], z_coords[end_idx]],
                            color='blue'
                        )

                    # Label the plot
                    ax.set_title(f'Pose Landmarks - Class {class_label} - {filename}')
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    plt.show()








                   """  # Extract the x, y, z coordinates of each landmark
                    x_coords = [landmark.x for landmark in result.pose_landmarks.landmark]
                    y_coords = [landmark.y for landmark in result.pose_landmarks.landmark]
                    z_coords = [landmark.z for landmark in result.pose_landmarks.landmark] 

                    # Create 3D plot and take snapshots from different angles
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(x_coords, y_coords, z_coords, c='r', marker='o')

                    # Draw lines between connected landmarks
                    for connection in mp_pose.POSE_CONNECTIONS:
                        start_idx, end_idx = connection
                        ax.plot(
                            [x_coords[start_idx], x_coords[end_idx]],
                            [y_coords[start_idx], y_coords[end_idx]],
                            [z_coords[start_idx], z_coords[end_idx]],
                            color='blue'
                        )

                    # Set x-axis as vertical by adjusting the view
                    #ax.view_init(elev=250, azim=-60)  # This will rotate the x-axis to be vertical
                    ax.view_init(elev=250)
                    #ax.invert_zaxis()  # Invert Z-axis to make it point upwards
                    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio for all axes

                    # Configure plot labels and limits
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    ax.set_xlim([0, 1])
                    ax.set_ylim([0, 1])
                    ax.set_zlim([-0.5, 0.5])
                    ax.set_title(f"{filename}.png")

                    plt.show()

                    # Remove axis, grids, and labels for a clean image
                    #ax.set_axis_off()  # Turn off axis
                    #ax.grid(False)     # Turn off grid 
                    #plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding

                    # Take snapshots from multiple angles
                    for angle in angles:
                        ax.view_init(elev=250, azim=angle)  # Rotate the view
                        output_filename = os.path.join(class_output_folder, f"{filename}_angle_{angle}.png")
                        plt.show()

                        #plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)

                    plt.close(fig) """ 

print("3D slices saved.")
                    
                    

                    


