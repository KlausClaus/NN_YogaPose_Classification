import cv2
import mediapipe as mp
import numpy as np
import os
import shutil

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define text offset values
x_offset = -4  # Adjust this value for horizontal offset
y_offset = 2   # Adjust this value for vertical offset

# Input and output folder paths
input_folder = r'D:\Term 3 2024\COMP9444\COMP9444_project\Dataset\Yoga-82\yoga_dataset_links'
output_folder = r'D:\Term 3 2024\COMP9444\COMP9444_project\Dataset\Yoga-82\yoga_dataset_links_skeleton_voxel'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process each subfolder in the input directory
for subfolder in os.listdir(input_folder):
    subfolder_path = os.path.join(input_folder, subfolder)
    
    # Only process if it's a directory
    if os.path.isdir(subfolder_path):
        # Create a corresponding subfolder in the output directory
        output_subfolder = os.path.join(output_folder, subfolder)
        os.makedirs(output_subfolder, exist_ok=True)
        
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
                    mp_drawing.draw_landmarks(
                        keypoints_image, 
                        result.pose_landmarks, 
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=3, circle_radius=3),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                    )

                    # Label each landmark with its name
                    for idx, landmark in enumerate(result.pose_landmarks.landmark):
                        x = int(landmark.x * image_width) + x_offset  # Apply x offset
                        y = int(landmark.y * image_height) + y_offset  # Apply y offset
                        label = str(idx) # Use the index number as the label
                        cv2.putText(keypoints_image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1, cv2.LINE_AA)

                    """ for landmark in result.pose_landmarks.landmark:
                        x = int(landmark.x * image_width)
                        y = int(landmark.y * image_height)
                        cv2.circle(keypoints_image, (x, y), 5, (255, 255, 255), -1) """

                    # Save the image with keypoints
                    output_path = os.path.join(output_subfolder, filename)
                    cv2.imwrite(output_path, keypoints_image)

print("Processing complete.")
