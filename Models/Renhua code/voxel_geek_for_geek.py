import cv2
import mediapipe as mp
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# input and output folder paths
input_folder = r'D:\Term 3 2024\COMP9444\COMP9444_project\Dataset\Yoga-82\yoga_dataset_links'
output_folder = r'D:\Term 3 2024\COMP9444\COMP9444_project\Dataset\Yoga-82\yoga_dataset_linkes_voxel_32'

# make an output folder 
os.makedirs(output_folder, exist_ok=True)

# define voxel grid size (I have chosen 32x32x32)
grid_size = 32


import numpy as np

def draw_line_3d(grid, start_voxel, end_voxel):
    """
    Draw a line in a 3D voxel grid using Bresenham's line algorithm

    Arguments:
        grid (np.ndarray): 3D numpy array representing the voxel grid
        start_voxel (tuple): The starting voxel coordinates (in x, y, z coordinates)
        end_voxel (tuple): The ending voxel coordinates (in x, y, z coordinates)

    Credit: https://www.geeksforgeeks.org/bresenhams-algorithm-for-3-d-line-drawing/
    """
    x1, y1, z1 = start_voxel
    x2, y2, z2 = end_voxel
    
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)
    
    xs = 1 if x2 > x1 else -1
    ys = 1 if y2 > y1 else -1
    zs = 1 if z2 > z1 else -1

    # driving axis is X-axis
    if dx >= dy and dx >= dz:
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while x1 != x2:
            grid[x1, y1, z1] = 1
            x1 += xs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dx
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
        grid[x1, y1, z1] = 1  # mark the end point

    # driving axis is Y-axis
    elif dy >= dx and dy >= dz:
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while y1 != y2:
            grid[x1, y1, z1] = 1
            y1 += ys
            if p1 >= 0:
                x1 += xs
                p1 -= 2 * dy
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
        grid[x1, y1, z1] = 1  # mark the end point

    # driving axis is Z-axis
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while z1 != z2:
            grid[x1, y1, z1] = 1
            z1 += zs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dz
            if p2 >= 0:
                x1 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
        grid[x1, y1, z1] = 1  # mark the end point


# function to normalize and map 3D landmarks to a voxel grid
def normalize_landmarks(landmarks, grid_size):
    """
    This function is designed to normalize the landmarks to fit within the voxel grid

    Arguments:
        landmarks: A list of landmarks with x, y, z coordinates
        grid_size: The size of the voxel grid along each dimension

    Returns:
        list: List of normalized voxel coordinates
    """
    # Retrieve the min and max values across all landmarks
    min_vals = np.array([min([landmark.x for landmark in landmarks]),
                         min([landmark.y for landmark in landmarks]),+
                         min([landmark.z for landmark in landmarks])])
    max_vals = np.array([max([landmark.x for landmark in landmarks]),
                         max([landmark.y for landmark in landmarks]),
                         max([landmark.z for landmark in landmarks])])
    
    # center the landmarks by subtracting min and scaling to the voxel grid size
    ranges = max_vals - min_vals
    normalized_landmarks = []
    for landmark in landmarks:
        # normalize and scale each coordinate
        norm_x = (landmark.x - min_vals[0]) / ranges[0] * (grid_size - 1)
        norm_y = (landmark.y - min_vals[1]) / ranges[1] * (grid_size - 1)
        norm_z = (landmark.z - min_vals[2]) / ranges[2] * (grid_size - 1)
        normalized_landmarks.append((int(norm_x), int(norm_y), int(norm_z)))
    
    return normalized_landmarks

def visualize_voxel_grid(filename, voxel_grid, connections):
    """
    Visualize a 3D voxel grid with landmark connections using matplotlib.

    Parameters:
        voxel_grid (np.ndarray): 3D numpy array representing the voxel grid, 
                                 where 1 indicates an "on" voxel and 0 indicates an "off" voxel.
        connections (list): List of landmark index pairs representing the connections between landmarks.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Get the indices of the voxels that are "on"
    x, y, z = np.where(voxel_grid == 1)
    
    # Plot the voxels
    ax.scatter(x, y, z, c='black', marker='s')


    # Set the labels and grid
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Voxel Grid Visualization with Connections {filename}')
    
    # Adjust limits to make sure grid is displayed as a cube
    max_dim = max(voxel_grid.shape)
    ax.set_xlim([0, max_dim])
    ax.set_ylim([0, max_dim])
    ax.set_zlim([0, max_dim])
    
    plt.show()


# process each subfolder in the input directory. Here a subfolder is also a class of poses
for subfolder in os.listdir(input_folder):
    subfolder_path = os.path.join(input_folder, subfolder)
    
    # make sure it's a directory
    if os.path.isdir(subfolder_path):
        # create a corresponding subfolder in the output directory
        output_subfolder = os.path.join(output_folder, subfolder)
        os.makedirs(output_subfolder, exist_ok=True)
        
        # process each image in the input folder
        for filename in os.listdir(subfolder_path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                # read the image
                image_path = os.path.join(subfolder_path, filename)
                image = cv2.imread(image_path)

                if image is None:
                    continue  # skip invalid images

                # convert the image to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # process the image using Mediapipe and find the pose
                result = pose.process(image_rgb)

                # create a 3D voxel grid initialized to zero
                voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)

                # draw the keypoints in the voxel grid
                if result.pose_landmarks:
                    
                    """ # plot the pose
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
                    ax.set_title(f'Pose Landmarks {filename}')
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    plt.show() """

                    # normalize landmarks to fit in the voxel grid
                    landmark_voxels = normalize_landmarks(result.pose_landmarks.landmark, grid_size)

                    # mark each normalized landmark in the voxel grid
                    for voxel in landmark_voxels:
                        voxel_grid[voxel] = 1

                    # draw lines for each connection in the 3D space using Bresenham's line algorithm
                    for connection in mp_pose.POSE_CONNECTIONS:
                        start_idx, end_idx = connection
                        start_voxel = landmark_voxels[start_idx]
                        end_voxel = landmark_voxels[end_idx]
                        draw_line_3d(voxel_grid, start_voxel, end_voxel)

                    
                    # reflect the voxel grid about the X-axis (invert Y-coordinates). 
                    # This is to make sure the pose in voxel grid is in the correct orientation
                    voxel_grid = voxel_grid[:, ::-1, :]

                    # save the voxel grid as a numpy array for later use
                    output_path = os.path.join(output_subfolder, filename.split('.')[0] + '.npy')
                    
                    np.save(output_path, voxel_grid)

                    """ # visualise voxel
                    visualize_voxel_grid(filename, voxel_grid, mp_pose.POSE_CONNECTIONS) """

print("Completed generating voxel grids")







