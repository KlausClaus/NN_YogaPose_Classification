import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

class YogaConv3d(nn.Module):
    def __init__(self, num_classes):
        super(YogaConv3d, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, padding=1)  # Assuming 1 channel for grayscale voxel grid
        self.pool = nn.MaxPool3d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout(0.25)
        
        self.flatten = nn.Flatten()
        # Update the linear layer's input size based on the new 64x64x64 grid
        self.fc1 = nn.Linear(256 * 4 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout3 = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout1(x)
        
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout2(x)

        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x
    
    
import numpy as np
from torch.utils.data import Dataset
import os

class VoxelDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []  # Store file paths for each sample
        self.class_to_idx = {}  # Dictionary to store class-to-index mapping

        # Automatically generate class_to_idx based on folder names
        for idx, class_dir in enumerate(sorted(os.listdir(data_dir))):
            class_path = os.path.join(data_dir, class_dir)
            if os.path.isdir(class_path):
                self.class_to_idx[class_dir] = idx  # Assign an integer index to each class
                for file in os.listdir(class_path):
                    if file.endswith('.npy'):
                        self.samples.append((os.path.join(class_path, file), class_dir))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, class_name = self.samples[idx]
        voxel_grid = np.load(file_path)  # Load voxel grid (assume numpy format)
        voxel_grid = torch.tensor(voxel_grid, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        label = self.class_to_idx[class_name]  # Get integer label from class name
        return voxel_grid, label

# Load voxel data
input_folder = r'D:\Term 3 2024\COMP9444\COMP9444_project\Dataset\Yoga-82\yoga_dataset_links_voxel_32'
dataset = VoxelDataset(data_dir=input_folder)

# Set the train-test split ratio
train_ratio = 0.8
train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size

# Split the dataset into training and testing sets
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# DataLoader for training and testing sets
batch_size = 16  # Adjust batch size if necessary for memory limits
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Load the trained model
num_classes = len(dataset.class_to_idx)
model = YogaConv3d(num_classes=num_classes)  # Ensure num_classes matches your model
model.load_state_dict(torch.load("model_checkpoint.pth"))  # Replace with the path to your saved model
model.eval()

# Training and Testing Loops (similar to previous implementation)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

model.to(device)

# Mapping class index to class name
class_names = {v: k for k, v in dataset.class_to_idx.items()}  # Reverse the dictionary

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Define the voxel grid size
grid_size = 32  # Match this with the grid size used during training

def normalize_landmarks(landmarks, grid_size):
    """
    Normalize the landmarks to fit within the voxel grid.

    Parameters:
        landmarks (list): List of landmarks with x, y, z attributes.
        grid_size (int): The size of the voxel grid along each dimension.

    Returns:
        list: List of normalized voxel coordinates.
    """
    # Extract min and max values across all landmarks
    min_vals = np.array([min([landmark.x for landmark in landmarks]),
                         min([landmark.y for landmark in landmarks]),
                         min([landmark.z for landmark in landmarks])])
    max_vals = np.array([max([landmark.x for landmark in landmarks]),
                         max([landmark.y for landmark in landmarks]),
                         max([landmark.z for landmark in landmarks])])
    
    # Center the landmarks by subtracting min and scaling to the voxel grid size
    ranges = max_vals - min_vals
    normalized_landmarks = []
    for landmark in landmarks:
        # Normalize and scale each coordinate
        norm_x = (landmark.x - min_vals[0]) / ranges[0] * (grid_size - 1)
        norm_y = (landmark.y - min_vals[1]) / ranges[1] * (grid_size - 1)
        norm_z = (landmark.z - min_vals[2]) / ranges[2] * (grid_size - 1)
        normalized_landmarks.append((int(norm_x), int(norm_y), int(norm_z)))
    
    return normalized_landmarks

def draw_line_3d(grid, start_voxel, end_voxel):
    """
    Draw a line in a 3D grid using Bresenham's line algorithm.

    Parameters:
        grid (np.ndarray): 3D numpy array representing the voxel grid.
        start_voxel (tuple): The starting voxel coordinates (x, y, z).
        end_voxel (tuple): The ending voxel coordinates (x, y, z).
    """
    # Unpack start and end coordinates
    x0, y0, z0 = start_voxel
    x1, y1, z1 = end_voxel

    # Calculate differences and directions
    dx, dy, dz = abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    sz = 1 if z1 > z0 else -1

    # Initialize error terms
    if dx >= dy and dx >= dz:  # X is the driving axis
        err_y, err_z = 2 * dy - dx, 2 * dz - dx
        for _ in range(dx):
            grid[x0, y0, z0] = 1  # Mark voxel
            if err_y >= 0:
                y0 += sy
                err_y -= 2 * dx
            if err_z >= 0:
                z0 += sz
                err_z -= 2 * dx
            x0 += sx
            err_y += 2 * dy
            err_z += 2 * dz
    elif dy >= dx and dy >= dz:  # Y is the driving axis
        err_x, err_z = 2 * dx - dy, 2 * dz - dy
        for _ in range(dy):
            grid[x0, y0, z0] = 1  # Mark voxel
            if err_x >= 0:
                x0 += sx
                err_x -= 2 * dy
            if err_z >= 0:
                z0 += sz
                err_z -= 2 * dy
            y0 += sy
            err_x += 2 * dx
            err_z += 2 * dz
    else:  # Z is the driving axis
        err_x, err_y = 2 * dx - dz, 2 * dy - dz
        for _ in range(dz):
            grid[x0, y0, z0] = 1  # Mark voxel
            if err_x >= 0:
                x0 += sx
                err_x -= 2 * dz
            if err_y >= 0:
                y0 += sy
                err_y -= 2 * dz
            z0 += sz
            err_x += 2 * dx
            err_y += 2 * dy
    
    grid[x1, y1, z1] = 1  # Ensure the end voxel is marked


# Start capturing from the webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    result = pose.process(image_rgb)

    if result.pose_landmarks:
        # Create a blank voxel grid
        voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
        
        # Normalize landmarks to fit within the voxel grid
        landmark_voxels = normalize_landmarks(result.pose_landmarks.landmark, grid_size)

        # Mark each normalized landmark in the voxel grid
        for voxel in landmark_voxels:
            voxel_grid[voxel] = 1

        # Draw lines for each connection in the 3D space
        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            start_voxel = landmark_voxels[start_idx]
            end_voxel = landmark_voxels[end_idx]
            draw_line_3d(voxel_grid, start_voxel, end_voxel)

        # Reflect the voxel grid about the X-axis (invert Y-coordinates)
        voxel_grid = voxel_grid[:, ::-1, :]

        # Convert voxel grid to tensor
        input_tensor = torch.tensor(voxel_grid.copy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted_idx = torch.max(output, 1)
            predicted_pose = class_names[predicted_idx.item()]

        # Draw landmarks on the frame
        mp.solutions.drawing_utils.draw_landmarks(
            frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Display the predicted class on the webcam frame
        cv2.putText(frame, f'Pose: {predicted_pose}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Yoga Pose Classification', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
