import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard
from torchsummary import summary


# Initialize TensorBoard writer
#writer = SummaryWriter(log_dir='runs/yoga_pose_experiment')
writer = SummaryWriter(f'runs/yoga_pose_experiment')

# updated YogaConv3d model to handle 3D voxel grids
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


# DataLoader setup should remain similar, but your voxel grid data needs to be loaded as 3D tensors.
# Example transformation for voxel data if stored as numpy arrays:
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

# Initialize model, loss function, and optimizer
num_classes = len(dataset.class_to_idx)
model = YogaConv3d(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training and Testing Loops (similar to previous implementation)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

model.to(device)

# Log the model graph
dummy_input = torch.randn(1, 1, 32, 32, 32).to(device)  # Adjust dimensions as needed
writer.add_graph(model, dummy_input)

# Training function
def train(model, train_loader, optimizer, criterion, epoch, step):
    model.train()
    total, correct = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # Move to device
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, preds = torch.max(outputs, 1)
        num_correcy = (preds == labels).sum().item()
        correct += num_correcy
        total += labels.size(0)
        running_train_accuracy = float(num_correcy)/float(images.shape[0])

        writer.add_scalar('Training lose', loss, global_step=step)
        writer.add_scalar('Training accuracy', running_train_accuracy, global_step=step)
        step += 1
    
    accuracy = 100 * correct / total
    print(f"Training Accuracy: {accuracy:.6f}%")
    #writer.add_scalar('Training Accuracy', accuracy, epoch)  # Log training accuracy
    #writer.add_scalar('Training Loss', loss.item(), epoch)    # Log training loss
    print(f"Training Accuracy: {accuracy:.6f}%")
    return accuracy

# Testing function
def test(model, test_loader, criterion, epoch):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # Move to device
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.6f}%")
    writer.add_scalar('Test Accuracy', accuracy, epoch)  # Log test accuracy
    writer.add_scalar('Test Loss', loss.item(), epoch)   # Log test loss
    return accuracy

print(model)
summary(model, input_size=(1, 32, 32, 32))

# Training loop
epochs = 50
step = 0
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    train_accuracy = train(model, train_loader, optimizer, criterion, epoch, step)
    if (epoch + 1) % 10 == 0:
        test_accuracy = test(model, test_loader, criterion, epoch)

# save the trained model to PC
""" torch.save(model.state_dict(), 'yoga_pose_classification_model.pth')
print("Model saved") """



# Flush and close the writer after training
writer.flush()
writer.close()