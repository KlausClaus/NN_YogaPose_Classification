import os
import shutil

# Paths to the input files and source image directory
train_file = r"D:\Term 3 2024\COMP9444\COMP9444_project\Dataset\Yoga-82\yoga_train.txt"
test_file = r"D:\Term 3 2024\COMP9444\COMP9444_project\Dataset\Yoga-82\yoga_test.txt"

""" # original images
source_dir = r"D:\Term 3 2024\COMP9444\COMP9444_project\Dataset\Yoga-82\yoga_dataset_links"
train_output_dir = r"D:\Term 3 2024\COMP9444\COMP9444_project\Dataset\Yoga-82\train_dataset_original_images"
test_output_dir = r"D:\Term 3 2024\COMP9444\COMP9444_project\Dataset\Yoga-82\test_dataset_original_images" """

""" # skeleton
source_dir = r"D:\Term 3 2024\COMP9444\COMP9444_project\Dataset\Yoga-82\yoga_dataset_links_skeleton"
train_output_dir = r"D:\Term 3 2024\COMP9444\COMP9444_project\Dataset\Yoga-82\train_dataset_skeleton"
test_output_dir = r"D:\Term 3 2024\COMP9444\COMP9444_project\Dataset\Yoga-82\test_dataset_skeleton" """

# voxel
source_dir = r"D:\Term 3 2024\COMP9444\COMP9444_project\Dataset\Yoga-82\yoga_dataset_links_voxel_32"
train_output_dir = r"D:\Term 3 2024\COMP9444\COMP9444_project\Dataset\Yoga-82\train_dataset_voxel"
test_output_dir = r"D:\Term 3 2024\COMP9444\COMP9444_project\Dataset\Yoga-82\test_dataset_voxel"

# Function to read image paths from a text file
def get_image_paths(file_path):
    with open(file_path, 'r') as f:
        return [os.path.splitext(line.split(',')[0].strip())[0] + '.npy' for line in f.readlines()]

# Function to copy images to a new directory structure
def copy_images(image_paths, source_dir, output_dir):
    for image_path in image_paths:
        class_name, filename = os.path.split(image_path)
        source_path = os.path.join(source_dir, image_path)
        
        # Create directory for the class in the output directory
        target_dir = os.path.join(output_dir, class_name)
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy the image to the target directory
        target_path = os.path.join(target_dir, filename)
        if source_path.endswith('.npy') and os.path.exists(source_path):
            shutil.copy(source_path, target_path)
        else:
            print(f"Warning: {source_path} not found.")

# Read image paths for train and test datasets
train_images = get_image_paths(train_file)
test_images = get_image_paths(test_file)

# Copy images for both train and test datasets
copy_images(train_images, source_dir, train_output_dir)
copy_images(test_images, source_dir, test_output_dir)

print("Image splitting and copying complete.")
