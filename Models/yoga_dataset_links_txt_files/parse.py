import os
import requests

# Set the path to the folder containing the text files
base_folder = os.path.dirname(os.path.abspath(__file__))

# Loop through each text file in the folder
for txt_file in os.listdir(base_folder):
    if txt_file.endswith(".txt"):
        # Create a directory for each pose based on the text file name (without .txt)
        pose_name = txt_file.replace(".txt", "")
        pose_dir = os.path.join(base_folder, pose_name)
        os.makedirs(pose_dir, exist_ok=True)

        # Open the text file and read each line
        with open(os.path.join(base_folder, txt_file), 'r') as file:
            for line in file:
                # Split each line to get the image name and URL
                try:
                    image_name, image_url = line.strip().split()
                    image_path = os.path.join(base_folder, image_name)

                    if os.path.exists(image_path):
                        print(f"Image {image_name} already exists. Skipping download.")
                        continue
                    # Download the image
                    response = requests.get(image_url, stream=True)
                    if response.status_code == 200:
                        # Save the image in the respective directory
                        #image_path = os.path.join(pose_dir, image_name)
                        
                        with open(image_path, 'wb') as img_file:
                            for chunk in response.iter_content(1024):
                                img_file.write(chunk)
                        print(f"Downloaded {image_name} to {pose_dir}")
                    else:
                        print(f"Failed to download {image_name} from {image_url}")
                except requests.exceptions.RequestException as e:
                    print(f"Error downloading {image_name} from {image_url}: {e}")
                except ValueError:
                    print(f"Invalid line format in {txt_file}: {line.strip()}")
