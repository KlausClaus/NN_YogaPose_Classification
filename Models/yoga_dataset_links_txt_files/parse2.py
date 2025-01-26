import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import webbrowser
import shutil


# Set the path to the folder containing the text files
base_folder = os.path.dirname(os.path.abspath(__file__))

downloaded_dir = os.path.join(base_folder, "Downloaded")

def download_image(image_name, image_url, pose_dir):
    """Function to download a single image."""
    image_path = os.path.join(base_folder, image_name)

    # Check if the image already exists
    """ if os.path.exists(image_path):
        print(f"Image {image_name} already exists. Skipping download.")
        return """

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"
    }

    try:
        if image_url == 'http://mugup.info/appimg/Yoga/Akarna%20Dhanurasana.jpg':
            print("DEBUG")
        # Download the image
        response = requests.get(image_url, headers=headers, stream=True, timeout=10)
        if response.status_code == 200 :
            # Save the image in the respective directory
            with open(image_path, 'wb') as img_file:
                for chunk in response.iter_content(1024):
                    img_file.write(chunk)
            print(f"Downloaded {image_name} to {pose_dir}")
        elif response.status_code in [404, 500, 503, 523, 400, 410, 401]:
            return
            #print(f"Failed to download {image_name} from {image_url}")
        else:
            webbrowser.open(image_url)
            print(f"Failed to download {image_name} from {image_url}")
    except requests.exceptions.Timeout:
        webbrowser.open(image_url)  # Only opens the browser for timeout errors
        print(f"Timeout occurred for {image_name} from {image_url}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {image_name} from {image_url}: {e}")
        
# List of blacklisted domains
blacklisted_domains = ["yogatrail.com", "fitnessgoals.com"]

# Loop through each text file in the folder
for txt_file in os.listdir(base_folder):
    if txt_file.endswith(".txt"):
        # Create a directory for each pose based on the text file name (without .txt)
        pose_name = txt_file.replace(".txt", "")
        pose_dir = os.path.join(base_folder, pose_name)
        
        # Skip the directory if it already exists
        if not (os.path.isdir(pose_dir)):
            os.makedirs(pose_dir, exist_ok=True)
        
        

        # Prepare a list of image download tasks
        tasks = []

        # Open the text file and read each line
        with open(os.path.join(base_folder, txt_file), 'r') as file:
            for line in file:
                # Split each line to get the image name and URL
                try:
                    image_name, image_url = line.strip().split('\t')

                    # Check if the URL contains any blacklisted domain
                    if any(domain in image_url for domain in blacklisted_domains):
                        print(f"Skipping {image_url} from blacklisted domain.")
                        continue

                    image_path = os.path.join(base_folder, image_name)
                    if os.path.exists(image_path):
                        print(f"Image {image_name} already exists. Skipping download.")
                        continue
                    tasks.append((image_name, image_url, pose_dir))
                except ValueError:
                    print(f"Invalid line format in {txt_file}: {line.strip()}")

        # Download images concurrently using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=30) as executor:  # Adjust max_workers as needed
            futures = [executor.submit(download_image, image_name, image_url, pose_dir) for image_name, image_url, pose_dir in tasks]
            
            for future in as_completed(futures):
                # Optionally handle any exception raised during the download
                try:
                    future.result()
                except Exception as e:
                    print(f"Error occurred during image download: {e}")

        # Move the text file to the "Downloaded" directory after processing
        shutil.move(os.path.join(base_folder, txt_file), os.path.join(downloaded_dir, txt_file))
        print(f"Moved {txt_file} to the 'Downloaded' directory.")