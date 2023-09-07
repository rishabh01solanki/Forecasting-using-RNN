from PIL import Image
import os

def check_images(root_folder):
    for folder in ['me', 'not_me']:
        folder_path = os.path.join(root_folder, folder)
        for filename in os.listdir(folder_path):
            if filename.startswith('.'):  # Skip hidden files
                continue
            file_path = os.path.join(folder_path, filename)
            try:
                with Image.open(file_path) as im:
                   print('Ok:', file_path)
            except:
                print('Invalid:', file_path)

# Check images in your dataset folder
check_images('train_data')
