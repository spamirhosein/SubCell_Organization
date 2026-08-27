from pathlib import Path
import shutil

# This script takes all files in the specified folder and organizes them into subfolders based on their filenames.

# Change this to the path of your folder
TARGET_FOLDER = r"D:\image_data\Ha Anh\stack2\sorted_by_quality\High" 

def organize_files(directory):
    dir_path = Path(directory)

    if not dir_path.exists():
        print(f"Error: The directory '{directory}' does not exist.")
        return

    # Iterate through all items in the folder
    for item in dir_path.iterdir():
        # Process only files (skip directories)
        if item.is_file():
            # Get filename without extension and the extension itself
            file_name_no_ext = item.stem
            file_ext = item.suffix

            # Define the new folder path
            new_folder_path = dir_path / file_name_no_ext

            # Create the folder (does nothing if it already exists)
            new_folder_path.mkdir(exist_ok=True)

            # Define the new file path (inside the new folder, named same as folder)
            new_file_path = new_folder_path / f"{file_name_no_ext}{file_ext}"

            # Move and rename the file
            shutil.move(str(item), str(new_file_path))
            print(f"Moved: '{item.name}' -> '{new_folder_path.name}/{file_name_no_ext}{file_ext}'")

if __name__ == "__main__":
    organize_files(TARGET_FOLDER)
    print("Done!")