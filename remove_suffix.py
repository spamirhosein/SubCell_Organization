import os

# Configuration
FOLDER_PATH = r"D:\image_data\Ha Anh\HLA-I_Channel3"
SUFFIX_TO_REMOVE = "_NaK_ATPase_HLA-I"  # E.g., "_old", "_backup", etc.

def remove_suffix_from_files(folder, suffix):
    """
    Remove suffix from all files in a folder.
    
    Args:
        folder: Path to the folder containing files
        suffix: Suffix string to remove from filenames (before extension)
    
    Returns:
        Dictionary with stats: {'renamed': count, 'skipped': count, 'errors': count}
    """
    if not os.path.isdir(folder):
        print(f"ERROR: Folder not found: {folder}")
        return {'renamed': 0, 'skipped': 0, 'errors': 0}
    
    stats = {'renamed': 0, 'skipped': 0, 'errors': 0}
    
    # List all files in the folder
    files = sorted(os.listdir(folder))
    
    if not files:
        print(f"No files found in {folder}")
        return stats
    
    print(f"Processing folder: {folder}")
    print(f"Suffix to remove: '{suffix}'\n")
    
    for filename in files:
        file_path = os.path.join(folder, filename)
        
        # Skip directories
        if os.path.isdir(file_path):
            continue
        
        # Split filename and extension
        name, ext = os.path.splitext(filename)
        
        # Check if name ends with suffix
        if not name.endswith(suffix):
            print(f"SKIP (no suffix): {filename}")
            stats['skipped'] += 1
            continue
        
        # Remove suffix
        new_name = name[:-len(suffix)]
        new_filename = new_name + ext
        new_file_path = os.path.join(folder, new_filename)
        
        # Check if new filename already exists
        if os.path.exists(new_file_path):
            print(f"ERROR (target exists): {filename} -> {new_filename}")
            stats['errors'] += 1
            continue
        
        try:
            os.rename(file_path, new_file_path)
            print(f"RENAMED: {filename} -> {new_filename}")
            stats['renamed'] += 1
        except Exception as e:
            print(f"ERROR ({str(e)}): {filename}")
            stats['errors'] += 1
    
    print(f"\n{'='*60}")
    print(f"Summary: Renamed={stats['renamed']}, Skipped={stats['skipped']}, Errors={stats['errors']}")
    print(f"{'='*60}")
    
    return stats

if __name__ == "__main__":
    # Run the function
    remove_suffix_from_files(FOLDER_PATH, SUFFIX_TO_REMOVE)
