import os
import shutil

def delete_pycache_dirs(base_path):
    """Recursively delete all __pycache__ directories in the given base path."""
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            if dir_name == "__pycache__":
                pycache_path = os.path.join(root, dir_name)
                print(f"Deleting: {pycache_path}")
                shutil.rmtree(pycache_path)

if __name__ == "__main__":
    this_path = os.path.dirname(os.path.abspath(__file__)) 
    parent_dir = os.path.dirname(this_path)
    delete_pycache_dirs(parent_dir)
    print("All __pycache__ directories have been deleted.")