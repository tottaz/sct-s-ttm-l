#!/usr/bin/env python3
"""
Cleanup script for temporary files and cache.
Run this if the app is running slow or before restarting.
"""
import os
import glob

def get_data_dir():
    """Get the data directory path."""
    return os.path.join(os.path.expanduser("~"), ".sct-sattmal-data")

def cleanup():
    data_dir = get_data_dir()
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist. Nothing to clean.")
        return
    
    # Clean temp OCR files
    temp_pattern = os.path.join(data_dir, "temp_page_*.png")
    temp_files = glob.glob(temp_pattern)
    
    if temp_files:
        print(f"Found {len(temp_files)} temp files to delete:")
        for f in temp_files:
            try:
                os.remove(f)
                print(f"  Deleted: {os.path.basename(f)}")
            except Exception as e:
                print(f"  Error deleting {f}: {e}")
    else:
        print("No temp files found.")
    
    print("\nCleanup complete!")
    print("Tip: Restart the Flask server to clear memory.")

if __name__ == "__main__":
    cleanup()
