import os
import shutil

# --- CONFIGURATION ---
# Path to the extracted MRL folder on your D: drive
# If your structure is archive (1)\mrlEyes_2018_01\s1...
SOURCE_FOLDER = r"D:\AI-Powered Truck Driver Safety Monitoring System\archive (1)\mrlEyes_2018_01"
# If your structure is archive (1)\s1... (no nested folder), use this instead:
# SOURCE_FOLDER = r"D:\AI-Powered Truck Driver Safety Monitoring System\archive (1)"
# The new folders where we will store the organized images
# This will be created inside your project folder (where this script is saved)
BASE_DEST = "dataset"
OPEN_DIR = os.path.join(BASE_DEST, "open")
CLOSED_DIR = os.paSth.join(BASE_DEST, "closed")

# Create the destination folders
os.makedirs(OPEN_DIR, exist_ok=True)
os.makedirs(CLOSED_DIR, exist_ok=True)

count_open = 0
count_closed = 0

print(f"Reading from: {SOURCE_FOLDER}")
print("Processing started. This may take a few minutes...")

if not os.path.exists(SOURCE_FOLDER):
    print("ERROR: Source folder does not exist!")
    print("Please check the path and try again.")
    exit()

# Walk through all directories and files
for root, dirs, files in os.walk(SOURCE_FOLDER):
    for file in files:
        if file.endswith(".png"):
            file_path = os.path.join(root, file)
            parts = file.split('_')
            
            # parts[2] indicates eye state: 1 = Open, 0 = Closed
            if len(parts) > 2:
                eye_state = parts[2]
                
                if eye_state == '1':
                    shutil.copy(file_path, OPEN_DIR)
                    count_open += 1
                elif eye_state == '0':
                    shutil.copy(file_path, CLOSED_DIR)
                    count_closed += 1

print(f"\nProcess Complete!")
print(f"Total Open images: {count_open}")
print(f"Total Closed images: {count_closed}")
print("Dataset is ready in the 'dataset' folder.")