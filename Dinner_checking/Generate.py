import cv2
import os
import json
import sys
import numpy as np

# --- Configuration ---
# 1. The main folder containing your club subfolders (e.g., IBOT_CLUB, RAFTAR)
INPUT_FOLDER = "Dinner_checking\ID_cards"

# 2. The main output folder. Subfolders will be created *inside* this.
MARKER_OUTPUT_FOLDER = "Research_conclave\Dinner_checking/Generated_markers"

# 3. The name of the map file to be created
MAP_FILE = "aruco_image_map.json"

# 4. The ArUco dictionary to use
ARUCO_DICTIONARY_TYPE = cv2.aruco.DICT_6X6_1000

# 5. Size of the output marker images (in pixels)
MARKER_SIZE_PX = 500
# ---------------------

def generate_markers_and_map_nested(input_dir, output_dir, map_filename, dict_type):
    """
    Scans an input directory *and all its subdirectories*, generates a
    unique ArUco marker for each image, saves the markers in a parallel
    directory structure, and creates a JSON map.
    """
    
    # 1. Initialize ArUco dictionary
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
    except AttributeError:
        print("Error: Could not initialize ArUco detector.")
        print("Please ensure you have 'opencv-contrib-python' installed.")
        print("Run: pip install opencv-contrib-python")
        return
        
    # 2. Check if input folder exists
    if not os.path.isdir(input_dir):
        print(f"Error: Input folder not found at: '{input_dir}'")
        return

    # 3. Create the *root* output folder if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created root output folder: '{output_dir}'")

    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_to_marker_map = {}
    current_marker_id = 0 # This ID increments for *every* image, staying unique

    print(f"Scanning '{input_dir}' and all subfolders...")

    # 4. Use os.walk() to go through all subfolders
    #    This is the key difference!
    for dirpath, dirnames, filenames in os.walk(input_dir):
        
        # dirpath = full path to current folder (e.g., ".../ID_cards/IBOT_CLUB")
        # filenames = list of files in that folder (e.g., ["Aurovind.jpg"])
        
        # 5. Determine the relative subfolder path (e.g., "IBOT_CLUB")
        relative_subfolder = os.path.relpath(dirpath, input_dir)
        
        # We skip the root folder itself ("."). We only care about
        # files *inside* the subfolders.
        if relative_subfolder == ".":
            continue
            
        # 6. Create the matching subfolder in the output directory
        target_marker_dir = os.path.join(output_dir, relative_subfolder)
        if not os.path.exists(target_marker_dir):
            os.makedirs(target_marker_dir)
            
        print(f"\n--- Scanning subfolder: {relative_subfolder} ---")
        
        if not filenames:
            print("   - No image files found here.")
            continue
        
        for filename in sorted(filenames):
            # Check if the file is a valid image
            if not filename.lower().endswith(valid_extensions):
                continue 
            
            # 7. Generate the marker image
            marker_image = np.zeros((MARKER_SIZE_PX, MARKER_SIZE_PX), dtype=np.uint8)
            marker_image = cv2.aruco.generateImageMarker(
                aruco_dict, 
                current_marker_id, 
                MARKER_SIZE_PX
            )
            
            # Add a white border (padding)
            padded_image = cv2.copyMakeBorder(
                marker_image, 50, 50, 50, 50, 
                cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )
            
            # 8. Save the new marker image
            base_name = os.path.splitext(filename)[0]
            marker_filename = f"{base_name}.png"
            
            # Save it inside the new target subfolder
            output_path = os.path.join(target_marker_dir, marker_filename)
            cv2.imwrite(output_path, padded_image)

            # 9. Add the link to our map
            # We map the ID to the *relative path* of the original file
            # e.g., "0": "IBOT_CLUB/Aurovind_Sadangi.jpg"
            relative_image_path = os.path.join(relative_subfolder, filename).replace("\\", "/")
            image_to_marker_map[str(current_marker_id)] = relative_image_path
            
            print(f"   - Generated {marker_filename} (ID: {current_marker_id}) for {relative_image_path}")
            
            # 10. Increment the ID for the next image
            current_marker_id += 1

    if not image_to_marker_map:
        print(f"No images found in any subfolders of '{input_dir}'. No map was created.")
        return

    # 11. Save the completed map to a JSON file
    with open(map_filename, 'w') as f:
        json.dump(image_to_marker_map, f, indent=4)

    print(f"\nSuccess! Map saved to '{map_filename}'")
    print(f"All marker images saved in '{output_dir}' in their respective subfolders.")

# --- Main execution ---
if __name__ == "__main__":
    generate_markers_and_map_nested(
        INPUT_FOLDER, 
        MARKER_OUTPUT_FOLDER, 
        MAP_FILE, 
        ARUCO_DICTIONARY_TYPE
    )