import cv2
import os
import json
import sys
import numpy as np

# --- Configuration ---
# 1. The folder with your original JPGs
INPUT_FOLDER = "ID_cards"

# 2. The new folder where the marker images will be saved
MARKER_OUTPUT_FOLDER = "generated_markers"

# 3. The name of the map file to be created
MAP_FILE = "aruco_image_map.json"

# 4. The ArUco dictionary to use
ARUCO_DICTIONARY_TYPE = cv2.aruco.DICT_6X6_250

# 5. Size of the output marker images (in pixels)
MARKER_SIZE_PX = 500
# ---------------------

def generate_markers_and_map(input_dir, output_dir, map_filename, dict_type):
    """
    Scans an input directory, generates a unique ArUco marker for each
    image, saves the markers, and creates a JSON map.
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
        print("Please make sure your folder is named 'wallpapers' and is in the same directory.")
        return

    # 3. Create the output folder if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output folder: '{output_dir}'")

    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_to_marker_map = {}
    current_marker_id = 0

    print(f"Scanning '{input_dir}' and generating markers...")

    # 4. Loop through all files in the input directory
    for filename in sorted(os.listdir(input_dir)): # Sorted for consistent order
        # Check if the file is a valid image
        if not filename.lower().endswith(valid_extensions):
            continue 
        
        # 5. Generate the marker image
        marker_image = np.zeros((MARKER_SIZE_PX, MARKER_SIZE_PX), dtype=np.uint8)
        marker_image = cv2.aruco.generateImageMarker(
            aruco_dict, 
            current_marker_id, 
            MARKER_SIZE_PX
        )
        
        # Add a white border (padding) for easier scanning
        padded_image = cv2.copyMakeBorder(
            marker_image, 50, 50, 50, 50, 
            cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )
        
        # 6. Save the new marker image
        marker_filename = f"marker_{current_marker_id}.png"
        output_path = os.path.join(output_dir, marker_filename)
        cv2.imwrite(output_path, padded_image)

        # 7. Add the link to our map
        # We map the ID (as a string) to the original image filename
        image_to_marker_map[str(current_marker_id)] = filename
        
        print(f"  - Generated {marker_filename} for {filename}")
        
        # 8. Increment the ID for the next image
        current_marker_id += 1

    if not image_to_marker_map:
        print(f"No images found in '{input_dir}'. No map was created.")
        return

    # 9. Save the completed map to a JSON file
    with open(map_filename, 'w') as f:
        json.dump(image_to_marker_map, f, indent=4)

    print(f"\nSuccess! Map saved to '{map_filename}'")
    print(f"All marker images saved in '{output_dir}'.")
    print("You can now print the markers from the 'generated_markers' folder.")

# --- Main execution ---
if __name__ == "__main__":
    generate_markers_and_map(
        INPUT_FOLDER, 
        MARKER_OUTPUT_FOLDER, 
        MAP_FILE, 
        ARUCO_DICTIONARY_TYPE
    )
