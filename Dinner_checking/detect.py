import cv2
import os
import json
import numpy as np

# --- Configuration ---
# 1. The folder with your original JPGs (the 'wallpapers' folder)
IMAGE_FOLDER_PATH = "ID_cards"

# 2. The map file you generated in the previous step
MAP_FILE = "aruco_image_map.json"

# 3. The ArUco dictionary you used (MUST match the generator script)
ARUCO_DICTIONARY_TYPE = cv2.aruco.DICT_6X6_250

# 4. Size for the display window
DISPLAY_WINDOW_HEIGHT = 600
DISPLAY_WINDOW_WIDTH = 800
# ---------------------

def load_resources(map_filepath, image_folder_path):
    """
    Loads the JSON map and all the image assets into memory.
    Returns a dictionary: { 0: image_data, 1: image_data, ... }
    """
    
    # 1. Load the JSON map
    print(f"Loading map from '{map_filepath}'...")
    try:
        with open(map_filepath, 'r') as f:
            json_map = json.load(f)
    except FileNotFoundError:
        print(f"Error: Map file not found at '{map_filepath}'")
        print("Please run the 'create_map.py' script first.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not read the map file '{map_filepath}'. It might be empty or corrupt.")
        return None
        
    # 2. Check if the image folder exists
    if not os.path.isdir(image_folder_path):
        print(f"Error: Image folder not found at '{image_folder_path}'")
        return None

    # 3. Load all images from the map into a new dictionary
    print(f"Loading image assets from '{image_folder_path}'...")
    image_asset_map = {}
    
    for marker_id_str, filename in json_map.items():
        try:
            marker_id_int = int(marker_id_str)
            image_path = os.path.join(image_folder_path, filename)
            
            image = cv2.imread(image_path)
            if image is not None:
                image_asset_map[marker_id_int] = image
                print(f"  - Loaded '{filename}' for ID {marker_id_int}")
            else:
                print(f"  - Warning: Could not load image '{filename}' at '{image_path}'")
                
        except ValueError:
            print(f"  - Warning: Invalid ID '{marker_id_str}' in map file. Skipping.")
        except Exception as e:
            print(f"  - Error loading {filename}: {e}")
            
    if not image_asset_map:
        print("Error: No images were successfully loaded. Check your map file and folder.")
        return None
        
    return image_asset_map

def main():
    # 1. Load all our images and the map
    image_map = load_resources(MAP_FILE, IMAGE_FOLDER_PATH)
    if image_map is None:
        return

    # 2. Initialize ArUco detector
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY_TYPE)
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    except AttributeError:
        print("Error: Could not initialize ArUco detector.")
        print("Please ensure you have 'opencv-contrib-python' installed.")
        return

    # 3. Start video camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    # 4. Create a blank black canvas for our display window
    display_canvas = np.zeros(
        (DISPLAY_WINDOW_HEIGHT, DISPLAY_WINDOW_WIDTH, 3), 
        dtype=np.uint8
    )
    cv2.imshow("AR Display", display_canvas) # Create the window

    print("\nCamera started. Show a printed ArUco marker to the camera.")
    print("Press 'q' to quit.")

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            break

        # Detect markers in the frame
        corners, ids, rejected = detector.detectMarkers(frame)
        
        # Reset the display to black
        current_display = display_canvas.copy()
        found_mapped_marker = False

        if ids is not None:
            # Draw borders around detected markers on the camera feed
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Use the first detected ID
            first_id = ids[0][0]
            
            # Check if this ID is in our map
            if first_id in image_map:
                found_mapped_marker = True
                image_to_show = image_map[first_id]
                
                # --- Resize and center the image ---
                img_h, img_w = image_to_show.shape[:2]
                disp_h, disp_w = current_display.shape[:2]

                scale = min(disp_h / img_h, disp_w / img_w)
                
                new_w = int(img_w * scale)
                new_h = int(img_h * scale)
                
                resized_img = cv2.resize(image_to_show, (new_w, new_h))
                
                y_offset = (disp_h - new_h) // 2
                x_offset = (disp_w - new_w) // 2
                
                # Place the image onto the black canvas
                current_display[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
                # -------------------------------------

        if not found_mapped_marker:
            # If no *mapped* marker is found, show "Scan" text
            cv2.putText(
                current_display, 
                "Scan an ArUco Code", 
                (50, DISPLAY_WINDOW_HEIGHT // 2), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.5, (255, 255, 255), 3
            )

        # Show the camera feed and the AR display
        cv2.imshow("Camera Feed", frame)
        cv2.imshow("AR Display", current_display)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if not os.path.exists(MAP_FILE) or not os.path.exists(IMAGE_FOLDER_PATH):
        print("Error: Missing 'aruco_image_map.json' or 'wallpapers' folder.")
        print("Please run the 'create_map.py' script first and ensure your 'wallpapers' folder exists.")
    else:
        main()
