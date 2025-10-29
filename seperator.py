import cv2
import os
import re
import pytesseract
import shutil
from thefuzz import process, fuzz
import numpy as np

# --- Configuration ---

# 1. SET THIS TO FALSE to run the script silently without pop-ups.
#    Set to TRUE if you need to debug your coordinates again.
DEBUG_MODE = False

# 2. The main folder with your ID card images
SOURCE_FOLDER = "Research_conclave\Dinner_checking\ID_cards"

# 3. (CRITICAL) List all valid club names here.
KNOWN_CLUBS = [
    "IBOT CLUB", "RAFTAAR", "SAHAAY", "AGNIRATH", "PROGRAMMING CLUB",
    "CYBERSECURITY CLUB", "ABHYUDAY", "ELECTRONICS CLUB", "ENVISAGE",
    "MATHEMATICS CLUB", "AVISHKAR", "BIOTECH CLUB", "AI CLUB", "AMOGH",
    "PRODUCT DESIGN CLUB", "WEBOPS AND BLOCKCHAIN", "ABHIYAN", "IGEM",
    "3D PRINTING CLUB", "AERO CLUB", "HORIZON", "ANVESHAK","BRANDING AND ENGAGEMENT",
]

# 4. Your fine-tuned coordinates.
CLUB_ROI = {
    "y1": 750,  # Top-most pixel of the club box
    "y2": 910,  # Bottom-most pixel of the club box
    "x1": 80,   # Left-most pixel of the club box
    "x2": 520   # Right-most pixel of the club box
}

# 5. Your fine-tuned match threshold.
MATCH_THRESHOLD = 40

# ---------------------

# --- Tesseract Configuration ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def sanitize_foldername(name_text):
    """Cleans the text to make a valid folder name."""
    clean_name = re.sub(r'[^a-zA-Z0-9 ]', '', name_text).strip()
    return clean_name.replace(" ", "_")

def preprocess_for_ocr(image_crop):
    """
    Cleans a crop segment to make it easier for Tesseract to read.
    """
    gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    (thresh, binarized) = cv2.threshold(gray_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(binarized)
    return inverted

def sort_images_by_club_validated(source_dir, roi, clubs, threshold):
    print(f"Scanning folder: {source_dir}...")
    
    valid_extensions = ('.png', '.jpg', '.jpeg')
    
    try:
        filenames = [f for f in os.listdir(source_dir) 
                     if os.path.isfile(os.path.join(source_dir, f)) 
                     and f.lower().endswith(valid_extensions)]
    except FileNotFoundError:
        print(f"Error: Source folder not found at {source_dir}")
        return

    if not filenames:
        print("No image files found in the source folder.")
        return

    for filename in filenames:
        old_file_path = os.path.join(source_dir, filename)

        try:
            img = cv2.imread(old_file_path)
            if img is None:
                print(f"  - Could not read image {filename}. Skipping.")
                continue

            # 1. Crop the image to the club ROI
            y1, y2 = roi["y1"], roi["y2"]
            x1, x2 = roi["x1"], roi["x2"]
            club_crop = img[y1:y2, x1:x2]
            
            # --- PRE-PROCESSING STEP ---
            processed_crop = preprocess_for_ocr(club_crop)
            # ---------------------------

            # --- DEBUG MODE (Now skipped) ---
            if DEBUG_MODE:
                print(f"\n[DEBUG] Processing {filename}")
                print("Showing original crop (left) and processed (right).")
                print("  ==> Press 'q' in the window to continue. <==")
                
                original_debug = cv2.resize(club_crop, (processed_crop.shape[1], processed_crop.shape[0]))
                processed_debug_color = cv2.cvtColor(processed_crop, cv2.COLOR_GRAY2BGR)
                debug_image = np.hstack((original_debug, processed_debug_color))
                
                cv2.imshow("Debug - Original vs Processed", debug_image)
                
                # --- BUG FIX ---
                # Wait until user presses 'q' (not equal to)
                while cv2.waitKey(0) != ord('q'):
                    pass
                cv2.destroyWindow("Debug - Original vs Processed")
            # --- END DEBUG MODE ---

            # 2. Read text from the *processed* image.
            config = "--psm 6" 
            ocr_text = pytesseract.image_to_string(processed_crop, config=config)
            ocr_text = ocr_text.strip().upper()

            if not ocr_text:
                print(f"  - No text detected in {filename}. Skipping.")
                continue

            # 3. Find the best match from our KNOWN_CLUBS list
            best_match = process.extractOne(ocr_text, clubs, scorer=fuzz.ratio)
            
            if best_match and best_match[1] >= threshold:
                # 4. We have a confident match!
                folder_name_base = best_match[0]
                folder_name = sanitize_foldername(folder_name_base)
                
                # 5. Create folder
                target_folder_path = os.path.join(source_dir, folder_name)
                if not os.path.exists(target_folder_path):
                    os.makedirs(target_folder_path)
                    print(f"  + Created new folder: {folder_name}")
                
                # 6. Move the file
                new_file_path = os.path.join(target_folder_path, filename)
                shutil.move(old_file_path, new_file_path)
                print(f"  -> Match: '{ocr_text}' -> '{folder_name}'. Moved {filename}.")

            else:
                # 7. Match is too poor.
                if best_match:
                    print(f"  - No match for '{ocr_text}' (Best: '{best_match[0]}' @ {best_match[1]}%). Skipping {filename}.")
                else:
                    print(f"  - No match for '{ocr_text}'. Skipping {filename}.")

        except Exception as e:
            print(f"  - Error processing {filename}: {e}")

# --- Main execution ---
if __name__ == "__main__":
    # IMPORTANT: Test this on a COPY of your folder first.
    sort_images_by_club_validated(
        SOURCE_FOLDER, 
        CLUB_ROI, 
        KNOWN_CLUBS, 
        MATCH_THRESHOLD
    )
    print("\nValidated club sorting complete.")