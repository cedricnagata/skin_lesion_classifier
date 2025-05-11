import os
import logging
from PIL import Image
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define base data directory
DATA_DIR = os.getenv("DATA_DIR")
logging.info(f"Using data directory: {DATA_DIR}")

# Define all paths relative to DATA_DIR
IMAGE_DIR = os.path.join(DATA_DIR, "images")
IMAGE_RAW_DIR = os.path.join(IMAGE_DIR, "raw")
IMAGE_PROCESSED_DIR = os.path.join(IMAGE_DIR, "processed")

def crop_and_verify_images(input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(input_folder) 
                  if f.lower().endswith(image_extensions)]
    
    processed_count = 0
    error_count = 0
    total_files = len(image_files)
    
    # Process each image
    for filename in image_files:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        try:
            with Image.open(input_path) as img:
                # Check if image is 600x450
                if img.size != (600, 450):
                    error_count += 1
                    continue
                
                # Calculate crop box (75px from each side)
                left = 75
                top = 0
                right = 525  # 600 - 75
                bottom = 450
                
                # Crop the image
                cropped_img = img.crop((left, top, right, bottom))
                
                # Save the cropped image
                cropped_img.save(output_path)
                processed_count += 1
                
                # Print live count
                sys.stdout.write(f"\rProcessed: {processed_count}/{total_files} images")
                sys.stdout.flush()
                
        except Exception as e:
            error_count += 1
    
    # Print final newline and summary
    print()  # Move to next line
    logging.info(f"Completed: {processed_count} processed, {error_count} errors")

if __name__ == "__main__":
    crop_and_verify_images(IMAGE_RAW_DIR, IMAGE_PROCESSED_DIR)