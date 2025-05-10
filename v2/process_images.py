import os
import logging
from PIL import Image
# Define base data directory
DATA_DIR = os.getenv("DATA_DIR")
print(DATA_DIR)

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
    
    logging.info(f"Found {len(image_files)} images to process")
    
    # Process each image
    for filename in image_files:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        try:
            with Image.open(input_path) as img:
                # Check if image is 600x450
                if img.size != (600, 450):
                    logging.warning(f"{filename} is not 600x450 (actual size: {img.size})")
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
                logging.info(f"Processed: {filename}")
                
        except Exception as e:
            logging.error(f"Error processing {filename}: {str(e)}")
    
    # Verify all images in output folder
    logging.info("\nVerifying cropped images...")
    all_correct = True
    for filename in os.listdir(output_folder):
        if filename.lower().endswith(image_extensions):
            try:
                with Image.open(os.path.join(output_folder, filename)) as img:
                    if img.size != (450, 450):
                        logging.error(f"{filename} is not 450x450 (actual size: {img.size})")
                        all_correct = False
            except Exception as e:
                logging.error(f"Error verifying {filename}: {str(e)}")
                all_correct = False
    
    if all_correct:
        logging.info("\nAll images have been successfully cropped to 450x450!")
    else:
        logging.error("\nSome images were not properly cropped. Please check the errors above.")

# Run image processing
if __name__ == "__main__":
    crop_and_verify_images(IMAGE_RAW_DIR, IMAGE_PROCESSED_DIR)