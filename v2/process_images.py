from PIL import Image
import os
import argparse
from pathlib import Path

def crop_and_verify_images(input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(input_folder) 
                  if f.lower().endswith(image_extensions)]
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for filename in image_files:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        try:
            with Image.open(input_path) as img:
                # Check if image is 600x450
                if img.size != (600, 450):
                    print(f"Warning: {filename} is not 600x450 (actual size: {img.size})")
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
                print(f"Processed: {filename}")
                
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    # Verify all images in output folder
    print("\nVerifying cropped images...")
    all_correct = True
    for filename in os.listdir(output_folder):
        if filename.lower().endswith(image_extensions):
            try:
                with Image.open(os.path.join(output_folder, filename)) as img:
                    if img.size != (450, 450):
                        print(f"Error: {filename} is not 450x450 (actual size: {img.size})")
                        all_correct = False
            except Exception as e:
                print(f"Error verifying {filename}: {str(e)}")
                all_correct = False
    
    if all_correct:
        print("\nAll images have been successfully cropped to 450x450!")
    else:
        print("\nSome images were not properly cropped. Please check the errors above.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crop images from 600x450 to 450x450')
    parser.add_argument('--image_dir', required=True, help='Path to the folder containing input images')
    parser.add_argument('--output_dir', required=True, help='Path to the folder where cropped images will be saved')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    input_folder = os.path.abspath(args.input_folder)
    output_folder = os.path.abspath(args.output_folder)
    
    # Process images
    crop_and_verify_images(input_folder, output_folder) 