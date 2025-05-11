import os
import logging
from PIL import Image
import sys
import numpy as np
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_image_sizes(input_folder):
    """Analyze image sizes in the dataset to find median dimensions."""
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(input_folder) 
                  if f.lower().endswith(image_extensions)]
    
    # Store dimensions
    dimensions = []
    
    for filename in image_files:
        try:
            with Image.open(os.path.join(input_folder, filename)) as img:
                width, height = img.size
                dimensions.append((width, height))
        except Exception as e:
            logging.error(f"Error analyzing {filename}: {str(e)}")
    
    if not dimensions:
        return None
    
    # Calculate statistics
    widths, heights = zip(*dimensions)
    max_width = max(widths)
    max_height = max(heights)
    min_width = min(widths)
    min_height = min(heights)
    median_width = int(np.median(widths))
    median_height = int(np.median(heights))
    
    # Use the larger of median width/height as target size
    target_size = max(median_width, median_height)
    
    # Count how many images will need padding vs cropping
    padded = 0
    cropped = 0
    total_padding = 0
    
    for width, height in dimensions:
        if width < target_size or height < target_size:
            padded += 1
            total_padding += (target_size - width) * (target_size - height)
        else:
            cropped += 1
    
    return {
        'optimal_size': target_size,
        'max_width': max_width,
        'max_height': max_height,
        'min_width': min_width,
        'min_height': min_height,
        'median_width': median_width,
        'median_height': median_height,
        'total_images': len(dimensions),
        'cropped_ratio': cropped / len(dimensions),
        'padding_stats': {
            'padded': padded,
            'cropped': cropped,
            'total_padding': total_padding
        }
    }

def process_image(img, target_size):
    """Process an image to be square with the target size using padding and cropping."""
    width, height = img.size
    
    # If image is smaller than target size, pad it
    if width < target_size or height < target_size:
        # Create a new black background image
        new_img = Image.new('RGB', (target_size, target_size), (0, 0, 0))
        
        # Calculate position to paste the original image
        paste_x = (target_size - width) // 2
        paste_y = (target_size - height) // 2
        
        # Paste the original image onto the center of the new image
        new_img.paste(img, (paste_x, paste_y))
        return new_img
    
    # If image is larger than target size, crop it
    else:
        # Calculate crop box for center crop
        left = (width - target_size) // 2
        top = (height - target_size) // 2
        right = left + target_size
        bottom = top + target_size
        
        return img.crop((left, top, right, bottom))

def process_images(input_folder, output_folder, target_size):
    """Process all images to be square with the specified target size using padding and cropping."""
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
                # Convert to RGB if necessary (for PNG with transparency)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Process the image (pad or crop)
                processed_img = process_image(img, target_size)
                
                # Save the processed image
                processed_img.save(output_path)
                processed_count += 1
                
                # Print live count
                sys.stdout.write(f"\rProcessed: {processed_count}/{total_files} images")
                sys.stdout.flush()
                
        except Exception as e:
            error_count += 1
            logging.error(f"Error processing {filename}: {str(e)}")
    
    # Print final newline and summary
    print()  # Move to next line
    logging.info(f"Completed: {processed_count} processed, {error_count} errors")

def main():
    # Define base data directory
    DATA_DIR = os.getenv("DATA_DIR")
    if not DATA_DIR:
        logging.error("DATA_DIR environment variable not set")
        return
    
    # Set up input and output directories
    input_dir = os.path.join(DATA_DIR, "images", "raw")
    output_dir = os.path.join(DATA_DIR, "images", "processed")
    
    # Analyze image sizes
    logging.info("Analyzing image sizes in dataset...")
    analysis = analyze_image_sizes(input_dir)
    
    if analysis:
        logging.info(f"Dataset analysis:")
        logging.info(f"Total images: {analysis['total_images']}")
        logging.info(f"Max dimensions: {analysis['max_width']}x{analysis['max_height']}")
        logging.info(f"Min dimensions: {analysis['min_width']}x{analysis['min_height']}")
        logging.info(f"Median dimensions: {analysis['median_width']}x{analysis['median_height']}")
        logging.info(f"Target size: {analysis['optimal_size']}x{analysis['optimal_size']}")
        if analysis['padding_stats']:
            logging.info(f"Images that will be padded: {analysis['padding_stats']['padded']}")
            logging.info(f"Images that will be cropped: {analysis['padding_stats']['cropped']}")
            logging.info(f"Percentage of images that will be cropped: {analysis['cropped_ratio']*100:.1f}%")
        
        # Use the optimal size
        target_size = analysis['optimal_size']
    else:
        logging.error("No images found to analyze")
        return
    
    logging.info(f"Processing images to size: {target_size}x{target_size}")
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")
    
    # Process all images
    process_images(input_dir, output_dir, target_size)

if __name__ == "__main__":
    main() 