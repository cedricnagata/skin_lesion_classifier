import os
import logging
from PIL import Image
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def resize_image(img, target_size):
    """Resize an image to the target square size using high-quality resampling."""
    return img.resize((target_size, target_size), Image.LANCZOS)

def resize_images(input_folder, output_folder, target_size):
    """Resize all images in the input folder to the specified square size and save to output folder."""
    os.makedirs(output_folder, exist_ok=True)
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(image_extensions)]
    processed_count = 0
    error_count = 0
    total_files = len(image_files)

    for filename in image_files:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        try:
            with Image.open(input_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                resized_img = resize_image(img, target_size)
                resized_img.save(output_path)
                processed_count += 1
                sys.stdout.write(f"\rProcessed: {processed_count}/{total_files} images")
                sys.stdout.flush()
        except Exception as e:
            error_count += 1
            logging.error(f"Error processing {filename}: {str(e)}")
    print()
    logging.info(f"Completed: {processed_count} processed, {error_count} errors")

    # Verification step: check all output images are the correct size
    wrong_size = []
    for filename in os.listdir(output_folder):
        if filename.lower().endswith(image_extensions):
            output_path = os.path.join(output_folder, filename)
            try:
                with Image.open(output_path) as img:
                    if img.size != (target_size, target_size):
                        wrong_size.append(filename)
            except Exception as e:
                logging.error(f"Error verifying {filename}: {str(e)}")
    if wrong_size:
        logging.warning(f"{len(wrong_size)} images are not the correct size ({target_size}x{target_size}): {wrong_size}")
    else:
        logging.info(f"All images in {output_folder} are the correct size: {target_size}x{target_size}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Resize all images in DATA_DIR/images/raw to a square size and save to DATA_DIR/images/processed.")
    parser.add_argument('--size', type=int, required=True, help='Target square size (e.g., 600)')
    args = parser.parse_args()

    DATA_DIR = os.getenv("DATA_DIR")
    if not DATA_DIR:
        logging.error("DATA_DIR environment variable not set. Please set it in your .env file.")
        return

    input_dir = os.path.join(DATA_DIR, "images", "raw")
    output_dir = os.path.join(DATA_DIR, "images", "processed")

    logging.info(f"Resizing images in {input_dir} to {args.size}x{args.size} and saving to {output_dir}")
    resize_images(input_dir, output_dir, args.size)

if __name__ == "__main__":
    main() 