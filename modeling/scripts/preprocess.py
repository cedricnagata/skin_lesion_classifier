import os
from PIL import Image

def resize_images(source_dir, target_dir, target_size=(224, 224)):
    """
    Resize images in the source directory and save them to the target directory,
    using the LANCZOS filter for high-quality downsampling.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                # Construct source and target file paths
                source_path = os.path.join(root, file)
                target_path = os.path.join(target_dir, file)

                # Open and resize the image
                with Image.open(source_path) as img:
                    img = img.resize(target_size, Image.LANCZOS)  # Use Image.LANCZOS for high-quality downsampling
                    
                    # Save the resized image
                    img.save(target_path)

resize_images('images/ISIC-images', 'images/processed')
