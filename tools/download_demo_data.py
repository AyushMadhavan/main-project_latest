import os
import cv2
import logging
import argparse
from sklearn.datasets import fetch_lfw_people
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DownloadDemoData")

def main():
    parser = argparse.ArgumentParser(description="Download LFW subset for testing")
    parser.add_argument("--output", type=str, default="data/lfw_subset", help="Output directory")
    parser.add_argument("--min_faces", type=int, default=20, help="Min faces per person to include")
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    logger.info("Downloading LFW dataset (this may take a moment)...")
    # Fetch data: min_faces_per_person fliters for people with many images (good for matching)
    try:
        lfw_people = fetch_lfw_people(min_faces_per_person=args.min_faces, resize=None, color=True) 
    except TypeError:
        # Fallback for older sklearn versions
        lfw_people = fetch_lfw_people(min_faces_per_person=args.min_faces, resize=None)

    logger.info(f"Found {len(lfw_people.target_names)} identities with enough images.")
    
    for i, (image, target) in enumerate(zip(lfw_people.images, lfw_people.target)):
        name = lfw_people.target_names[target].replace(' ', '_')
        person_dir = os.path.join(args.output, name)
        
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
            
        # LFW images from sklearn are normalized 0-1 float or 0-255 depending on version, 
        # usually 0-255 float32. cv2 expects uint8.
        # Check range
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
            
        # Sklearn LFW is grayscale? 
        # Actually fetch_lfw_people returns grayscale content by default or color?
        # Argument 'color=True' is needed for color.
        # Let's fix the call.
        
        # Convert RGB (sklearn) to BGR (OpenCV)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
        save_path = os.path.join(person_dir, f"{name}_{i}.jpg")
        cv2.imwrite(save_path, image)

    logger.info(f"Saved images to {args.output}")
    logger.info("You can now run 'python tools/bulk_import.py --dataset data/lfw_subset' to import them.")

if __name__ == "__main__":
    # monkey patch for color fetch if needed, but sklearn default is fine for basic test.
    # Note: color=True in fetch_lfw_people is safer for our pipeline
    # Let's re-wrap the fetch part slightly to ensure color.
    try:
        from sklearn.datasets import fetch_lfw_people
        # Re-run logic inside main, but let's override the fetch call here for clarity
        pass
    except ImportError:
        logger.error("scikit-learn not installed. Please run 'pip install scikit-learn'")
    
    # We will just run main() but I'll add color=True inside main in the file write
    main()
