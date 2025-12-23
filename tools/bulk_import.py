import os
import cv2
import glob
import logging
import argparse
import numpy as np
from insightface.app import FaceAnalysis

# Adjust path to import local modules if run as script
import sys
sys.path.append(os.getcwd())

from database.vector_db import VectorDB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BulkImport")

def parse_args():
    parser = argparse.ArgumentParser(description="Bulk Import Criminal Faces")
    parser.add_argument("--dataset", type=str, required=True, help="Path to directory with images (folders as names or filename as name)")
    parser.add_argument("--collection", type=str, default="criminals", help="Milvus collection name")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Init DB
    db = VectorDB(collection_name=args.collection)
    
    # Init Detection for embedding
    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_thresh=0.3)
    
    # Scan files
    image_paths = glob.glob(os.path.join(args.dataset, "*", "*.*")) + glob.glob(os.path.join(args.dataset, "*.*"))
    
    # Get existing names to skip duplicates
    existing_names = set(item.get('name') for item in db.data)
    logger.info(f"Found {len(existing_names)} identities already in database.")
    
    records = []
    
    try:
        for path in image_paths:
            if not path.lower().endswith(('.jpg', '.jpeg', '.png')):
                 continue
                 
            # Derive name
            parent = os.path.dirname(path)
            if os.path.basename(parent) == os.path.basename(args.dataset):
                name = os.path.splitext(os.path.basename(path))[0].split('_')[0]
            else:
                name = os.path.basename(parent)
                
            if name in existing_names:
                # Naive check: skip if we have any record of this person
                # For LFW where we have multiple images per person, this skips subsequent images of same person 
                # strictly speaking we might want all images, but for a demo, one embedding per person is often enough 
                # or we just want to skip if we already processed this *specific* file.
                # However, our DB doesn't store filename.
                # Let's just skip if name is present to save time for now, or maybe not?
                # The user asked "why import all".
                # If we want to allow multiple images, we need a better check. 
                # But to save time, let's just log and continue if we have > 0 samples.
                continue
                
            logger.info(f"Processing {name} from {path}...")
            
            img = cv2.imread(path)
            if img is None:
                logger.warning(f"Could not read {path}")
                continue
                
            faces = app.get(img)
            
            if not faces:
                logger.warning(f"No face detected in {path}")
                continue
                
            # Take the largest face
            faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
            embedding = faces[0].embedding
            
            records.append({
                "vector": embedding,
                "name": name,
                "id": len(db.data) + len(records)
            })
            
            # Save every 10 records
            if len(records) >= 10:
                db.insert_embeddings(records)
                # Update existing names so we don't re-add if we encounter same person again 
                # (unless we want multiple samples). 
                # For this specific logic, I'll add the names to existing_names set
                for r in records:
                    existing_names.add(r['name'])
                records = []
                
    except KeyboardInterrupt:
        logger.info("Import interrupted by user. Saving pending records...")
    finally:
        if records:
            db.insert_embeddings(records)
            
    logger.info("Import process finished.")

if __name__ == "__main__":
    main()
