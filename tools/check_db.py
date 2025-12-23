import json
import os
import argparse

def main():
    db_path = "./milvus_demo_local.json"
    
    if not os.path.exists(db_path):
        print(f"Database file not found at {db_path}")
        return

    print(f"Loading database from {db_path}...")
    try:
        with open(db_path, 'r') as f:
            data = json.load(f)
            
        # Structure is {"criminals": [...]} or just the dict if I implemented it simply?
        # VectorDB implementation: 
        # all_data = {"collection_name": [list of records]}
        # records = {"name": str, "vector": list, "id": int}
        
        collection_name = "criminals" 
        records = data.get(collection_name, [])
        
        if not records:
            # Fallback check if keys are different
            print(f"No records found under collection '{collection_name}'. Keys: {list(data.keys())}")
            return

        print(f"Total Records: {len(records)}")
        
        # Get unique names
        names = {}
        for r in records:
            n = r.get('name', 'Unknown')
            names[n] = names.get(n, 0) + 1
            
        print("\nIdentities found:")
        print("-" * 30)
        for name, count in sorted(names.items()):
            print(f"{name}: {count} images")
        print("-" * 30)
            
    except Exception as e:
        print(f"Error reading database: {e}")

if __name__ == "__main__":
    main()
