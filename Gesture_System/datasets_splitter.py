import shutil
import random
from pathlib import Path
from tqdm import tqdm  

# --- CONFIGURATION ---
SOURCE_DIR = Path("dataset_1_6_2026")    # Raw data folder
DEST_DIR = Path("dataset_processed")     # Where the sorted data will go
SPLIT_RATIO = 0.8                        # 80% Train, 20% Validation
SEED = 42                                # For reproducibility

def organize_dataset():
    # 1. Setup
    if not SOURCE_DIR.exists():
        print(f"Error: Source directory '{SOURCE_DIR}' not found.")
        return

    # Clean destination if it exists to avoid duplicates
    if DEST_DIR.exists():
        user_input = input(f"Warning: '{DEST_DIR}' exists. Delete and overwrite? (y/n): ")
        if user_input.lower() == 'y':
            shutil.rmtree(DEST_DIR)
        else:
            print("Operation cancelled.")
            return

    random.seed(SEED)
    
    # 2. Iterate through each Class Folder (e.g., "SwipeUp", "Pinch")
    class_folders = [f for f in SOURCE_DIR.iterdir() if f.is_dir()]
    
    print(f"Found {len(class_folders)} classes: {[f.name for f in class_folders]}")

    for class_folder in tqdm(class_folders, desc="Processing Classes"):
        class_name = class_folder.name
        
        # Get all Sequence Folders (e.g., "001", "002") inside this class
        sequence_folders = [f for f in class_folder.iterdir() if f.is_dir()]
        
        # Shuffle sequences to ensure random split
        random.shuffle(sequence_folders)
        
        # Calculate split index
        split_idx = int(len(sequence_folders) * SPLIT_RATIO)
        train_seqs = sequence_folders[:split_idx]
        valid_seqs = sequence_folders[split_idx:]
        
        # 3. Copy Folders to Destination
        # Helper function to copy
        def copy_sequences(sequences, split_name):
            for seq_path in sequences:
                # Target: dataset_processed / train / SwipeUp / 001
                target_path = DEST_DIR / split_name / class_name / seq_path.name
                
                # Copy the entire directory tree (Sequence + Images inside)
                shutil.copytree(seq_path, target_path)

        copy_sequences(train_seqs, "train")
        copy_sequences(valid_seqs, "valid")

    print("\nOrganization Complete!")
    print(f"Dataset ready at: {DEST_DIR.resolve()}")
    print("Structure created:")
    print(f"  {DEST_DIR}/train/...")
    print(f"  {DEST_DIR}/valid/...")

if __name__ == "__main__":
    organize_dataset()