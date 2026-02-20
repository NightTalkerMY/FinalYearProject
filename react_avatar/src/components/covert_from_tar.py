import os
import zipfile
import tarfile
import shutil
from pathlib import Path

# --- CONFIGURATION ---
SOURCE_DIR = Path("public/amazon_3d_models_up_final") # Your Zips
TARGET_DIR = Path("public/products")                  # Clean destination

def extract_recursive(asin):
    zip_path = SOURCE_DIR / f"{asin}.zip"
    output_folder = TARGET_DIR / asin
    
    # 1. Skip if already done
    if output_folder.exists():
        print(f"⏩ Skipping {asin} (Already extracted)")
        return

    if not zip_path.exists():
        print(f"⚠️  Missing Zip: {asin}")
        return

    print(f"📦 Processing {asin}...")
    
    # Create temp workspace
    temp_dir = Path("temp_extract")
    if temp_dir.exists(): shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    try:
        # 2. Unzip the Outer ZIP
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # 3. Find the 'metadata.tar'
        # Priority: Right > Left > Root
        tar_path = None
        for sub in ["Right", "Left", "."]:
            candidate = temp_dir / sub / "metadata.tar"
            if candidate.exists():
                tar_path = candidate
                break
        
        if not tar_path:
            print(f"   ❌ No metadata.tar found in {asin}")
            shutil.rmtree(temp_dir)
            return

        # 4. Extract the TAR
        # We extract it DIRECTLY to the target folder to save a move step later
        # But we need to rename the files after
        tar_extract_dir = temp_dir / "tar_content"
        tar_extract_dir.mkdir()
        
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(tar_extract_dir)

        # 5. Find the .gltf inside the extracted TAR content
        # It could be named "B00XYZ.gltf" or anything.
        found_gltf = None
        for file in tar_extract_dir.iterdir():
            if file.suffix.lower() == ".gltf":
                found_gltf = file
                break
        
        if found_gltf:
            output_folder.mkdir(parents=True, exist_ok=True)
            
            # MOVE and RENAME to 'model.gltf'
            # We must also move the .bin and .png/.jpg files!
            for file in tar_extract_dir.iterdir():
                if file.name == found_gltf.name:
                    shutil.move(str(file), str(output_folder / "model.gltf"))
                else:
                    shutil.move(str(file), str(output_folder / file.name))
            
            print(f"   ✅ Success! Extracted {found_gltf.name} -> model.gltf")
        else:
            print(f"   ❌ No .gltf found inside metadata.tar")

    except Exception as e:
        print(f"   ❌ Error processing {asin}: {e}")
    
    # Cleanup
    if temp_dir.exists(): shutil.rmtree(temp_dir)

if __name__ == "__main__":
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    zips = list(SOURCE_DIR.glob("*.zip"))
    print(f"Found {len(zips)} product archives.")

    for z in zips:
        extract_recursive(z.stem)

    print("\n✨ Asset Prep Complete!")