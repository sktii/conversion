import os
import sys
import glob
import time
import tempfile
import shutil

# Try importing necessary libraries
try:
    import cadquery as cq
    import trimesh
    import numpy as np
except ImportError as e:
    print(f"❌ Critical Error: Missing dependency: {e}")
    print("   Please install: conda install -c cadquery -c conda-forge cadquery=master")
    print("   and: pip install trimesh numpy")
    sys.exit(1)

# Default paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(BASE_DIR, "STP")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "STL")

# Parameters
TOLERANCE = 0.05
ANGULAR_TOLERANCE = 0.1

def ensure_binary_stl(filepath):
    """
    Checks if the STL is binary. If not, converts it using Trimesh.
    Binary STL starts with 80 bytes header.
    ASCII STL starts with 'solid'.
    """
    try:
        with open(filepath, 'rb') as f:
            header = f.read(5)

        # Check for ASCII marker
        if header.startswith(b'solid'):
            print(f"ℹ️  Detected ASCII STL: {filepath}. Converting to Binary...")
            mesh = trimesh.load(filepath)
            # Export as binary STL
            mesh.export(filepath, file_type='stl') # Defaults to binary
            print(f"✅ Converted to Binary STL.")
        else:
            pass

    except Exception as e:
        print(f"⚠️  Error verifying/converting STL format: {e}")

def convert_step_to_stl(input_filename=None):
    """
    Converts a STEP file to STL.
    Outputs:
    1. Individual parts in STL/{basename}_parts/
    2. Merged mesh in STL/{basename}.stl
    """
    # 1. Setup Input Path
    if input_filename:
        input_path = input_filename
    else:
        if not os.path.exists(INPUT_FOLDER):
             print(f"❌ Input folder not found: {INPUT_FOLDER}")
             return None
        # Find step files
        files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.step', '.stp'))]
        if not files:
            print(f"❌ No STEP files found in {INPUT_FOLDER}")
            return None
        input_path = os.path.join(INPUT_FOLDER, files[0])

    print(f"🚀 Starting Conversion Process")
    print(f"📂 Input File: {input_path}")

    if not os.path.exists(input_path):
        print(f"❌ File does not exist: {input_path}")
        return None

    # 2. Setup Output Path
    if not os.path.exists(OUTPUT_FOLDER):
        try:
            os.makedirs(OUTPUT_FOLDER)
        except OSError as e:
            print(f"❌ Error creating output directory: {e}")
            return None

    file_name = os.path.basename(input_path)
    base_name = os.path.splitext(file_name)[0]

    # Merged Output
    merged_output_path = os.path.join(OUTPUT_FOLDER, base_name + ".stl")

    # Parts Output Directory
    parts_dir = os.path.join(OUTPUT_FOLDER, f"{base_name}_parts")
    if not os.path.exists(parts_dir):
        os.makedirs(parts_dir)

    print(f"📂 Merged Output: {merged_output_path}")
    print(f"📂 Parts Output: {parts_dir}/")

    # 3. Import STEP
    try:
        print("⏳ Importing STEP file (this may take a while)...")
        start_time = time.time()
        model = cq.importers.importStep(input_path)
        print(f"   Import took {time.time() - start_time:.2f}s")

        # Check solids
        solids = model.solids().vals()
        print(f"ℹ️  Found {len(solids)} solid(s) in the model.")

        if len(solids) == 0:
             print("⚠️  Warning: No solids found!")
             return None

        # --- Process Solids ---
        meshes = []
        success_count = 0

        for i, solid in enumerate(solids):
            part_file = os.path.join(parts_dir, f"part_{i}.stl")
            try:
                # Wrap solid in Workplane
                wp = cq.Workplane("XY").newObject([solid])

                # Export Individual Part
                cq.exporters.export(
                    wp,
                    part_file,
                    exportType="STL",
                    tolerance=TOLERANCE,
                    angularTolerance=ANGULAR_TOLERANCE
                )

                # Ensure Binary
                ensure_binary_stl(part_file)

                if os.path.exists(part_file) and os.path.getsize(part_file) > 0:
                    # Load back for merging
                    m = trimesh.load(part_file)
                    meshes.append(m)
                    success_count += 1
                    print(f"   ✅ Processed solid {i+1}/{len(solids)} -> {os.path.basename(part_file)}", end="\r")
                else:
                    print(f"   ⚠️  Failed to export solid {i+1} (empty file)")
            except Exception as e:
                print(f"   ⚠️  Error processing solid {i+1}: {e}")

        print(f"\nℹ️  Successfully processed {success_count}/{len(solids)} parts.")

        if not meshes:
            print("❌ No parts could be converted.")
            return None

        # Concatenate for Merged File
        print("⏳ Merging meshes for pillar calculation...")
        combined_mesh = trimesh.util.concatenate(meshes)

        print(f"⏳ Saving combined mesh to {merged_output_path}...")
        combined_mesh.export(merged_output_path) # Trimesh default is binary
        ensure_binary_stl(merged_output_path) # Double check

    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Final Verification
    if verify_file(merged_output_path):
        print("✅ Conversion Successful!")
        return merged_output_path
    else:
        print("❌ Critical Error: Export failed.")
        return None

def verify_file(path):
    if os.path.exists(path):
        size = os.path.getsize(path)
        if size > 100:
            return True
        else:
            print(f"   (File exists but is too small: {size} bytes)")
    return False

if __name__ == "__main__":
    result = None
    if len(sys.argv) > 1:
        result = convert_step_to_stl(sys.argv[1])
    else:
        result = convert_step_to_stl()

    if result is None:
        sys.exit(1)
