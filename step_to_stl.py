import os
import sys

# Try importing cadquery. If it fails, we can't do the conversion.
try:
    import cadquery as cq
except ImportError:
    print("❌ Critical Error: 'cadquery' library not found.")
    print("   Please install it using: conda install -c cadquery -c conda-forge cadquery=master")
    print("   or check your python environment.")
    sys.exit(1)

# Default paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(BASE_DIR, "STP")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "STL")

# Parameters
TOLERANCE = 0.05  # Lower is finer but heavier
ANGULAR_TOLERANCE = 0.1

def convert_step_to_stl(input_filename=None):
    """
    Converts a STEP file to STL.
    """
    # 1. Setup Input Path
    if input_filename:
        input_path = input_filename
    else:
        # Auto-detect .step or .stp in STP folder
        if not os.path.exists(INPUT_FOLDER):
             print(f"❌ Input folder not found: {INPUT_FOLDER}")
             return None

        files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.step', '.stp'))]
        if not files:
            print(f"❌ No STEP files found in {INPUT_FOLDER}")
            return None
        # Pick the first one for now, or loop? The user implies one main file.
        input_path = os.path.join(INPUT_FOLDER, files[0])

    print(f"🚀 Starting Conversion Process")
    print(f"📂 Input File: {input_path}")

    if not os.path.exists(input_path):
        print(f"❌ File does not exist: {input_path}")
        return None

    # 2. Setup Output Path
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"📁 Created Output Folder: {OUTPUT_FOLDER}")

    file_name = os.path.basename(input_path)
    base_name = os.path.splitext(file_name)[0]
    output_path = os.path.join(OUTPUT_FOLDER, base_name + ".stl")

    print(f"📂 Target Output: {output_path}")

    # 3. Import STEP
    try:
        print("⏳ Importing STEP file (this may take a while)...")
        model = cq.importers.importStep(input_path)

        # 4. Debugging Content
        # Check if we actually got solids
        solids = model.solids().vals()
        print(f"ℹ️  Found {len(solids)} solid(s) in the model.")

        if len(solids) == 0:
             print("⚠️  Warning: No solids found in STEP file! Attempting to export anyway, but output might be empty.")

        # 5. Export STL
        print("⏳ Exporting to STL...")
        cq.exporters.export(
            model,
            output_path,
            exportType="STL",
            tolerance=TOLERANCE,
            angularTolerance=ANGULAR_TOLERANCE
        )

    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return None

    # 6. Verification
    if os.path.exists(output_path):
        size = os.path.getsize(output_path)
        print(f"✅ Conversion Successful!")
        print(f"📄 Output File: {output_path}")
        print(f"📊 File Size: {size / (1024*1024):.2f} MB")
        if size < 100:
            print("⚠️  Warning: File size is extremely small. The export might be empty.")
        return output_path
    else:
        print("❌ Critical Error: Export function finished but file was not created.")
        return None

if __name__ == "__main__":
    # Allow passing file via command line
    if len(sys.argv) > 1:
        convert_step_to_stl(sys.argv[1])
    else:
        convert_step_to_stl()
