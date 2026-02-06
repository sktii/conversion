import os
import sys
import subprocess

# Define script names
SCRIPT_STEP_TO_STL = "step_to_stl.py"
SCRIPT_STL_TO_PILLARS = "stl_to_pillar_xml.py"

def run_step(command):
    print(f"\n🔹 Running: {command}")
    try:
        # Run and capture output to show it in real-time
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Read output
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())

        rc = process.poll()
        if rc != 0:
            err = process.stderr.read()
            print(f"❌ Command failed with return code {rc}")
            print(f"Error output:\n{err}")
            return False
        return True
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        return False

def main():
    print("=========================================")
    print("   STEP to MuJoCo Pillars Converter      ")
    print("=========================================")

    # 1. STEP -> STL
    print("\n[Step 1/2] Converting STEP to STL...")
    # This script automatically finds the STEP file in STP folder
    success = run_step(f"python {SCRIPT_STEP_TO_STL}")
    if not success:
        print("❌ Step 1 failed. Aborting.")
        sys.exit(1)

    # Identify the generated STL file
    # We look into the output of step_to_stl or just check the STL folder
    stl_dir = "STL"
    if not os.path.exists(stl_dir):
        print(f"❌ Error: STL directory {stl_dir} not found.")
        sys.exit(1)

    stl_files = [f for f in os.listdir(stl_dir) if f.lower().endswith('.stl') and "test_box" not in f]

    # Filter out test_box if it exists, to find the real file.
    # If the user has multiple files, we might pick the wrong one.
    # Let's try to be smarter or just ask the user?
    # For now, pick the largest one or just the first non-test one.

    target_stl = None
    if stl_files:
        target_stl = os.path.join(stl_dir, stl_files[0])

    if not target_stl:
        # Maybe the user only has the test file or the file generation failed silently?
        # But step_to_stl checks for file existence.
        # Let's check if the user provided an argument to step_to_stl?
        # The main_workflow calls it without args, so it processes the first STEP found.
        pass

    # If we can't determine the file easily from here without parsing stdout,
    # let's just assume the user wants to process the *latest* generated STL.
    # Or simply:
    stl_files_all = [os.path.join(stl_dir, f) for f in os.listdir(stl_dir) if f.lower().endswith('.stl')]
    if not stl_files_all:
        print("❌ No STL files found to process.")
        sys.exit(1)

    # Pick the most recently modified file
    target_stl = max(stl_files_all, key=os.path.getmtime)

    print(f"\n✅ STL File ready: {target_stl}")

    # 2. STL -> XML
    print("\n[Step 2/2] Generating Pillars XML...")
    output_xml = "fitted_pillars.xml"
    success = run_step(f"python {SCRIPT_STL_TO_PILLARS} {target_stl} {output_xml}")

    if success:
        print("\n🎉 WORKFLOW COMPLETE!")
        print(f"👉 Final XML saved to: {os.path.abspath(output_xml)}")
    else:
        print("❌ Step 2 failed.")

if __name__ == "__main__":
    main()
