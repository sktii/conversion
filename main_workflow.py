import os
import sys
import subprocess
import glob

# Define script names
SCRIPT_STEP_TO_STL = "step_to_stl.py"
SCRIPT_STL_TO_PILLARS = "stl_to_pillar_xml.py"
SCRIPT_FIND_CENTER = "find_disk_center.py"

def run_step(command):
    print(f"\n🔹 Running: {command}")
    try:
        # Run and capture output
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        last_line = ""
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                last_line = output.strip()

        rc = process.poll()
        if rc != 0:
            err = process.stderr.read()
            print(f"❌ Command failed with return code {rc}")
            print(f"Error output:\n{err}")
            return False, None

        return True, last_line
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        return False, None

def main():
    print("=========================================")
    print("   STEP to MuJoCo Pillars Converter      ")
    print("=========================================")

    # 1. STEP -> STL
    print("\n[Step 1/3] Converting STEP to STL...")
    success, _ = run_step(f"python {SCRIPT_STEP_TO_STL}")
    if not success:
        print("❌ Step 1 failed. Aborting.")
        sys.exit(1)

    # Identify generated STL
    stl_dir = "STL"
    if not os.path.exists(stl_dir):
        print(f"❌ Error: STL directory {stl_dir} not found.")
        sys.exit(1)

    stl_files_all = [os.path.join(stl_dir, f) for f in os.listdir(stl_dir) if f.lower().endswith('.stl') and "part_" not in f]
    if not stl_files_all:
        print("❌ No STL files found.")
        sys.exit(1)

    target_stl = max(stl_files_all, key=os.path.getmtime)
    print(f"✅ STL File ready: {target_stl}")

    # 2. Find Disk Center
    print("\n[Step 2/3] Detecting Disk Center for Robot Placement...")
    # Capture output from find_disk_center.py which prints "CENTER_RESULT:x,y,z"
    # We need to run it and parse stdout.

    # We use subprocess.check_output to capture all output easily
    try:
        output = subprocess.check_output(f"python {SCRIPT_FIND_CENTER} {target_stl}", shell=True, text=True)
        print(output)

        robot_pos = "0 0 0"
        for line in output.splitlines():
            if "CENTER_RESULT:" in line:
                coords = line.split(":")[1].strip().split(",")
                robot_pos = f"{coords[0]} {coords[1]} {coords[2]}"
                break

        print(f"📍 Detected Robot Position: {robot_pos}")

    except subprocess.CalledProcessError as e:
        print(f"⚠️  Center detection failed ({e}). Defaulting to 0 0 0.")
        robot_pos = "0 0 0"

    # 3. STL -> XML
    print("\n[Step 3/3] Generating Pillars XML...")
    output_xml = "fitted_pillars.xml"

    # Pass robot_pos to the script
    success, _ = run_step(f"python {SCRIPT_STL_TO_PILLARS} {target_stl} {output_xml} {robot_pos}")

    if success:
        print("\n🎉 WORKFLOW COMPLETE!")
        print(f"👉 Final XML saved to: {os.path.abspath(output_xml)}")
    else:
        print("❌ Step 3 failed.")

if __name__ == "__main__":
    main()
