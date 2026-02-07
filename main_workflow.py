import os
import sys
import subprocess
import glob

# Define script names
SCRIPT_STEP_TO_STL = "step_to_stl.py"

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
    print("   STEP to STL Converter Only            ")
    print("   (Use 'stl_to_mujoco.py' for XML)      ")
    print("=========================================")

    # 1. STEP -> STL
    print("\n[Step 1] Converting STEP to STL...")
    success, _ = run_step(f"python {SCRIPT_STEP_TO_STL}")
    if not success:
        print("❌ Step 1 failed. Aborting.")
        sys.exit(1)

    print("\n✅ STL Generation Complete.")
    print("👉 Now run: python stl_to_mujoco.py <path_to_stl>")

if __name__ == "__main__":
    main()
