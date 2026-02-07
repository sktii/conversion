# STEP to MuJoCo Environment Converter

This project converts `.step` CAD files into MuJoCo XML environments, specifically designed to approximate geometry with a set of 32 pillars (cylinders and boxes) for AI model inputs. It also automatically places a UR5e robot on a detected "disk" surface.

## Features

*   **STEP to STL**: Robust conversion using `cadquery` (with fallback to `trimesh`).
*   **Automatic Scale Detection**: Automatically detects if the input model is in millimeters (e.g., table width > 10m) and scales it to meters.
*   **Smart Pillar Fitting**: Approximates the geometry using up to 32 pillars. Uses a smart merging algorithm to group adjacent clusters, reducing the number of pillars for simple shapes (like tables) to improve simulation performance and visual clarity. Unused pillars are moved out of the workspace.
*   **Robot Placement**: Automatically detects a "disk" or table surface and places the UR5e robot base at that center point.
*   **Dual Output**: Generates both a `fitted_pillars.xml` (the approximation) and a `raw_mesh.xml` (ground truth for debugging).

## Prerequisites

Ensure you have the following installed:

*   Python 3.x
*   Dependencies:
    ```bash
    pip install trimesh numpy scikit-learn scipy
    ```
*   `cadquery` (via conda recommended):
    ```bash
    conda install -c cadquery -c conda-forge cadquery=master
    ```

## Usage

### 1. Place Input Files
Put your `.step` or `.stp` file into the `STP/` directory.

### 2. Convert STEP to STL
Run the main workflow to handle the CAD conversion:
```bash
python main_workflow.py
```
This will generate `.stl` files in the `STL/` directory.

### 3. Generate MuJoCo XML
Run the unified converter script on the generated STL file:
```bash
python stl_to_mujoco.py STL/your_file.stl
```

### 4. Check Outputs
The generated XML files will be in the `XML/` directory:

*   **`XML/fitted_pillars.xml`**: The main environment file. It contains:
    *   The UR5e robot (positioned on the table/disk).
    *   The approximated pillar geometry (boxes and cylinders).
    *   *Note: Unused pillars are hidden at position (10, 0, 0).*
*   **`XML/raw_mesh.xml`**: A simple viewer file that loads the original STL mesh (scaled correctly) for verification.
*   **`XML/ur5e_fitted.xml`**: An intermediate file generated to position the robot correctly.

## Project Structure

*   `main_workflow.py`: Orchestrates the STEP -> STL conversion.
*   `stl_to_mujoco.py`: Handles STL -> XML conversion, scale detection, robot placement, and geometry fitting.
*   `step_to_stl.py`: Helper script for CAD conversion.
*   `STP/`: Input directory.
*   `STL/`: Intermediate output directory.
*   `XML/`: Final output directory.
