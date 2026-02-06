# STEP to MuJoCo Pillars Converter

This project converts a 3D CAD model (`.step`) into a simplified MuJoCo XML environment containing 32 pillars (16 cylinders and 16 boxes) to approximate the structure.

## Usage

1. **Place your STEP file** in the `STP/` folder.
2. **Run the main workflow**:
   ```bash
   python main_workflow.py
   ```

   This will:
   1. Convert the STEP file to STL using `step_to_stl.py`.
   2. Generate the approximated pillars XML using `stl_to_pillar_xml.py`.
   3. Save the result to `fitted_pillars.xml`.

## Scripts Explanation

*   **`step_to_stl.py`**: Handles the conversion from STEP to STL format. It uses `cadquery` and includes robust error checking.
*   **`stl_to_pillar_xml.py`**: Reads an STL file, samples points from its surface, clusters them into 32 groups using K-Means, and generates an XML file with corresponding geometric primitives.
*   **`main_workflow.py`**: Orchestrates the above two scripts.

## Requirements

*   Python 3.10+
*   `cadquery`
*   `trimesh`
*   `scikit-learn`
*   `numpy`

Install dependencies:
```bash
conda install -c cadquery -c conda-forge cadquery=master
pip install trimesh scikit-learn numpy
```
