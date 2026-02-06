import os
import sys
import trimesh
import numpy as np

# Scale Threshold: 10 meters. If object is bigger, it's likely mm.
SCALE_THRESHOLD = 10.0

def generate_raw_xml(stl_path, output_xml="raw_mesh.xml"):
    """
    Wraps an STL mesh file into a simple MuJoCo XML.
    Auto-detects scale (mm vs m) and adds scale attribute if needed.
    """

    if not os.path.exists(stl_path):
        print(f"❌ Error: STL file not found: {stl_path}")
        return

    stl_filename = os.path.basename(stl_path)
    stl_dir = os.path.dirname(os.path.abspath(stl_path))

    # Check scale
    try:
        mesh = trimesh.load(stl_path)
        bounds = mesh.bounds
        max_dim = np.max(bounds[1] - bounds[0])

        scale_str = "1 1 1"
        if max_dim > SCALE_THRESHOLD:
            print(f"⚠️  Detected large dimensions (max={max_dim:.2f}). Assuming Millimeters.")
            print(f"   -> Adding scale='0.001 0.001 0.001' to mesh asset.")
            scale_str = "0.001 0.001 0.001"

    except Exception as e:
        print(f"⚠️  Could not analyze mesh for scaling: {e}. Defaulting to 1 1 1.")
        scale_str = "1 1 1"

    mesh_name = "target_mesh"

    xml_content = f"""<mujoco model="raw_mesh_view">
  <compiler angle="radian" meshdir="{stl_dir}"/>

  <option timestep="0.002" gravity="0 0 -9.81"/>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>

    <mesh name="{mesh_name}" file="{stl_filename}" scale="{scale_str}"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="2 2 .05" type="plane" material="grid"/>

    <!-- The Raw Mesh -->
    <geom name="imported_part" type="mesh" mesh="{mesh_name}" rgba="0.8 0.8 0.8 1"/>
  </worldbody>
</mujoco>
"""

    with open(output_xml, "w") as f:
        f.write(xml_content)

    print(f"✅ Generated Raw XML: {os.path.abspath(output_xml)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        stl = sys.argv[1]
        out = "raw_mesh.xml"
        if len(sys.argv) > 2:
            out = sys.argv[2]
        generate_raw_xml(stl, out)
    else:
        print("Usage: python generate_raw_mesh_xml.py <stl_path> <output_xml>")
