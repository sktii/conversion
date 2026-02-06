import os
import sys

def generate_raw_xml(stl_path, output_xml="raw_mesh.xml"):
    """
    Wraps an STL mesh file into a simple MuJoCo XML.
    This corresponds to 'stp transfer to xml file, don't need anyother modify'.
    """

    if not os.path.exists(stl_path):
        print(f"❌ Error: STL file not found: {stl_path}")
        return

    # Use basename for asset reference to avoid absolute path issues in asset loading if desired,
    # but relative path is better.
    stl_filename = os.path.basename(stl_path)
    stl_dir = os.path.dirname(os.path.abspath(stl_path))

    # We need to construct a relative path or absolute path for the mesh file attribute
    # Since XML might be run from anywhere, absolute path is safest for this generated file,
    # or relative if we assume a specific run directory.
    # Let's use the provided path directly.

    mesh_name = "target_mesh"

    xml_content = f"""<mujoco model="raw_mesh_view">
  <compiler angle="radian" meshdir="{stl_dir}"/>

  <option timestep="0.002" gravity="0 0 -9.81"/>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>

    <mesh name="{mesh_name}" file="{stl_filename}"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="2 2 .05" type="plane" material="grid"/>

    <!-- The Raw Mesh -->
    <!-- Position is default (0,0,0) as per 'don't need anyother modify' -->
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
