import numpy as np
import trimesh
from sklearn.cluster import KMeans
from scipy.spatial.transform import Rotation as R
import os
import sys
import shutil
import glob
import xml.etree.ElementTree as ET

# Parameters
NUM_CYLINDERS = 16
NUM_BOXES = 16
TOTAL_CLUSTERS = NUM_CYLINDERS + NUM_BOXES
SAMPLE_COUNT = 5000
SCALE_THRESHOLD = 10.0 # If > 10m, assume millimeters
MUJOCO_FACE_LIMIT = 20000

# Output Paths (relative to script dir)
XML_DIR = "XML"
RAW_XML_FILE = os.path.join(XML_DIR, "raw_mesh.xml")
RAW_XML2_FILE = os.path.join(XML_DIR, "raw_mesh2.xml")
FITTED_XML_FILE = os.path.join(XML_DIR, "fitted_pillars.xml")
PATCHED_UR5E_FILE = os.path.join(XML_DIR, "ur5e_fitted.xml")

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def ensure_binary_stl(filepath):
    """
    Checks if the STL is binary and converts if necessary.
    Also attempts mesh simplification if faces > limit.
    """
    if not os.path.exists(filepath):
        return False, None

    try:
        needs_save = False
        reason = ""

        # 1. Check Header (ASCII vs Binary)
        with open(filepath, 'rb') as f:
            header = f.read(5)
        if header.startswith(b'solid'):
            needs_save = True
            reason = "ASCII format detected"

        # 2. Check Face Count (Load mesh)
        mesh = trimesh.load(filepath)
        face_count = len(mesh.faces)

        if face_count > MUJOCO_FACE_LIMIT:
            print(f"⚠️  High face count ({face_count}) in {os.path.basename(filepath)}. Attempting simplification...")
            simplification_success = False

            # Try simplification methods
            try:
                if hasattr(mesh, 'simplify_quadric_decimation'):
                    try:
                        mesh = mesh.simplify_quadric_decimation(MUJOCO_FACE_LIMIT)
                        simplification_success = True
                        needs_save = True
                        reason += f" & Simplified (Count Method) -> {len(mesh.faces)} faces"
                    except:
                        ratio = 1.0 - (MUJOCO_FACE_LIMIT / face_count)
                        if ratio < 0: ratio = 0.0
                        if ratio > 1: ratio = 0.99
                        try:
                            mesh = mesh.simplify_quadric_decimation(ratio)
                            simplification_success = True
                            needs_save = True
                            reason += f" & Simplified (Ratio Method {ratio:.2f}) -> {len(mesh.faces)} faces"
                        except:
                            pass
            except:
                pass

            # Fallback: Convex Hull if still too big
            if len(mesh.faces) > MUJOCO_FACE_LIMIT:
                print(f"⚠️  Mesh still too large ({len(mesh.faces)}). Generating Hull for safety.")
                hull = mesh.convex_hull
                base, ext = os.path.splitext(filepath)
                hull_path = f"{base}_hull{ext}"
                hull.export(hull_path)
                print(f"✅ Generated Hull: {hull_path}")
                return False, hull_path # Return hull path for raw_mesh.xml

        if needs_save:
            print(f"ℹ️  Optimizing STL ({reason})...")
            mesh.export(filepath, file_type='stl')
            print(f"✅ Saved optimized Binary STL: {filepath}")

        return True, None

    except Exception as e:
        print(f"⚠️  Error verifying/converting STL format: {e}")
        return False, None

def collect_stls(stl_dir):
    """
    Collects all STL files to be processed.
    Prioritizes parts in *_parts directories over merged files in root.
    """
    if not os.path.exists(stl_dir):
        print(f"❌ STL directory not found: {stl_dir}")
        return []

    collected_files = []

    # 1. Find all parts directories
    parts_dirs = glob.glob(os.path.join(stl_dir, "*_parts"))
    processed_bases = set()

    for p_dir in parts_dirs:
        if os.path.isdir(p_dir):
            base_name = os.path.basename(p_dir).replace("_parts", "")
            processed_bases.add(base_name)

            # Add all STLs in this parts directory
            parts = glob.glob(os.path.join(p_dir, "*.stl"))
            if parts:
                print(f"ℹ️  Found parts in {os.path.basename(p_dir)} ({len(parts)} files).")
                collected_files.extend(parts)

    # 2. Find root STLs (merged files or others)
    root_stls = glob.glob(os.path.join(stl_dir, "*.stl"))
    for stl_path in root_stls:
        base_name = os.path.splitext(os.path.basename(stl_path))[0]

        # If we already processed parts for this base name, skip the merged file
        if base_name in processed_bases:
            print(f"ℹ️  Skipping merged file {os.path.basename(stl_path)} (parts found).")
            continue

        print(f"ℹ️  Found standalone/merged file: {os.path.basename(stl_path)}")
        collected_files.append(stl_path)

    # 3. Recursively find other STLs if needed (Fallback)
    known_files = set(os.path.abspath(f) for f in collected_files)

    for root, dirs, files in os.walk(stl_dir):
        for file in files:
            if file.lower().endswith(".stl"):
                abs_path = os.path.abspath(os.path.join(root, file))
                if abs_path not in known_files:
                    print(f"ℹ️  Found additional STL: {os.path.relpath(abs_path, stl_dir)}")
                    collected_files.append(abs_path)
                    known_files.add(abs_path)

    return collected_files

def generate_xmls(stl_files, scale_factor=1.0):
    ensure_dir(XML_DIR)

    if not stl_files:
        print("❌ No STL files to generate XML for.")
        return

    print(f"🚀 Generating XMLs for {len(stl_files)} files...")
    scale_str = f"{scale_factor} {scale_factor} {scale_factor}"

    # First Pass: Calculate Global Min Z after rotation AND Locate plane.stl
    print("ℹ️  Calculating global bounds to auto-position on floor...")
    global_min_z = np.inf
    plane_bounds = None # (min_x, min_y, min_z, max_x, max_y, max_z)

    # Rotation matrix for 90 degrees around X
    r = R.from_euler('x', 90, degrees=True)
    rot_matrix = r.as_matrix() # 3x3

    for stl_path in stl_files:
        try:
            mesh = trimesh.load(stl_path)
            # Apply Scale
            if scale_factor != 1.0:
                mesh.apply_scale(scale_factor)

            # Apply Rotation (Transform vertices)
            T = np.eye(4)
            T[:3, :3] = rot_matrix
            mesh.apply_transform(T)

            min_z = mesh.bounds[0][2]
            if min_z < global_min_z:
                global_min_z = min_z

            # Check for plane.stl
            if "plane.stl" in os.path.basename(stl_path).lower():
                print(f"   -> Found target plane mesh: {os.path.basename(stl_path)}")
                plane_bounds = (
                    mesh.bounds[0][0], mesh.bounds[0][1], mesh.bounds[0][2],
                    mesh.bounds[1][0], mesh.bounds[1][1], mesh.bounds[1][2]
                )

        except Exception as e:
            print(f"⚠️  Skipping bounds check for {os.path.basename(stl_path)}: {e}")

    z_offset = 0.0
    if global_min_z != np.inf:
        z_offset = -global_min_z
        print(f"   -> Global Min Z detected: {global_min_z:.4f}m. Applying Offset: +{z_offset:.4f}m")
    else:
        print("   -> Could not calculate bounds. Defaulting to 0 offset.")

    # Calculate Robot Position (Top Surface Centroid)
    robot_pos = np.array([0.0, 0.0, 0.0])

    # Re-process plane.stl to find the true top center (disk center)
    plane_stl_path = None
    for p in stl_files:
        if "plane.stl" in os.path.basename(p).lower():
            plane_stl_path = p
            break

    if plane_stl_path:
        try:
            # Load and transform plane mesh again
            pm = trimesh.load(plane_stl_path)
            if scale_factor != 1.0: pm.apply_scale(scale_factor)
            T = np.eye(4); T[:3,:3] = rot_matrix
            pm.apply_transform(T)

            # Sample points on surface to find high spots
            points, _ = trimesh.sample.sample_surface(pm, 10000)
            z_coords = points[:, 2]
            max_z = np.max(z_coords)

            # Filter for points within 2mm (0.002m) of the top
            # This isolates the "disk" if it is raised
            top_mask = z_coords > (max_z - 0.002)
            top_points = points[top_mask]

            if len(top_points) > 0:
                center_x = np.mean(top_points[:, 0])
                center_y = np.mean(top_points[:, 1])
                # The robot should sit ON TOP of this surface
                final_z = max_z + z_offset

                robot_pos = np.array([center_x, center_y, final_z])
                print(f"🤖 Calculated Robot Position on Disk (Surface Sampling): {robot_pos}")
            else:
                 # Fallback to bounding box center if sampling fails
                 print("⚠️  Warning: Could not sample top surface. Falling back to bounding box.")
                 center_x = (pm.bounds[0][0] + pm.bounds[1][0]) / 2.0
                 center_y = (pm.bounds[0][1] + pm.bounds[1][1]) / 2.0
                 final_z = pm.bounds[1][2] + z_offset
                 robot_pos = np.array([center_x, center_y, final_z])

        except Exception as e:
            print(f"⚠️  Error calculating disk center: {e}")
    else:
        print("⚠️  Warning: 'plane.stl' not found. Robot position defaults to origin.")

    # Patch UR5e XML
    try:
        ur5e_source = "ur5e.xml"
        if os.path.exists(ur5e_source):
            tree = ET.parse(ur5e_source)
            root = tree.getroot()
            found = False
            # Search for robot base body
            # Assuming structure: worldbody -> body name="robot0:ur5e:base"
            # ET.iter works recursively
            for body in root.iter('body'):
                if body.get('name') == "robot0:ur5e:base":
                    pos_str_new = f'{robot_pos[0]:.4f} {robot_pos[1]:.4f} {robot_pos[2]:.4f}'
                    body.set('pos', pos_str_new)
                    found = True
                    break

            if not found:
                print("⚠️  Warning: Could not find body 'robot0:ur5e:base' in ur5e.xml")

            tree.write(PATCHED_UR5E_FILE)
            print(f"✅ Created patched UR5e XML: {os.path.abspath(PATCHED_UR5E_FILE)}")
        else:
            print(f"❌ ur5e.xml not found at {os.path.abspath(ur5e_source)}")
    except Exception as e:
        print(f"⚠️  Failed to patch ur5e.xml: {e}")

    # Generate XML Content
    geoms_xml = ""
    abs_xml_dir = os.path.abspath(XML_DIR)

    # Track used names to prevent duplicates
    used_mesh_names = set()

    for stl_path in stl_files:
        is_valid, hull_path = ensure_binary_stl(stl_path)
        final_part_path = hull_path if hull_path else stl_path

        # Get relative path for XML (Project Root Relative)
        try:
            rel_path = os.path.relpath(final_part_path, os.getcwd())
            rel_path = rel_path.replace("\\", "/")
        except ValueError:
             rel_path = final_part_path.replace("\\", "/")

        final_part_name = os.path.basename(final_part_path)
        base_mesh_id = os.path.splitext(final_part_name)[0].replace(" ", "_").replace(".", "_")

        # Uniquify Name
        mesh_id = base_mesh_id
        counter = 1
        while mesh_id in used_mesh_names:
            mesh_id = f"{base_mesh_id}_{counter}"
            counter += 1
        used_mesh_names.add(mesh_id)

        # Apply Default Rotation: 90 degrees around X-axis (1.5707963 rad)
        # Ensure path does not start with ../
        if rel_path.startswith("../"):
            rel_path = rel_path[3:]

        # Position Correction
        pos_str = f"0 0 {z_offset:.4f}"

        geoms_xml += f'    <mesh name="{mesh_id}" file="{rel_path}" scale="{scale_str}"/>\n'
        geoms_xml += f'    <geom name="geom_{mesh_id}" type="mesh" mesh="{mesh_id}" rgba="0.8 0.8 0.8 1" euler="1.5707963 0 0" pos="{pos_str}"/>\n'

    mesh_lines = [line for line in geoms_xml.splitlines() if '<mesh' in line]
    geom_lines = [line for line in geoms_xml.splitlines() if '<geom' in line]

    assets_block = "\n".join(mesh_lines)
    world_block = "\n".join(geom_lines)

    # 1. raw_mesh.xml (Viewer only)
    xml_content_raw = f"""<mujoco model="raw_mesh_view">
  <compiler angle="radian"/>

  <option timestep="0.002" gravity="0 0 -9.81"/>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>

{assets_block}
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="2 2 .05" type="plane" material="grid"/>

{world_block}
  </worldbody>
</mujoco>
"""
    with open(RAW_XML_FILE, "w") as f:
        f.write(xml_content_raw)
    print(f"✅ Generated Raw XML: {os.path.abspath(RAW_XML_FILE)}")

    # 2. raw_mesh2.xml (Viewer + Robot)
    ur5e_ref = os.path.basename(PATCHED_UR5E_FILE)

    xml_content_2 = f"""<mujoco model="raw_mesh_view_with_robot">
  <compiler angle="radian"/>

  <include file="{ur5e_ref}"/>

  <option timestep="0.002" gravity="0 0 -9.81"/>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>

{assets_block}
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="2 2 .05" type="plane" material="grid"/>

{world_block}
  </worldbody>
</mujoco>
"""
    with open(RAW_XML2_FILE, "w") as f:
        f.write(xml_content_2)
    print(f"✅ Generated Raw XML 2 (with Robot): {os.path.abspath(RAW_XML2_FILE)}")


def main():
    stl_dir = "STL"
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if os.path.isdir(arg):
            stl_dir = arg
        elif os.path.isfile(arg):
             # If user passes a file, use its directory
             stl_dir = os.path.dirname(arg)

    print(f"📂 Scanning for STL files in: {stl_dir} ...")

    collected_stls = collect_stls(stl_dir)

    if not collected_stls:
        print("❌ No STL files found to process.")
        return

    print(f"✅ Found {len(collected_stls)} files to process.")

    # Check Scale (using the first file as reference)
    scale_factor = 1.0
    try:
        if collected_stls:
            # Check the first few files
            for test_file in collected_stls[:3]:
                 m = trimesh.load(test_file)
                 bounds = m.bounds
                 max_dim = np.max(bounds[1] - bounds[0])
                 if max_dim > SCALE_THRESHOLD:
                     print(f"⚠️  Detected large dimensions ({max_dim:.2f}) in {os.path.basename(test_file)}. Scaling ALL by 0.001 (mm -> m).")
                     scale_factor = 0.001
                     break
    except Exception as e:
        print(f"⚠️  Could not check scale: {e}")

    generate_xmls(collected_stls, scale_factor)

    print("ℹ️  Skipped fitted_pillars.xml generation (focusing on raw_mesh.xml/raw_mesh2.xml).")

if __name__ == "__main__":
    main()
