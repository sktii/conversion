import numpy as np
import trimesh
from sklearn.cluster import KMeans
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

def generate_raw_xml(stl_files, scale_factor=1.0):
    ensure_dir(XML_DIR)

    if not stl_files:
        print("❌ No STL files to generate XML for.")
        return

    print(f"🚀 Generating raw_mesh.xml for {len(stl_files)} files...")
    scale_str = f"{scale_factor} {scale_factor} {scale_factor}"

    geoms_xml = ""

    abs_xml_dir = os.path.abspath(XML_DIR)

    for stl_path in stl_files:
        is_valid, hull_path = ensure_binary_stl(stl_path)
        final_part_path = hull_path if hull_path else stl_path

        # Get relative path for XML (Project Root Relative)
        # We want "STL/filename.stl" not "../STL/filename.stl"
        # Since we are running from project root, we can just use the relative path from there.
        try:
            rel_path = os.path.relpath(final_part_path, os.getcwd())
            rel_path = rel_path.replace("\\", "/")
        except ValueError:
             rel_path = final_part_path.replace("\\", "/")

        final_part_name = os.path.basename(final_part_path)
        mesh_id = os.path.splitext(final_part_name)[0].replace(" ", "_").replace(".", "_")

        # Apply Default Rotation: 90 degrees around X-axis (1.5707963 rad)
        # Ensure path does not start with ../
        if rel_path.startswith("../"):
            rel_path = rel_path[3:]

        geoms_xml += f'    <mesh name="{mesh_id}" file="{rel_path}" scale="{scale_str}"/>\n'
        geoms_xml += f'    <geom name="geom_{mesh_id}" type="mesh" mesh="{mesh_id}" rgba="0.8 0.8 0.8 1" euler="1.5707963 0 0"/>\n'

    mesh_lines = [line for line in geoms_xml.splitlines() if '<mesh' in line]
    geom_lines = [line for line in geoms_xml.splitlines() if '<geom' in line]

    assets_block = "\n".join(mesh_lines)
    world_block = "\n".join(geom_lines)

    xml_content = f"""<mujoco model="raw_mesh_view">
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
        f.write(xml_content)
    print(f"✅ Generated Raw XML: {os.path.abspath(RAW_XML_FILE)}")

def find_disk_center(mesh, stl_path, scale_factor=1.0):
    stl_dir = os.path.dirname(stl_path)
    base_name = os.path.splitext(os.path.basename(stl_path))[0]
    parts_dir = os.path.join(stl_dir, f"{base_name}_parts")

    best_candidate_center = None
    max_z = -np.inf

    if os.path.exists(parts_dir):
        parts = glob.glob(os.path.join(parts_dir, "*.stl"))
        for part in parts:
            try:
                part_mesh = trimesh.load(part)
                if scale_factor != 1.0: part_mesh.apply_scale(scale_factor)
                z_max = part_mesh.bounds[1][2]
                center = part_mesh.centroid
                if z_max > max_z:
                    max_z = z_max
                    best_candidate_center = center
            except: pass

        if best_candidate_center is not None:
             return best_candidate_center

    if mesh:
        points, _ = trimesh.sample.sample_surface(mesh, 2000)
        z_coords = points[:, 2]
        top_percentile_z = np.percentile(z_coords, 90)
        top_points = points[z_coords > top_percentile_z]
        if len(top_points) == 0: return np.mean(points, axis=0)
        return np.mean(top_points, axis=0)
    return np.array([0,0,0])

def generate_fitted_xml(stl_path, scale_factor=1.0):
    # LEGACY / UNUSED in this version
    ensure_dir(XML_DIR)
    # print(f"🚀 Starting Pillar Generation")

    mesh = None
    points = None

    if os.path.exists(stl_path):
        try:
            ensure_binary_stl(stl_path)
            mesh = trimesh.load(stl_path)
            if scale_factor != 1.0:
                mesh.apply_scale(scale_factor)
            points, _ = trimesh.sample.sample_surface(mesh, SAMPLE_COUNT)
        except Exception:
            pass

    if points is None:
        return

    robot_pos = find_disk_center(mesh, stl_path, scale_factor)

    kmeans = KMeans(n_clusters=TOTAL_CLUSTERS, n_init=10, random_state=42)
    kmeans.fit(points)
    labels = kmeans.labels_

    # ... (Truncating this logic as it is unused, but keeping the function stub valid)
    # To be safe, I'll just return here since we don't use it.
    return

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

    generate_raw_xml(collected_stls, scale_factor)

    print("ℹ️  Skipped fitted_pillars.xml generation (focusing on raw_mesh.xml).")

if __name__ == "__main__":
    main()
