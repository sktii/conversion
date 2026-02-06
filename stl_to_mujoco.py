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

# Output Paths (relative to script dir)
XML_DIR = "XML"
RAW_XML_FILE = os.path.join(XML_DIR, "raw_mesh.xml")
FITTED_XML_FILE = os.path.join(XML_DIR, "fitted_pillars.xml")
PATCHED_UR5E_FILE = os.path.join(XML_DIR, "ur5e_fitted.xml")

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def generate_raw_xml(stl_path, scale_factor=1.0):
    """
    Generates a raw XML wrapper for the STL mesh.
    """
    ensure_dir(XML_DIR)

    stl_filename = os.path.basename(stl_path)
    scale_str = f"{scale_factor} {scale_factor} {scale_factor}"

    abs_stl_path = os.path.abspath(stl_path)
    stl_dir = os.path.dirname(abs_stl_path)

    xml_content = f"""<mujoco model="raw_mesh_view">
  <compiler angle="radian" meshdir="{stl_dir}"/>

  <option timestep="0.002" gravity="0 0 -9.81"/>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>

    <mesh name="target_mesh" file="{stl_filename}" scale="{scale_str}"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="2 2 .05" type="plane" material="grid"/>

    <!-- The Raw Mesh -->
    <geom name="imported_part" type="mesh" mesh="target_mesh" rgba="0.8 0.8 0.8 1"/>
  </worldbody>
</mujoco>
"""
    with open(RAW_XML_FILE, "w") as f:
        f.write(xml_content)
    print(f"✅ Generated Raw XML: {os.path.abspath(RAW_XML_FILE)}")


def find_disk_center(mesh):
    points, _ = trimesh.sample.sample_surface(mesh, 2000)
    z_coords = points[:, 2]
    top_percentile_z = np.percentile(z_coords, 90)
    top_points = points[z_coords > top_percentile_z]

    if len(top_points) == 0:
        return np.mean(points, axis=0)

    center = np.mean(top_points, axis=0)
    return center


def generate_fitted_xml(stl_path, scale_factor=1.0):
    ensure_dir(XML_DIR)

    print(f"🚀 Starting Pillar Generation")
    print(f"📂 Input STL: {stl_path}")

    if not os.path.exists(stl_path):
        print(f"❌ Error: STL file not found: {stl_path}")
        return

    try:
        # 1. Load Mesh
        mesh = trimesh.load(stl_path)

        # Apply Scale if needed
        if scale_factor != 1.0:
            mesh.apply_scale(scale_factor)
            print(f"   -> Applied scale factor {scale_factor} to internal mesh.")

        # 2. Find Robot Position (on scaled mesh)
        robot_pos = find_disk_center(mesh)
        print(f"🤖 Detected Robot Placement: {robot_pos}")

        # 3. Sample Points
        points, _ = trimesh.sample.sample_surface(mesh, SAMPLE_COUNT)

        # 4. K-Means Clustering
        kmeans = KMeans(n_clusters=TOTAL_CLUSTERS, n_init=10, random_state=42)
        kmeans.fit(points)
        labels = kmeans.labels_

        # 5. Smart Merging Logic
        unique, counts = np.unique(labels, return_counts=True)
        cluster_counts = dict(zip(unique, counts))

        clusters = []
        for i in range(TOTAL_CLUSTERS):
            if i not in cluster_counts: continue
            pts = points[labels == i]
            min_p = np.min(pts, axis=0)
            max_p = np.max(pts, axis=0)
            center = (min_p + max_p) / 2.0
            dims = max_p - min_p
            clusters.append({
                'id': i,
                'center': center,
                'min': min_p,
                'max': max_p,
                'dims': dims,
                'count': cluster_counts[i],
                'merged': False
            })

        # Iterative Merge
        final_pillars = []

        while clusters:
            clusters.sort(key=lambda x: x['count'], reverse=True)
            current = clusters.pop(0)

            c_min = current['min']
            c_max = current['max']

            i = 0
            while i < len(clusters):
                other = clusters[i]
                o_min = other['min']
                o_max = other['max']

                tol = 0.05
                overlap_x = (c_min[0] - tol <= o_max[0]) and (c_max[0] + tol >= o_min[0])
                overlap_y = (c_min[1] - tol <= o_max[1]) and (c_max[1] + tol >= o_min[1])
                overlap_z = (c_min[2] - tol <= o_max[2]) and (c_max[2] + tol >= o_min[2])

                if overlap_x and overlap_y and overlap_z:
                    current['min'] = np.minimum(current['min'], other['min'])
                    current['max'] = np.maximum(current['max'], other['max'])
                    current['count'] += other['count']
                    clusters.pop(i)
                else:
                    i += 1

            dims = current['max'] - current['min']
            center = (current['min'] + current['max']) / 2.0
            dims = np.maximum(dims, 0.01)

            final_pillars.append({
                'center': center,
                'dims': dims
            })

            if len(final_pillars) >= TOTAL_CLUSTERS:
                break

        print(f"✅ Merged into {len(final_pillars)} distinct pillars.")

        # 6. Patch UR5e XML
        try:
            tree = ET.parse("ur5e.xml")
            root = tree.getroot()

            found = False
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

        except Exception as e:
            print(f"⚠️  Failed to patch ur5e.xml: {e}.")
            if not os.path.exists(PATCHED_UR5E_FILE):
                 if os.path.exists("ur5e.xml"):
                     shutil.copy("ur5e.xml", PATCHED_UR5E_FILE)
                 else:
                     print("❌ Critical: ur5e.xml not found for fallback.")

        # 7. Generate XML Strings
        pillar_geoms = []

        def make_geom(name, type_str, size, pos, rgba):
            return f'    <geom name="{name}" type="{type_str}" size="{size}" pos="{pos}" rgba="{rgba}" contype="1" conaffinity="1"/>'

        unused_pos = "10 0 0"
        unused_size = "0.01 0.01 0.01"
        unused_size_cyl = "0.01 0.01"
        unused_rgba = "0.5 0.5 0.5 0"
        used_rgba = "0.5 0.5 0.5 1"

        for i in range(1, NUM_BOXES + 1):
            if final_pillars:
                p = final_pillars.pop(0)
                c = p['center']
                d = p['dims'] / 2.0
                pos_str = f"{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}"
                size_str = f"{d[0]:.4f} {d[1]:.4f} {d[2]:.4f}"
                pillar_geoms.append(make_geom(f"pillar_box_{i}", "box", size_str, pos_str, used_rgba))
            else:
                pillar_geoms.append(make_geom(f"pillar_box_{i}", "box", unused_size, unused_pos, unused_rgba))

        for i in range(1, NUM_CYLINDERS + 1):
            if final_pillars:
                p = final_pillars.pop(0)
                c = p['center']
                d = p['dims'] / 2.0
                radius = max(d[0], d[1])
                height = d[2]
                pos_str = f"{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}"
                size_str = f"{radius:.4f} {height:.4f}"
                pillar_geoms.append(make_geom(f"pillar_cyl_{i}", "cylinder", size_str, pos_str, used_rgba))
            else:
                pillar_geoms.append(make_geom(f"pillar_cyl_{i}", "cylinder", unused_size_cyl, unused_pos, unused_rgba))

        # 8. Construct Final XML
        ur5e_ref = os.path.basename(PATCHED_UR5E_FILE)

        xml_content = f"""<mujoco model="approximated_pillars">
  <compiler angle="radian" meshdir="../"/>

  <include file="{ur5e_ref}"/>

  <option timestep="0.002" gravity="0 0 -9.81"/>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="2 2 .05" type="plane" material="grid"/>

    <!-- Generated Pillars -->
{chr(10).join(pillar_geoms)}

  </worldbody>
</mujoco>
"""

        with open(FITTED_XML_FILE, "w") as f:
            f.write(xml_content)

        print(f"✅ XML Generated Successfully: {os.path.abspath(FITTED_XML_FILE)}")

    except Exception as e:
        print(f"❌ Error during pillar generation: {e}")
        import traceback
        traceback.print_exc()

def main():
    stl_path = None

    # Arg parsing logic
    if len(sys.argv) > 1:
        stl_path = sys.argv[1]
    else:
        # Auto-scan STL directory
        stl_dir = "STL"
        if os.path.exists(stl_dir):
            files = glob.glob(os.path.join(stl_dir, "*.stl"))
            if files:
                # Sort by modification time, newest first
                files.sort(key=os.path.getmtime, reverse=True)
                stl_path = files[0]
                print(f"ℹ️  No input file provided. Using newest STL found: {stl_path}")
            else:
                print(f"❌ No STL files found in {stl_dir}")
                return
        else:
            print(f"❌ STL directory not found: {stl_dir}")
            return

    if not stl_path:
        print("Usage: python stl_to_mujoco.py [stl_path]")
        return

    # Check Scale
    scale_factor = 1.0
    try:
        mesh = trimesh.load(stl_path)
        bounds = mesh.bounds
        max_dim = np.max(bounds[1] - bounds[0])
        if max_dim > SCALE_THRESHOLD:
            print(f"⚠️  Detected large dimensions ({max_dim:.2f}). Scaling by 0.001 (mm -> m).")
            scale_factor = 0.001
    except Exception as e:
        print(f"⚠️  Could not check scale: {e}")

    generate_raw_xml(stl_path, scale_factor)
    generate_fitted_xml(stl_path, scale_factor)

if __name__ == "__main__":
    main()
