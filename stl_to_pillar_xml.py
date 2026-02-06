import numpy as np
import trimesh
from sklearn.cluster import KMeans
import os
import sys
import xml.etree.ElementTree as ET

# Parameters
NUM_CYLINDERS = 16
NUM_BOXES = 16
TOTAL_CLUSTERS = NUM_CYLINDERS + NUM_BOXES
SAMPLE_COUNT = 5000

# Scale Threshold
# If the maximum dimension of the object exceeds this value (in meters),
# we assume it's in millimeters and scale by 0.001.
# A table is rarely > 10 meters.
SCALE_THRESHOLD = 10.0

def generate_pillars_xml(stl_path, output_xml="fitted_pillars.xml", robot_pos=None):
    """
    Approximates the geometry in the STL file using 32 pillars.
    If the geometry is simple, unused pillars are moved to (10, 0, 0).
    Also includes the UR5e robot at 'robot_pos'.
    """
    print(f"🚀 Starting Pillar Generation")
    print(f"📂 Input STL: {stl_path}")

    if robot_pos is None:
        robot_pos = [0.0, 0.0, 0.0]

    # Store original robot_pos to check later, but we will mutate it if we detect scale
    robot_pos = list(robot_pos)

    if not os.path.exists(stl_path):
        print(f"❌ Error: STL file not found: {stl_path}")
        return

    try:
        # 1. Load Mesh
        mesh = trimesh.load(stl_path)

        # --- SCALE DETECTION ---
        bounds = mesh.bounds
        extents = bounds[1] - bounds[0]
        max_dim = np.max(extents)

        scale_factor = 1.0
        if max_dim > SCALE_THRESHOLD:
            print(f"⚠️  Detected large dimensions (max={max_dim:.2f}). Assuming Millimeters.")
            print(f"   -> Scaling mesh by 0.001 to convert to Meters.")
            scale_factor = 0.001
            mesh.apply_scale(scale_factor)

            # Also scale the robot_pos if it looks large?
            # Usually find_disk_center returns coords in the same unit as the mesh.
            # So if mesh was mm, robot_pos is likely mm.
            if max(abs(x) for x in robot_pos) > SCALE_THRESHOLD:
                print(f"   -> Scaling robot position by 0.001.")
                robot_pos = [x * scale_factor for x in robot_pos]
            else:
                 print(f"   -> Robot position seems small ({robot_pos}), keeping as is (check this!).")

        print(f"🤖 Robot Placement (Final): {robot_pos}")

        # 2. Sample Points
        points, _ = trimesh.sample.sample_surface(mesh, SAMPLE_COUNT)

        # 3. K-Means Clustering
        kmeans = KMeans(n_clusters=TOTAL_CLUSTERS, n_init=10, random_state=42)
        kmeans.fit(points)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_

        # 4. Filter logic & Properties Calculation
        unique, counts = np.unique(labels, return_counts=True)
        cluster_counts = dict(zip(unique, counts))

        cluster_props = []
        for i in range(TOTAL_CLUSTERS):
            if i not in cluster_counts:
                # Empty cluster
                cluster_props.append({'idx': i, 'valid': False, 'reason': 'empty'})
                continue

            cluster_points = points[labels == i]
            count = cluster_counts[i]

            # --- AABB Calculation ---
            min_x, min_y, min_z = np.min(cluster_points, axis=0)
            max_x, max_y, max_z = np.max(cluster_points, axis=0)

            center = [(min_x + max_x) / 2.0, (min_y + max_y) / 2.0, (min_z + max_z) / 2.0]

            dx = max_x - min_x
            dy = max_y - min_y
            height = (max_z - min_z) / 2.0

            # Constraints
            dx = max(dx, 0.01)
            dy = max(dy, 0.01)
            height = max(height, 0.01)

            # Calculate metrics for shape selection
            elongation = max(dx, dy) / min(dx, dy)
            radius_tight = (dx + dy) / 4.0
            radius_cover = np.sqrt(dx*dx + dy*dy) / 2.0

            cluster_props.append({
                'idx': i,
                'valid': True,
                'center': np.array(center),
                'dx': dx,
                'dy': dy,
                'height': height,
                'radius': radius_tight, # Use tight radius by default
                'radius_cover': radius_cover,
                'elongation': elongation,
                'count': count,
                'reason': 'ok'
            })

        # --- Refined Pruning Logic ---
        # 1. Sort by point count (importance)
        cluster_props.sort(key=lambda x: x.get('count', 0), reverse=True)

        # 2. Check for Overlaps and redundancy
        valid_props = [p for p in cluster_props if p['valid']]

        for i in range(len(valid_props)):
            p1 = valid_props[i]
            if not p1['valid']: continue

            # Prune small speckles
            if p1['count'] < (SAMPLE_COUNT / 200) and p1['radius'] < 0.02:
                p1['valid'] = False
                p1['reason'] = 'too_small'
                continue

            for j in range(i + 1, len(valid_props)):
                p2 = valid_props[j]
                if not p2['valid']: continue

                # Distance between centers
                dist = np.linalg.norm(p1['center'] - p2['center'])

                # Overlap threshold
                if dist < 0.05:
                    p2['valid'] = False
                    p2['reason'] = f'overlap_with_{p1["idx"]}'

        valid_list = [p for p in cluster_props if p['valid']]

        # If we have NO valid clusters, keep the largest
        if not valid_list and cluster_props:
             cluster_props[0]['valid'] = True
             valid_list = [cluster_props[0]]

        print(f"✅ Found {len(valid_list)} valid pillars out of {TOTAL_CLUSTERS}.")

        # 5. Patch UR5e XML
        # Read ur5e.xml and update robot pos using XML parser
        try:
            tree = ET.parse("ur5e.xml")
            root = tree.getroot()

            found = False
            # Search recursively for body with name
            for body in root.iter('body'):
                if body.get('name') == "robot0:ur5e:base":
                    pos_str_new = f'{robot_pos[0]:.4f} {robot_pos[1]:.4f} {robot_pos[2]:.4f}'
                    body.set('pos', pos_str_new)
                    found = True
                    break

            if not found:
                print("⚠️  Warning: Could not find body 'robot0:ur5e:base' in ur5e.xml")

            patched_ur5e_filename = "ur5e_fitted.xml"
            tree.write(patched_ur5e_filename)
            print(f"✅ Created patched UR5e XML: {patched_ur5e_filename}")

        except Exception as e:
            print(f"⚠️  Failed to patch ur5e.xml: {e}. Using original.")
            patched_ur5e_filename = "ur5e.xml"

        # 6. Generate XML Strings
        pillar_geoms = []

        cyl_slots = list(range(1, NUM_CYLINDERS + 1))
        box_slots = list(range(1, NUM_BOXES + 1))

        # Categorize valid clusters
        box_candidates = []
        cyl_candidates = []

        for p in valid_list:
            if p['elongation'] > 1.5: # Clearly rectangular
                box_candidates.append(p)
            else:
                cyl_candidates.append(p)

        # Assignment Logic

        # Fill Box Slots
        for _ in range(len(box_slots)):
            if not box_candidates and not cyl_candidates: break

            p = None
            if box_candidates:
                p = box_candidates.pop(0)
            elif cyl_candidates:
                p = cyl_candidates.pop(0)

            if p:
                idx = box_slots.pop(0)
                c = p['center']
                pos_str = f"{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}"
                size_str = f"{p['dx']/2.0:.4f} {p['dy']/2.0:.4f} {p['height']:.4f}"
                rgba = "0.5 0.5 0.5 1"
                geom = f'    <geom name="pillar_box_{idx}" type="box" size="{size_str}" pos="{pos_str}" rgba="{rgba}" contype="1" conaffinity="1"/>'
                pillar_geoms.append(geom)

        # Fill Cylinder Slots
        for _ in range(len(cyl_slots)):
            if not cyl_candidates and not box_candidates: break

            p = None
            if cyl_candidates:
                p = cyl_candidates.pop(0)
            elif box_candidates:
                p = box_candidates.pop(0)

            if p:
                idx = cyl_slots.pop(0)
                c = p['center']
                pos_str = f"{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}"
                size_str = f"{p['radius']:.4f} {p['height']:.4f}"
                rgba = "0.5 0.5 0.5 1"
                geom = f'    <geom name="pillar_cyl_{idx}" type="cylinder" size="{size_str}" pos="{pos_str}" rgba="{rgba}" contype="1" conaffinity="1"/>'
                pillar_geoms.append(geom)

        # Fill remaining unused slots
        for idx in box_slots:
            geom = f'    <geom name="pillar_box_{idx}" type="box" size="0.01 0.01 0.01" pos="10 0 0" rgba="0.5 0.5 0.5 0" contype="1" conaffinity="1"/>'
            pillar_geoms.append(geom)

        for idx in cyl_slots:
            geom = f'    <geom name="pillar_cyl_{idx}" type="cylinder" size="0.01 0.01" pos="10 0 0" rgba="0.5 0.5 0.5 0" contype="1" conaffinity="1"/>'
            pillar_geoms.append(geom)

        # 7. Construct Final XML

        xml_content = f"""<mujoco model="approximated_pillars">
  <compiler angle="radian"/>

  <include file="{patched_ur5e_filename}"/>

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
    <!-- Positioned at absolute coordinates from STL (Robot placed at detected disk center via ur5e_fitted.xml) -->
{chr(10).join(pillar_geoms)}

  </worldbody>
</mujoco>
"""

        with open(output_xml, "w") as f:
            f.write(xml_content)

        print(f"✅ XML Generated Successfully: {os.path.abspath(output_xml)}")

    except Exception as e:
        print(f"❌ Error during pillar generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        stl_file = sys.argv[1]
        out_file = "fitted_pillars.xml"
        r_pos = [0.0, 0.0, 0.0]

        if len(sys.argv) > 2:
            out_file = sys.argv[2]

        if len(sys.argv) > 5:
            try:
                r_pos = [float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])]
            except ValueError:
                print("⚠️  Invalid robot position arguments. Using default [0,0,0].")

        generate_pillars_xml(stl_file, out_file, r_pos)
    else:
        print("Usage: python stl_to_pillar_xml.py <input_stl> <output_xml> [rx ry rz]")
