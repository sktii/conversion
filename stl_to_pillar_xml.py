import numpy as np
import trimesh
from sklearn.cluster import KMeans
import os
import sys

# Parameters
NUM_CYLINDERS = 16
NUM_BOXES = 16
TOTAL_CLUSTERS = NUM_CYLINDERS + NUM_BOXES
SAMPLE_COUNT = 5000

def generate_pillars_xml(stl_path, output_xml="fitted_pillars.xml", robot_pos=None):
    """
    Approximates the geometry in the STL file using 32 pillars.
    If the geometry is simple, unused pillars are moved to (10, 0, 0).
    Also includes the UR5e robot.
    """
    print(f"🚀 Starting Pillar Generation")
    print(f"📂 Input STL: {stl_path}")

    if robot_pos is None:
        robot_pos = [0, 0, 0] # Default
    print(f"🤖 Robot Placement: {robot_pos}")

    if not os.path.exists(stl_path):
        print(f"❌ Error: STL file not found: {stl_path}")
        return

    try:
        # 1. Load Mesh
        mesh = trimesh.load(stl_path)

        # 2. Sample Points
        points, _ = trimesh.sample.sample_surface(mesh, SAMPLE_COUNT)

        # 3. K-Means Clustering
        # We always ask for 32 clusters to satisfy the "32 pillar" constraint of the neural net input
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
            center = centers[i]
            count = cluster_counts[i]

            # FIX: Use 2D (XY) distance for radius
            dists_xy = np.linalg.norm(cluster_points[:, :2] - center[:2], axis=1)
            radius = np.max(dists_xy)

            min_z = np.min(cluster_points[:, 2])
            max_z = np.max(cluster_points[:, 2])
            height = (max_z - min_z) / 2.0
            center[2] = (max_z + min_z) / 2.0 # Update center Z to mid-height

            # Constraints
            radius = max(radius, 0.01)
            height = max(height, 0.01)

            cluster_props.append({
                'idx': i,
                'valid': True,
                'center': center,
                'radius': radius,
                'height': height,
                'count': count,
                'reason': 'ok'
            })

        # --- Refined Pruning Logic ---
        # 1. Sort by point count (importance)
        # We want to keep the most substantial clusters.
        cluster_props.sort(key=lambda x: x.get('count', 0), reverse=True)

        # 2. Check for Overlaps and redundancy
        valid_props = [p for p in cluster_props if p['valid']]

        for i in range(len(valid_props)):
            p1 = valid_props[i]
            if not p1['valid']: continue

            # Prune small speckles (unless we have very few clusters)
            # If radius is tiny relative to height or absolute tiny
            if p1['radius'] < 0.01 and p1['count'] < (SAMPLE_COUNT / 200):
                # Very small and few points
                p1['valid'] = False
                p1['reason'] = 'too_small'
                continue

            for j in range(i + 1, len(valid_props)):
                p2 = valid_props[j]
                if not p2['valid']: continue

                # Distance between centers
                dist = np.linalg.norm(p1['center'] - p2['center'])

                # Overlap threshold: if distance is less than the larger radius?
                # Or if they are very close.
                # Heuristic: If they are closer than 5cm, AND one is significantly smaller or they are similar
                # Let's say if dist < 0.05 (5cm)
                if dist < 0.05:
                    # Merge/Kill p2 (since p1 is larger due to sort)
                    # We strictly kill p2 to "use 1 or 2" pillars if possible
                    p2['valid'] = False
                    p2['reason'] = f'overlap_with_{p1["idx"]}'

        # Re-sort by index to restore order if needed? Not strictly necessary for XML, but
        # we assign cylinders 1-16 and boxes 1-16.
        # We need to map the VALID ones to slots, and INVALID to far away.
        # But wait, 'idx' in cluster_props was the original K-Means label.
        # We have 32 slots total (16 cyl, 16 box).
        # We should fill the slots with VALID props first.

        valid_list = [p for p in cluster_props if p['valid']]

        # If we have NO valid clusters (shouldn't happen), keep at least one
        if not valid_list and cluster_props:
             cluster_props[0]['valid'] = True
             valid_list = [cluster_props[0]]

        print(f"✅ Found {len(valid_list)} valid pillars out of {TOTAL_CLUSTERS}.")

        # 5. Generate XML Strings
        pillar_geoms = []

        shift_vector = np.array(robot_pos)

        # We have 16 cylinders and 16 boxes to fill.
        # We distribute the valid clusters among them.
        # Since K-means doesn't distinguish shape, we just assign the first N to cylinders, next M to boxes.

        valid_iter = iter(valid_list)

        # Helper to get next valid prop or None
        def get_next_valid():
            try:
                return next(valid_iter)
            except StopIteration:
                return None

        # Generate Cylinders (1-16)
        for i in range(1, NUM_CYLINDERS + 1):
            prop = get_next_valid()
            if prop:
                # Valid Pillar
                c = prop['center'] - shift_vector
                pos_str = f"{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}"
                size_str = f"{prop['radius']:.4f} {prop['height']:.4f}"
                rgba = "0.5 0.5 0.5 1"
            else:
                # Unused / Far Away
                pos_str = "10 0 0"
                size_str = "0.01 0.01"
                rgba = "0.5 0.5 0.5 0" # Invisible

            geom = f'    <geom name="pillar_cyl_{i}" type="cylinder" size="{size_str}" pos="{pos_str}" rgba="{rgba}" contype="1" conaffinity="1"/>'
            pillar_geoms.append(geom)

        # Generate Boxes (1-16)
        for i in range(1, NUM_BOXES + 1):
            prop = get_next_valid()
            if prop:
                # Valid Pillar
                c = prop['center'] - shift_vector
                pos_str = f"{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}"
                # Box size is half-width
                half_w = prop['radius'] / 1.414
                size_str = f"{half_w:.4f} {half_w:.4f} {prop['height']:.4f}"
                rgba = "0.5 0.5 0.5 1"
            else:
                # Unused / Far Away
                pos_str = "10 0 0"
                size_str = "0.01 0.01 0.01"
                rgba = "0.5 0.5 0.5 0"

            geom = f'    <geom name="pillar_box_{i}" type="box" size="{size_str}" pos="{pos_str}" rgba="{rgba}" contype="1" conaffinity="1"/>'
            pillar_geoms.append(geom)

        # 6. Construct Final XML
        # Note: ur5e.xml includes its own worldbody, so we include it at top level.
        # We assume ur5e.xml defines the robot at 0,0,0.
        # Our pillars are shifted so the desired robot location (on the table) matches 0,0,0.

        xml_content = f"""<mujoco model="approximated_pillars">
  <compiler angle="radian"/>

  <include file="ur5e.xml"/>

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
    <!-- Shifted by {-shift_vector} relative to original mesh to align robot site to 0,0,0 -->
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
    # Args: input_stl output_xml [rx ry rz]
    if len(sys.argv) > 1:
        stl_file = sys.argv[1]
        out_file = "fitted_pillars.xml"
        r_pos = [0.0, 0.0, 0.0]

        if len(sys.argv) > 2:
            out_file = sys.argv[2]

        if len(sys.argv) > 5:
            r_pos = [float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])]

        # Handle the case where args might be passed as "x y z" or "x" "y" "z"
        # main_workflow passes "x y z" as 3 args usually if shell splitting works,
        # but the previous code parsing was:
        # if len > 5: r_pos = [argv[3], argv[4], argv[5]]
        # Let's ensure we support that.

        generate_pillars_xml(stl_file, out_file, r_pos)
    else:
        print("Usage: python stl_to_pillar_xml.py <input_stl> <output_xml> [rx ry rz]")
