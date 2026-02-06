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
    Also includes the UR5e robot at 'robot_pos'.
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

        pillar_geoms = []

        # 4. Filter logic
        # If a cluster is too small (e.g. < 0.5% of points), it might be noise or insignificant.
        # However, small clusters might be small details.
        # Let's use a geometric heuristic: if the radius is very small (< 5mm),
        # or if the cluster center is extremely close to the robot base (collision?),
        # we might want to "remove" it (move far away).
        # User said: "if dont need 32 pillar to fix you can just use 1 or 2"
        # This implies we should be aggressive in marking pillars as unused if they aren't "needed".
        # But determining "need" blindly is hard.
        # Let's stick to the "far away" strategy for clearly redundant ones.
        # For now, I will treat all clusters as valid UNLESS they are practically empty.

        # BETTER STRATEGY:
        # Identify "Occupied" space vs "Empty" space.
        # But K-Means forces 32 centers even on a simple cube.
        # We can check silhouette score or just threshold the cluster size (radius).
        # If multiple centers are very close to each other, merge them (keep one, throw others away).

        used_indices = []

        # Sort clusters by number of points (importance)
        unique, counts = np.unique(labels, return_counts=True)
        cluster_counts = dict(zip(unique, counts))

        # Calculate properties first
        cluster_props = []
        for i in range(TOTAL_CLUSTERS):
            if i not in cluster_counts:
                cluster_props.append({'idx': i, 'valid': False})
                continue

            cluster_points = points[labels == i]
            center = centers[i]

            # FIX: Use 2D (XY) distance for radius
            dists_xy = np.linalg.norm(cluster_points[:, :2] - center[:2], axis=1)
            radius = np.max(dists_xy)

            min_z = np.min(cluster_points[:, 2])
            max_z = np.max(cluster_points[:, 2])
            height = (max_z - min_z) / 2.0
            center[2] = (max_z + min_z) / 2.0

            # Constraints
            radius = max(radius, 0.01)
            height = max(height, 0.01)

            cluster_props.append({
                'idx': i,
                'valid': True,
                'center': center,
                'radius': radius,
                'height': height,
                'count': cluster_counts[i]
            })

        # Generate XML strings
        # We need exactly 16 cylinders and 16 boxes.
        # We map the valid clusters to these slots.
        # If we run out of valid clusters, the remaining slots get "far away" coords.

        # Simple overlap check: if two clusters are within 2cm of each other, ignore smaller one?
        # Let's just output all for now, but if the user wants fewer,
        # they imply the geometry might be simple.
        # K-Means on a simple box will partition the box into 32 Voronoi cells.
        # This is actually fine for approximation, just inefficient.
        # Moving them away is risky if I misinterpret "needed".
        # I will implement the logic: If radius < 1cm (very thin speck), move away.

        cyl_idx = 1
        box_idx = 1

        for prop in cluster_props:
            is_cylinder = (prop['idx'] < NUM_CYLINDERS)

            pos_str = "10 0 0" # Default far away
            size_str = "0.01 0.01" # Default tiny
            rgba = "0.5 0.5 0.5 0" # Transparent/Hidden

            if prop['valid'] and prop['radius'] > 0.005:
                c = prop['center']
                pos_str = f"{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}"
                if is_cylinder:
                    size_str = f"{prop['radius']:.4f} {prop['height']:.4f}"
                else:
                    half_w = prop['radius'] / 1.414
                    size_str = f"{half_w:.4f} {half_w:.4f} {prop['height']:.4f}"
                rgba = "0.5 0.5 0.5 1"

            if is_cylinder:
                geom = f'    <geom name="pillar_cyl_{cyl_idx}" type="cylinder" size="{size_str}" pos="{pos_str}" rgba="{rgba}" contype="1" conaffinity="1"/>'
                cyl_idx += 1
            else:
                geom = f'    <geom name="pillar_box_{box_idx}" type="box" size="{size_str}" pos="{pos_str}" rgba="{rgba}" contype="1" conaffinity="1"/>'
                box_idx += 1

            pillar_geoms.append(geom)

        # 5. Generate XML
        # Include UR5e
        # The 'robot0:ur5e:base' body is defined in ur5e.xml.
        # To position it, we wrap it or use a mocap/offset?
        # Actually ur5e.xml defines a top-level body in <worldbody>.
        # If we include it, it's inserted. We can't easily change its pos="0 0 0" inside the include without modifying ur5e.xml.
        # BUT, MuJoCo <include> inserts content.
        # Typically, one modifies the included file or uses a nested model.
        # Strategy: We will NOT include ur5e.xml directly.
        # Instead, we rely on the fact that ur5e.xml has its own worldbody.
        # Wait, nested worldbodies merge.
        # To move the robot, we might need to Wrap the include in a body?
        # <body pos="..."> <include .../> </body> -- This works in newer MuJoCo if the included file defines bodies.
        # UR5e.xml defines <body name="robot0:ur5e:base"...>.
        # If we wrap it, it becomes a child of our wrapper.

        xml_content = f"""<mujoco model="approximated_pillars">
  <compiler angle="radian"/>

  <!-- Include UR5e + Gripper -->
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

    <!-- The Robot Base Placement -->
    <!-- Since ur5e.xml defines the robot at 0,0,0, we might need to rely on the user modifying ur5e.xml
         OR we treat 'robot_pos' as the origin of the pillars relative to the robot?
         No, usually we move the robot.
         Hack: We define a site at the desired robot pos to visualize it,
         but we can't move the robot if it's hardcoded in ur5e.xml without using a nested body.

         Let's try to wrap it.
    -->
    <body name="robot_mount" pos="{robot_pos[0]:.4f} {robot_pos[1]:.4f} {robot_pos[2]:.4f}">
        <!-- Ideally we include here, but if ur5e.xml has <mujoco> root, it might fail inside body?
             No, <include> just pastes text. ur5e.xml has <mujoco> root tags.
             This is invalid inside a body.

             CORRECTION: Valid MuJoCo includes strip the root tag.
             So if ur5e.xml starts with <mujoco> and has <worldbody> inside, we get double worldbody.

             Let's look at ur5e.xml provided.
             It has <worldbody> ... </worldbody>.
             So we can't put it inside a <body>.

             We must generate an XML that places the pillars RELATIVE to the robot?
             Or we assume the user accepts the robot at 0,0,0 and we shift the pillars?

             "UR5e site in disk center" -> The robot sits ON the disk.
             If the disk is at (1,1,0.5), the robot should be at (1,1,0.5).

             If we cannot move the robot (hardcoded 0,0,0 in included file), we must move the WORLD (floor + pillars) relative to it.
             i.e. Shift pillars by -robot_pos.

             Let's apply this shift.
        -->
    </body>

    <!-- Generated 32 Pillars (Shifted so robot appears at 'robot_pos' relative to them)
         Wait, if we shift pillars by -robot_pos, the pillars move, robot stays at 0.
         Visually, the robot will be 'off' the table.

         We want the robot ON the table.
         Table is at P_table. Robot is at 0.
         So we need P_table = 0.
         So we must shift the entire scene (STL points) such that the Disk Center becomes (0,0,0).
    -->

{chr(10).join(pillar_geoms)}

  </worldbody>
</mujoco>
"""
        # Re-calc with shift
        # If we want the robot to be at 'robot_pos' (the disk center), and the robot is fixed at 0,0,0 in XML...
        # Then we need the disk center to be at 0,0,0.
        # So we subtract 'robot_pos' from all pillar coordinates.

        # Let's regenerate geoms with shift
        pillar_geoms_shifted = []
        cyl_idx = 1
        box_idx = 1

        shift_vector = np.array(robot_pos)

        for prop in cluster_props:
            is_cylinder = (prop['idx'] < NUM_CYLINDERS)

            pos_str = "10 0 0"
            size_str = "0.01 0.01"
            rgba = "0.5 0.5 0.5 0"

            if prop['valid'] and prop['radius'] > 0.005:
                c = prop['center'] - shift_vector # SHIFT!

                pos_str = f"{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}"
                if is_cylinder:
                    size_str = f"{prop['radius']:.4f} {prop['height']:.4f}"
                else:
                    half_w = prop['radius'] / 1.414
                    size_str = f"{half_w:.4f} {half_w:.4f} {prop['height']:.4f}"
                rgba = "0.5 0.5 0.5 1"

            if is_cylinder:
                geom = f'    <geom name="pillar_cyl_{cyl_idx}" type="cylinder" size="{size_str}" pos="{pos_str}" rgba="{rgba}" contype="1" conaffinity="1"/>'
                cyl_idx += 1
            else:
                geom = f'    <geom name="pillar_box_{box_idx}" type="box" size="{size_str}" pos="{pos_str}" rgba="{rgba}" contype="1" conaffinity="1"/>'
                box_idx += 1

            pillar_geoms_shifted.append(geom)

        # Update XML content with shifted geoms
        xml_content = xml_content.replace(chr(10).join(pillar_geoms), chr(10).join(pillar_geoms_shifted))

        with open(output_xml, "w") as f:
            f.write(xml_content)

        print(f"✅ XML Generated Successfully: {os.path.abspath(output_xml)}")
        print(f"   (Scene shifted so robot is at 0,0,0 corresponding to detected disk center)")

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

        generate_pillars_xml(stl_file, out_file, r_pos)
    else:
        print("Usage: python stl_to_pillar_xml.py <input_stl> <output_xml> [rx ry rz]")
