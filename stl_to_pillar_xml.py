import numpy as np
import trimesh
from sklearn.cluster import KMeans
import os
import sys

# Parameters
NUM_CYLINDERS = 16
NUM_BOXES = 16
TOTAL_CLUSTERS = NUM_CYLINDERS + NUM_BOXES
SAMPLE_COUNT = 5000  # Number of points to sample from mesh surface

def generate_pillars_xml(stl_path, output_xml="pillars.xml"):
    """
    Approximates the geometry in the STL file using 32 pillars (16 cylinders, 16 boxes)
    and saves them to a MuJoCo XML file.
    """
    print(f"🚀 Starting Pillar Generation")
    print(f"📂 Input STL: {stl_path}")

    if not os.path.exists(stl_path):
        print(f"❌ Error: STL file not found: {stl_path}")
        return

    try:
        # 1. Load Mesh
        mesh = trimesh.load(stl_path)
        print(f"ℹ️  Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces.")

        # 2. Sample Points
        print(f"⏳ Sampling {SAMPLE_COUNT} points from surface...")
        points, _ = trimesh.sample.sample_surface(mesh, SAMPLE_COUNT)

        # 3. K-Means Clustering
        print(f"⏳ Running K-Means clustering (k={TOTAL_CLUSTERS})...")
        kmeans = KMeans(n_clusters=TOTAL_CLUSTERS, n_init=10, random_state=42)
        kmeans.fit(points)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_

        # 4. Calculate Dimensions for each pillar
        # We try to fit the points in each cluster
        pillar_geoms = []

        print("⏳ Calculating pillar dimensions...")
        for i in range(TOTAL_CLUSTERS):
            cluster_points = points[labels == i]
            center = centers[i]

            if len(cluster_points) == 0:
                # Fallback for empty cluster (unlikely)
                radius = 0.02
                height = 0.2
            else:
                # Calculate bounding box of the cluster relative to center
                # We want the radius/half-size
                dists = np.linalg.norm(cluster_points - center, axis=1)
                radius = np.max(dists)

                # Z-height approximation: find min/max Z of cluster
                min_z = np.min(cluster_points[:, 2])
                max_z = np.max(cluster_points[:, 2])
                height = (max_z - min_z) / 2.0
                center[2] = (max_z + min_z) / 2.0 # Adjust center Z to midpoint

                # Minimum constraints to avoid invisible objects
                radius = max(radius, 0.01)
                height = max(height, 0.01)

            # Assign type
            if i < NUM_CYLINDERS:
                # Cylinder
                # size="radius half_height" (MuJoCo convention for cylinder is radius half_length usually?
                # Check documentation: type="cylinder", size="radius half_length")
                # Wait, cylinder size param: "radius half-length"

                geom_str = f'    <geom name="pillar_cyl_{i+1}" type="cylinder" size="{radius:.4f} {height:.4f}" pos="{center[0]:.4f} {center[1]:.4f} {center[2]:.4f}" rgba="0.5 0.5 0.5 1" contype="1" conaffinity="1"/>'
                pillar_geoms.append(geom_str)
            else:
                # Box
                # size="half_x half_y half_z"
                # We approximate as a cube of side radius
                # Note: 'radius' here is distance from center, so it's roughly half-diagonal.
                # Let's treat it as half-width.
                half_width = radius / 1.414 # rough approximation
                box_idx = i - NUM_CYLINDERS + 1
                geom_str = f'    <geom name="pillar_box_{box_idx}" type="box" size="{half_width:.4f} {half_width:.4f} {height:.4f}" pos="{center[0]:.4f} {center[1]:.4f} {center[2]:.4f}" rgba="0.6 0.6 0.6 1" contype="1" conaffinity="1"/>'
                pillar_geoms.append(geom_str)

        # 5. Generate XML
        xml_content = f"""<mujoco model="approximated_pillars">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="2 2 .05" type="plane" material="grid"/>

    <!-- Generated 32 Pillars -->
{chr(10).join(pillar_geoms)}

  </worldbody>
</mujoco>
"""

        with open(output_xml, "w") as f:
            f.write(xml_content)

        print(f"✅ XML Generated Successfully: {os.path.abspath(output_xml)}")
        print(f"📊 Created {NUM_CYLINDERS} cylinders and {NUM_BOXES} boxes.")

    except Exception as e:
        print(f"❌ Error during pillar generation: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        stl_file = sys.argv[1]
        out_file = "pillars.xml"
        if len(sys.argv) > 2:
            out_file = sys.argv[2]
        generate_pillars_xml(stl_file, out_file)
    else:
        print("Usage: python stl_to_pillar_xml.py <input_stl_file> [output_xml_file]")
