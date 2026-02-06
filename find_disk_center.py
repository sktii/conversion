import numpy as np
import trimesh
import sys
import os

def find_disk_center(stl_path):
    """
    Analyzes the STL file to find a "disk" structure and returns its center.
    A disk is characterized as a flat, circular object.

    Heuristic:
    1. Sample points from the mesh.
    2. Cluster points by Normal vector. Look for a large cluster with Normal approx (0,0,1).
    3. Project these points to 2D (XY).
    4. Fit a circle or find the centroid.

    Alternative simpler heuristic for "Table with Disk":
    - The table is likely the largest flat surface.
    - The disk might be a raised platform or a specific part.
    - If the user says "UR5e site in disk center", and the robot is usually on top of something...
    - Let's look for circular features.
    """
    if not os.path.exists(stl_path):
        print(f"❌ STL file not found: {stl_path}")
        return None

    try:
        mesh = trimesh.load(stl_path)

        # If the mesh is a Scene (multiple objects), we might want to analyze them separately
        # But step_to_stl often merges them. Let's assume it's one mesh for now.
        if isinstance(mesh, trimesh.Scene):
            # If it's a scene, dump all geometries into one for analysis
            # Or iterate? Iterating is safer to find specific parts.
            geometries = list(mesh.geometry.values())
        else:
            geometries = [mesh]

        candidates = []

        for geom in geometries:
            # Check if this geometry looks like a disk
            # 1. Check bounds: roughly equal X and Y range, small Z range?
            bounds = geom.bounds
            extents = bounds[1] - bounds[0] # [dx, dy, dz]

            # Aspect ratio check
            # Disk: dx ~ dy, dz << dx
            if extents[2] < 0.2 * max(extents[0], extents[1]): # Flat
                 if 0.8 < extents[0] / extents[1] < 1.2: # Square/Circle footprint
                     # It's a candidate.
                     center = (bounds[0] + bounds[1]) / 2.0
                     # Refine Z: usually the top surface
                     center[2] = bounds[1][2]
                     candidates.append((center, geom.area))

        if not candidates:
            # Fallback: Just return the global center of the bounding box,
            # but projected to the top surface?
            print("⚠️  No specific disk-like part found. Using global center.")
            bounds = mesh.bounds
            center = (bounds[0] + bounds[1]) / 2.0
            # User said "UR5e site in disk center".
            # If we can't find it, defaulting to (0,0,0) or center might be wrong.
            # But let's try to be helpful.
            return center

        # Return the candidate with largest area (likely the table/disk main surface)
        # Or if the "disk" is smaller than the "table", maybe the second largest?
        # "Table AND a disk". Disk is usually on the table.
        # So maybe the highest candidate?
        candidates.sort(key=lambda x: x[0][2], reverse=True) # Sort by Z height (highest first)

        best_center = candidates[0][0]
        print(f"✅ Detected potential placement site at: {best_center}")
        return best_center

    except Exception as e:
        print(f"❌ Error analyzing mesh: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        c = find_disk_center(sys.argv[1])
        if c is not None:
            print(f"CENTER_RESULT:{c[0]:.4f},{c[1]:.4f},{c[2]:.4f}")
    else:
        print("Usage: python find_disk_center.py <stl_file>")
