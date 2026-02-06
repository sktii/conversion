import numpy as np
import trimesh
from sklearn.cluster import KMeans
import os

# --- 設定 ---
STL_FILE = "STL_Static/machine_combined.stl"  # 指向你轉好的 STL (如果是多個零件，建議先在 MeshLab 合併成一個，或修改代碼讀取多個)
OUTPUT_XML = "fitted_pillars.xml"
NUM_CYLINDERS = 16
NUM_BOXES = 16
TOTAL_PILLARS = NUM_CYLINDERS + NUM_BOXES

def generate_approximate_pillars():
    print(f"Loading STL from {STL_FILE}...")
    
    # 1. 讀取 STL (處理多個檔案的情況)
    # 這裡假設你有辦法提供一個合併後的 Mesh，如果沒有，此處需修改為讀取資料夾
    # 為了示範，我們先生成一個隨機點雲來模擬你的機台
    # 在實際使用時，請用: mesh = trimesh.load(STL_FILE)
    
    # --- 模擬讀取 STL (請替換為真實讀取代碼) ---
    # 假設機台是一個在 (0.4, 0.2) 附近的複雜形狀
    try:
        if os.path.exists(STL_FILE):
            mesh = trimesh.load(STL_FILE)
            # 在網格表面採樣 5000 個點
            points, _ = trimesh.sample.sample_surface(mesh, 5000)
        else:
            print("⚠️ 找不到 STL 檔案，使用模擬數據演示...")
            points = np.random.rand(5000, 3) * 0.5 + np.array([0.2, 0.0, 0.0])
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return
    # ---------------------------------------------

    print(f"Sampling {len(points)} points and running K-Means for {TOTAL_PILLARS} clusters...")

    # 2. 使用 K-Means 將點雲分成 32 群
    kmeans = KMeans(n_clusters=TOTAL_PILLARS, n_init=10)
    kmeans.fit(points)
    centers = kmeans.cluster_centers_

    # 3. 為了安全，計算每個群的半徑 (包覆球)
    # 我們算出每個中心點到其所屬群內最遠點的距離，作為柱子的半徑/尺寸
    labels = kmeans.labels_
    radii = []
    for i in range(TOTAL_PILLARS):
        cluster_points = points[labels == i]
        if len(cluster_points) > 0:
            # 計算該群所有點到中心的距離
            dists = np.linalg.norm(cluster_points - centers[i], axis=1)
            # 取最大值作為半徑，並稍微放大一點做安全餘量 (Safety Margin)
            r = np.max(dists) * 1.1 
            # 限制最小半徑，避免太細
            r = max(r, 0.02)
        else:
            r = 0.02
        radii.append(r)

    # 4. 生成 XML
    xml_content = "<mujoco>\n<worldbody>\n"
    
    # 為了讓機台是靜態的，通常不需要 body，直接用 geom 即可，但為了配合你的 RL 輸入格式
    # 你的原檔是 geom name="pillar_cyl_1"...
    
    # 生成 16 個圓柱
    for i in range(NUM_CYLINDERS):
        pos_str = f"{centers[i][0]:.4f} {centers[i][1]:.4f} {centers[i][2]:.4f}"
        # 你的原檔圓柱是 Z 軸朝上，size="半徑 半長"
        # 這裡我們簡化：假設柱子是直立的，高度設為 0.2 (或根據 bounding box 計算)
        r = radii[i]
        xml_content += f'    <geom name="pillar_cyl_{i+1}" type="cylinder" size="{r:.3f} 0.2" pos="{pos_str}" rgba="0.5 0.5 0.5 1" contype="1" conaffinity="1"/>\n'

    # 生成 16 個方柱
    for i in range(NUM_BOXES):
        idx = NUM_CYLINDERS + i
        pos_str = f"{centers[idx][0]:.4f} {centers[idx][1]:.4f} {centers[idx][2]:.4f}"
        # 方柱 size="x y z" (半長)
        r = radii[idx]
        xml_content += f'    <geom name="pillar_box_{i+1}" type="box" size="{r:.3f} {r:.3f} 0.2" pos="{pos_str}" rgba="0.5 0.5 0.5 1" contype="1" conaffinity="1"/>\n'

    xml_content += "</worldbody>\n</mujoco>"

    with open(OUTPUT_XML, "w") as f:
        f.write(xml_content)
    
    print(f"✅ Generated {OUTPUT_XML} with {TOTAL_PILLARS} approximated pillars.")

if __name__ == "__main__":
    generate_approximate_pillars()