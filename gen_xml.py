import os

# --- 設定路徑 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STL_DIR_NAME = "STL"  # 你的 STL 資料夾名稱
OUTPUT_XML = "model.xml"

def generate_mujoco_xml():
    stl_dir_path = os.path.join(BASE_DIR, STL_DIR_NAME)
    
    # 檢查 STL 資料夾是否存在
    if not os.path.exists(stl_dir_path):
        print(f"❌ 找不到 STL 資料夾: {stl_dir_path}")
        return

    # 取得所有 stl 檔案
    stl_files = [f for f in os.listdir(stl_dir_path) if f.lower().endswith('.stl')]
    
    if not stl_files:
        print("⚠️  找不到任何 STL 檔案，無法生成 XML。")
        return

    # --- 開始撰寫 XML 內容 ---
    # 這裡使用 f-string 直接組裝 XML 字串，這是最直觀的方法
    
    xml_content = f"""<mujoco model="auto_generated_robot">
  <compiler meshdir="{STL_DIR_NAME}" balanceinertia="true"/>
  
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
    
    """

    # 1. 寫入 Asset (告訴 MuJoCo 有哪些檔案)
    for stl in stl_files:
        name = os.path.splitext(stl)[0]
        # scale="0.001 0.001 0.001" 是因為 SolidWorks 通常是 mm，MuJoCo 是 m
        xml_content += f'    <mesh name="{name}_mesh" file="{stl}" scale="0.001 0.001 0.001"/>\n'

    xml_content += """  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 .05" type="plane" material="grid" condim="3"/>

    """

    # 2. 寫入 Body (把零件放進場景)
    for stl in stl_files:
        name = os.path.splitext(stl)[0]
        xml_content += f"""
    <body name="{name}" pos="0 0 0.5">
      <freejoint/> <geom type="mesh" mesh="{name}_mesh" rgba="0.8 0.6 0.4 1"/>
    </body>
"""

    xml_content += """
  </worldbody>
</mujoco>
"""

    # 寫入檔案
    output_path = os.path.join(BASE_DIR, OUTPUT_XML)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_content)

    print(f"🎉 成功生成 MuJoCo XML: {output_path}")
    print("👉 下一步：請使用 './simulate model.xml' 開啟並調整位置。")

if __name__ == "__main__":
    generate_mujoco_xml()