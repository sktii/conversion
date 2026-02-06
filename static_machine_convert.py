import os
import cadquery as cq

# --- 設定區 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "STP")
OUTPUT_DIR = os.path.join(BASE_DIR, "STL_Static")
XML_FILE = "static_machine.xml"

# 碰撞精度設定
# 如果你的機台需要非常精確的碰撞（例如有很小的孔），請把 TOLERANCE 設小
TOLERANCE = 0.02  
ANGULAR_TOLERANCE = 0.1

def convert_static_machine():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    stp_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.stp', '.step'))]
    if not stp_files:
        print("❌ 找不到 STP 檔案")
        return

    xml_assets = ""
    xml_geoms = ""
    
    print(f"🚀 開始處理靜態機台轉換...")

    total_parts = 0
    
    for file_name in stp_files:
        input_path = os.path.join(INPUT_DIR, file_name)
        base_name = os.path.splitext(file_name)[0]
        
        print(f"📖 讀取: {file_name} ...")
        
        try:
            # 讀取 STP
            model = cq.importers.importStep(input_path)
            solids = model.solids().vals()
            
            print(f"   ↳ 拆解出 {len(solids)} 個組件")

            for i, solid in enumerate(solids):
                part_name = f"{base_name}_{i}"
                stl_name = f"{part_name}.stl"
                output_path = os.path.join(OUTPUT_DIR, stl_name)
                
                # 1. 匯出 STL (保留絕對座標)
                cq.exporters.export(
                    cq.Workplane(obj=solid),
                    output_path, 
                    exportType="STL", 
                    tolerance=TOLERANCE, 
                    angularTolerance=ANGULAR_TOLERANCE
                )
                
                # 2. 寫入 XML Asset (定義網格)
                # scale="0.001..." 假設 STP 是 mm，轉成公尺
                xml_assets += f'    <mesh name="{part_name}_mesh" file="{stl_name}" scale="0.001 0.001 0.001"/>\n'
                
                # 3. 寫入 XML Geom (定義實體)
                # 注意：這裡直接放在 worldbody，沒有 joint，所以它是靜態的(不會動)
                # group="1" 用於碰撞分組 (可選)
                xml_geoms += f"""
    <geom type="mesh" mesh="{part_name}_mesh" rgba="0.7 0.7 0.7 1" />
"""
                total_parts += 1
                
        except Exception as e:
            print(f"❌ 錯誤: {e}")

    # 生成完整 XML
    full_xml = f"""<mujoco model="static_machine">
  <compiler meshdir="{os.path.basename(OUTPUT_DIR)}" balanceinertia="true"/>
  
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
    
    {xml_assets}
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="2 2 .05" type="plane" material="grid"/>

    {xml_geoms}
  </worldbody>
</mujoco>
"""

    with open(os.path.join(BASE_DIR, XML_FILE), "w", encoding="utf-8") as f:
        f.write(full_xml)

    print(f"🎉 完成！共處理 {total_parts} 個零件。")
    print(f"👉 請開啟 {XML_FILE} 檢查碰撞狀況。")

if __name__ == "__main__":
    convert_static_machine()