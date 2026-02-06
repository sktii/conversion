import os
import cadquery as cq

# --- 設定路徑 (根據你的檔案結構) ---
# 取得目前腳本所在的目錄
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "STP")  # 讀取 STP 資料夾
OUTPUT_DIR = os.path.join(BASE_DIR, "STL") # 寫入 STL 資料夾

# --- 轉換參數 (可依需求調整) ---
# tolerance (公差): 數值越小模型越精細，但檔案越大。
# 給 MuJoCo 用建議：0.01 ~ 0.05
TOLERANCE = 0.05 
ANGULAR_TOLERANCE = 0.1

def convert_stp_to_stl():
    # 檢查輸入資料夾是否存在
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 錯誤: 找不到輸入資料夾 '{INPUT_DIR}'")
        return

    # 確保輸出資料夾存在 (雖然你已經有了，但加這行比較保險)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 已建立輸出資料夾: {OUTPUT_DIR}")

    # 取得資料夾內所有 .stp 或 .step 檔案
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.stp', '.step'))]
    
    if not files:
        print(f"⚠️  在 '{INPUT_DIR}' 裡找不到任何 STP 檔案。")
        return

    print(f"🚀 開始轉換，共找到 {len(files)} 個檔案...")
    print(f"📂 來源: {INPUT_DIR}")
    print(f"📂 目標: {OUTPUT_DIR}")
    print("-" * 40)

    success_count = 0

    for file_name in files:
        input_path = os.path.join(INPUT_DIR, file_name)
        # 輸出的檔名將副檔名改為 .stl
        output_name = os.path.splitext(file_name)[0] + ".stl"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        
        try:
            print(f"🔄 正在處理: {file_name} ...", end="\r")
            
            # 1. 匯入 STP
            model = cq.importers.importStep(input_path)
            
            # 2. 匯出 STL
            cq.exporters.export(
                model, 
                output_path, 
                exportType="STL", 
                tolerance=TOLERANCE, 
                angularTolerance=ANGULAR_TOLERANCE
            )
            
            print(f"✅ 完成: {output_name}      ") # 空格是為了蓋掉上一行的文字
            success_count += 1
            
        except Exception as e:
            print(f"\n❌ 失敗: {file_name}")
            print(f"   錯誤訊息: {e}")

    print("-" * 40)
    print(f"🎉 全部完成！成功轉換 {success_count}/{len(files)} 個檔案。")

if __name__ == "__main__":
    convert_stp_to_stl()