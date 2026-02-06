Env building
# 建立一個新環境 (建議)
conda create -n cad_conversion python=3.10
conda activate cad_conversion

# 安裝 CadQuery (它包含了轉換所需的 OCP 核心)
conda install -c cadquery -c conda-forge cadquery=master
mkdir -p CAD_CONVERSION
pip install trimesh scikit-learn numpy

pip install path

Use

python cad_conversion.py
python static_machine_convert.py
python origin_to_pillar.py
