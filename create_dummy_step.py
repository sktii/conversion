import cadquery as cq
import os

if not os.path.exists("STP"):
    os.makedirs("STP")

# Create a simple box
result = cq.Workplane("XY").box(10, 10, 10)

# Export to STEP
cq.exporters.export(result, "STP/test_box.step")
print("Created STP/test_box.step")
