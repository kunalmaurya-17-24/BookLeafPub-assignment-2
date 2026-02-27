import sys
import json
from PIL import Image
from detection import load_and_preprocess, run_geometry_engine

img_path = 'temp_covers/9789373147994.png'
img = load_and_preprocess(img_path)
geo = run_geometry_engine(img, author_name="Benny James SDB")

out = {
    "passed": geo.passed,
    "max_overlap_mm": geo.max_overlap_mm,
    "issues": [{"desc": i.description, "sev": getattr(i.severity, "name", str(i.severity))} for i in geo.issues]
}

with open("geo_debug.json", "w") as f:
    json.dump(out, f, indent=2)
