import cv2
import pytesseract
import json

img_path = 'temp_covers/9789373147499.png'
img = cv2.imread(img_path)
d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config='--psm 11')

height = img.shape[0]
dpi_y = height / 8.0
badge_height_px = int((9.0 / 25.4) * dpi_y)
badge_y = height - badge_height_px

out = {
    "image_height": height,
    "badge_zone_y": badge_y,
    "badge_height_px": badge_height_px,
    "texts": []
}

for i in range(len(d['text'])):
    text = d['text'][i].strip()
    conf = float(d['conf'][i])
    if conf >= 40 and text:
        x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
        bottom_edge = y + h
        distance_from_bottom = height - bottom_edge
        distance_mm = (distance_from_bottom / dpi_y) * 25.4
        
        out["texts"].append({
            "text": text,
            "y": y,
            "bottom_edge": bottom_edge,
            "distance_from_bottom_px": distance_from_bottom,
            "distance_mm": round(distance_mm, 2)
        })

with open("analysis.json", "w") as f:
    json.dump(out, f, indent=2)
print("Done")
