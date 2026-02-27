import cv2
import pytesseract

img_path = 'temp_covers/9789373147994.png'
img = cv2.imread(img_path)
if img is None:
    with open('debug_ocr.txt', 'w') as f:
        f.write(f"Failed to load {img_path}")
    exit(1)

d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config='--psm 11')
with open('debug_ocr.txt', 'w', encoding='utf-8') as f:
    f.write(f"Total boxes: {len(d['text'])}\n")
    for i in range(len(d['text'])):
        text = d['text'][i].strip()
        conf = float(d['conf'][i])
        x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
        if text:
            f.write(f"Text: '{text}', Conf: {conf}, Box: (x={x}, y={y}, w={w}, h={h})\n")
