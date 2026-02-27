import cv2
import pytesseract

img_path = 'temp_covers/9789373147994.png'
img = cv2.imread(img_path)
if img is None:
    print(f"Failed to load {img_path}")
    exit(1)

d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config='--psm 11')
print("Total boxes:", len(d['text']))
for i in range(len(d['text'])):
    text = d['text'][i].strip()
    conf = float(d['conf'][i])
    x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
    if text:
        print(f"Text: '{text}', Conf: {conf}, Box: (x={x}, y={y}, w={w}, h={h})")
