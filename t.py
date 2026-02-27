import cv2
import pytesseract
print("Imported")
img = cv2.imread('temp_covers/9789373147499.png')
print(f"Loaded {img.shape}")
d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config='--psm 11')
print(f"Found {len(d['text'])} texts")
with open("testout.txt", "w") as f:
    f.write(str(d['text']))
print("Done")
