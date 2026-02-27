import traceback
import sys
import os

try:
    with open("crash_log.txt", "w") as f:
        f.write(f"Starting script... cwd: {os.getcwd()}\n")
        
    import cv2
    import pytesseract
    
    img_path = 'temp_covers/9789373147499.png'
    if not os.path.exists(img_path):
        with open("crash_log.txt", "a") as f: f.write(f"File not found: {img_path}\n")
        sys.exit(1)
        
    img = cv2.imread(img_path)
    d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config='--psm 11')
    
    height = img.shape[0]
    dpi_y = height / 8.0
    badge_height_px = int((9.0 / 25.4) * dpi_y)
    badge_y = height - badge_height_px
    
    with open("crash_log.txt", "a") as f:
        f.write(f"OCR success. Found {len(d['text'])} boxes. height={height} badge_y={badge_y}\n")
        for i in range(len(d['text'])):
            text = d['text'][i].strip()
            conf = float(d['conf'][i])
            if conf >= 40 and text:
                x, y, h = d['left'][i], d['top'][i], d['height'][i]
                bot = y + h
                in_zone = bot > badge_y
                f.write(f"[{text}] bot={bot} in_zone={in_zone}\n")
                
except Exception as e:
    with open("crash_log.txt", "a") as f:
        f.write("ERROR:\n")
        f.write(traceback.format_exc())
