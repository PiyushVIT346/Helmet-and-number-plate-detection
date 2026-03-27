import os
import sys
import cv2
import numpy as np
import imutils
import easyocr
import argparse
from ultralytics import YOLO


print("Loading YOLO model...")
yolo_model = YOLO('best.pt') 
print(f"YOLO Model loaded successfully! Classes: {yolo_model.names}")

print("Initializing EasyOCR...")

reader = easyocr.Reader(['en'], gpu=False) 
print("EasyOCR loaded successfully!")

HELMET_CLASS_NAME = 'helmet'

import cv2
import numpy as np
import imutils
import re

def read_license_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bfilter = cv2.bilateralFilter(gray, 11, 17, 17) 
    edged = cv2.Canny(bfilter, 30, 200) 

    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    if location is None:
        print("  -> [OCR] Could not find a rectangular contour matching a license plate.")
        return img

    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [location], 0, 255, -1)
    
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]
    
    cropped_resized = cv2.resize(cropped_image, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    
    blur = cv2.GaussianBlur(cropped_resized, (5, 5), 0)
    
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    alphanumeric = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    results = reader.readtext(thresh, allowlist=alphanumeric, detail=0, paragraph=False)

    if not results:
        print("  -> [OCR] EasyOCR could not detect text in the cropped region.")
        return img
    
    raw_text = "".join(results).replace(" ", "")
    
    clean_text = re.sub(r'[^A-Z0-9]', '', raw_text)
    
    print(f"  -> [OCR] Detected Number Plate Text: {clean_text}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y, w, h = cv2.boundingRect(location)
    
    res = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
    (text_width, text_height), _ = cv2.getTextSize(clean_text, font, 1, 2)
    cv2.rectangle(res, (x, y - text_height - 10), (x + text_width, y), (0, 255, 0), -1)
    
    res = cv2.putText(res, text=clean_text, org=(x, y - 5), 
                    fontFace=font, fontScale=1, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

    return res

def process_single_image(image_path, output_path):
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at {image_path}")
        sys.exit(1)
        
    print(f"\nProcessing image: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read the image file. It might be corrupted or in an unsupported format.")
        sys.exit(1)

    results = yolo_model.predict(source=img, conf=0.5, save=False, verbose=False)
    
    annotated_img = results[0].plot()
    
    detected_class_indices = results[0].boxes.cls.cpu().numpy()
    detected_class_names = [yolo_model.names[int(cls_idx)] for cls_idx in detected_class_indices]
    
    if HELMET_CLASS_NAME not in detected_class_names:
        print("  -> Helmet NOT detected. Triggering Number Plate Recognition...")
        final_img = read_license_plate(annotated_img)
    else:
        print("  -> Helmet detected. Skipping Number Plate Recognition.")
        final_img = annotated_img

    cv2.imwrite(output_path, final_img)
    print(f"Pipeline Completed! Processed image saved to: {output_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Helmet Detection and License Plate Recognition Pipeline")
    parser.add_argument("-i", "--image", required=True, help="Path to the input image file")
    parser.add_argument("-o", "--output", required=False, default="output.jpg", help="Path to save the output image (default: output.jpg)")
    
    args = parser.parse_args()
    
    process_single_image(args.image, args.output)