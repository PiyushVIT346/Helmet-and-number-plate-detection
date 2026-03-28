import cv2
import numpy as np
import imutils
import easyocr
import argparse
import sys
import os

def process_image(image_path, output_path=None):
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at {image_path}")
        sys.exit(1)
        
    img = cv2.imread(image_path)
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
        print("Could not find a rectangular contour matching a license plate.")
        sys.exit(1)
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]

    print("Initializing EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(cropped_image)

    if not result:
        print("EasyOCR could not detect any text in the cropped region.")
        sys.exit(1)

    text = result[0][1]
    print(f"Detected Text: {text}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    res = cv2.putText(img, text=text, org=(location[0][0][0], location[1][0][1] + 60), 
                      fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    
    x, y, w, h = cv2.boundingRect(location)
    res = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    if output_path:
        cv2.imwrite(output_path, res)
        print(f"Processed image saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from a license plate in an image.")
    parser.add_argument("-i", "--image", required=True, help="Path to the input image")
    parser.add_argument("-o", "--output", required=False, default="output.jpg", help="Path to save the output image (default: output.jpg)")
    
    args = parser.parse_args()
    
    # Run the pipeline
    process_image(args.image, args.output)
