from ultralytics import YOLO
import os
import cv2

model = YOLO('best.pt')
print("Model loaded successfully!")

print(f"\nModel: {model.model_name}")
print(f"Classes: {model.names}")
print(f"Number of Classes: {len(model.names)}")

print("\n Model is ready for testing. ")


source_path = r"D:\VIT\CV project\Helmet dataset\test\images"
output_path = r"D:\VIT\CV project\output"

results = model.predict(
    source=source_path,
    conf=0.5,
    project = output_path,
    name = "Predicted Images",
    save=True
)

print("Detection Completed")
print(f"Result saved to: {os.path.join(output_path, 'Predicted Images')}")