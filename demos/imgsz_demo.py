"""
Demo to visualize the fidelity of the predicted mask at different image sizes.
"""

from ultralytics import YOLO

models = [
    '/Users/sebnico/Desktop/CIS4900/injection-tracker/weights/second_dataset/640/best_train6.pt',
    '/Users/sebnico/Desktop/CIS4900/injection-tracker/weights/second_dataset/1280/best_train12.pt',
    '/Users/sebnico/Desktop/CIS4900/injection-tracker/weights/second_dataset/2560/best_train15.pt'
]

imgsz_values = [640, 1280, 2560]

img_path = "/Users/sebnico/Desktop/CIS4900/injection-tracker/processed_second_dataset/images/train/outdoors/frame_00011.png"
for imgsz, model in zip(imgsz_values, models):
    model = YOLO(model)
    results = model.predict(img_path, imgsz=imgsz)
    results[0].plot(labels=False, boxes=False, masks=True, save=True, filename=f"./report/result_{imgsz}.jpg")