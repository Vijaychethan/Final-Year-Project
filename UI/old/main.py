from ultralytics import YOLO
model_path = 'yolov8n-seg.pt'
model = YOLO(model_path)
model.predict(source="C:/Users/VICH/Desktop/Ulcerdetection/train/images/4-Figure3-used_png.rf.5b88ce9292a74bdea14410db5e12b065.jpg",show=True,save=True,hide_labels=False,hide_conf=False,conf=0.5, save_txt=True,save_crop=True,line_thickness=2)