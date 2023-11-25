from ultralytics import YOLO
#from ultralytics.models.yolo.v8.detect.predict import DetectionPredictor
import cv2


def predict(source: str="/Users/navid/work_station/github/Pedestrian-detection/data/pedestrian/train/images/PennPed00040_png.rf.537dd180b59562e73949b96dc4154c6a.jpg",
            imgsz: str=416,
            conf: int=0.9,
            iou: int=0.05,
            show: bool=True,
            model_dir: str="/Users/navid/work_station/github/Pedestrian-detection/output/detect/train2/weights/best.pt",
            ):

    #if not model_dir:
    #    model = YOLO("yolov8n.pt")
    #else:
    model = YOLO(model_dir)
    model.predict(source=source, 
                  imgsz=imgsz,  
                  show=show, 
                  conf=conf, 
                  iou=iou,
                  )
    #results = model.predict(source="0", show=True)

    #print(results)

predict()
