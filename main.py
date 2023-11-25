import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2
import os
from xml.etree import ElementTree
from matplotlib import pyplot as plt



import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras import datasets, layers, models
keras = tf.keras


from src.eda import eda
from src.train import train
from src.predict import predict



predict_check = True
train_check = False
# Apply EDA on dataset
eda(yaml_file_path="/Users/navid/work_station/github/Pedestrian-detection/data/PedestriansDetection/data.yaml",
        data_dirs="/Users/navid/work_station/github/Pedestrian-detection/data/PedestriansDetection/",
        hist_output="../output/data_eda/",
        size_output="../output/data_eda/",
        )





if train_check:
    # call train funciton to train yolov8
    train(model="yolov8.yaml",
          optimizer="Adam",
          patience=80,
          verbose=True,
          yaml_data="/Users/navid/work_station/github/Pedestrian-detection/data/PedestriansDetection/data.yaml",
          epochs=50,
          )


if predict_check:
    # call predict function to predict using trained model/weight
    predict(source="screen",
            imgsz=416,
            conf=0.2,
            iou=0.2,
            show=True,
            model_dir="/Users/navid/work_station/github/Pedestrian-detection/output/detect/train2/weights/best.pt",
            )


