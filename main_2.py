import argparse
import numpy as np
import pandas as pd
import cv2
import os
import ast

from xml.etree import ElementTree
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras import datasets, layers, models

from src.eda import eda
from src.train import train
from src.predict import predict

keras = tf.keras

parser = argparse.ArgumentParser(description='Your script description')
parser.add_argument('--predict_check',
                    type=ast.literal_eval,
                    help='Enable prediction'
                    )

parser.add_argument('--train_check', 
                    type=ast.literal_eval,
                    help='Enable prediction'
                    )

parser.add_argument('--yaml_file_path', 
                    type=str, 
                    default="/Users/navid/work_station/github/Pedestrian-detection/data/pedestrian/data.yaml", 
                    help='Path to YAML file'
                    )

parser.add_argument('--data_dirs', 
                    type=str, 
                    default="/Users/navid/work_station/github/Pedestrian-detection/data/pedestrian/", 
                    help='Data directories'
                    )

parser.add_argument('--hist_output', 
                    type=str, 
                    default="../output/data_eda/", 
                    help='Path for histogram output'
                    )

parser.add_argument('--size_output', 
                    type=str, 
                    default="../output/data_eda/", 
                    help='Path for size output'
                    )

parser.add_argument('--model', 
                    type=str, 
                    default="yolov8.yaml", 
                    help='Model name'
                    )

parser.add_argument('--optimizer', 
                    type=str, 
                    default="Adam", 
                    help='Optimizer'
                    )

parser.add_argument('--patience', 
                    type=float, 
                    default=80, 
                    help='Patience for training'
                    )

parser.add_argument('--verbose', 
                    type=ast.literal_eval,
                    default=True,
                    help='Verbose mode for training'
                    )

parser.add_argument('--yaml_data', 
                    type=str, 
                    default="/Users/navid/work_station/github/Pedestrian-detection/data/pedestrian/data.yaml", 
                    help='Path to YAML data'
                    )

parser.add_argument('--epochs', 
                    type=float, 
                    default=50, 
                    help='Number of epochs for training'
                    )

parser.add_argument('--source', 
                    type=str, 
                    default="screen", 
                    help='Source for prediction'
                    )

parser.add_argument('--imgsz', 
                    type=float, 
                    default=416, 
                    help='Image size for prediction'
                    )

parser.add_argument('--conf', 
                    type=float, 
                    default=0.9, 
                    help='Confidence for prediction'
                    )

parser.add_argument('--iou', 
                    type=float, 
                    default=0.05, 
                    help='IOU for prediction'
                    )

parser.add_argument('--show', 
                    type=ast.literal_eval,
                    default=True, 
                    help='Show prediction'
                    )

parser.add_argument('--modeldir', 
                    type=str, 
                    default="/Users/navid/work_station/github/Pedestrian-detection/output/train/weights/last.pt", 
                    help='Model directory for prediction'
                    )

parser.parse_args()

def main(args):
    # Apply EDA on dataset
    eda(yaml_file_path=args.yaml_file_path,
        data_dirs=args.data_dirs,
        hist_output=args.hist_output,
        size_output=args.size_output)

    if args.train_check:
        # Call train function to train yolov8
        train(model=args.model,
              optimizer=args.optimizer,
              patience=int(args.patience),
              verbose=args.verbose,
              yaml_data=args.yaml_data,
              epochs=int(args.epochs)
              )

    if args.predict_check:
        # Call predict function to predict using trained model/weight
        predict(source=args.source,
                imgsz=int(args.imgsz),
                conf=args.conf,
                iou=args.iou,
                show=args.show,
                model_dir=args.modeldir
                )

if __name__ == "__main__":
    main(parser.parse_args())



