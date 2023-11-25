# Pedestrian-detection
Pedestrian detection using YOLOv8 for accurate and real-time results in computer vision applications. 🚶‍♂️👀 #YOLOv8 #PedestrianDetection

Certainly! Here's a combined README.md file that includes information about the purpose of the code and the YOLOv8 model used for pedestrian detection:

```markdown
# Pedestrian Detection Project

This repository contains code for a pedestrian detection project using the YOLOv8 model. The project includes scripts for exploratory data analysis (EDA), training, and prediction.

## Purpose

The purpose of this project is to detect pedestrians in images using the YOLOv8 model. YOLO (You Only Look Once) is an object detection algorithm that can detect and classify multiple objects in an image in real-time. YOLOv8 is the latest version of the YOLO series and is known for its efficiency and accuracy.

## Directory Structure

The project directory structure is organized as follows:

```
/github/Pedestrian-detection
├── data
│   ├── pedestrian
│   └── archive-3
│   └── PedestriansDetection
├── src
│   ├── eda.py
│   ├── predict.py
│   └── train.py
├── output
│   ├── data_eda
│   ├── detect
│   │   ├── train2
│   │   └── val
├── main.py
├── main_2.py
├── run.sh
└── README.md
```

## Scripts

- **`main.py`**: Main script to perform EDA, training, and prediction. Set flags `predict_check` and `train_check` accordingly.

- **`main_2.py`**: Alternative main script with command-line arguments for more flexibility in EDA, training, and prediction.

- **`eda.py`**: Script for exploratory data analysis, including label distribution, image size analysis, and average image size calculation.

- **`predict.py`**: Script for making predictions using a pre-trained YOLOv8 model.

- **`train.py`**: Script for training a YOLOv8 model on the provided dataset.

## Usage

### EDA

```bash
python main.py
```

or with more options:

```bash
python main_2.py --predict_check True --train_check False --yaml_file_path /path/to/data.yaml --data_dirs /path/to/dataset/ --hist_output /path/to/histograms/ --size_output /path/to/sizes/
```

### Training

```bash
python main.py
```

or with more options:

```bash
python main_2.py --predict_check False --train_check True --model yolov8.yaml --optimizer Adam --patience 80 --verbose True --yaml_data /path/to/data.yaml --epochs 50
```

### Prediction

```bash
python main.py
```

or with more options:

```bash
python main_2.py --predict_check True --train_check False --source /path/to/image.jpg --imgsz 416 --conf 0.9 --iou 0.05 --show True --modeldir /path/to/model/weights/best.pt
```

## Model

The model used for this project is YOLOv8, a state-of-the-art object detection model. YOLOv8 is known for its speed and accuracy, making it suitable for real-time applications. The model is trained on pedestrian datasets to detect and localize pedestrians in images.

## Dependencies

- numpy
- pandas
- matplotlib
- cv2
- seaborn
- tensorflow
- scikit-learn
- ultralytics

Install the dependencies using:

```bash
pip install -r requirements.txt
```

# Training Results

## Overview

This repository contains the training results of the YOLOv8 model for pedestrian detection. The model was trained for 50 epochs using the following datasets:

- Number of Train Images: 4059
- Number of Validation Images: 397
- Number of Test Images: 185

## Training Results

### Epoch: 50

- **Training Loss:**
  - Box Loss: 1.0281
  - Classification Loss: 0.71156
  - DFL (Dynamic Feature Learning) Loss: 1.4072

- **Metrics:**
  - Precision (B): 0.80545
  - Recall (B): 0.84532
  - mAP50 (B): 0.88894
  - mAP50-95 (B): 0.59321

- **Validation Loss:**
  - Box Loss: 1.2284
  - Classification Loss: 0.75598
  - DFL Loss: 1.5826

- **Learning Rates:**
  - pg0: 0.000496
  - pg1: 0.000496
  - pg2: 0.000496



## Confusion Matrix - Normalized

![Confusion Matrix - Normalized](/output/detect/train2/confusion_matrix_normalized.png)

## Confusion Matrix

![Confusion Matrix](/output/detect/train2/confusion_matrix.png)

## F1 Curve

![F1 Curve](/output/detect/train2/F1_curve.png)

## Labels Correlogram

![Labels Correlogram](/output/detect/train2/labels_correlogram.jpg)

## Labels

![Labels](/output/detect/train2/labels.jpg)

## P Curve

![P Curve](/output/detect/train2/P_curve.png)

## PR Curve

![PR Curve](/output/detect/train2/PR_curve.png)

## R Curve

![R Curve](/output/detect/train2/R_curve.png)

## Results

![Results](/output/detect/train2/results.png)



# Validation Results

## Confusion Matrix - Normalized

![Confusion Matrix - Normalized](/output/detect/val/confusion_matrix_normalized.png)

## Confusion Matrix

![Confusion Matrix](/output/detect/val/confusion_matrix.png)

## F1 Curve

![F1 Curve](/output/detect/val/F1_curve.png)

## P Curve

![P Curve](/output/detect/val/P_curve.png)

## PR Curve

![PR Curve](/output/detect/val/PR_curve.png)

## R Curve

![R Curve](/output/detect/val/R_curve.png)

## Validated Images

![Validated Image 1](/output/detect/val/val_batch0_pred.jpg)



## License

This project is licensed under the [MIT License](LICENSE).
```

Feel free to copy and paste this into your README.md file on GitHub. Customize any additional details or information specific to your project.
