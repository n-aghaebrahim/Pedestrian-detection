from ultralytics import YOLO



def train(model: str="yolov8.yaml", 
          optimizer: str= 'Adam', 
          patience: int=50,
          verbose: bool=True,
          epochs: int=10,
          yaml_data: str=None
          ):

    # Load a model
    model = YOLO(model)  # build a new model from scratch
    #model = YOLO("yolov8.pt")

    # Use the model to train

    model.train(verbose=verbose, 
                optimizer=optimizer, 
                patience=patience, 
                data=yaml_data, 
                epochs=epochs,
                )
    

    metrics = model.val()  # evaluate model performance on the validation set
