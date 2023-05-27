import torch
from .config import settings
from ultralytics import YOLO


class LoadModel():
    model_id: str
    model_type: str

    def __init__(self, model_type: str, model_id: str) -> None:
        global model
        self.model_id = model_id
        self.model_type = model_type
        if model_type == "yolov5":
            model = torch.hub.load('app\models\yolov5', 'custom',
                                   path=f'{settings.MODEL_DIR}/{model_id}.pt',
                                   source='local',
                                   verbose=False,
                                   force_reload=True)
        elif model_type == "yolov5_onnx":
            model = torch.hub.load('app\models\yolov5', 'custom',
                                   path=f'{settings.MODEL_DIR}/{model_id}.onnx',
                                   source='local',
                                   verbose=False,
                                   force_reload=True)
        elif model_type == "yolov8":
            model = YOLO(f'{settings.MODEL_DIR}/{model_id}.pt')

    def get_model(self):
        return model
