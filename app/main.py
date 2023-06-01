import shutil
from typing import Annotated
import uuid
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from app.utils import create_grid
from .config import settings
from .database import *
import os
from contextlib import asynccontextmanager
import aiofiles
import torch
from ultralytics import YOLO

from enum import Enum

CHUNK_SIZE = 1024 * 1024
db_model = get_all_model()

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    ml_models["yolov5"] = torch.hub.load('app\models\yolov5', 'custom',
                                         path=r'app\weight\yolov5s.pt',
                                         source='local', verbose=False)
    ml_models["yolov5onnx"] = torch.hub.load('app\models\yolov5', 'custom',
                                             path=r'app\weight\yolov5.onnx',
                                             source='local', verbose=False)
    ml_models["yolov8"] = YOLO("app\weight\yolov8.pt")
    ml_models["yolov8onnx"] = YOLO("app\weight\yolov8.onnx", task="detect")
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()


app = FastAPI(
    title="Quick Detect",
    description="""
    G64190075.
    Detection Only and Input must Image.
    Classification and Segmentation are not supported.
    Supported model: 
        - yolov5 (pyTorch and Onnx)
        - yolov8 (pyTorch Only)
    Supported image format: JPG and JPEG
    """,
    lifespan=lifespan
)


def check_file_type(file: UploadFile = File()):
    """Check the file type. The file must be a JPG or JPEG file."""
    if not file.content_type.startswith("image/jpg") and not file.content_type.startswith("image/jpeg"):
        raise HTTPException(
            status_code=400, detail="File must be an image (JPG or JPEG format)")


# @app.on_event("startup")
# async def startup_event():
#     # construct class
#     global yolov5, yolov5onnx, yolov8, yolov8onnx
#     yolov5 = torch.hub.load('app\models\yolov5', 'custom',
#                             path=r'app\weight\yolov5s.pt',
#                             source='local', verbose=False)
#     yolov5onnx = torch.hub.load('app\models\yolov5', 'custom',
#                                 path=r'app\weight\yolov5.onnx',
#                                 source='local', verbose=False)
#     yolov8 = YOLO("app\weight\yolov8.pt")
#     yolov8onnx = YOLO("app\weight\yolov8.onnx", task="detect")


@app.get("/")
def root():
    return {"API Ready": "yeah"}


class SupportedModelType(str, Enum):
    yolov5 = "yolov5"
    yolov5_onnx = "yolov5_onnx"
    yolov8 = "yolov8"
    yolov8_onnx = "yolov8_onnx"


@app.post("/yolov5", tags=["Predict How Many Trees in the picture"])
async def useyolov5(image_file: UploadFile = File()):
    check_file_type(image_file)

    uniqie_id = str(uuid.uuid1())
    output_path = settings.OUTPUT_DIR + '/' + uniqie_id

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Save the image to the output folder
    image_path = output_path + '/' + 'original.jpg'
    with open(image_path, "wb") as buffer:
        buffer.write(await image_file.read())

    try:
        result = ml_models['yolov5'](image_path)
        img_result = create_grid(image_path, result)
        cv2.imwrite(f"{output_path}/result.jpg", img_result)
        time = {
            "pre-process": result.t[0],
            "interface": result.t[1],
            "NMS": result.t[2]
        }

        resultData = result.pandas().xyxy[0].value_counts('name').to_dict()

        # Save the result to the database
        returniddb = insert_result(uniqie_id, "yolov5", time, resultData)

        if not returniddb.inserted_id:
            raise HTTPException(
                status_code=500, detail="Error while saving the result to the database")

        return {
            "status": "success",
            "message": f"you can see the result in http://localhost:8000/get_result/{uniqie_id}",
            "data": {
                "id": str(returniddb.inserted_id),
                "model": "yolov5",
                "speed": time,
                "result": resultData,
            }}
    except Exception as e:
        # Delete the output folder if there is an error
        shutil.rmtree(output_path)
        return {"status": "error", "message": str(e)}


@app.post("/yolov5onnx", tags=["Predict How Many Trees in the picture"])
async def useyolov5onnx(image_file: UploadFile = File()):
    check_file_type(image_file)

    uniqie_id = str(uuid.uuid1())
    output_path = settings.OUTPUT_DIR + '/' + uniqie_id

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Save the image to the output folder
    image_path = output_path + '/' + 'original.jpg'
    with open(image_path, "wb") as buffer:
        buffer.write(await image_file.read())

    try:
        result = ml_models['yolov5onnx'](image_path)
        img_result = create_grid(image_path, result)
        cv2.imwrite(f"{output_path}/result.jpg", img_result)
        time = {
            "pre-process": result.t[0],
            "interface": result.t[1],
            "NMS": result.t[2]
        }

        resultData = result.pandas().xyxy[0].value_counts('name').to_dict()

        # Save the result to the database
        returniddb = insert_result(uniqie_id, "yolov5", time, resultData)

        if not returniddb.inserted_id:
            raise HTTPException(
                status_code=500, detail="Error while saving the result to the database")

        return {
            "status": "success",
            "message": f"you can see the result in http://localhost:8000/get_result/{uniqie_id}",
            "data": {
                "id": str(returniddb.inserted_id),
                "model": "yolov5onxx",
                "speed": time,
                "result": resultData,
            }}
    except Exception as e:
        # Delete the output folder if there is an error
        shutil.rmtree(output_path)
        return {"status": "error", "message": str(e)}


@app.post("/yolov8", tags=["Predict How Many Trees in the picture"])
async def useyolov8(image_file: UploadFile = File()):
    check_file_type(image_file)

    uniqie_id = str(uuid.uuid1())
    output_path = settings.OUTPUT_DIR + '/' + uniqie_id

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Save the image to the output folder
    image_path = output_path + '/' + 'original.jpg'
    with open(image_path, "wb") as buffer:
        buffer.write(await image_file.read())

    try:
        result = ml_models['yolov8'](image_path)
        cv2.imwrite(f'{output_path}/result.jpg', result[0].plot())
        result = result[0]

        resultData = {}
        if result.boxes:
            for c in result.boxes.cls.unique():
                n = (result.boxes.cls == c).sum()
                resultData[result.names[int(c)]] = int(n)

        time = {
            "pre-process": result.speed['preprocess'],
            "inference": result.speed['inference'],
            "post-process": result.speed['postprocess']
        }

        returniddb = insert_result(uniqie_id, "yolov8", time, resultData)

        if not returniddb.inserted_id:
            raise HTTPException(
                status_code=500, detail="Error while saving the result to the database")

        return {
            "status": "success",
            "message": f"you can see the result in http://localhost:8000/get_result/{uniqie_id}",
            "data": {
                "id": str(returniddb.inserted_id),
                "model": "yolov8",
                "speed": time,
                "result": resultData,
            }}
    except Exception as e:
        # Delete the output folder if there is an error
        shutil.rmtree(output_path)
        return {"status": "error", "message": str(e)}


@app.post("/yolov8onnx", tags=["Predict How Many Trees in the picture"])
async def useyolov8onnx(image_file: UploadFile = File()):
    check_file_type(image_file)

    uniqie_id = str(uuid.uuid1())
    output_path = settings.OUTPUT_DIR + '/' + uniqie_id

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Save the image to the output folder
    image_path = output_path + '/' + 'original.jpg'
    with open(image_path, "wb") as buffer:
        buffer.write(await image_file.read())

    try:
        result = ml_models['yolov8onnx'](image_path)
        cv2.imwrite(f'{output_path}/result.jpg', result[0].plot())
        result = result[0]

        resultData = {}
        if result.boxes:
            for c in result.boxes.cls.unique():
                n = (result.boxes.cls == c).sum()
                resultData[result.names[int(c)]] = int(n)

        time = {
            "pre-process": result.speed['preprocess'],
            "inference": result.speed['inference'],
            "post-process": result.speed['postprocess']
        }

        returniddb = insert_result(uniqie_id, "yolov8", time, resultData)

        if not returniddb.inserted_id:
            raise HTTPException(
                status_code=500, detail="Error while saving the result to the database")

        return {
            "status": "success",
            "message": f"you can see the result in http://localhost:8000/get_result/{uniqie_id}",
            "data": {
                "id": str(returniddb.inserted_id),
                "model": "yolov8",
                "speed": time,
                "result": resultData,
            }}
    except Exception as e:
        # Delete the output folder if there is an error
        shutil.rmtree(output_path)
        return {"status": "error", "message": str(e)}


@app.post("/upload_model",
          description=f"Upload model to server",
          tags=["Model Management"])
async def upload_model(
    model_file: Annotated[UploadFile, File()],
    model_description: Annotated[str, Form()],
    model_type: Annotated[SupportedModelType, Form()]
):
    if model_type is SupportedModelType.yolov5_onnx or model_type is SupportedModelType.yolov8_onnx:
        if not model_file.filename.endswith(".onnx"):
            raise HTTPException(
                status_code=400,
                detail="Model is not ONNX")
    elif model_type is SupportedModelType.yolov5 or model_type is SupportedModelType.yolov8:
        if not model_file.filename.endswith(".pt"):
            raise HTTPException(
                status_code=400,
                detail="Model is not PyTorch")
    else:
        raise HTTPException(
            status_code=400,
            detail="Model is not supported or it's not match the file extension")

    unique_id = str(uuid.uuid1())

    _, extension = os.path.splitext(model_file.filename)

    # Save the image to the output folder
    model_path = settings.MODEL_DIR + '/' + unique_id + extension

    # IF TO LARGE check Reverse Proxy
    # https://fastapi.tiangolo.com/advanced/behind-a-proxy/

    async with aiofiles.open(model_path, 'wb') as f:
        while chunk := await model_file.read(CHUNK_SIZE):
            await f.write(chunk)
    try:
        returniddb = insert_model(
            unique_id, model_description, model_type.value)

        if not returniddb.inserted_id:
            raise HTTPException(
                status_code=500, detail="Error while saving the result to the database")

        return {
            "status": "success",
            "message": "model uploaded",
            "data": {
                "id": str(returniddb.inserted_id),
                "description": model_description,
                "type": model_type.value}
        }
    except Exception as e:
        # remove file if there is an error
        os.remove(model_path)
        return {"status": "error", "message": str(e)}


@app.get("/model",
         description="""
                Get model by id or type
                """,
         tags=["Model Management"])
async def get_model(
        model_id: str = None,
        model_type: Annotated[SupportedModelType, None] = None):

    if model_id is not None and model_type is None:
        model = get_model_by_id(model_id)
        message = "get model by id :" + model_id
    elif model_type:
        model = get_model_by_type(model_type)
        message = "get model by type : " + model_type.value
    else:
        model = get_all_model()
        message = "get all model"

    if model is None:
        raise HTTPException(status_code=200, detail="model not found")

    try:
        return {
            "status": "success",
            "message": message,
            "data": model
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/model",
          tags=["Model Management"])
async def model(model_id: str, input: UploadFile = File()):
    check_file_type(input)
    # Get model type from database
    metadata = get_model_by_id(model_id)

    if metadata is None:
        raise HTTPException(
            status_code=404,
            detail="Model not found in database.")

    type = metadata['model_type']
    if type == SupportedModelType.yolov5_onnx or type == SupportedModelType.yolov8_onnx:
        model_path = settings.MODEL_DIR + '/' + model_id + '.onnx'
    else:
        model_path = settings.MODEL_DIR + '/' + model_id + '.pt'

    if not os.path.isfile(model_path):
        raise HTTPException(
            status_code=500,
            detail="Model not found in server.")

    uniqie_id = str(uuid.uuid1())

    output_path = settings.OUTPUT_DIR + '/' + uniqie_id
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    image_path = output_path + '/' + 'original.jpg'

    with open(image_path, "wb") as buffer:
        buffer.write(await input.read())

    try:
        if type == SupportedModelType.yolov5_onnx or type == SupportedModelType.yolov5:
            model = torch.hub.load('app\models\yolov5', 'custom',
                                   path=model_path,
                                   source='local',
                                   verbose=False,
                                   force_reload=True)
            result = model(image_path)
            img_result = create_grid(image_path, result)
            cv2.imwrite(f"{output_path}/result.jpg", img_result)
            time = {
                "pre-process": result.t[0],
                "interface": result.t[1],
                "NMS": result.t[2]
            }

            resultData = result.pandas().xyxy[0].value_counts('name').to_dict()

        elif type == SupportedModelType.yolov8 or type == SupportedModelType.yolov8_onnx:
            model = YOLO(model_path)
            result = model.predict(source=image_path)
            cv2.imwrite(f'{output_path}/result.jpg', result[0].plot())

            result = result[0]

            resultData = {}
            if result.boxes:
                for c in result.boxes.cls.unique():
                    n = (result.boxes.cls == c).sum()
                    resultData[result.names[int(c)]] = int(n)

            time = {
                "pre-process": result.speed['preprocess'],
                "inference": result.speed['inference'],
                "post-process": result.speed['postprocess']
            }

        returniddb = insert_result(
            uniqie_id, type, time, resultData)

        if not returniddb.inserted_id:
            raise HTTPException(
                status_code=500, detail="Error while saving the result to the database")

        return {
            "status": "success",
            "message": f"you can see the result in http://localhost:8000/get_result/{uniqie_id}",
            "data": {
                "id": str(returniddb.inserted_id),
                "model": type,
                "speed": time,
                "result": resultData,
            }}
    except Exception as e:
        # Delete the output folder if there is an error
        shutil.rmtree(output_path)
        return {"status": "error", "message": str(e)}


@app.put("/model", tags=["Model Management"])
async def update_model(
    model_id: str,
    model_description: Annotated[str, Form()] = None,
    model_type: Annotated[SupportedModelType, Form()] = None,
    model_file: Annotated[UploadFile, File()] = None,
):

    metadata = get_model_by_id(model_id)
    if not model:
        raise HTTPException(
            status_code=404, detail="Model not found in database")

    if model_type:
        if not model_file:
            pass
        elif model_type is SupportedModelType.yolov5_onnx or model_type is SupportedModelType.yolov8_onnx:
            if not model_file.filename.endswith(".onnx"):
                raise HTTPException(
                    status_code=400,
                    detail="Model is not ONNX")
        elif model_type is SupportedModelType.yolov5 or model_type is SupportedModelType.yolov8:
            if not model_file.filename.endswith(".pt"):
                raise HTTPException(
                    status_code=400,
                    detail="Model is not PyTorch")
        else:
            raise HTTPException(
                status_code=400,
                detail="Model is not supported or it's not match the file extension")
    else:
        model_type = metadata['model_type']

    if model_file:
        _, extension = os.path.splitext(model_file.filename)
        model_path = settings.MODEL_DIR + '/' + model_id + extension

        os.remove(model_path)

        async with aiofiles.open(model_path, 'wb') as f:
            while chunk := await model_file.read(CHUNK_SIZE):
                await f.write(chunk)

    if not model_description:
        model_description = metadata['model_description']

    try:
        update_model_by_id(model_id, model_description, model_type)
        if not get_model_by_id(model_id):
            raise HTTPException(
                status_code=500,
                detail="Error while updating the model in database")

        return {
            "status": "success",
            "message": "model updated",
            "data": {
                "id": model_id,
                "description": model_description,
                "type": model_type}
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.delete("/model", tags=["Model Management"])
async def delete_model(model_id: str):

    model = get_model_by_id(model_id)
    if not model:
        raise HTTPException(
            status_code=404, detail="Model not found in database")

    model_type = model['model_type']
    model_path = settings.MODEL_DIR + '/' + model_id + '.pt'
    if model_type == SupportedModelType.yolov5_onnx.value:
        model_path = settings.MODEL_DIR + '/' + model_id + '.onnx'

    if not os.path.isfile(model_path):
        raise HTTPException(
            status_code=500,
            detail="Model File not found")
    try:

        delete_model_by_id(model_id)
        if get_model_by_id(model_id):
            raise HTTPException(
                status_code=500,
                detail="Error while deleting the model in database")

        os.remove(model_path)

        return {
            "status": "success",
            "message": "model deleted",
            "data": model
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/get_image", tags=["Get Result"])
def get_image(unique_id: str):
    if unique_id:
        HTTPException(status_code=404, detail="unique id is required")
    if not os.path.exists(f"{settings.OUTPUT_DIR}/{unique_id}"):
        raise HTTPException(
            status_code=500, detail="file not found. either the unique id is wrong or the model is not yet finished processing the image")
    try:
        file = f"{settings.OUTPUT_DIR}/{unique_id}/result.jpg"
        return FileResponse(file)
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/get_data", tags=["Get Result"])
def get_result(unique_id: str):
    if unique_id:
        HTTPException(status_code=404, detail="unique id is required")
    result = get_result_by_id(unique_id)
    if not result:
        raise HTTPException(
            status_code=500, detail="result not found. either the unique id is wrong or the model is not yet finished processing the image")
    try:
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/result", tags=["Get Result"])
async def result(unique_id: str):
    Response = """
    <!DOCTYPE html>
    <html>
    <head>
    <title>Result</title>
    </head>
    <body>
    <h1>Result</h1>
    <img src="http://localhost:8000/get_image?unique_id={unique_id}" alt="Result">
    <embed src="http://localhost:8000/get_data?unique_id={unique_id}" width="100%" height="100%">
    </body>
    </html>
    """
    return HTMLResponse(content=Response.replace("{unique_id}", unique_id), status_code=200)
