from asyncio import sleep
from .output import Output, Speed, Data
from .model import Model
from .config import settings
from .database import init_db
import beanie
from enum import Enum
from ultralytics import YOLO
import torch
import aiofiles
from contextlib import asynccontextmanager
import os
from app.utils import create_grid
from pydantic import BaseModel
from fastapi.responses import FileResponse, HTMLResponse
import datetime
from datetime import timedelta
import shutil
from typing import Annotated, Optional
import uuid
import cv2
from fastapi import FastAPI, HTTPException, File, Form, UploadFile
import logging

from fastapi.middleware.cors import CORSMiddleware

CHUNK_SIZE = 1024 * 1024

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Memuat Model
    ml_models["yolov5"] = torch.hub.load('app\models\yolov5', 'custom',
                                         path=r'app\weight\yolov5lafin.pt',
                                         source='local', verbose=False)
    ml_models["yolov5onnx"] = torch.hub.load('app\models\yolov5', 'custom',
                                             path=r'app\weight\yolov5lafin.onnx',
                                             source='local', verbose=False)
    ml_models["yolov8"] = YOLO("app\weight\yolov8afin.pt")
    ml_models["yolov8onnx"] = YOLO("app\weight\yolov8afin.onnx", task="detect")
    # Inisialisasi Database
    await init_db()
    yield
    # Hapus Model
    ml_models.clear()


origins = [
    "http://localhost:3000",
    "localhost:3000"
]


app = FastAPI(
    title="Quick Detect",
    description="""
    G64190075.
    Hanya dapat mendeteksi citra.
    Model yang didukung: 
        - yolov5 (pyTorch and ONNX)
        - yolov8 (pyTorch dan ONNX)
    Format Citra yang didukung :
        - JPG atau JPEG
    """,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


def check_file_type(file: UploadFile = File()):
    """Check the file type. The file must be a JPG or JPEG file."""
    if not file.content_type.startswith("image/jpg") and not file.content_type.startswith("image/jpeg"):
        raise HTTPException(
            status_code=400, detail="File must be an image (JPG or JPEG format)")


@app.get("/")
def root():
    return {"API Ready": "yeah"}


class SupportedModelType(str, Enum):
    yolov5 = "yolov5"
    yolov5_onnx = "yolov5_onnx"
    yolov8 = "yolov8"
    yolov8_onnx = "yolov8_onnx"


@app.post("/detect/yolov5", tags=["Predict How Many Trees in the picture"])
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
    except Exception as e:
        shutil.rmtree(output_path)
        raise HTTPException(
            status_code=400, detail=f"Error when running the model.\n {str(e)}")

    img_result = create_grid(image_path, result)
    cv2.imwrite(f"{output_path}/result.jpg", img_result)

    docInsert = Output(
        id=uniqie_id,
        model_id="yolov5 by Afin tachiar",
        model_type="yolov5",
        created_at=datetime.datetime.utcnow(),
        speed=Speed(
            preprocess=result.t[0],
            interface=result.t[1],
            postprocess=result.t[2]
        ),
        result=Data(
            detect=result.pandas().xyxy[0].value_counts('name').to_dict())
    )
    success = await Output.insert_one(docInsert)

    if not success:
        shutil.rmtree(output_path)
        raise HTTPException(
            status_code=500, detail="Error while saving the result to the database")

    return {
        "status": "success",
        "message": f"URI to get the result http://localhost:8000/result?unique_id={success.id}",
        "data": {
            "id": str(success.id),
            "model_id": success.model_id,
            "model_type": success.model_type,
            "created_at": success.created_at,
            "speed": success.speed,
            "result": success.result,
        }}


@app.post("/detect/yolov5onnx", tags=["Predict How Many Trees in the picture"])
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
    except Exception as e:
        shutil.rmtree(output_path)
        raise HTTPException(
            status_code=400, detail=f"Error when running the model.\n {str(e)}")

    img_result = create_grid(image_path, result)
    cv2.imwrite(f"{output_path}/result.jpg", img_result)
    docInsert = Output(
        id=uniqie_id,
        model_id="yolov5 onnx by Afin tachiar",
        model_type="yolov5 onnx",
        created_at=datetime.datetime.utcnow(),
        speed=Speed(
            preprocess=result.t[0],
            interface=result.t[1],
            postprocess=result.t[2]
        ),
        result=Data(
            detect=result.pandas().xyxy[0].value_counts('name').to_dict())
    )
    success = await Output.insert_one(docInsert)
    if not success:
        shutil.rmtree(output_path)
        raise HTTPException(
            status_code=500, detail="Error while saving the result to the database")
    return {
        "status": "success",
        "message": f"URI to get the result http://localhost:8000/result?unique_id={success.id}",
        "data": {
            "id": str(success.id),
            "model_id": success.model_id,
            "model_type": success.model_type,
            "created_at": success.created_at,
            "speed": success.speed,
            "result": success.result,
        }}


@app.post("/detect/yolov8", tags=["Predict How Many Trees in the picture"])
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
    except Exception as e:
        shutil.rmtree(output_path)
        raise HTTPException(
            status_code=400, detail=f"Error when running the model.\n {str(e)}")

    cv2.imwrite(f'{output_path}/result.jpg', result[0].plot())
    result = result[0]
    resultData = {}
    if result.boxes:
        for c in result.boxes.cls.unique():
            n = (result.boxes.cls == c).sum()
            resultData[result.names[int(c)]] = int(n)
    docInsert = Output(
        id=uniqie_id,
        model_id="yolov8 onnx by Afin tachiar",
        model_type="yolov8 onnx",
        created_at=datetime.datetime.utcnow(),
        speed=Speed(
            preprocess=result.speed['preprocess'],
            interface=result.speed['inference'],
            postprocess=result.speed['postprocess']
        ),
        result=Data(detect=resultData)
    )
    success = await Output.insert_one(docInsert)
    if not success:
        shutil.rmtree(output_path)
        raise HTTPException(
            status_code=500, detail="Error while saving the result to the database")
    return {
        "status": "success",
        "message": f"URI to get the result http://localhost:8000/result?unique_id={success.id}",
        "data": {
            "id": str(success.id),
            "model_id": success.model_id,
            "model_type": success.model_type,
            "created_at": success.created_at,
            "speed": success.speed,
            "result": success.result,
        }
    }


@app.post("/detect/yolov8onnx", tags=["Predict How Many Trees in the picture"])
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
    except Exception as e:
        shutil.rmtree(output_path)
        raise HTTPException(
            status_code=400, detail=f"Error when running the model.\n {str(e)}")

    cv2.imwrite(f'{output_path}/result.jpg', result[0].plot())
    result = result[0]
    resultData = {}
    if result.boxes:
        for c in result.boxes.cls.unique():
            n = (result.boxes.cls == c).sum()
            resultData[result.names[int(c)]] = int(n)
    docInsert = Output(
        id=uniqie_id,
        model_id="yolov8 onnx by Afin tachiar",
        model_type="yolov8 onnx",
        created_at=datetime.datetime.utcnow(),
        speed=Speed(
            preprocess=result.speed['preprocess'],
            interface=result.speed['inference'],
            postprocess=result.speed['postprocess']
        ),
        result=Data(detect=resultData)
    )
    success = await Output.insert_one(docInsert)
    if not success:
        shutil.rmtree(output_path)
        raise HTTPException(
            status_code=500, detail="Error while saving the result to the database")
    return {
        "status": "success",
        "message": f"URI to get the result http://localhost:8000/result?unique_id={success.id}",
        "data": {
            "id": str(success.id),
            "model_id": success.model_id,
            "model_type": success.model_type,
            "created_at": success.created_at,
            "speed": success.speed,
            "result": success.result,
        }}


@app.post("/model",
          description=f"Upload model to server",
          tags=["Model Management"])
async def upload_model(
    model_file: Annotated[UploadFile, File()],
    model_description: Annotated[str, Form()],
    model_type: Annotated[SupportedModelType, Form()]
):
    if model_type == SupportedModelType.yolov5_onnx or model_type == SupportedModelType.yolov8_onnx:
        if not model_file.filename.endswith(".onnx"):
            raise HTTPException(
                status_code=400,
                detail="Model is not ONNX")
    elif model_type == SupportedModelType.yolov5 or model_type == SupportedModelType.yolov8:
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

    docInsert = Model(
        id=unique_id,
        description=model_description,
        type=model_type.value,
        created_at=datetime.datetime.utcnow(),
        updated_at=datetime.datetime.utcnow(),
    )
    succces = await Model.insert_one(docInsert)
    if not succces:
        shutil.rmtree(model_path)
        raise HTTPException(
            status_code=500, detail="Error while saving the model metadata to the database")
    return {
        "status": "success",
        "message": "model uploaded",
        "data": {
            "id": str(succces.id),
            "description": succces.description,
            "type": succces.type,
            "created_at": succces.created_at,
        },
    }


@app.get("/model",
         description="""
                Get model by id or type
                """,
         tags=["Model Management"],)
async def get_model(
        model_id: str = None,
        model_type: Annotated[SupportedModelType, None] = None):

    if model_id is not None and model_type is None:
        try:
            uuid_obj = uuid.UUID(model_id, version=1)
        except ValueError:
            raise HTTPException(
                status_code=400, detail="unique id is not valid")
        model = await Model.get(model_id)
        message = "get model by id : " + model_id
    elif model_type:
        model = await Model.find(Model.type == model_type).to_list()
        message = "get model by type : " + model_type.value
    else:
        model = await Model.find({}).to_list()
        message = "get all model"

    return {
        "status": "success",
        "message": message,
        "data": model
    }


@app.post("/detect",
          tags=["Model Management"])
async def model(model_id: str, image_file: UploadFile = File()):

    try:
        uuid_obj = uuid.UUID(model_id, version=1)
    except ValueError:
        raise HTTPException(
            status_code=400, detail="unique id is not valid")

    check_file_type(image_file)
    # Get model type from database
    metadata = await Model.get(model_id)

    if metadata is None:
        raise HTTPException(
            status_code=404,
            detail="Model not found in database.")

    if metadata.type == SupportedModelType.yolov5_onnx or metadata.type == SupportedModelType.yolov8_onnx:
        model_path = settings.MODEL_DIR + '/' + model_id + '.onnx'
    else:
        model_path = settings.MODEL_DIR + '/' + model_id + '.pt'

    if not os.path.isfile(model_path):
        raise HTTPException(
            status_code=404,
            detail="Model not found in server.")

    uniqie_id = str(uuid.uuid1())

    output_path = settings.OUTPUT_DIR + '/' + uniqie_id
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    image_path = output_path + '/' + 'original.jpg'

    with open(image_path, "wb") as buffer:
        buffer.write(await image_file.read())

    try:
        if metadata.type == SupportedModelType.yolov5_onnx or metadata.type == SupportedModelType.yolov5:
            model = torch.hub.load('app\models\yolov5', 'custom',
                                   path=model_path,
                                   source='local',
                                   verbose=False,
                                   force_reload=True)
            result = model(image_path)
            img_result = create_grid(image_path, result)
            cv2.imwrite(f"{output_path}/result.jpg", img_result)
            docInsert = Output(
                id=uniqie_id,
                model_id=str(metadata.id),
                model_type=metadata.type,
                created_at=datetime.datetime.utcnow(),
                speed=Speed(
                    preprocess=result.t[0],
                    interface=result.t[1],
                    postprocess=result.t[2]
                ),
                result=Data(
                    detect=result.pandas().xyxy[0].value_counts('name').to_dict())
            )

        elif metadata.type == SupportedModelType.yolov8 or metadata.type == SupportedModelType.yolov8_onnx:
            model = YOLO(model_path)
            result = model.predict(source=image_path)
            cv2.imwrite(f'{output_path}/result.jpg', result[0].plot())

            result = result[0]

            resultData = {}
            if result.boxes:
                for c in result.boxes.cls.unique():
                    n = (result.boxes.cls == c).sum()
                    resultData[result.names[int(c)]] = int(n)

            docInsert = Output(
                id=uniqie_id,
                model_id=str(metadata.id),
                model_type=metadata.type,
                created_at=datetime.datetime.utcnow(),
                speed=Speed(
                    preprocess=result.speed['preprocess'],
                    interface=result.speed['inference'],
                    postprocess=result.speed['postprocess']
                ),
                result=Data(detect=resultData)
            )

    except Exception as e:
        # Delete the output folder if there is an error
        shutil.rmtree(output_path)
        raise HTTPException(
            status_code=400, detail=f"Error when running the model.\n {str(e)}")

    success = await Output.insert_one(docInsert)
    if not success:
        shutil.rmtree(output_path)
        raise HTTPException(
            status_code=500, detail="Error while saving the result to the database")
    return {
        "status": "success",
        "message": f"URI to get the result http://localhost:8000/result?unique_id={success.id}",
        "data": {
            "id": str(success.id),
            "model_id": success.model_id,
            "model_type": success.model_type,
            "created_at": success.created_at,
            "speed": success.speed,
            "result": success.result,
        }}


@app.put("/model", tags=["Model Management"])
async def update_model(
    model_id: str,
    model_description: Annotated[str, Form()] = None,
    model_type: Annotated[SupportedModelType, Form()] = None,
    model_file: Annotated[UploadFile, File()] = None,
):

    try:
        uuid_obj = uuid.UUID(model_id, version=1)
    except ValueError:
        raise HTTPException(
            status_code=400, detail="unique id is not valid")

    metadata = await Model.get(model_id)
    if not metadata:
        raise HTTPException(
            status_code=404, detail="Model not found in database")

    if model_type is None and model_file is None and model_description is None and metadata is not None:
        raise HTTPException(
            status_code=400,
            detail="Model found in database, But nothing to Update. Please provide at least one field to update.")

    model_path = f"{settings.MODEL_DIR}/{model_id}.pt"
    if metadata.type == SupportedModelType.yolov5_onnx or metadata.type == SupportedModelType.yolov8_onnx:
        model_path = f"{settings.MODEL_DIR}/{model_id}.onnx"

    if not os.path.isfile(model_path) and metadata is not None:
        raise HTTPException(
            status_code=404,
            detail="Model File not found in server, But found in database. Please upload the model file and choose the model type")

    if model_type is not None and model_type != metadata.type:
        if model_file is None:
            raise HTTPException(
                status_code=404,
                detail="Model type is different with the model in the server. Please upload the model file and choose the model type")

        if model_type == SupportedModelType.yolov5_onnx or model_type == SupportedModelType.yolov8_onnx:
            if not model_file.filename.endswith(".onnx"):
                raise HTTPException(
                    status_code=400,
                    detail="Model Uploaded is not ONNX")
        elif model_type == SupportedModelType.yolov5 or model_type == SupportedModelType.yolov8:
            if not model_file.filename.endswith(".pt"):
                raise HTTPException(
                    status_code=400,
                    detail="Model Uploaded is not PyTorch")
        else:
            raise HTTPException(
                status_code=400,
                detail="Model is not supported or it's not match the file extension")

    if model_file is not None and model_type is None:
        raise HTTPException(
            status_code=400,
            detail="Model Uploaded is provided but model type is not. Please choose the model type")

    if model_description is not None:
        metadata.description = model_description

    if model_type is not None and model_file is not None:
        _, extension = os.path.splitext(model_file.filename)

        # remove old model
        if os.path.isfile(model_path):
            os.remove(model_path)

        model_path = settings.MODEL_DIR + '/' + model_id + extension

        # IF TO LARGE check Reverse Proxy
        # https://fastapi.tiangolo.com/advanced/behind-a-proxy/

        async with aiofiles.open(model_path, 'wb') as f:
            while chunk := await model_file.read(CHUNK_SIZE):
                await f.write(chunk)

        metadata.type = model_type.value

    try:
        metadata.updated_at = datetime.datetime.utcnow()
        await metadata.replace()

        return {
            "status": "success",
            "message": f"model updated at {metadata.updated_at}",
            "data": {
                "id": metadata.id,
                "description": metadata.description,
                "type": metadata.type, }
        }
    except (ValueError, beanie.exceptions.DocumentNotFound):
        raise HTTPException(
            status_code=404, detail="Model not found in database")
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.delete("/model", tags=["Model Management"])
async def delete_model(model_id: str):

    try:
        uuid_obj = uuid.UUID(model_id, version=1)
    except ValueError:
        raise HTTPException(
            status_code=400, detail="unique id is not valid")

    metadata = await Model.get(model_id)
    if not metadata:
        raise HTTPException(
            status_code=404, detail="Model not found in database")

    model_path = f"{settings.MODEL_DIR}/{model_id}.pt"
    if metadata.type == SupportedModelType.yolov5_onnx or metadata.type == SupportedModelType.yolov8_onnx:
        model_path = f"{settings.MODEL_DIR}/{model_id}.onnx"

    if not os.path.isfile(model_path):
        raise HTTPException(
            status_code=404,
            detail="Model File not found in the server")
    try:
        await metadata.delete()
        if await Model.get(model_id):
            raise HTTPException(
                status_code=500,
                detail="Error while deleting the model in database")

        os.remove(model_path)

        return {
            "status": "success",
            "message": f"model deleted : " + model_id,
            "data": {
                "id": metadata.id,
                "description": metadata.description,
                "type": metadata.type, }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/result/image", tags=["Get Result"])
async def get_image(unique_id: str):

    try:
        uuid_obj = uuid.UUID(unique_id, version=1)
    except ValueError:
        raise HTTPException(
            status_code=400, detail="unique id is not valid")

    if not os.path.exists(f"{settings.OUTPUT_DIR}/{unique_id}"):
        raise HTTPException(
            status_code=404, detail="file not found. either the unique id is wrong or the model is not yet finished processing the image")
    try:
        file = f"{settings.OUTPUT_DIR}/{unique_id}/result.jpg"
        return FileResponse(file)
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/result/data", tags=["Get Result"])
async def get_result(unique_id: str):

    try:
        uuid_obj = uuid.UUID(unique_id, version=1)
    except ValueError:
        raise HTTPException(
            status_code=400, detail="unique id is not valid")

    result = await Output.get(unique_id)
    if result is None:
        raise HTTPException(
            status_code=404, detail="Result not found in database")

    return {
        "status": "success",
        "data": {
            "id": str(result.id),
            "model_id": result.model_id,
            "model_type": result.model_type,
            "created_at": result.created_at,
            "speed": result.speed,
            "result": result.result,
        }}


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
    <img src="http://localhost:8000/result/image?unique_id={unique_id}" alt="Result">
    <embed src="http://localhost:8000/result/data?unique_id={unique_id}" width="100%" height="100%">
    </body>
    </html>
    """
    return HTMLResponse(content=Response.replace("{unique_id}", unique_id), status_code=200)

# Delete all the output folder and in database


@app.delete("/delete_all", tags=["Delete All"])
async def delete_all():
    try:
        await Output.delete_all()
        shutil.rmtree(settings.OUTPUT_DIR)
        return {
            "status": "success",
            "message": "all result deleted"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
