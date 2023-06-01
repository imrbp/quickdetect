import datetime
from .config import settings
from pymongo.mongo_client import MongoClient

if settings.MONGO_INITDB_ROOT_USERNAME and settings.MONGO_INITDB_ROOT_PASSWORD:
    uri = f"mongodb://{settings.MONGO_INITDB_ROOT_USERNAME}:{settings.MONGO_INITDB_ROOT_PASSWORD}@localhost:27017"
else:
    uri = f"mongodb://localhost:27017"

client = MongoClient(uri)
db = client[settings.MONGO_INITDB_DATABASE]

print('🚀 Connected to MongoDB...')

Model = db['model']
Output = db['output']


def insert_result(uniqie_id: str, version: str, time: dict, result: dict):
    return Output.insert_one({
        "_id": uniqie_id,
        "version": version,
        "created_at": datetime.datetime.utcnow(),
        "speed": time,
        "data": result
    })


def get_result_by_id(uuid: str):
    return Output.find_one({"_id": uuid})


def insert_model(uniqie_id: str, model_description: str, model_type: str):
    return Model.insert_one({
        "_id": uniqie_id,
        "model_description": model_description,
        "model_type": model_type,
        "created_at": datetime.datetime.utcnow(),
        "updated_at": datetime.datetime.utcnow(),
    })


def get_all_model():
    models = []
    for model in Model.find():
        models.append(model)
    return models


def get_model_by_type(model_type: str):
    models = []
    for model in Model.find({"model_type": model_type}):
        models.append(model)
    return models


def get_model_by_id(model_id: str):
    return Model.find_one({"_id": model_id})


def update_model_by_id(model_id: str, model_description: str, model_type: str):
    # check if model is updated
    metadata = Model.find_one({"_id": model_id})
    Model.update_one({"_id": model_id}, {
        "$set": {
            "model_description": model_description,
            "model_type": model_type,
            "updated_at": datetime.datetime.utcnow(),
        }
    })
    if metadata is None:
        return None
    else:
        if metadata['model_description'] == model_description and metadata['model_type'] == model_type:
            return None
        else:
            return Model.find_one({"_id": model_id})


def delete_model_by_id(model_id: str):
    return Model.delete_one({"_id": model_id})


def delete_all_model():
    return Model.delete_many({})


def delete_all_output():
    return Output.delete_many({})
