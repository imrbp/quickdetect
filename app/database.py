from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient

from .model import Model
from .output import Output
from .config import settings

if settings.MONGO_INITDB_ROOT_USERNAME and settings.MONGO_INITDB_ROOT_PASSWORD:
    uri = f"mongodb://{settings.MONGO_INITDB_ROOT_USERNAME}:{settings.MONGO_INITDB_ROOT_PASSWORD}@localhost:27017"
else:
    uri = f"mongodb://localhost:27017"


async def init_db():
    # Create Motor client
    client = AsyncIOMotorClient(
        uri
    )

    # Initialize beanie with the Product document class and a database
    await init_beanie(database=client.dbquickdetect, document_models=[Model, Output])
    print('🚀 Connected to MongoDB...')
