from pydantic import BaseSettings
import os


class Settings(BaseSettings):
    MONGO_INITDB_ROOT_USERNAME: str
    MONGO_INITDB_ROOT_PASSWORD: str
    MONGO_INITDB_DATABASE: str

    OUTPUT_DIR: str
    MODEL_DIR: str

    class Config:
        env_file = os.path.join(os.getcwd(), '.env')


settings = Settings()
