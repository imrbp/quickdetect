\
from pydantic import BaseModel

from beanie import Document
from uuid import UUID, uuid1

from datetime import datetime
from pydantic import Field


class Data(BaseModel):
    detect: dict


class Speed(BaseModel):
    preprocess: float
    interface: float
    postprocess: float


class Output(Document):
    id: UUID = Field(default_factory=uuid1)
    model_id: str
    model_type: str
    created_at: datetime
    speed: Speed
    result: Data

    class Settings:
        name = "output"
