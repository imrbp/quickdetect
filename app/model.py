from beanie import Document
from uuid import UUID, uuid1

from datetime import datetime
from pydantic import Field


class Model(Document):
    id: UUID = Field(default_factory=uuid1)
    description: str
    type: str
    created_at: datetime
    updated_at: datetime

    class Settings:
        name = "model"
