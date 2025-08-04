from pydantic import BaseModel, Field


class Input(BaseModel):
    question: str = Field(..., description="The question to be answered.")