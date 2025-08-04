from pydantic import BaseModel, Field

class AnswerOutput(BaseModel):
    answer: str = Field(description="The answer to the question.")
    reason: str = Field(description="The reason for the answer.")