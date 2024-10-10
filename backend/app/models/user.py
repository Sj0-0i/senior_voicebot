from pydantic import BaseModel

class UserInput(BaseModel):
    user_id: str

class AnswerInput(BaseModel):
    user_id: str
    answer: str
