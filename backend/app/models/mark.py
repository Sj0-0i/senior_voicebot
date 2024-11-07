from pydantic import BaseModel

class QuestionInput(BaseModel):
    user_id: str
    question_id: str

class InterestInput(BaseModel):
    interest_id: str