from fastapi import APIRouter, HTTPException, BackgroundTasks
from models.mark import QuestionInput, InterestInput
from services.question_service import mark_question
from services.interest_service import mark_interest

mark_router = APIRouter()

@mark_router.post('/question')
async def question(question_input: QuestionInput):
    try:
        response = await mark_question(question_input)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@mark_router.post('/interest')
async def interest(interset_input: InterestInput):
    try:
        response = await mark_interest(interset_input.interest_id)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))