from fastapi import APIRouter, HTTPException, BackgroundTasks
from models.user import UserInput, AnswerInput
from services.conversation_service import process_first_conversation, process_second_conversation, finalize_conversation

conversation_router = APIRouter()

@conversation_router.post('/first')
async def conversation_first(user_input: UserInput):
    try:
        response = await process_first_conversation(user_input)
        return {"status": "success", **response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@conversation_router.post('/second')
async def conversation_second(answer_input: AnswerInput, background_tasks: BackgroundTasks):
    try:
        response = await process_second_conversation(answer_input, background_tasks)
        return {"status": "success", **response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))