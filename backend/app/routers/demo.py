from fastapi import APIRouter, HTTPException
from services.conversation_service import get_summary
from services.demo_service import parse_memory_txt_file, get_user_interests
from models.demo import DemoResponse

demo_router = APIRouter()

@demo_router.get("/{user_id}", response_model=DemoResponse)
async def demo(user_id: str):
    try:
        summaries = get_summary(user_id)
        interests = await get_user_interests(user_id)
        memories = parse_memory_txt_file("./data/90Incheon.txt")
        return {
            "summary": [chunk for chunk in summaries],
            "interests": interests,
            "memory": memories
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))