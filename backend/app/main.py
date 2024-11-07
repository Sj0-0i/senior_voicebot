from fastapi import FastAPI
from routers.conversation import conversation_router
from routers.mark import mark_router

app = FastAPI()

app.include_router(conversation_router, prefix="/conversation")
app.include_router(mark_router, prefix="/mark")

@app.get('/')
def home():
    return 'This is home!'
