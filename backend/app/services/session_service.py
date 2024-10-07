from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

store = {}

def get_history(session_ids: str) -> BaseChatMessageHistory:
    print("session_id : " + session_ids)
    if session_ids not in store:  
        store[session_ids] = ChatMessageHistory()
    return store[session_ids] 

def clear_history(session_id: str):
    if session_id in store:
        store[session_id].clear()  
        print(f"session {session_id} 대화 내역 초기화")
    else:
        print(f"{session_id}의 session이 존재하지 않음.")
