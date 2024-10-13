from langchain_openai import ChatOpenAI
from core import config

class OpenAIModelManager:
    _instances = {}

    @classmethod
    def get_model(cls, user_id):
        if user_id not in cls._instances:
            cls._instances[user_id] = ChatOpenAI(
                model="gpt-4o",
                temperature=0,
                openai_api_key=config.openai_api_key
            )
        return cls._instances[user_id]

    @classmethod
    def clear_model(cls, user_id):
        if user_id in cls._instances:
            del cls._instances[user_id]
