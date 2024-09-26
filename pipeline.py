import openai
import os
import shutil
import requests, json
from dotenv import load_dotenv
from uuid import uuid4
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema import Document


load_dotenv()

model = ChatOpenAI(model="gpt-4o")

store = {}
name = "가나다"
age = 70
chroma_db = None
LOCATION = 'Seoul'
API_KEY = os.getenv('OPENWEATHER_API_KEY')
WEATHER_API_URL = f"http://api.openweathermap.org/data/2.5/weather?q={LOCATION}&appid={API_KEY}&lang=kr&units=metric"

class Conversation(BaseModel):
    message: str = Field(description="노인분의 응답에 대한 대화를 이어갈 수 있는 적절한 답변")
    score: int = Field(description="이전 질문에 대한 노인분의 응답이 상식적으로 이어질 만한 응답인지 평가하여 1~10의 숫자로만 표현. 이어지지 않게 대답을 했다면 점수를 낮게 1에 가깝게 줘야 함.")


def get_weather():
    response = requests.get(WEATHER_API_URL)
    if response.status_code == 200:
        data = response.json()
        weather_description = data['weather'][0]['description']
        temperature = data['main']['temp']
        return f"어르신이 지금 계시는 {LOCATION}의 날씨는 {weather_description}이며, 기온은 {temperature}도이다."
    else:
        return "날씨 정보를 불러오는 데 실패했습니다."

def get_session_history(session_ids: str) -> BaseChatMessageHistory:
    print("session_id : " + session_ids)
    if session_ids not in store:  
        store[session_ids] = ChatMessageHistory()
    return store[session_ids] 

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True,
    )
    return text_splitter.split_documents(documents)

def get_chroma_instance(chroma_path):
    global chroma_db
    if chroma_db is None:
        chroma_db = Chroma(persist_directory=chroma_path, embedding_function=OpenAIEmbeddings())
    return chroma_db

def save_to_chroma(chunks, chroma_path):
    uuids = [str(uuid4()) for _ in range(len(chunks))]
    vector_store = get_chroma_instance(chroma_path)
    vector_store.add_documents(documents=chunks, ids=uuids)
    vector_store.update_documents(documents=chunks, ids=uuids)

def query_rag(query_text):
    db = get_chroma_instance("./chroma")
    results = db.similarity_search_with_relevance_scores(query_text, k=2)
    context_text = results
    if len(results) == 0 or results[0][1] < 0.8:
        context_text = ""
    else:
        context_text = results
    return context_text

weather_info = get_weather()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            당신은 유능한 노인복지사입니다.
            당신의 역할은 다음과 같습니다.
            1. 항상 노인의 이름, 나이, 집 위치를 고려하여 대화를 진행해야 합니다. 
            2. 당신은 당신과의 대화를 통해 노인분의 고독함, 외로움을 사라지게 하는 것을 목표로 하여 노인분과 대화를 지속해야 합니다.
            3. 대화를 하면서 노인분의 일상 정보를 알아내야 합니다.
            4. 오늘의 날씨 정보를 참고하여 대화를 진행해도 좋습니다. 
            5. Context를 참고해서 사용자와 대화해주세요.
            
            노인 정보는 다음과 같습니다. 
            이름은 {name}이고, 나이는 {age}살이며, 집의 위치는 {LOCATION}임.

            오늘 날씨 정보는 다음과 같습니다. 
            {weather}
            
            이제 노인과 대화하기에 괜찮은 발화문 5가지를 생각해서 랜덤하게 하나를 뽑아 대화를 시작해주세요. 
            출력 형태는 반드시 다음과 같아야 합니다.
            {{
                "message": "생성한 대화 시작 발화문"
            }}
            """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

runnable = prompt | model

with_message_history = (
    RunnableWithMessageHistory(  
        runnable,  
        get_session_history,  
        input_messages_key="input",  
        history_messages_key="history",  
    )
)

response = with_message_history.invoke(
    {
        "age": age, "name": name, "LOCATION": LOCATION, 
        "weather": weather_info, "input": ""
    },
    config={"configurable": {"session_id": name + str(age) + LOCATION}},
)
print(response.content)



prompt2 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            [Context]를 참고해서 사용자와 대화해주세요.
            반드시 노인분의 응답에 대해 자연스럽게 대화를 이어가야 합니다.

            [Context]
            {context}
    
            출력 형태는 반드시 다음과 같아야 합니다.
            {format_instructions}
            """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

output_parser = JsonOutputParser(pydantic_object=Conversation)
runnable2 = prompt2 | model | output_parser

with_message_history = (
    RunnableWithMessageHistory(
        runnable2,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
)

while True:
    input_str = input("맥락에 안 맞을 시 종료:")

    if input_str == '그만':
        break

    str_context = query_rag(input_str)
    print("context: \n")
    for i in range(len(str_context)):
        print(str_context[i])

    response = with_message_history.invoke(
        {
            "input": input_str, 
            "context": str_context,
            "format_instructions": output_parser.get_format_instructions(),
        },
        config={"configurable": {"session_id": name + str(age) + LOCATION}},
    )

    response_dict = response.content
    print("response: \n" + response.content)

    message = response_dict.get('message')
    score = response_dict.get('score')

    if score <= 5:
        print(f"Score가 {score}로 낮아서 대화를 종료합니다.")
        break


prompt3 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            대화가 종료됐습니다.
            대화 내용을 분석하여 어떤 일이 있었는지 중요한 일들을 개행식으로 요약해주세요.
            또한 노인분의 정보가 파악되면 작은 따옴표 안에 단어로 정리해서 보여주세요.
            출력 형태는 반드시 다음과 같아야 합니다. 
            {{
                "summary": "대화 내용 요약 정리",
                "info": "노인분의 정보를 정리한 것"
            }}

            예시:
            {{
                "summary": "허리를 다쳐 병원을 갔다 오셨다. 다음 주 목요일에 다시 병원을 가기로 했다.",
                "info": "병원 가는 길에 있는 카페에 가는 것을 좋아한다. 집 앞 산책로 걷는 것을 좋아한다. 토마토를 좋아한다. 옥수수를 좋아한다. 버섯을 싫어한다.
            }}
            """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

runnable3 = prompt3 | model

with_message_history = (
    RunnableWithMessageHistory(
        runnable3,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
)

response = with_message_history.invoke(
    {"input": ""},
    config={"configurable": {"session_id": name + str(age) + LOCATION}},
)
print(response.content)

document = Document(page_content=response.content)
chunks = split_text([document])
save_to_chroma(chunks, "./chroma")
