import openai
import os
import shutil
import pickle
import requests, json
import pymysql
from dotenv import load_dotenv
from uuid import uuid4
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.docstore.in_memory import InMemoryDocstore
from sentence_transformers import SentenceTransformer
from langchain.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.utils import filter_complex_metadata


load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
password = os.getenv('MYSQL')
model = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key)

store = {}
name = "가나다"
age = 70
LOCATION = 'Seoul'
API_KEY = os.getenv('OPENWEATHER_API_KEY')
WEATHER_API_URL = f"http://api.openweathermap.org/data/2.5/weather?q={LOCATION}&appid={API_KEY}&lang=kr&units=metric"


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
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=10,
        chunk_overlap=0,
        length_function=len,
        add_start_index=True,
    )
    split_docs = text_splitter.split_documents(documents)
    return split_docs

def query_ensemble(query_text, data_path):
    document_loader = TextLoader(data_path, encoding='UTF8')
    pages = document_loader.load()
    docs = split_text(pages)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 2

    chroma_vectorstore = Chroma.from_documents(docs, embeddings)
    chroma_retriever = chroma_vectorstore.as_retriever(search_kwargs={'k': 2})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.5, 0.5],
    )

    results = ensemble_retriever.invoke(query_text)
    return results    

def save_chunks_to_file(chunks, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for chunk in chunks:
            content = chunk.page_content
            file.write(content + '\n')

conn = pymysql.connect(host='localhost', user='root', password=password, db='chatbot', charset='utf8')

user_id = "abcdef"
sql1 = """
    SELECT q.question_id, q.question_text 
    FROM Questions q 
    LEFT JOIN UserQuestions uq 
    ON q.question_id = uq.question_id AND uq.user_id = %s
    WHERE uq.question_id IS NULL 
    ORDER BY RAND() 
    LIMIT 1;
    """
with conn:
    with conn.cursor() as cur:
        cur.execute(sql1, (user_id,))
        question = cur.fetchone()
        print(question)

### 대화가 잘 진행되면 UserQuestions 테이블에 저장.
sql2 = """
    INSERT INTO UserQuestions (user_id, question_id, asked_at)
    VALUES (?, ?, NOW())
    ON DUPLICATE KEY UPDATE asked_at = NOW();
    """
# record = ("abcdef", 1)
# cursor.execute(sql2, record)


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
            
            question을 대화 시작문으로 사용하기에 더 적합한 문장으로 바꿔주세요.
            {question}

            출력 형태는 반드시 다음과 같아야 합니다.
            {{
                "message": "question을 적합하게 바꾼 문장"
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
        "weather": weather_info, "question": question, "input": ""
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
            {{
                "message": "노인분의 응답에 대한 대화를 이어갈 수 있는 적절한 답변",
                "score": "이전 질문에 대한 내용에 상식적으로 이어질만한 대답을 했는지 정도를 1~10의 숫자로만 표현, 이전 질문에 이어지지 않는 내용의 대답을 했다면 점수를 낮게 1~5 사이로 줘야 함. 다른 표현 하지 않음."
            }}
            """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

runnable2 = prompt2 | model 

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
    
    path = str(age) + LOCATION
    str_context = query_ensemble(input_str, f"./data/{path}.txt")
    print("context: \n")
    for i in range(len(str_context)):
        print(str_context[i])

    response = with_message_history.invoke(
        {
            "input": input_str, 
            "context": str_context,
        },
        config={"configurable": {"session_id": name + str(age) + LOCATION}},
    )

    response_json = json.loads(response.content)
    message = response_json.get('message')
    score = response_json.get('score')

    print(message)
    print(score)

    if int(score) <= 5:
        print(f"Score가 {score}로 낮아서 대화를 종료합니다.")
        break


prompt3 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            대화가 종료됐습니다.
            대화 내용을 분석하여 다음 대화를 위해 기억하고 있어야 하는 중요한 정보를 요약해주세요. 
            요약을 완료하고 나면, 대화 주제로 사용할 수 있는 노인분의 취미 등의 키워드를 리스트로 정리해서 보여주세요.
            출력 형태는 반드시 다음과 같아야 합니다. 
            {{
                기억해야 할 요약 내용\n
                기억해야 할 요약 내용2\n
                ...\n
                ["노인분의 취미 키워드1", "노인분의 취미 키워드2", ...]
            }}

            예시:
            {{
                허리를 다쳐 병원을 갔다 왔다.\n
                다음 주 목요일에 다시 병원을 가기로 했다\n
                병원 가는 길에 있는 카페에 가는 것을 좋아한다\n
                집 앞 산책로 걷는 것을 좋아한다\n
                토마토를 좋아한다\n
                옥수수를 좋아한다\n
                버섯을 싫어한다\n
                ["산책", "카페"]
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
cleaned_text = response.content.replace("{", "").replace("}", "")
print(store[name + str(age) + LOCATION])

document = Document(page_content=cleaned_text)
chunks = split_text([document])

for chunk in chunks:
    print(chunk)
    
save_chunks_to_file(chunks, "./data/" + str(age) + LOCATION + ".txt")