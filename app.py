import openai
import os
import shutil
import pickle
import requests, json
import pymysql
from dotenv import load_dotenv
from uuid import uuid4
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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
from flask import Flask, request, jsonify


app = Flask(__name__)

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
password = os.getenv('MYSQL')
API_KEY = os.getenv('OPENWEATHER_API_KEY')
model = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key)

store = {}
chroma_db = None


def get_weather(weather_api_url, location):
    response = requests.get(weather_api_url)
    if response.status_code == 200:
        data = response.json()
        weather_description = data['weather'][0]['description']
        temperature = data['main']['temp']
        return f"어르신이 지금 계시는 {location}의 날씨는 {weather_description}이며, 기온은 {temperature}도이다."
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
    db = get_chroma_instance("./db/chroma")
    results = db.similarity_search_with_relevance_scores(query_text, k=2)
    context_text = results
    if len(results) == 0 or results[0][1] < 0.8:
        context_text = ""
    else:
        context_text = results
    return context_text

@app.route('/')
def home():
    return 'This is home!'

@app.route('/question/<string:user_id>', methods=['GET'])
def generate_question(user_id):
    conn = pymysql.connect(host='localhost', user='root', password=password, db='chatbot', charset='utf8')

    sql1 = """
        SELECT q.question_id, q.question_text 
        FROM Questions q 
        LEFT JOIN UserQuestions uq 
        ON q.question_id = uq.question_id AND uq.user_id = %s
        WHERE uq.question_id IS NULL 
        ORDER BY RAND() 
        LIMIT 1;
        """
    try:
        with conn.cursor() as cur:
            cur.execute(sql1, (user_id,))
            question = cur.fetchone()
            if question is None:
                return jsonify({"status": "error", "message": "No more questions available for this user"}), 404
            return jsonify({"question_id": question[0], "question_text": question[1]}), 200

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        conn.close()


@app.route('/question/<int:question_id>/<string:user_id>', methods=['POST'])
def mark_question(question_id, user_id):
    conn = pymysql.connect(host='localhost', user='root', password=password, db='chatbot', charset='utf8')

    sql2 = """
        INSERT INTO UserQuestions (user_id, question_id, asked_at)
        VALUES (%s, %s, NOW())
        ON DUPLICATE KEY UPDATE asked_at = NOW();
        """
    try:
        with conn.cursor() as cur:
            record = (user_id, question_id)
            cur.execute(sql2, record)
            conn.commit()
            print(f"Question {question_id} asked to user {user_id}")
            return jsonify({"status": "success", "message": f"질문 {question_id}번을 {user_id}에게 물어봄."}), 200
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        conn.close()


@app.route('/conversation-first', methods=['POST'])
def conversation_first():
    data = request.json
    question = data.get('question')
    name = data.get('name')  
    age = data.get('age')    
    location = data.get('location')
    WEATHER_API_URL = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={API_KEY}&lang=kr&units=metric"

    weather_info = get_weather(WEATHER_API_URL, location)

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
                이름은 {name}이고, 나이는 {age}살이며, 집의 위치는 {location}임.

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
            "age": age, "name": name, "location": location, 
            "weather": weather_info, "question": question, "input": ""
        },
        config={"configurable": {"session_id": name + str(age) + str(location)}},
    )
    print(response.content)
    return jsonify({"message": response.content}), 200


@app.route('/conversation-second', methods=['POST'])
def conversation_second():
    data = request.json
    answer = data.get('answer')
    name = data.get('name')  
    age = data.get('age')    
    location = data.get('location')

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

    str_context = query_rag(answer)
    print("context: \n")
    for i in range(len(str_context)):
        print(str_context[i])

    response = with_message_history.invoke(
        {
            "input": answer, 
            "context": str_context,
        },
        config={"configurable": {"session_id": name + str(age) + location}},
    )

    response_json = json.loads(response.content)
    message = response_json.get('message')
    score = response_json.get('score')

    print(message)
    print(score)

    if int(score) <= 5:
        print(f"Score가 {score}로 낮아서 대화를 종료합니다.")
        return jsonify({"message": f"Score가 {score}로 낮아서 대화를 종료합니다.", "score": score}), 200
    
    return jsonify({"message": message, "score": score}), 200


@app.route('/conversation-final', methods=['POST'])
def conversation_final():
    data = request.json
    name = data.get('name')  
    age = data.get('age')    
    location = data.get('location')

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
        config={"configurable": {"session_id": name + str(age) + location}},
    )
    print(response.content)
    cleaned_text = response.content.replace("{", "").replace("}", "")
    print(store[name + str(age) + location])

    document = Document(page_content=cleaned_text)
    chunks = split_text([document])

    for chunk in chunks:
        print(chunk)

    save_to_chroma(chunks, "./db/chroma")
    return jsonify({"message": "대화 분석 완료 및 저장됨"}), 200