import openai
import os
import requests, json
import pymysql
import binascii
import copy
import datetime
from dotenv import load_dotenv
from uuid import uuid4
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import Document
from flask import Flask, request, jsonify, session
from Exception.UserNotFoundError import UserNotFoundError


app = Flask(__name__)

app.secret_key = binascii.hexlify(os.urandom(24)).decode()

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
password = os.getenv('MYSQL')
API_KEY = os.getenv('OPENWEATHER_API_KEY')
model = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key)

store = {}
chroma_db = None

def get_weather(location):
    weather_api_url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={API_KEY}&lang=kr&units=metric"
    try:
        response = requests.get(weather_api_url)
        response.raise_for_status()  # 오류 발생 시 예외 처리
        data = response.json()
        weather_description = data['weather'][0]['description']
        temperature = data['main']['temp']
        return f"어르신이 지금 계시는 {location}의 날씨는 {weather_description}이며, 기온은 {temperature}도입니다."
    except requests.exceptions.RequestException as e:
        return f"날씨 정보를 불러오는 데 실패했습니다. 오류: {e}"

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

def query_ensemble(query_text, data_path):
    document_loader = TextLoader(data_path, encoding='UTF8')
    pages = document_loader.load()
    docs = split_text(pages)

    embeddings = OpenAIEmbeddings()

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 2

    chroma_vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./db/chroma")
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

def conversation_final():
    name = session.get('name')
    age = session.get('age')   
    location = session.get('location')

    session_id = name + str(age) + location
    original_history = get_session_history(session_id)
    history_copy = copy.deepcopy(original_history)

    prompt3 = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                대화가 종료됐습니다.
                대화 내용을 분석하여 다음 대화를 위해 기억하고 있어야 하는 중요한 정보를 요약해주세요.
                요약문 자체는 AI의 질문에 따른 Human의 응답이 자연스럽게 이루어진 내용에 대해서만 요약합니다.
                시간적 표현을 나타내는 표현이 나올 경우, 반드시 아래에 주어지는 현재 시간 정보를 토대로 년,월,일 정보로 바꾸어 기록합니다. (예 : 오늘 -> OO년 O월 O일)
                현재 시간 정보 : {current_time}

                출력 형태는 반드시 다음과 같아야 합니다. 
                {{
                    기억해야 할 요약 내용\n
                    기억해야 할 요약 내용2\n
                    ...\n
                }}

                예시:
                {{
                    허리를 다쳐 병원을 갔다 왔다.\n
                    2024년 10월 08일에 다시 병원을 가기로 했다.\n
                    병원 가는 길에 있는 카페에 가는 것을 좋아한다.\n
                    집 앞 산책로 걷는 것을 좋아한다.\n
                    토마토를 좋아한다.\n
                    옥수수를 좋아한다.\n
                    버섯을 싫어한다.\n
                }}
                """,
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    messages = prompt3.format_messages(input="", history=history_copy.messages, current_time=datetime.datetime.now().strftime('%Y-%m-%d'))

    response = model.invoke(messages)

    print("\n요약문 : ")
    print(response.content)
    cleaned_text = response.content.replace("{", "").replace("}", "")

    document = Document(page_content=cleaned_text)
    chunks = split_text([document])
    chunks.extend

    for chunk in chunks:
        print(chunk)

    save_chunks_to_file(chunks, f"./data/{age}{location}.txt")

    prompt4 = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                대화가 종료됐습니다.
                대화 내용을 분석하여 차후 대화에 참고할 수 있을만한 AI와 Human의 주된 대화 대주제만을 키워드 형태의 list형식으로 반환하세요.
                키워드는 사전적 정의가 되는 일반명사에 속하는 것으로만 작성합니다.
                특히 고유명사에 속하는 것들을 제외합니다.
                또한 작성한 키워드 내에서 상위 개념이 존재하는 키워드는 제외합니다.
                예를 들면, [줄넘기, 운동] 이 있다면 줄넘기는 운동에 속하는 키워드이기 때문에 제거됩니다.

                출력 형태는 반드시 다음과 같아야 합니다. 
                {{
                    ["대화 주제1", "대화 주제2", ...]
                }}

                예시:
                {{
                    ["카페", "뉴스", "부동산", ...]
                }}
                """,
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    messages = prompt4.format_messages(input="", history=history_copy.messages)

    response = model.invoke(messages)

    print("\n관심사 키워드 : ")
    print(response.content)



def get_user_info(user_id):
    conn = pymysql.connect(host='localhost', user='root', password=password, db='chatbot', charset='utf8')

    sql = """
    SELECT user_name, age, location
    FROM Users
    WHERE user_id = %s;
    """

    try:
        with conn.cursor() as cur:
            cur.execute(sql, (user_id,))
            user_info = cur.fetchone() 
            if user_info:
                return {
                    "name": user_info[0],
                    "age": user_info[1],
                    "location": user_info[2]
                }
            else:
                raise UserNotFoundError(f"존재하지 않는 user_id: {user_id}")
    except Exception as e:
        print(f"Error: {str(e)}")
        return e
    finally:
        conn.close()


@app.route('/')
def home():
    return 'This is home!'

@app.route('/users', methods=['POST'])
def generate_session():
    data = request.json
    user_id = data.get('user_id')

    try:
        user_info = get_user_info(user_id)
        session['user_id'] = user_id
        session['name'] = user_info['name']
        session['age'] = user_info['age']
        session['location'] = user_info['location']

        return jsonify({"message": "세션 생성 완료"}), 200
    except UserNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": "서버 오류가 발생했습니다."}), 500


@app.route('/question', methods=['GET'])
def generate_question():
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
            cur.execute(sql1, (session.get('user_id'),))
            question = cur.fetchone()
            if question is None:
                return jsonify({"status": "error", "message": "No more questions available for this user"}), 404
            return jsonify({"question_id": question[0], "question_text": question[1]}), 200

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        conn.close()


@app.route('/question/<int:question_id>', methods=['POST'])
def mark_question(question_id):
    conn = pymysql.connect(host='localhost', user='root', password=password, db='chatbot', charset='utf8')

    sql2 = """
        INSERT INTO UserQuestions (user_id, question_id, asked_at)
        VALUES (%s, %s, NOW())
        ON DUPLICATE KEY UPDATE asked_at = NOW();
        """
    try:
        with conn.cursor() as cur:
            record = (session.get('user_id'), question_id)
            cur.execute(sql2, record)
            conn.commit()
            print(f"Question {question_id} asked to user {session.get('name')}")
            return jsonify({"status": "success", "message": f"질문 {question_id}번을 {session.get('name')}에게 물어봄."}), 200
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        conn.close()


@app.route('/conversation-first', methods=['POST'])
def conversation_first():
    data = request.json
    question = data.get('question')
    name = session.get('name')
    age = session.get('age')   
    location = session.get('location')

    weather_info = get_weather(location)

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
        config={"configurable": {"session_id": name + str(age) + location}},
    )
    print(response.content)
    return jsonify({"message": response.content}), 200


@app.route('/conversation-second', methods=['POST'])
def conversation_second():
    data = request.json
    answer = data.get('answer')
    name = session.get('name')
    age = session.get('age')   
    location = session.get('location')

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
                    "score": "Human의 응답이 이전 message 질문의 문맥에 맞게 대답을 했는지의 정도를 0~10의 숫자로만 평가(Higher is better), 어르신이 아닌 다른 화자가 말을 한다고 느껴지거나, 이전 질문과 다른 내용의 대답을 했다면 가차없이 0점을 주며, 0~4점 사이로 나와야 함. 어르신이 대화를 그만하자는 의사를 밝히면 0점을 주어야 함. 다른 표현 하지 않음."
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

    path = str(age) + location
    context_text = query_ensemble(answer, f"./data/{path}.txt")
    print("context: \n")
    for i in range(len(context_text)):
        print(context_text[i])

    response = with_message_history.invoke(
        {
            "input": answer, 
            "context": context_text,
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
        conversation_final()
        return jsonify({"message": "지금 대화가 어려우신가봐요. 대화를 종료하겠습니다.", "score": score}), 200
    
    return jsonify({"message": message, "score": score}), 200