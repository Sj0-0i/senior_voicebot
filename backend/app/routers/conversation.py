import json
import copy
import datetime
from core import config
from fastapi import APIRouter, HTTPException
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import Document
from services.user_service import get_user_info, save_user_interests
from services.question_service import generate_question, mark_question
from services.weather_service import get_weather
from services.document_service import query_ensemble, save_chunks_to_file, init_file, update_summaries
from services.session_service import get_history, set_history, clear_history
from utils.utils import split_text
from utils.prompts import prompt1, prompt2, prompt3, prompt4, prompt5
from models.user import UserInput, AnswerInput


conversation_router = APIRouter()


model = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=config.openai_api_key)


# test data
name = "박호산"
age = 70
location = "Seoul"

async def decide_modifications(new_summary, similar_summaries):
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            prompt5,
        ),
        ("human", "{input}"),
        ]
    )

    messages = prompt.format_messages(input="",
                                    new_summary=new_summary,
                                    existing_summary=similar_summaries)
    
    response = model.invoke(messages)
    print(response.content)
    try:
        modifications = json.loads(response.content)
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}")
        modifications = {"summary_modifications": []}

    return modifications

async def update_memory_module(new_summaries, data_path):    
    existing_summaries = []
    
    for new_summary in new_summaries:
        with open(data_path, 'r', encoding='utf-8') as file:
            existing_summaries = file.readlines()
        
        print(f"existing summaries : {existing_summaries}")
    
        similar_summaries = await query_ensemble(new_summary.page_content, data_path, 5)
        unique_summaries = []
        seen_contents = set()
        
        print('--------------')
        for result in similar_summaries:
            if result.page_content not in seen_contents:
                unique_summaries.append(result.page_content)
                seen_contents.add(result.page_content)
        print(f"unique_summaries : {unique_summaries}")
        
        modifications = await decide_modifications(new_summary.page_content, unique_summaries)
        print(f"update 대상 요약문 : {new_summary.page_content}")
        existing_summaries = update_summaries(existing_summaries, modifications, data_path)

async def final(user_id):
    session_id = user_id
    original_history = get_history(session_id)
    history_copy = copy.deepcopy(original_history)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                prompt3,
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    messages = prompt.format_messages(input="", history=history_copy.messages, current_time=datetime.datetime.now().strftime('%Y-%m-%d'))
    response = model.invoke(messages)

    print("\n요약문 : ")
    print(response.content)

    cleaned_text = response.content.replace("{", "").replace("}", "")
    document = Document(page_content=cleaned_text)
    chunks = split_text([document])
    chunks.extend

    for chunk in chunks:
        print(chunk)

    await update_memory_module(chunks, f"./data/{age}{location}.txt")
    save_chunks_to_file(chunks, f"./data/{age}{location}.txt")

    prompt_ = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                prompt4,
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    messages = prompt_.format_messages(input="", history=history_copy.messages)
    response = model.invoke(messages)

    print("\n관심사 키워드 : ")
    print(response.content)
    cleaned = response.content.replace("{", "").replace("}", "")
    interests = json.loads(cleaned)
    await save_user_interests(user_id, interests)


@conversation_router.post('/first')
async def conversation_first(user_input: UserInput):
    user_id = user_input.user_id

    # jwt 로그인 구현 후 name, age, location 정보 저장 (session) 수정 필요
    try:
        user_info = await get_user_info(user_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    session_id = user_id
    clear_history(session_id)
    
    try:
        question = await generate_question(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        weather_info = await get_weather(user_info['location'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    init_file(age,location)
    
    question_id = question['question_id']
    question_text = question['question_text']

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                prompt1,
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    runnable = prompt | model
    with_message_history = RunnableWithMessageHistory(
        runnable, get_history, input_messages_key="input", history_messages_key="history"
    )

    response = with_message_history.invoke(
        {
            "age": user_info['age'], "name": user_info['name'], "location": user_info['location'],
            "weather": weather_info, "question": question_text, "input": ""
        },
        config={"configurable": {"session_id": session_id}},
    )

    return {"status": "success", "message": response.content, "question": question_id}



@conversation_router.post('/second')
async def conversation_second(answer_input: AnswerInput):
    user_id = answer_input.user_id
    answer = answer_input.answer

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                prompt2,
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    runnable2 = prompt | model 

    with_message_history = (
        RunnableWithMessageHistory(
            runnable2,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )
    )

    path = str(age) + location
    context_text = await query_ensemble(answer, f"./data/{path}.txt")
    print("context: ")
    for i in range(len(context_text)):
        print(context_text[i].page_content)

    response = with_message_history.invoke(
        {
            "input": answer, 
            "context": context_text,
        },
        config={"configurable": {"session_id": user_id}},
    )

    response_json = json.loads(response.content)
    message = response_json.get('message')
    score = response_json.get('score')

    print(message)
    print(score)

    if int(score) <= 8:
        print(f"Score가 {score}로 낮아서 대화를 종료합니다.")

        history = get_history(name + str(age) + location)
        if len(history.messages) >= 2:
            history.messages = history.messages[:-2]
        print(get_history(name + str(age) + location))

        await final(user_id)
        return {"status": "success", "message": "지금 대화가 어려우신가봐요. 대화를 종료하겠습니다.", "score": score}
    
    return {"status": "success", "message": message, "score": score}
