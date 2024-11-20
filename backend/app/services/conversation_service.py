import json
import copy
import datetime
from services.user_service import get_user_info
from services.question_service import generate_question
from services.interest_service import save_user_interests, get_user_interest
from services.weather_service import get_weather
from services.document_service import query_ensemble, clear_user_cache, save_chunks_to_file, init_file, update_summaries
from services.session_service import get_history, set_history, clear_history
from utils.openai_model_manager import OpenAIModelManager
from utils.utils import split_text
from utils.prompts import prompt_morning, prompt_question, prompt_interest, prompt_night, prompt2, prompt3, prompt4, prompt5
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import Document

from typing import List, Dict


stored_summaries: Dict[str, List[str]] = {}


async def fetch_user(user_id):
    user_info = await get_user_info(user_id)
    return user_info


async def fetch_user_context(user_id):
    user_info = await get_user_info(user_id)
    weather_info = await get_weather(user_info['location'])
    return user_info, weather_info


async def process_first_conversation(user_input):
    user_id = user_input.user_id
    model = OpenAIModelManager.get_model(user_id)
    user_info, weather_info = await fetch_user_context(user_id)
    clear_history(user_id)
    init_file(user_info['age'], user_info['location'])

    # current_hour = datetime.now().hour
    current_hour = 20

    if 7 <= current_hour <= 10:
        response = await execute_conversation(prompt_morning, model, user_info, weather_info, user_id)

    if 10 < current_hour <= 19:
        question = await generate_question(user_id)
        if question is None:
            response, interest_id = await handle_no_question_case(user_info, model, user_id)
            response_json = json.loads(response.content)
            return {"message": response_json.get('message'), "interest_id": interest_id}

        response = await handle_question_case(user_info, model, user_id, question)
        response_json = json.loads(response.content)
        return {"message": response_json.get('message'), "question_id": question["question_id"]}
    
    if 19 < current_hour <= 22:
        response = await execute_conversation(prompt_night, model, user_info, weather_info, user_id)
    
    response_json = json.loads(response.content)
    return {"message": response_json.get('message')}


async def execute_conversation(prompt_template, model, user_info, weather_info, session_id, **kwargs):
    prompt = ChatPromptTemplate.from_messages(
        [("system", prompt_template), MessagesPlaceholder(variable_name="history"), ("human", "{input}")]
    )

    runnable = prompt | model
    with_message_history = RunnableWithMessageHistory(
        runnable, get_history, input_messages_key="input", history_messages_key="history"
    )

    return with_message_history.invoke(
        {
            "age": user_info['age'], "name": user_info['name'],
            "location": user_info['location'], "weather": weather_info,
            "input": "",
            **kwargs
        },
        config={"configurable": {"session_id": session_id}}
    )


async def handle_no_question_case(user_info, model, user_id):
    interest = await get_user_interest(user_id)
    print(f"interest : {interest['interest']}")
    path = f"{user_info['age']}{user_info['location']}"
    context_text = await query_ensemble(user_id, interest['interest'], f"./data/{path}.txt")

    print("context: ")
    for context in context_text:
        print(context.page_content)

    response = await execute_conversation(
        prompt_interest, model, user_info, None, user_id, 
        interest=interest, context=context_text
    )
    return response, interest['interest_id']
    # 대화가 잘 진행되면 mark_interest 해야함


async def handle_question_case(user_info, model, user_id, question):
    response = await execute_conversation(
        prompt_question, model, user_info, None, user_id, question=question['question_text']
    )
    return response
    # 대화가 잘 진행되면 mark_question 해야함


async def process_second_conversation(answer_input, background_tasks):
    user_id = answer_input.user_id
    user_info = await fetch_user(user_id)
    answer = answer_input.answer
    model = OpenAIModelManager.get_model(user_id)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt2),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    runnable2 = prompt | model
    with_message_history = RunnableWithMessageHistory(
        runnable2,
        get_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    
    path = f"{user_info['age']}{user_info['location']}"
    context_text = await query_ensemble(user_id, answer, f"./data/{path}.txt")

    print("context: ")
    for i in range(len(context_text)):
        print(context_text[i].page_content)

    response = with_message_history.invoke(
        {"input": answer, "context": context_text},
        config={"configurable": {"session_id": user_id}},
    )

    response_json = json.loads(response.content)
    message = response_json.get('message')
    score = response_json.get('score')
    reference = response_json.get('reference')
    ref_num = response_json.get('ref_num')
       

    print(message)
    print(score)
    print(reference)
    print(ref_num)

    if int(score):
        print(f"score: {score}")

        history = get_history(user_id)
        if len(history.messages) >= 2:
            history.messages = history.messages[:-1]
        print(get_history(user_id))

        background_tasks.add_task(finalize_conversation, user_id)
        return {"message": message, "score": score}
    
    return {"message": message, "score": score, "reference" : reference, "ref_num" : ref_num}


async def finalize_conversation(user_id):
    session_id = user_id
    history_copy = copy.deepcopy(get_history(session_id))
    user_info = await fetch_user(user_id)
    path = f"{user_info['age']}{user_info['location']}" 
    summary = await generate_summary(user_id, history_copy)
    await update_memory_module(summary, f"./data/{path}.txt", user_id)
    save_chunks_to_file(summary, f"./data/{path}.txt")

    interests = await extract_user_interests(user_id, history_copy)
    await save_user_interests(user_id, interests)

    ## demo 위한 코드
    stored_summaries[user_id] = [chunk.page_content for chunk in summary]

    clear_user_cache(user_id)

## demo 위한 코드
def get_summary(user_id: str):
    return stored_summaries.get(user_id, [])


async def generate_summary(user_id, history):
    model = OpenAIModelManager.get_model(user_id)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt3),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    messages = prompt.format_messages(input="", history=history.messages, 
                                      current_time=datetime.datetime.now().strftime('%Y-%m-%d'))
    response = model.invoke(messages)
    print("\n요약문 : ")
    print(response.content)

    cleaned_text = response.content.replace("{", "").replace("}", "").replace('\\"', '"')
    document = Document(page_content=cleaned_text)
    chunks = split_text([document])

    return chunks


async def extract_user_interests(user_id, history):
    model = OpenAIModelManager.get_model(user_id)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt4),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    messages = prompt.format_messages(input="", history=history.messages)
    response = model.invoke(messages)
    print(f"사용자의 관심사 키워드 : {response.content}")
    cleaned = response.content.replace("{", "").replace("}", "")

    return json.loads(cleaned)


async def update_memory_module(new_summaries, data_path, user_id):
    existing_summaries = []
    
    for new_summary in new_summaries:
        with open(data_path, 'r', encoding='utf-8') as file:
            existing_summaries = file.readlines()
        
        print(f"existing summaries : {existing_summaries}")
    
        similar_summaries = await query_ensemble(user_id, new_summary.page_content, data_path, 5)
        unique_summaries = []
        seen_contents = set()
        
        print('--------------')
        for result in similar_summaries:
            if result.page_content not in seen_contents:
                unique_summaries.append(result.page_content)
                seen_contents.add(result.page_content)
        print(f"unique_summaries : {unique_summaries}")
        
        modifications = await decide_modifications(new_summary.page_content, unique_summaries, user_id)
        print(f"update 대상 요약문 : {new_summary.page_content}")
        existing_summaries = update_summaries(existing_summaries, modifications, data_path)


async def decide_modifications(new_summary, similar_summaries, user_id):
    model = OpenAIModelManager.get_model(user_id)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt5,),
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