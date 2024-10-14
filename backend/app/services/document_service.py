from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_openai import OpenAIEmbeddings
from utils.utils import split_text
import os


async def query_ensemble(query_text, data_path, top_k = 2):
    document_loader = TextLoader(data_path, encoding='UTF8')
    pages = document_loader.load()
    docs = split_text(pages)

    if len(docs) < top_k:
        return []
   
    embeddings = OpenAIEmbeddings()
    
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = top_k

    chroma_vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./db/chroma")
    chroma_retriever = chroma_vectorstore.as_retriever(search_kwargs={'k': top_k})
        
    try:
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever],
            weights=[0.5, 0.5],
        )
        final_results = ensemble_retriever.invoke(query_text)

        return final_results

    except ValueError as e:
        print(f"{e}")
        return []

def save_chunks_to_file(chunks, file_path):
    with open(file_path, 'a', encoding='utf-8') as file:
        for chunk in chunks:
            content = chunk.page_content
            file.write(content + '\n')  
            
def init_file(age, location):
    path = str(age) + location
    if not os.path.exists(f"./data"):
        os.makedirs("./data")
        
    if not os.path.exists(f"./data/{path}.txt"):
        with open(f"./data/{path}.txt", 'w'):
            pass
        print("new user file is made")

def update_summaries(origin_summaries, modifications, data_path):
    modified = False;
    
    for modification in modifications["summary_modifications"]:
        modified = True
        origin_sentence = modification["origin_sentence"] + '\n'
        action = modification["action"]
        
        if action == "keep":
            continue
        elif action == "update":
            new_content = modification["modification"]
            if origin_sentence in origin_summaries:
                idx = origin_summaries.index(origin_sentence)
                origin_summaries[idx] = new_content + "\n"
                print("updated : " + origin_summaries[idx])
        elif action == "delete":
            if origin_sentence in origin_summaries:
                print("deleted " + origin_sentence)
                origin_summaries.remove(origin_sentence)

    if modified:
        with open(data_path, 'w', encoding='utf-8') as file:
            for line in origin_summaries:
                if not line.endswith('\n'):
                    line += '\n'
                file.write(line.strip('"'))