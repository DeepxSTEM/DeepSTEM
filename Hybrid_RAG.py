from langchain_google_genai import GoogleGenerativeAI
#import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
import os
import streamlit as st

#Import load PDF function and embedding function
from load_pdf import load_pdf
from get_embedding_func import get_embedding_function
from dotenv import load_dotenv

# take environment variables from .env
load_dotenv()  

CHROMA_PATH = "vdb"
embedding = "hf"

st.markdown("""
<style>
    .custom-border {
        border: 2px solid #007eff; /* Set border color and width */
        padding: 10px; /* Add some padding for spacing */
        border-radius: 5px; /* Add rounded corners */
    }
</style>
""", unsafe_allow_html=True)

db = Chroma(
            persist_directory=CHROMA_PATH, embedding_function=get_embedding_function(embedding)
        )
retriever = db.as_retriever()

ggllm = GoogleGenerativeAI(temperature=0, model="models/gemini-1.5-pro-latest", google_api_key=os.getenv('GOOGLE_API_KEY'))

Mllm = Ollama(temperature=0, model="mistral")
L2llm = Ollama(temperature=0, model="llama2")
Gllm = Ollama(temperature=0, model="gemma:7b")

NVllm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")

PROMPT_TEMPLATE = """
Read the context from the provided text carefully. 
{context}
-----------------------
Answer the question only based on the provided text. Please include the metadata 'id' of 
2 documents that are most closely match the question in the response. 
Don't missed out important information. Don't hallucinate and DON'T make up your own answer. 
Only provide answer to what is asked in the question. Don't add any additional information that
is not found in the provided text.
{question}
"""
PROMPT_TEMPLATE1 = """
Read the context from the provided text carefully. 
{context}
-----------------------
Answer the question only based on the provided text. Please include the metadata of 
the document in the response. 
Don't missed out important information. Don't hallucinate and DON'T make up your own answer. 
Only provide answer to what is asked in the question. Don't add any additional information that
is not found in the provided text.
{question}
"""

prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE)
prompt_template_1 = PromptTemplate.from_template(PROMPT_TEMPLATE1)

results = []
one_doc = []
chunk_ids = []

def print_chunks(results):
    j=1
    if results:
        for x in results:
            st.markdown(f'\nChunk {j}: {x}\n', unsafe_allow_html=True)
            j += 1

def find_chunk_id(resonseStr):
    chunk_id = "None"
   
    #st.write(f'Response Data: {resonseStr}')
    if resonseStr:
        if "'" in resonseStr:
            strlist = resonseStr.split("'")
            i=0
            for item in strlist:
                if len(item) == 32:
                    chunk_id = item.strip()
                    chunk_ids.append(chunk_id)
                    #st.write(f'Found chunk ID: {chunk_id}')
                    i += 1
                    if i == 2:
                        break
        elif "id:" in resonseStr:
            start_idx = resonseStr.lower().find("id:") + 5
            n_idx = start_idx + 32
            chunk_id = resonseStr[start_idx:n_idx].strip()
    else:
        pass
    return chunk_ids

def call_lead_llm(prompt):
    response = Mllm.invoke(prompt)

    chunk_id = find_chunk_id(response)
    
    if chunk_ids:
        for item in chunk_ids:
            for x in results:
                if item == x.metadata["id"]:
                    one_doc.append(x)
                    #st.write(f'find chunk: {x}')
    else:
        st.write(f'chunk id not found. Send all 4 chunks to LLMs')
            
st.header("Let's Chat with AI Assistant DeepSTEM")

query_text = st.text_input(label="Query Text", placeholder="Enter your Query here:", label_visibility="collapsed")
if query_text:
    
    results = db.similarity_search_by_vector(get_embedding_function(embedding).embed_query(query_text), k=4)
    
    prompt = prompt_template.format(context=results, question=query_text)

    call_lead_llm(prompt)

    if not one_doc:
        one_doc = results

    prompt2 = prompt_template_1.format(context=one_doc, question=query_text)

    st.subheader("LLMs Response:")
    
    response1 = Gllm.invoke(prompt2)
    st.write(f'Gemma:7b: {response1} \n')

    response2 = Mllm.invoke(prompt2)
    st.write(f'\nMistral: {response2} \n')

    response3 = L2llm.invoke(prompt2)
    st.write(f'\nLallma2: {response3}')

    response4 = NVllm.invoke(prompt2)
    st.write(f'\nMixtral-8x7b-instruct-v0.1: {response4}')

    response5 = ggllm.invoke(prompt2)
    st.write(f'gemini-1.5-pro-latest: {response5} \n')

    print_chunks(one_doc)


    