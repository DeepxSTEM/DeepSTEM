from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import NeMoEmbeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings


import os
from dotenv import load_dotenv

load_dotenv()

# Among the following costs free embeddings, HuggingFaceEmbeddings seems to perform best

HFembeddings = HuggingFaceEmbeddings()
Olmembeddings = OllamaEmbeddings(model="nomic-embed-text")
Nvdembeddings = NVIDIAEmbeddings(model="NV-Embed-QA", embed_documents="passage")

def get_embedding_function(emb: str):
    if emb == "hf":
        return HFembeddings
    elif emb == "ollama":
        return Olmembeddings
    elif emb == "nvd":
        return Nvdembeddings
    else:
        return HFembeddings