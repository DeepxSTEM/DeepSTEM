from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

import os
from dotenv import load_dotenv

load_dotenv()

# Among the following costs free embeddings, HuggingFaceEmbeddings seems to perform better others
# Add your prefer Embeddings model here as you need

HFembeddings = HuggingFaceEmbeddings()
Olmembeddings = OllamaEmbeddings(model="nomic-embed-text")

def get_embedding_function(emb: str):
    if emb == "hf":
        return HFembeddings
    elif emb == "ollama":
        return Olmembeddings
    else:
        return HFembeddings #default to HuggingFaceEmbeddings if no other embedding model is specified