from langchain_chroma import Chroma
from langchain.schema.document import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from load_pdf import load_pdf
from get_embedding_func import get_embedding_function
import hashlib
import streamlit as st

vectorDB_PATH = "vdb"
chunk_size = 1000
chunk_overlap = 100
embedding = "hf"

def split_docs(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ['\n\n', '\n', '.'],
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function=len,
    )

    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=vectorDB_PATH, embedding_function=get_embedding_function(embedding)
    )

    # Check if chunk already exist or not. If not, add to vector database.
    existing_items = db.get(include=[]) 
    existing_ids = existing_items["ids"]
    
    if len(existing_items['ids']) > 0:
        existing_ids = existing_items["ids"]
        with st.sidebar:
            st.write(f"Number of existing documents in DB: {len(existing_ids)}")

    else:
        with st.sidebar:
            st.write("NO existing items in database")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    
    # Hash each chunk and use the hash value as chunk id to identify the chunk
    for chunk in chunks:
        new_chunk = f"{chunk}".encode()
        
        # Create a new hash object for each chunk
        md5_hash = hashlib.md5()
        md5_hash.update(new_chunk)

        # Append the hash of the chunk to the list
        hash_chunk = md5_hash.hexdigest()
        
        if hash_chunk not in existing_ids:
            new_chunks.append(chunk)
            chunk_id = hash_chunk

            # Add it to the page meta-data.
            chunk.metadata["id"] = chunk_id

    if len(new_chunks):
        with st.sidebar:
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
            st.write(f"✅ Added new documents: {len(new_chunks)}")
        
    else:
        with st.sidebar:
            st.write("👉 No new documents added")

def createDB():
    docs = load_pdf()
    docs = split_docs(docs)

    db = Chroma(
                persist_directory=vectorDB_PATH, embedding_function=get_embedding_function(embedding)
            )
    #add documents to vector database
    add_to_chroma(docs)

if __name__=="__main__":
    createDB()
