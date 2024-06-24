from langchain_community.document_loaders import PyPDFDirectoryLoader
import os

pdf_folder_path = "pdf_files/"

def load_pdf():
    loader = PyPDFDirectoryLoader(pdf_folder_path)

    docs = loader.load()

    return docs

if __name__ == "__main__":
    print(os.listdir(pdf_folder_path))
    docs = load_pdf()

    for i in range(2):
        print(docs[i].page_content)