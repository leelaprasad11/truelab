from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from ingest import load_and_split

def create_vectorstore(pdf_path):
    chunks = load_and_split(pdf_path)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local("vectorstore")

    print("Vector store created successfully!")

if __name__ == "__main__":
    create_vectorstore("data/sample.pdf")
