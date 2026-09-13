from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()
from langchain_core.documents import Document
docs = [
    Document(
        page_content="Retrieval-Augmented Generation enhances LLM outputs with external factual sources.",
        metadata={"source": "rag_intro", "topic": "architecture"},
    ),
    Document(
        page_content="Vector databases store high-dimensional embeddings to enable low-latency semantic search.",
        metadata={"source": "vector_db", "topic": "storage"},
    ),
    Document(
        page_content="Hugging Face sentence-transformers map text strings into dense numerical vectors.",
        metadata={"source": "embeddings_guide", "topic": "models"},
    ),
]
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore=Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="chroma-db"
)
result=vectorstore.similarity_search("what is RAG",k=2)
for r in result:
    print(r.page_content)