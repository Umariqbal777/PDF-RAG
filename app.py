import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="RAG Document Assistant", page_icon="🤖", layout="wide")

st.title("📄 RAG Assistant with Gemini & ChromaDB")
st.markdown("Upload your research paper or PDF in the sidebar, then ask questions about it below!")

# --- Sidebar: File Upload & Ingestion ---
st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Save uploaded file to a temporary location so PyPDFLoader can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    if st.sidebar.button("Process & Index PDF"):
        with st.spinner("Extracting text and building vector database..."):
            try:
                # Load and split document (from your database.py logic)
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = splitter.split_documents(docs)
                
                # Embeddings and Vectorstore creation
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory="chroma-db"
                )
                st.sidebar.success("✅ PDF successfully indexed into ChromaDB!")
            except Exception as e:
                st.sidebar.error(f"Error processing file: {e}")
            finally:
                os.unlink(tmp_path) # Clean up temp file

# --- Main Chat Interface ---
db_path = "chroma-db"

if os.path.exists(db_path):
    # Initialize components
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings_model
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
    )
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant.
Use only the provided context to answer the question.
If the answer is not present in the context, say: "i could not find the answer in the document"
"""),
        ("human", """Context:
{context}

Question: {question}
""")
    ])

    # Initialize chat history state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history messages on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input chat
    if query := st.chat_input("Ask a question about your uploaded document..."):
        # Add user message to state history
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching document and generating answer..."):
                docs = retriever.invoke(query)
                context = " ".join([doc.page_content for doc in docs])
                
                final_prompt = prompt_template.invoke({
                    "context": context,
                    "question": query
                })
                
                response = llm.invoke(final_prompt)
                
                # Safely parse text response content
                if isinstance(response.content, list):
                    answer = "".join([item.get("text", "") for item in response.content if isinstance(item, dict)])
                else:
                    answer = response.content
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("👈 Please upload and process a PDF file using the sidebar to begin chatting.")