from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

template=ChatPromptTemplate.from_messages([
    ("system","yur are ai that summarizes the text"),
    ("human","{data}")
])
model = ChatGroq(model_name="qwen/qwen3.8-27b",)


