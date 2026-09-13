from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()
data=PyPDFLoader("documentsloaders/Final Research Paper.pdf")
docs=data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks=splitter.split_documents(docs)


template=ChatPromptTemplate.from_messages([
    ("system","yur are ai that summarizes the text"),
    ("human","{data}")
])
prompt=template.format_messages(data=docs[0])

model = ChatGroq(model_name="qwen/qwen3.8-27b",)


result = model.invoke(prompt)
print(result.content)