# from langchain_text_splitters import TokenTextSplitter
# from langchain_community.document_loaders import TextLoader

# splitter=TokenTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=10
# )
# data= TextLoader("documentsloaders/Final Research Paper.pdf")
# docs=data.load()
# chunks=splitter.split_documents(docs)
# print(len(chunks))
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size=1000,
    chunk_overlap=10
)

# Use PyPDFLoader instead of TextLoader
data = PyPDFLoader("documentsloaders/Final Research Paper.pdf")
docs = data.load()
chunks = splitter.split_documents(docs)
print(len(chunks))