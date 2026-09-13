from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

splitter=CharacterTextSplitter(
    separator="",
    chunk_size=100,
    chunk_overlap=1
)
data= TextLoader("documentsloaders/notes.txt")
docs=data.load()
chunks=splitter.split_documents(docs)
print(chunks[0].page_content)