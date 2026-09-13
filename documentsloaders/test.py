from langchain_community.document_loaders import TextLoader
data=TextLoader("documentsloaders/notes.txt")
docs=data.load()
print(docs[0])