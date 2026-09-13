from langchain_community.document_loaders import PyPDFLoader
data=PyPDFLoader("documentsloaders/Umar_Iqbal_Resume.pdf")
docs=data.load()
print(len(docs))