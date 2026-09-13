from langchain_community.document_loaders import WebBaseLoader
data=WebBaseLoader("https://leetcode.com/u/umariqbal/")
docs=data.load()
print(docs[0].page_content)