# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_core.prompts import ChatPromptTemplate
# load_dotenv()

# embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# vectorstore=Chroma(
#     persist_directory="chroma_db",
#     embedding_function=embeddings_model
# )

# retriever=vectorstore.as_retriever(
#     search_type="mmr",
#     search_kwargs={
#         "k":4,
#         "fetch_k":10,
#         "lambda_mult":0.5
#     }
# )
# llm=ChatGoogleGenerativeAI(model="gemini-3.6-flash",temperature=0.3)

# prompt=ChatPromptTemplate.from_messages(
#     [
#         ("system","""you are a helpful AI assistant.
#         use only the provided context to answer the question.
#         if the answer is not present in the context,
#         say:"i could not find the answer in the document"
#         """),
#         ("human","""Context:{context}
#         Question:{question}
#         """)

#     ]
# )
# print("Rag System Created ")
# print("Press 0 to exit ")
# while True:
#     query=input("you:")
#     if query=="0":
#         break
#     docs=retriever.invoke(query)
#     context="".join([doc.page_content for doc in docs])
#     final_prompt=prompt.invoke({
#         "context":context,
#         "question":query
#     })
#     response=llm.invoke(final_prompt)
#     print(f"\n AI:{response.content}")
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma  # Updated standalone import to fix deprecation warning
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory="chroma-db",
    embedding_function=embeddings_model
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

# Use active model string
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a helpful AI assistant.
Use only the provided context to answer the question.
If the answer is not present in the context, say: "i could not find the answer in the document"
"""),
        ("human", """Context:
{context}

Question: {question}
""")
    ]
)

print("Rag System Created ")
print("Press 0 to exit ")

while True:
    query = input("\nyou: ")
    if query == "0":
        break
    
    docs = retriever.invoke(query)
    
    # Add spacing between chunks so words don't merge together
    context = " ".join([doc.page_content for doc in docs])
    
    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })
    
    response = llm.invoke(final_prompt)
    
    # Safely extract text whether content is a string or structured list blocks
    if isinstance(response.content, list):
        answer = "".join([item.get("text", "") for item in response.content if isinstance(item, dict)])
    else:
        answer = response.content

    print(f"\nAI: {answer}")