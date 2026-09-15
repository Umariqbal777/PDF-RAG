# import os
# from dotenv import load_dotenv
# from groq import Groq

# load_dotenv()
# client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# completion = client.chat.completions.create(
#     model="llama-3.1-8b-instant",
#     messages=[{"role": "user", "content": "Hello"}]
# )
# print(completion.choices[0].message.content)
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Check if chroma_db directory exists
db_path = "chroma-db"
if not os.path.exists(db_path):
    print(f"❌ Error: The directory '{db_path}' does not exist in your project folder.")
    print("This means you haven't successfully saved documents to ChromaDB yet.")
    exit()

print(f"✅ Found '{db_path}' directory on disk.")

# 2. Initialize embeddings
print("Loading embedding model...")
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Load the vector store
print("Connecting to ChromaDB...")
vectorstore = Chroma(
    persist_directory=db_path,
    embedding_function=embeddings_model
)

# 4. Inspect collection size / stored items
try:
    # Fetch collection data safely
    collection = vectorstore._collection
    count = collection.count()
    print(f"📊 Total chunks stored in ChromaDB: {count}")

    if count == 0:
        print("⚠️ Warning: Your database is empty! No document chunks were found.")
    else:
        # 5. Run a test similarity search
        test_query = "agroscan ai"
        print(f"\n🔍 Running test similarity search for: '{test_query}'")
        
        results = vectorstore.similarity_search(test_query, k=2)
        print(f"Found {len(results)} matching chunks:\n")
        
        for idx, doc in enumerate(results):
            print(f"--- Result {idx+1} ---")
            print(doc.page_content[:400])
            print(f"Metadata: {doc.metadata}\n")

except Exception as e:
    print(f"❌ Error inspecting database: {e}")