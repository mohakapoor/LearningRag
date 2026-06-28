import chromadb

chroma_client = chromadb.Client()
 
collection_name = "test_collection"

collection = chroma_client.get_or_create_collection(name=collection_name)

print("Collection created successfully")
documents = [
    {"id": "doc1","text": "Hello World!"},
    {"id": "doc2","text": "My name is Mohak"},
    {"id": "doc3","text": "I am from Delhi"},
    {"id": "doc4","text": "I am a student"},
    {"id": "doc5","text": "I am a developer"},
]

for doc in documents:
    collection.upsert(ids = [doc["id"]],documents = [doc["text"]])

print("Documents added successfully")

query_text = "Hello World!"

results = collection.query(
    query_texts = [query_text],
    n_results = 3,
    )

print(results)