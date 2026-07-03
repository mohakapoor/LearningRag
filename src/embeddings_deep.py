# pyrefly: ignore [missing-import]
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
import numpy as np 
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2"
)

def basic_embeddings():

    text = "What is Machine Learning"
    single_embedding = embeddings_model.embed_query(text)
    print(f"Vector Dimensions: {len(single_embedding)}")
    print(f"First 5 values: {single_embedding[:5]}")
    print(f"Vector norm: {np.linalg.norm(single_embedding):.4f}")

def batch_embeddings():
    text = [
        "What is Machine Learning",
        "What is Deep Learning",
        "What is Artificial Intelligence",
    ]
    embeddings = embeddings_model.embed_documents(text)
    for i,emb in enumerate(embeddings):
        print(f"Text {i+1} : {text[i]}")
        print(f"Vector Dimensions: {len(emb)}")
        print(f"First 5 values: {emb[:5]}")
        print(f"Vector norm: {np.linalg.norm(emb):.4f}")
        

def cosine_similarity(v1,v2):
    return np.dot(v1,v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def similarity_search():
    
    #Documents
    docs = [
        "The Eiffel Tower is located in Paris, France.",
        "The Statue of Liberty is located in New York, USA.",
        "The Great Wall of China is located in China.",
        "The Colosseum is located in Rome, Italy.",
        "The Taj Mahal is located in Agra, India.",
    ]
    
    query = "Where is the Eiffel Tower located?"

    query_vector = embeddings_model.embed_query(query)

    doc_vectors = embeddings_model.embed_documents(docs)

    
    similarties = [cosine_similarity(query_vector,vec)for vec in doc_vectors]

    ranked_docs = sorted(zip(docs,similarties),key = lambda x : x[1],reverse = True)

    print(f"Query: {query}\n")
    print("Ranked by similarity:")
    for doc , score in ranked_docs:
        print(f"Score: {score:.4f}, Doc: {doc}")

if __name__ == "__main__":
    # basic_embeddings()
    # batch_embeddings()
    similarity_search()