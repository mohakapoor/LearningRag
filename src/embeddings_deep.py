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
        


if __name__ == "__main__":
    # basic_embeddings()
    batch_embeddings()