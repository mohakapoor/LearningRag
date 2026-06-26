import os
from dotenv import load_dotenv
load_dotenv()


from importlib.metadata import version

lg_version = version("langgraph")
core_version = version('langchain-core')


print(f"Langchain-core Version: {core_version}")

print(f"Langgraph version: {lg_version}")


from google import genai

def main():
    client = genai.Client(api_key=os.getenv("gemini_api_key"))
    # response = client.models.generate_content(model = "gemini-2.5-flash", contents = "What is the capital of Iceland",config = genai.types.GenerateContentConfig(system_instruction="You are a helpful assistant "))
    # print(response.text)

    response = client.models.embed_content(
        contents= "Your text string goes here",model = "gemini-embedding-2"
    )
    print(response)
    print(len(response.embeddings[0].values))

if __name__ == "__main__":
    main()
